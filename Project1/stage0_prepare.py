from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import modal


APP_NAME = "egoverse-action-recognition-stage0"
VOLUME_NAME = "egoverse-zarrs-v2"
VOLUME_ROOT = Path("/egoverse")
OUTPUT_ROOT = VOLUME_ROOT / "projects" / "action-recognition" / "stage0" / "episodes"
SEGMENTS_PATH = VOLUME_ROOT / "metadata" / "segments.parquet"

MEASUREMENT_KEYS = (
    "obs_head_pose",
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "left.obs_keypoints",
    "right.obs_keypoints",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "duckdb==1.4.3",
    "imageio-ffmpeg==0.6.0",
    "numpy==2.3.2",
    "zarr==3.1.3",
)


def _download_video(episode_hash: str, destination: Path) -> None:
    url = (
        "https://partners.mecka.ai/api/egoverse/uploads/"
        f"{episode_hash}/video?redirect=1"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "EgoVerse-Stage0/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.headers.get_content_type() != "video/mp4":
            raise ValueError(
                f"Expected video/mp4 for {episode_hash}, got "
                f"{response.headers.get_content_type()}"
            )
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _load_annotations(episode_hash: str) -> tuple[list[dict], dict]:
    import duckdb

    query = """
        SELECT operator, lab, scene, task, label, start_seconds, end_seconds
        FROM read_parquet(?)
        WHERE episode_hash = ?
        ORDER BY start_seconds
    """
    rows = duckdb.connect().execute(query, [str(SEGMENTS_PATH), episode_hash]).fetchall()
    if not rows:
        raise ValueError(f"No annotations found for {episode_hash}")

    annotations = [
        {
            "segment_index": index,
            "start_seconds": round(float(row[5]), 6),
            "end_seconds": round(float(row[6]), 6),
            "label": row[4],
        }
        for index, row in enumerate(rows)
    ]
    episode_fields = {
        "operator": rows[0][0],
        "lab": rows[0][1],
        "scene": rows[0][2],
        "task": rows[0][3],
    }
    return annotations, episode_fields


def _valid_fraction(array) -> float:
    import numpy as np

    flat = np.asarray(array).reshape(len(array), -1)
    valid = np.isfinite(flat).all(axis=1) & (np.abs(flat) < 1e8).all(axis=1)
    return float(valid.mean())


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=2,
    memory=4096,
    timeout=20 * 60,
)
def prepare_episode(episode_hash: str) -> dict:
    import imageio_ffmpeg
    import numpy as np
    import zarr

    episode_path = VOLUME_ROOT / "episodes" / episode_hash
    if not episode_path.is_dir():
        raise FileNotFoundError(f"Episode not found in Modal Volume: {episode_hash}")

    group = zarr.open_group(str(episode_path), mode="r")
    missing = [key for key in ("annotations", *MEASUREMENT_KEYS) if key not in group]
    if missing:
        raise ValueError(f"{episode_hash} is missing required arrays: {missing}")

    attributes = dict(group.attrs)
    fps = float(attributes["fps"])
    total_frames = int(attributes["total_frames"])
    arrays = {key: np.asarray(group[key][:]) for key in MEASUREMENT_KEYS}
    wrong_lengths = {key: len(value) for key, value in arrays.items() if len(value) != total_frames}
    if wrong_lengths:
        raise ValueError(
            f"{episode_hash} measurement lengths do not equal {total_frames}: {wrong_lengths}"
        )

    annotations, episode_fields = _load_annotations(episode_hash)
    source_annotation_count = int(group["annotations"].shape[0])

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        video_path = temporary_path / "rgb.mp4"
        measurements_path = temporary_path / "measurements.npz"
        annotations_path = temporary_path / "annotations.json"
        metadata_path = temporary_path / "metadata.json"

        _download_video(episode_hash, video_path)
        video_frames, video_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
        video_frames = int(video_frames)
        if video_frames != total_frames:
            raise ValueError(
                f"{episode_hash} video has {video_frames} frames but measurements have "
                f"{total_frames}; refusing to claim they are aligned"
            )

        reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
        video_info = next(reader)
        reader.close()
        video_fps = float(video_info["fps"])
        if abs(video_fps - fps) > 0.01:
            raise ValueError(
                f"{episode_hash} video FPS {video_fps} differs from measurement FPS {fps}"
            )

        time_seconds = np.arange(total_frames, dtype=np.float64) / fps
        np.savez_compressed(
            measurements_path,
            frame_index=np.arange(total_frames, dtype=np.int32),
            time_seconds=time_seconds,
            **arrays,
        )
        annotations_path.write_text(
            json.dumps(
                {
                    "episode_hash": episode_hash,
                    "time_base": "seconds from the first RGB frame",
                    "segments": annotations,
                },
                indent=2,
            )
            + "\n"
        )

        annotation_end = float(annotations[-1]["end_seconds"])
        expected_duration = total_frames / fps
        if abs(annotation_end - expected_duration) > 1.0:
            raise ValueError(
                f"{episode_hash} annotations end at {annotation_end:.3f}s but the "
                f"episode duration is {expected_duration:.3f}s"
            )

        metadata = {
            "episode_hash": episode_hash,
            **episode_fields,
            "fps": fps,
            "frames": total_frames,
            "duration_seconds": round(expected_duration, 6),
            "video_duration_reported_seconds": round(float(video_duration), 6),
            "frame_size": list(video_info["size"]),
            "annotation_segments": len(annotations),
            "source_zarr_annotation_segments": source_annotation_count,
            "annotation_end_seconds": annotation_end,
            "annotation_source": "metadata/segments.parquet",
            "annotation_version_note": (
                "The embedded Zarr annotation count may differ from the current "
                "segments table. This bundle uses the current segments table and "
                "records both counts instead of treating them as interchangeable."
            ),
            "time_alignment": (
                "RGB frame i, row i of every measurement array, and time_seconds[i] "
                "refer to the same instant; timestamps are derived as i / fps because "
                "this source does not contain an RGB timestamp array."
            ),
            "pose_note": (
                "Use the first three values of each *_pose row as XYZ position; retain "
                "the full row so orientation is available for later projects."
            ),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "valid_row_fraction": round(_valid_fraction(value), 6),
                }
                for key, value in arrays.items()
            },
            "files": {
                "video": "rgb.mp4",
                "measurements": "measurements.npz",
                "annotations": "annotations.json",
            },
            "source": {
                "dataset": "EgoVerse",
                "explorer": "https://partners.mecka.ai/egoverse",
                "license": "CC BY-SA 4.0",
            },
            "sha256": {
                "rgb.mp4": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                "measurements.npz": hashlib.sha256(
                    measurements_path.read_bytes()
                ).hexdigest(),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        destination = OUTPUT_ROOT / episode_hash
        destination.mkdir(parents=True, exist_ok=True)
        for source in (video_path, measurements_path, annotations_path, metadata_path):
            shutil.copy2(source, destination / source.name)

    volume.commit()
    return metadata


@app.local_entrypoint()
def main(episode_hash: str = ""):
    manifest_path = Path(__file__).with_name("stage0_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    selected = [entry["episode_hash"] for entry in manifest["episodes"]]
    if episode_hash:
        if episode_hash not in selected:
            raise ValueError(f"{episode_hash} is not in stage0_manifest.json")
        selected = [episode_hash]

    results = list(prepare_episode.map(selected, order_outputs=True))
    summary = [
        {
            "episode_hash": result["episode_hash"],
            "frames": result["frames"],
            "duration_seconds": result["duration_seconds"],
            "annotation_segments": result["annotation_segments"],
        }
        for result in results
    ]
    print(json.dumps(summary, indent=2))
