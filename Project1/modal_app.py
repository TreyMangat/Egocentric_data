from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import modal


APP_NAME = "egoverse-folding-clips"
VOLUME_NAME = "egoverse-zarrs-v2"
VOLUME_ROOT = Path("/egoverse")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "imageio==2.37.0",
    "imageio-ffmpeg==0.6.0",
    "numpy==2.3.2",
    "pillow==11.3.0",
    "zarr==3.1.3",
)

NUMERIC_KEYS = (
    "obs_head_pose",
    "obs_rgb_timestamps_ns",
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "left.obs_keypoints",
    "right.obs_keypoints",
)

REQUIRED_KEYS = ("images.front_1", *NUMERIC_KEYS)


def _valid_rows(array):
    import numpy as np

    flat = np.asarray(array).reshape(len(array), -1)
    return np.isfinite(flat).all(axis=1) & (np.abs(flat) < 1e8).all(axis=1)


def _choose_motion_window(arrays, total_frames: int, fps: int, seconds: int):
    import numpy as np

    window = min(seconds * fps, total_frames)
    if window < 1:
        raise ValueError("Episode contains no frames")

    valid = np.ones(total_frames, dtype=bool)
    for key in (
        "left.obs_ee_pose",
        "right.obs_ee_pose",
        "left.obs_keypoints",
        "right.obs_keypoints",
    ):
        valid &= _valid_rows(arrays[key][:total_frames])

    motion = np.zeros(total_frames, dtype=np.float64)
    for key in ("left.obs_ee_pose", "right.obs_ee_pose"):
        positions = arrays[key][:total_frames, :3]
        step_valid = valid[1:] & valid[:-1]
        delta = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        motion[1:] += np.where(step_valid, delta, 0.0)

    margin = min(3 * fps, max(0, (total_frames - window) // 2))
    starts = np.arange(margin, total_frames - window - margin + 1)
    if starts.size == 0:
        starts = np.arange(0, total_frames - window + 1)

    motion_prefix = np.concatenate(([0.0], np.cumsum(motion)))
    valid_prefix = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    ends = starts + window
    motion_scores = motion_prefix[ends] - motion_prefix[starts]
    valid_fractions = (valid_prefix[ends] - valid_prefix[starts]) / window

    nearly_complete = valid_fractions >= 0.99
    if nearly_complete.any():
        candidate_indexes = np.flatnonzero(nearly_complete)
        best_index = candidate_indexes[np.argmax(motion_scores[nearly_complete])]
    else:
        best_validity = valid_fractions.max()
        candidate_indexes = np.flatnonzero(valid_fractions == best_validity)
        best_index = candidate_indexes[np.argmax(motion_scores[candidate_indexes])]

    start = int(starts[best_index])
    end = start + window
    return start, end, float(valid_fractions[best_index]), float(motion_scores[best_index])


def _jpeg_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=4,
    memory=8192,
    timeout=30 * 60,
)
def build_clips(episode_hashes: list[str], duration_seconds: int = 8) -> list[dict]:
    import imageio.v2 as imageio
    import numpy as np
    import zarr
    from PIL import Image

    results = []
    output_root = VOLUME_ROOT / "projects" / "folding-clothes-clips"
    output_root.mkdir(parents=True, exist_ok=True)

    for episode_hash in episode_hashes:
        episode_path = VOLUME_ROOT / "episodes" / episode_hash
        if not episode_path.is_dir():
            raise FileNotFoundError(f"Episode not found: {episode_hash}")

        group = zarr.open_group(str(episode_path), mode="r")
        missing = [key for key in REQUIRED_KEYS if key not in group]
        if missing:
            raise ValueError(f"{episode_hash} is missing required keys: {missing}")

        attributes = dict(group.attrs)
        total_frames = int(attributes["total_frames"])
        fps = int(attributes["fps"])
        intrinsics = attributes.get("intrinsics", {}).get("front_1")
        if np.asarray(intrinsics).shape != (3, 4):
            raise ValueError(f"{episode_hash} has invalid front-camera intrinsics")

        arrays = {key: np.asarray(group[key][:total_frames]) for key in NUMERIC_KEYS}
        wrong_lengths = {
            key: len(value) for key, value in arrays.items() if len(value) < total_frames
        }
        rgb_frames = int(group["images.front_1"].shape[0])
        if rgb_frames < total_frames:
            wrong_lengths["images.front_1"] = rgb_frames
        if wrong_lengths:
            raise ValueError(
                f"{episode_hash} arrays are not frame-aligned: {wrong_lengths}; "
                f"expected {total_frames}"
            )

        start, end, valid_fraction, motion_score = _choose_motion_window(
            arrays, total_frames, fps, duration_seconds
        )
        clip_id = f"{episode_hash}_f{start:06d}-{end:06d}"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "rgb.mp4"
            metrics_path = temporary_path / "metrics.npz"
            metadata_path = temporary_path / "metadata.json"

            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=16,
            )
            try:
                for encoded_frame in group["images.front_1"][start:end]:
                    frame = np.asarray(
                        Image.open(io.BytesIO(_jpeg_bytes(encoded_frame))).convert("RGB")
                    )
                    writer.append_data(frame)
            finally:
                writer.close()

            clip_arrays = {key: value[start:end] for key, value in arrays.items()}
            np.savez_compressed(
                metrics_path,
                frame_indices=np.arange(start, end),
                intrinsics_front_1=np.asarray(intrinsics),
                **clip_arrays,
            )

            invalid_fractions = {
                key: round(float(1.0 - _valid_rows(value[start:end]).mean()), 6)
                for key, value in arrays.items()
                if value.ndim > 1
            }
            metadata = {
                "clip_id": clip_id,
                "episode_hash": episode_hash,
                "task_name": attributes.get("task_name"),
                "task_description": attributes.get("task_description"),
                "fps": fps,
                "frame_size": [480, 640],
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_seconds": round(start / fps, 3),
                "end_seconds": round(end / fps, 3),
                "duration_seconds": round((end - start) / fps, 3),
                "valid_hand_fraction": round(valid_fraction, 6),
                "hand_motion_score": round(motion_score, 6),
                "invalid_fractions": invalid_fractions,
                "pose_layout": "XYZ position followed by WXYZ quaternion",
                "keypoint_layout": "21 MANO landmarks flattened as XYZ values",
                "files": {"rgb": "rgb.mp4", "measurements": "metrics.npz"},
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

            destination = output_root / clip_id
            destination.mkdir(parents=True, exist_ok=True)
            for source in (video_path, metrics_path, metadata_path):
                shutil.copy2(source, destination / source.name)

        results.append(metadata)

    volume.commit()
    return results


@app.local_entrypoint()
def main(duration_seconds: int = 8):
    manifest_path = Path(__file__).with_name("episode_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    episode_hashes = [entry["episode_hash"] for entry in manifest["episodes"]]
    results = build_clips.remote(episode_hashes, duration_seconds)
    print(json.dumps(results, indent=2))
