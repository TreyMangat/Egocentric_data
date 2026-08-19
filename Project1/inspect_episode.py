from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _xyz(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2 and array.shape[1] >= 3:
        return array[:, :3]
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        return array[:, :3, 3]
    raise ValueError(f"Cannot extract XYZ positions from shape {array.shape}")


def _plot_wrist_trajectories(
    measurements: dict[str, np.ndarray], annotations: list[dict], destination: Path
) -> None:
    times = measurements["time_seconds"]
    left = _xyz(measurements["left.obs_wrist_pose"])
    right = _xyz(measurements["right.obs_wrist_pose"])
    colors = ("#0072B2", "#E69F00", "#009E73")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, values, side in zip(axes, (left, right), ("Left", "Right")):
        for index, (coordinate, color) in enumerate(zip("XYZ", colors)):
            axis.plot(times, values[:, index], color=color, label=coordinate, linewidth=1)
        for segment in annotations[1:]:
            axis.axvline(segment["start_seconds"], color="0.8", linewidth=0.5)
        axis.set_ylabel(f"{side} wrist\nposition")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", ncol=3)

    axes[-1].set_xlabel("Seconds from the first RGB frame")
    figure.suptitle("Wrist trajectories; gray lines are annotation boundaries")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _write_report(
    episode_dir: Path, metadata: dict, annotations: list[dict], destination: Path
) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{segment['segment_index']}</td>"
        f"<td>{segment['start_seconds']:.2f}</td>"
        f"<td>{segment['end_seconds']:.2f}</td>"
        f"<td>{html.escape(segment['label'])}</td>"
        "</tr>"
        for segment in annotations
    )
    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Stage 0 — {html.escape(metadata['episode_hash'])}</title>
<style>
body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1050px; margin: 2rem auto; padding: 0 1rem; }}
video, img {{ width: 100%; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: .5rem; border-bottom: 1px solid #ddd; text-align: left; }}
.facts {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
</style>
<h1>Stage 0: one complete EgoVerse episode</h1>
<div class="facts">
  <p><strong>Task</strong><br>{html.escape(metadata['task'])}</p>
  <p><strong>Frames</strong><br>{metadata['frames']:,}</p>
  <p><strong>FPS</strong><br>{metadata['fps']:g}</p>
  <p><strong>Annotations</strong><br>{metadata['annotation_segments']}</p>
</div>
<video controls preload="metadata" src="rgb.mp4"></video>
<h2>Left and right wrist positions</h2>
<img src="wrist_trajectory.png" alt="Wrist XYZ positions over time">
<h2>Timestamped annotations</h2>
<table><thead><tr><th>#</th><th>Start (s)</th><th>End (s)</th><th>Raw label</th></tr></thead>
<tbody>{rows}</tbody></table>
</html>
"""
    destination.write_text(document, encoding="utf-8")


def inspect_episode(episode_dir: Path, open_report: bool) -> None:
    required = ("rgb.mp4", "measurements.npz", "annotations.json", "metadata.json")
    missing = [name for name in required if not (episode_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {episode_dir}: {', '.join(missing)}")

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    annotation_document = json.loads((episode_dir / "annotations.json").read_text())
    annotations = annotation_document["segments"]
    with np.load(episode_dir / "measurements.npz") as archive:
        measurements = {key: archive[key] for key in archive.files}

    _plot_wrist_trajectories(
        measurements, annotations, episode_dir / "wrist_trajectory.png"
    )
    report_path = episode_dir / "episode_report.html"
    _write_report(episode_dir, metadata, annotations, report_path)

    print(f"Episode: {metadata['episode_hash']}")
    print(f"Task: {metadata['task']}")
    print(
        f"Video: {metadata['frames']} frames, {metadata['fps']:g} FPS, "
        f"{metadata['duration_seconds']:.2f} seconds"
    )
    print(f"Annotations: {len(annotations)} timestamped segments")
    print("Measurements:")
    for name, value in measurements.items():
        print(f"  {name}: {list(value.shape)} {value.dtype}")
    print(f"Report: {report_path.resolve()}")

    if open_report:
        webbrowser.open(report_path.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one prepared EgoVerse Stage 0 episode."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument(
        "--open", action="store_true", help="Open the generated HTML report in a browser."
    )
    arguments = parser.parse_args()
    inspect_episode(arguments.episode_dir, arguments.open)


if __name__ == "__main__":
    main()
