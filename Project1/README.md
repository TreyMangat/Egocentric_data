# Project 1: Folding-Clothes Clips

This project starts with three eight-second EgoVerse clips containing synchronized first-person RGB and hand-tracking data.

## Setup

From this folder in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
modal run modal_app.py
```

The Modal job validates the source episodes, selects active eight-second windows, and stores the results in the `egoverse-zarrs-v2` Volume.

To copy the generated bundles into the local project folder:

```powershell
modal volume get egoverse-zarrs-v2 projects/folding-clothes-clips data/egoverse/clips --force
```

## Clip Contents

Each clip directory contains:

- `rgb.mp4`: 240 first-person RGB frames at 30 FPS.
- `metrics.npz`: the matching 240 frames of head pose, hand/wrist poses, MANO keypoints, RGB timestamps, and camera intrinsics.
- `metadata.json`: the source episode, frame range, validity checks, and data conventions.

The video and measurements use the same frame index. Position data is in the metric SLAM world frame; camera intrinsics convert 3D camera-frame points into image pixels.

Raw and derived data under `data/egoverse` is intentionally excluded from Git.
