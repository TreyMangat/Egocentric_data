# Project 1: EgoVerse Action Recognition

Build a model that watches a short egocentric clip and predicts the action being performed. We are currently at **Stage 0: understand and prepare the data**. No model training yet.

## Roadmap

1. Stage 0 — align RGB video, wrist trajectories, timestamps, and annotations.
2. Stage 1 — predict actions from one-second wrist-motion windows.
3. Stage 2 — predict actions from video clips.
4. Stage 3 — compare wrist-only, video-only, and fused models.

An **input** is what the model sees. A **label** is the answer it should learn. One future training example will be a short video or wrist window paired with one action label.

## Stage 0 Workflow

Activate the environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the five episodes listed in `stage0_manifest.json` on Modal:

```powershell
modal run stage0_prepare.py
```

The job refuses to save an episode unless video frames and measurement rows match exactly. Copy just the first episode locally:

```powershell
New-Item -ItemType Directory -Force data/egoverse/stage0/episodes | Out-Null
modal volume get egoverse-zarrs-v2 projects/action-recognition/stage0/episodes/69bb01bf11e9b1cd78d2945d data/egoverse/stage0/episodes --force
python inspect_episode.py data/egoverse/stage0/episodes/69bb01bf11e9b1cd78d2945d --open
```

The generated report lets you play the RGB video, view both wrist trajectories, and read each timestamped annotation.

## Episode Bundle

- `rgb.mp4` — first-person RGB video.
- `measurements.npz` — one row per video frame for time, head pose, wrist pose, hand end-effector pose, and hand keypoints, plus the front-camera intrinsics.
- `annotations.json` — raw human-written labels with start and end times.
- `metadata.json` — shapes, validity checks, alignment rules, hashes, and provenance.

For the first episode, wrist poses have shape `[2275, 7]`: 2,275 timestamps, each with XYZ position and four orientation values. Each hand-keypoint array has shape `[2275, 63]`, or 21 landmarks × XYZ.

The annotations are free-text and sometimes name several actions in one segment. Turning them into classes such as `REACH`, `GRASP`, and `FOLD` is a later data-design step, not something Stage 0 silently guesses.

Raw videos and generated reports stay outside Git. Source: [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) and the [EgoVerse Explorer](https://partners.mecka.ai/egoverse).
