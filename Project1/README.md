# Project 1: EgoVerse Action Recognition

Build a model that watches a short egocentric clip and predicts the action being performed. **Stage 0 is complete** for five folding-clothes episodes; Stage 1 dataset construction is next. No model training yet.

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
```

Create and open the synchronized Stage 0 demo:

```powershell
python inspect_episode.py `
  data/egoverse/stage0/episodes/69bb01bf11e9b1cd78d2945d `
  --output-dir DEMOS/00_stage0_viewer `
  --open
```

The graph playhead, completed trajectory, and active annotation follow the video in real time. Project results live in [`DEMOS`](./DEMOS/); short findings live in [`notes`](./notes/).

## Assisted Labeling

Start the local labeling server:

```powershell
python label_episode.py data/egoverse/stage0/episodes/69bb01bf11e9b1cd78d2945d
```

It converts raw annotation phrases into automatic labels using transparent text rules. They are accepted for training by default, but they are not model predictions. You can change the class or time range, name an `OTHER`, select active hands, mark uncertainty, or create a new interval.

Accepted labels and your edits are saved to [`labels/accepted`](./labels/accepted/) as Git-tracked JSON. Press `Ctrl+C` in the terminal to stop the server.

## Episode Bundle

- `rgb.mp4` — first-person RGB video.
- `measurements.npz` — one row per video frame for time, head pose, wrist pose, hand end-effector pose, and hand keypoints, plus the front-camera intrinsics.
- `annotations.json` — raw human-written labels with start and end times.
- `metadata.json` — shapes, validity checks, alignment rules, hashes, and provenance.

For the first episode, wrist poses have shape `[2275, 7]`: 2,275 timestamps, each with XYZ position and four orientation values. Each hand-keypoint array has shape `[2275, 63]`, or 21 landmarks × XYZ.

The annotations are free-text and sometimes name several actions in one segment. Transparent text rules convert them into the initial accepted classes while retaining an uncertainty flag for estimated boundaries. Finer labels such as `REACH` and `GRASP` can be explored later if the data supports them.

Raw videos and generated reports stay outside Git. Source: [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) and the [EgoVerse Explorer](https://partners.mecka.ai/egoverse).
