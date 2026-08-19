# EgoVerse Action Recognition

Project 1 teaches the full ML lifecycle by first understanding EgoVerse data, then building wrist-only, video-only, and combined action-recognition models. The broader sequence should eventually connect recognized human activity to a rendered robot motion or simulated task.

## Language

**Episode**:
A complete synchronized recording with video, measurements, and metadata.
_Avoid_: Clip, video

**Clip**:
A shorter consecutive part of an episode used as one model input.
_Avoid_: Episode

**Modality**:
One kind of input, such as RGB video or wrist motion.
_Avoid_: Metric

**Annotation segment**:
A start time, end time, and human-written description of what happens in that interval.
_Avoid_: Class label

**Class label**:
A standardized answer such as `FOLD` that a model is trained to predict.
_Avoid_: Raw annotation

**Training example**:
One fixed-length input window paired with its correct class label.
_Avoid_: Episode

**Wrist trajectory**:
The wrist's XYZ position across consecutive frames.
_Avoid_: Hand video

**Dataset split**:
Separate training, validation, and test groups; episodes from one recording must not leak across groups.
_Avoid_: Random frames

**Synchronized viewer**:
An inspection view where video playback, annotation highlighting, and a moving cursor on sensor graphs share one clock.
_Avoid_: Static report

**Demo artifact**:
An easy-to-open visual result that shows what one project stage produced, such as a synchronized episode viewer or model-prediction clip.
_Avoid_: Raw output

**Kinematic retargeting**:
Mapping tracked human motion onto feasible robot joint motion without claiming that the robot physically completes the task.
_Avoid_: Robot policy, task execution

**Simulated task execution**:
A robot interacting with objects in a physics simulator to complete the task, not merely replaying a plausible arm trajectory.
_Avoid_: Robot rendering, kinematic playback

## Current State

- Update this file at the end of every work session so a new chat can resume quickly.
- Project goal: action recognition, progressing from wrist-only to video-only and then multimodal models.
- Current stage: Stage 0 data understanding and preparation; do not train a model yet.
- Modal 1.5.4 is installed in `Project1/.venv` and authenticated as `treymangat`.
- The existing `egoverse-zarrs-v2` Modal Volume is the data store; do not use the exposed AWS keys from the upstream README.
- The original three Aria clips have excellent RGB and hand tracking but no action annotations, so they are exploratory data rather than supervised examples.
- `stage0_manifest.json` selects five annotated Mecka `folding_clothes` episodes with public RGB video and both-hand measurements.
- All five bundles are prepared in Modal under `projects/action-recognition/stage0/episodes`: 11,790 frames, 110 current annotation segments, and about 6.55 minutes total.
- `stage0_prepare.py` creates complete bundles and verifies exact video/measurement frame alignment.
- Episode `69bb01bf11e9b1cd78d2945d` is downloaded locally and inspected: 2,275 frames at 30 FPS, 75.83 seconds, 21 annotations, 320×180 RGB, and 100% finite rows in every selected pose/keypoint array.
- Its wrist arrays are `[2275, 7]` (XYZ + orientation) and its keypoint arrays are `[2275, 63]` (21 XYZ landmarks).
- Each bundle also preserves `intrinsics_front_1`, the calibration matrix needed to project 3D camera-frame points onto RGB pixels.
- `inspect_episode.py` generated `episode_report.html` and `wrist_trajectory.png` beside the local episode data.
- The episode `696e84048a176d6397a7a11e` exposes a version difference: 10 embedded Zarr annotation records versus 17 current segment-table records. Bundles use the current table and record both counts.
- Raw free-text annotations often contain multiple actions. A canonical vocabulary such as `REACH`, `GRASP`, `MOVE`, `FOLD`, `RELEASE`, and `OTHER` has not been defined yet.
- Requested next visualization: synchronize the video with a moving cursor on the wrist/hand graphs and the active annotation.
- Requested project organization: a prominent demo gallery for stage results plus a concise notes folder; exact structure is still being discussed.
- Long-term goal: progress from recognizing human actions to visualizing a robot performing related motion, then later distinguish simple kinematic retargeting from true simulated task execution.
- Raw `.zarr`, `.mp4`, and derived data remain outside Git history.

## Next Step

Agree on the viewer/gallery structure, then use the synchronized viewer to audit a small set of labels. Do not start model training until the label vocabulary and example boundaries have been checked.
