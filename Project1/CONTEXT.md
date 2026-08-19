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

**Manual label**:
A reviewer-created or edited class and time range stored separately from the immutable raw annotation.
_Avoid_: Corrected annotation

**Automatic label**:
A class inferred from raw annotation text and accepted as training truth unless a reviewer edits or deletes it.
_Avoid_: Model prediction, manual label, suggestion

**Accepted label**:
A class and time range included as training truth, whether it came from the automatic rules or a manual edit.
_Avoid_: Raw annotation, suggestion

**Weak label**:
An accepted training label inferred without frame-by-frame human verification, so its class may be useful even when its exact boundary is noisy.
_Avoid_: Verified ground truth

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
