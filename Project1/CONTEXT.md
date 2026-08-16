# Folding-Clothes Clips

This project uses a small, validated subset of EgoVerse folding-clothes recordings to learn the structure of egocentric video data.

## Language

**Episode**:
A complete synchronized EgoVerse recording containing RGB frames, measurements, poses, and metadata.
_Avoid_: Clip, video

**Clip**:
A shorter consecutive frame range extracted from an episode for one experiment.
_Avoid_: Episode

**RGB stream**:
The first-person color-image sequence stored as `images.front_1`.
_Avoid_: RGB metrics

**Hand end-effector pose**:
A seven-value 3D hand position and orientation stored per frame as `left.obs_ee_pose` or `right.obs_ee_pose`.
_Avoid_: Hand coordinates

**MANO keypoints**:
Twenty-one 3D landmarks for one hand, stored as 63 values per frame in `*.obs_keypoints` when available.
_Avoid_: Hand pose

**Camera intrinsics**:
The calibration matrix used to project 3D points onto RGB pixels.
_Avoid_: Camera pose

**Episode manifest**:
The tracked list of selected episode hashes and why each episode was chosen; it contains no raw video data.
_Avoid_: Dataset

## Current State

- Update this file at the end of every work session so a new chat can resume quickly.
- Modal 1.5.4 is installed in `Project1/.venv` and authenticated under the `treymangat` profile.
- The existing `egoverse-zarrs-v2` Modal Volume contains 1,115 episode directories, so no new bulk download or R2 credential is needed.
- Three validated eight-second Aria folding-clothes clips are stored in Modal under `projects/folding-clothes-clips` and locally under `Project1/data/egoverse/clips/folding-clothes-clips`.
- Every selected clip has 240 aligned RGB/measurement frames and 100% valid head, wrist, hand-pose, and MANO-keypoint data.
- Checked Mecka folding candidates advertised RGB in metadata but did not contain a materialized `images.front_1` array; the current selection therefore uses complete Aria episodes.
- Raw `.zarr` episodes and derived `.mp4` clips must remain outside Git history.

## Next Step

Load one `metrics.npz` file and overlay its 3D hand information on the matching RGB frames.
