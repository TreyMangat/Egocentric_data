# Stage 0 Data Findings

- Five folding-clothes episodes: 11,790 frames, 110 annotation segments, about 6.55 minutes.
- Video frames and measurement rows align exactly at 30 FPS.
- Both wrists, hand end-effectors, 21 hand keypoints, head pose, and camera intrinsics are present.
- Raw annotations can contain several actions; they are not yet model-ready class labels.
- One episode has 10 embedded Zarr annotations but 17 current segment-table annotations; use the current table and record both counts.
- Agreed initial classes: `PICK_UP`, `PLACE`, `MOVE`, `FOLD`, `SMOOTH`, and `OTHER`.
- Manual labels should preserve the raw annotation and reference synchronized episode timestamps.
- Text rules create 24 first-episode labels; all are accepted by default, while estimated multi-action boundaries remain marked uncertain.
- The user confirmed the first episode's labels are acceptable and manually confirmed both `OTHER` labels as `UNFOLD`.
- All five prepared episodes are now local and checksum-verified: 11,790 aligned RGB/measurement frames and 152 accepted labels.
- Main-class balance: `MOVE` 53, `PICK_UP` 27, `SMOOTH` 24, `FOLD` 20, `PLACE` 14, `OTHER` 14.
- Named `OTHER` actions are `CLEAN` 6, `UNFOLD` 5, `OPEN` 2, and `SEAL` 1.
- 88 labels have estimated boundaries marked uncertain; they remain accepted as weak labels.
- Next milestone: build fixed-length wrist training examples and split them by episode.
