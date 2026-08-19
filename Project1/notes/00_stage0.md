# Stage 0 Data Findings

- Five folding-clothes episodes: 11,790 frames, 110 annotation segments, about 6.55 minutes.
- Video frames and measurement rows align exactly at 30 FPS.
- Both wrists, hand end-effectors, 21 hand keypoints, head pose, and camera intrinsics are present.
- Raw annotations can contain several actions; they are not yet model-ready class labels.
- One episode has 10 embedded Zarr annotations but 17 current segment-table annotations; use the current table and record both counts.
- Agreed initial classes: `PICK_UP`, `PLACE`, `MOVE`, `FOLD`, `SMOOTH`, and `OTHER`.
- Manual labels should preserve the raw annotation and reference synchronized episode timestamps.
- Text rules create 24 first-episode suggestions; all begin unreviewed, and estimated multi-action boundaries are uncertain.
- The two automatic `OTHER` suggestions are proposed as `UNFOLD`, which can later become a class if review supports it.
