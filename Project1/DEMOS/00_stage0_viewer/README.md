# Stage 0: Synchronized Episode Viewer

Open `index.html` locally to inspect the first folding-clothes episode. The graph playhead and highlighted annotation follow the video; clicking either seeks the video.

Regenerate it from `Project1`:

```powershell
python inspect_episode.py `
  data/egoverse/stage0/episodes/69bb01bf11e9b1cd78d2945d `
  --output-dir DEMOS/00_stage0_viewer
```

`rgb.mp4` is copied here locally but excluded from Git. Download the episode first using the command in the project README if it is missing.
