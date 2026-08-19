from __future__ import annotations

import argparse
import html
import json
import shutil
import webbrowser
from pathlib import Path

import numpy as np

from label_schema import generate_label_document, validate_label_document


def _xyz(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2 and array.shape[1] >= 3:
        return array[:, :3]
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        return array[:, :3, 3]
    raise ValueError(f"Cannot extract XYZ positions from shape {array.shape}")


def _rounded_rows(array: np.ndarray) -> list[list[float]]:
    return np.round(array.astype(np.float64), 5).tolist()


def _viewer_data(
    metadata: dict,
    annotations: list[dict],
    measurements: dict[str, np.ndarray],
    label_document: dict | None = None,
    editable: bool = False,
) -> dict:
    labels = label_document or generate_label_document(metadata, annotations)
    return {
        "episode": {
            "id": metadata["episode_hash"],
            "task": metadata["task"],
            "frames": metadata["frames"],
            "fps": metadata["fps"],
            "duration": metadata["duration_seconds"],
            "frameSize": metadata["frame_size"],
        },
        "times": np.round(measurements["time_seconds"], 5).tolist(),
        "poses": {
            "wrist": {
                "left": _rounded_rows(_xyz(measurements["left.obs_wrist_pose"])),
                "right": _rounded_rows(_xyz(measurements["right.obs_wrist_pose"])),
            },
            "hand": {
                "left": _rounded_rows(_xyz(measurements["left.obs_ee_pose"])),
                "right": _rounded_rows(_xyz(measurements["right.obs_ee_pose"])),
            },
        },
        "annotations": annotations,
        "labelDocument": labels,
        "editable": editable,
    }


def render_viewer(data: dict) -> str:
    episode = data["episode"]
    annotation_rows = "\n".join(
        (
            f'<button class="annotation-row" type="button" '
            f'data-index="{segment["segment_index"]}" '
            f'data-start="{segment["start_seconds"]}">'
            f'<span>{segment["segment_index"] + 1}</span>'
            f'<time>{segment["start_seconds"]:.2f}–{segment["end_seconds"]:.2f}s</time>'
            f'<strong>{html.escape(segment["label"])}</strong>'
            "</button>"
        )
        for segment in data["annotations"]
    )
    serialized_data = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 0 viewer — {html.escape(episode['id'])}</title>
<style>
:root {{ color-scheme: light dark; --bg: #f7f7f5; --surface: #ffffff; --text: #181817; --muted: #666661; --border: #d9d9d3; --active: #fff1c7; --active-border: #d18b00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #151514; --surface: #20201e; --text: #f0f0ec; --muted: #aaa9a2; --border: #3b3b36; --active: #3b3018; --active-border: #f2b84b; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, sans-serif; }}
main {{ max-width: 1280px; margin: auto; padding: 24px; }}
h1, h2, p {{ margin-top: 0; }}
h1 {{ margin-bottom: 4px; font-size: clamp(22px, 3vw, 32px); }}
h2 {{ margin: 28px 0 10px; font-size: 20px; }}
.subtitle, .small {{ color: var(--muted); }}
.layout {{ display: grid; grid-template-columns: minmax(280px, 0.8fr) minmax(420px, 1.2fr); gap: 20px; align-items: start; margin-top: 20px; }}
.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
video {{ width: 100%; display: block; background: #000; border-radius: 6px; }}
.now {{ margin-top: 12px; padding-left: 10px; border-left: 4px solid var(--active-border); min-height: 58px; }}
.now-label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
#active-annotation, #active-training-label {{ display: block; margin-top: 3px; }}
.training-now {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }}
.chart-toolbar {{ display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 8px; }}
label {{ display: grid; gap: 4px; font-weight: 600; }}
select, input[type="number"], input[type="text"] {{ font: inherit; padding: 7px 8px; background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 5px; }}
#time-display {{ font-variant-numeric: tabular-nums; color: var(--muted); }}
.canvas-wrap {{ width: 100%; height: 430px; }}
canvas {{ width: 100%; height: 100%; display: block; cursor: crosshair; }}
.hint {{ margin: 8px 0 0; color: var(--muted); font-size: 13px; }}
.label-editor h2 {{ margin-top: 0; }}
.form-grid {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 12px; }}
.form-grid .wide {{ grid-column: span 2; }}
.checks {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin: 14px 0; }}
.checks label {{ display: flex; flex-direction: row; align-items: center; gap: 6px; font-weight: 500; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
button.action {{ font: inherit; padding: 7px 11px; color: var(--text); background: var(--surface); border: 1px solid var(--border); border-radius: 5px; cursor: pointer; }}
button.primary {{ background: var(--active); border-color: var(--active-border); font-weight: 600; }}
button.danger {{ color: #c33; }}
#save-status {{ color: var(--muted); margin-left: 4px; }}
.training-labels {{ display: grid; gap: 4px; }}
.training-row {{ width: 100%; display: grid; grid-template-columns: 110px 100px 1fr 100px; gap: 10px; align-items: center; padding: 9px; color: var(--text); background: transparent; border: 1px solid transparent; border-bottom-color: var(--border); text-align: left; font: inherit; cursor: pointer; }}
.training-row:hover, .training-row.selected {{ border-color: var(--active-border); border-radius: 6px; }}
.training-row.active {{ background: var(--active); }}
.training-row .class-name {{ font-weight: 700; }}
.status {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.hidden {{ display: none !important; }}
.annotations {{ display: grid; gap: 4px; }}
.annotation-row {{ width: 100%; display: grid; grid-template-columns: 28px 130px 1fr; gap: 8px; align-items: start; border: 1px solid transparent; border-bottom-color: var(--border); background: transparent; color: var(--text); padding: 9px; text-align: left; font: inherit; cursor: pointer; }}
.annotation-row time {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
.annotation-row strong {{ font-weight: 600; }}
.annotation-row.active {{ background: var(--active); border-color: var(--active-border); border-radius: 6px; }}
.annotation-row:hover {{ border-color: var(--active-border); border-radius: 6px; }}
footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; }}
@media (max-width: 820px) {{ .layout {{ grid-template-columns: 1fr; }} .canvas-wrap {{ height: 390px; }} .form-grid {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 520px) {{ main {{ padding: 14px; }} .annotation-row {{ grid-template-columns: 24px 1fr; }} .annotation-row strong {{ grid-column: 2; }} .training-row {{ grid-template-columns: 92px 1fr; }} .training-row span:nth-child(n+3) {{ grid-column: 2; }} .canvas-wrap {{ height: 350px; }} .form-grid {{ grid-template-columns: 1fr; }} .form-grid .wide {{ grid-column: auto; }} }}
</style>
</head>
<body>
<main>
  <h1>Stage 0 synchronized episode viewer</h1>
  <p class="subtitle">{html.escape(episode['task'])} · {episode['frames']:,} frames · {episode['fps']:g} FPS · {episode['duration']:.2f}s</p>
  <section class="layout">
    <div class="panel">
      <video id="episode-video" controls preload="metadata" src="rgb.mp4"></video>
      <div class="now" aria-live="polite">
        <span class="now-label">Active raw annotation</span>
        <strong id="active-annotation">Loading…</strong>
        <div class="training-now">
          <span class="now-label">Current accepted label</span>
          <strong id="active-training-label">Loading…</strong>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="chart-toolbar">
        <label>Tracked position
          <select id="pose-select">
            <option value="wrist">Wrist</option>
            <option value="hand">Hand / end-effector</option>
          </select>
        </label>
        <output id="time-display">0.00 / {episode['duration']:.2f}s</output>
      </div>
      <div class="canvas-wrap"><canvas id="trajectory-chart" role="img" aria-label="Left and right XYZ position trajectories synchronized to the video"></canvas></div>
      <p class="hint">The strong trace is the motion already seen. Click the graph or an annotation to seek the video.</p>
    </div>
  </section>
  <section class="panel label-editor hidden" id="label-editor">
    <h2>Edit or create a label</h2>
    <p class="small">Automatic labels are accepted for training by default. Saving records your changes as a manual edit.</p>
    <div class="form-grid">
      <label>Start (seconds)<input id="label-start" type="number" min="0" step="0.033"></label>
      <label>End (seconds)<input id="label-end" type="number" min="0" step="0.033"></label>
      <label>Class<select id="label-class"></select></label>
      <label class="wide">Name when OTHER<input id="custom-label" type="text" maxlength="80" placeholder="Example: UNFOLD"></label>
    </div>
    <div class="checks">
      <label><input id="hand-left" type="checkbox"> Left hand</label>
      <label><input id="hand-right" type="checkbox"> Right hand</label>
      <label><input id="label-uncertain" type="checkbox"> Uncertain</label>
    </div>
    <div class="actions">
      <button class="action" id="use-start" type="button">Use video as start</button>
      <button class="action" id="use-end" type="button">Use video as end</button>
      <button class="action" id="new-label" type="button">New label here</button>
      <button class="action primary" id="save-label" type="button">Save changes</button>
      <button class="action danger" id="delete-label" type="button">Delete</button>
      <output id="save-status" aria-live="polite"></output>
    </div>
  </section>
  <h2>Accepted training labels</h2>
  <p class="small" id="label-summary"></p>
  <div class="training-labels" id="training-label-list"></div>
  <h2>Timestamped annotations</h2>
  <div class="annotations" id="annotation-list">{annotation_rows}</div>
  <footer>Source: EgoVerse · Episode {html.escape(episode['id'])} · CC BY-SA 4.0</footer>
</main>
<script>
const DATA={serialized_data};
const video=document.getElementById('episode-video');
const canvas=document.getElementById('trajectory-chart');
const context=canvas.getContext('2d');
const poseSelect=document.getElementById('pose-select');
const activeAnnotation=document.getElementById('active-annotation');
const activeTrainingLabelElement=document.getElementById('active-training-label');
const timeDisplay=document.getElementById('time-display');
const annotationRows=[...document.querySelectorAll('.annotation-row')];
const labelEditor=document.getElementById('label-editor');
const labelList=document.getElementById('training-label-list');
const labelSummary=document.getElementById('label-summary');
const labelStart=document.getElementById('label-start');
const labelEnd=document.getElementById('label-end');
const labelClass=document.getElementById('label-class');
const customLabel=document.getElementById('custom-label');
const handLeft=document.getElementById('hand-left');
const handRight=document.getElementById('hand-right');
const labelUncertain=document.getElementById('label-uncertain');
const saveStatus=document.getElementById('save-status');
const coordinates=['X','Y','Z'];
const lightColors=['#0072b2','#d77b00','#00875a'];
const darkColors=['#55b8ff','#ffb14e','#4bd6a0'];
let labelDocument=structuredClone(DATA.labelDocument);
let selectedLabelId=labelDocument.labels[0]?.id || null;
let draftId=null;
let animationFrame=null;

function theme() {{
  const dark=window.matchMedia('(prefers-color-scheme: dark)').matches;
  return {{
    text:dark?'#f0f0ec':'#181817', muted:dark?'#aaa9a2':'#666661',
    grid:dark?'#3b3b36':'#ddddda', boundary:dark?'#66665f':'#b8b8b2',
    playhead:dark?'#ffd166':'#a85d00', colors:dark?darkColors:lightColors
  }};
}}

function activeSegment(time) {{
  return DATA.annotations.find(segment => time >= segment.start_seconds && time < segment.end_seconds)
    || DATA.annotations.at(-1);
}}

function trainingLabelAt(time) {{
  return labelDocument.labels.find(label => time >= label.start_seconds && time < label.end_seconds) || null;
}}

function labelName(label) {{
  return label.label==='OTHER' && label.custom_label ? `OTHER · ${{label.custom_label}}` : label.label;
}}

function updateCustomLabelState() {{
  const isOther=labelClass.value==='OTHER';
  customLabel.disabled=!isOther;
  if(!isOther) customLabel.value='';
}}

function populateEditor(label) {{
  if(!label) return;
  selectedLabelId=label.id;
  draftId=null;
  labelStart.value=label.start_seconds.toFixed(3);
  labelEnd.value=label.end_seconds.toFixed(3);
  labelClass.value=label.label;
  customLabel.value=label.custom_label || '';
  handLeft.checked=label.active_hands.includes('left');
  handRight.checked=label.active_hands.includes('right');
  labelUncertain.checked=Boolean(label.uncertain);
  updateCustomLabelState();
  renderLabelList();
}}

function renderLabelList() {{
  labelList.replaceChildren();
  const accepted=labelDocument.labels.filter(label=>label.accepted).length;
  const manual=labelDocument.labels.filter(label=>label.source!=='automatic_text_rule').length;
  const other=labelDocument.labels.filter(label=>label.label==='OTHER').length;
  labelSummary.textContent=`${{accepted}} accepted · ${{manual}} manually edited · ${{other}} OTHER`;
  labelDocument.labels.forEach(label=>{{
    const row=document.createElement('button');
    row.type='button'; row.className='training-row'; row.dataset.id=label.id;
    if(label.id===selectedLabelId) row.classList.add('selected');
    const time=document.createElement('span'); time.textContent=`${{label.start_seconds.toFixed(2)}}–${{label.end_seconds.toFixed(2)}}s`;
    const className=document.createElement('span'); className.className='class-name'; className.textContent=labelName(label);
    const detail=document.createElement('span'); detail.textContent=label.suggestion_text || label.raw_annotation || 'Manual interval';
    const status=document.createElement('span'); status.className='status'; status.textContent=label.source==='automatic_text_rule'?'Auto · accepted':'Manual edit';
    row.append(time,className,detail,status);
    row.addEventListener('click',()=>{{
      populateEditor(label);
      video.currentTime=label.start_seconds;
      update();
    }});
    labelList.append(row);
  }});
}}

async function persistLabels(nextDocument, successMessage) {{
  saveStatus.textContent='Saving…';
  const response=await fetch('/api/labels',{{
    method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(nextDocument)
  }});
  const result=await response.json();
  if(!response.ok) throw new Error(result.error || 'Save failed');
  labelDocument=result;
  renderLabelList();
  saveStatus.textContent=successMessage;
}}

function draw() {{
  const ratio=window.devicePixelRatio || 1;
  const width=Math.max(320,canvas.clientWidth);
  const height=Math.max(300,canvas.clientHeight);
  if(canvas.width!==Math.round(width*ratio) || canvas.height!==Math.round(height*ratio)) {{
    canvas.width=Math.round(width*ratio); canvas.height=Math.round(height*ratio);
  }}
  context.setTransform(ratio,0,0,ratio,0,0);
  context.clearRect(0,0,width,height);
  const style=theme();
  const left=58, right=14, top=28, bottom=30, gap=44;
  const panelHeight=(height-top-bottom-gap)/2;
  const plotWidth=width-left-right;
  const duration=DATA.episode.duration;
  const current=Math.min(video.currentTime || 0,duration);
  const pose=DATA.poses[poseSelect.value];

  ['left','right'].forEach((side,panelIndex) => {{
    const values=pose[side];
    const panelTop=top+panelIndex*(panelHeight+gap);
    const flat=values.flat();
    let min=Math.min(...flat), max=Math.max(...flat);
    const padding=Math.max((max-min)*0.08,0.01); min-=padding; max+=padding;
    const x=time => left+(time/duration)*plotWidth;
    const y=value => panelTop+panelHeight-((value-min)/(max-min))*panelHeight;

    context.strokeStyle=style.grid; context.lineWidth=1; context.fillStyle=style.muted;
    context.font='12px system-ui'; context.textAlign='right'; context.textBaseline='middle';
    for(let tick=0;tick<=4;tick++) {{
      const value=min+(max-min)*tick/4; const py=y(value);
      context.beginPath(); context.moveTo(left,py); context.lineTo(width-right,py); context.stroke();
      context.fillText(value.toFixed(2),left-7,py);
    }}
    DATA.annotations.slice(1).forEach(segment => {{
      const px=x(segment.start_seconds); context.strokeStyle=style.boundary; context.globalAlpha=.28;
      context.beginPath(); context.moveTo(px,panelTop); context.lineTo(px,panelTop+panelHeight); context.stroke();
    }});
    context.globalAlpha=1;

    function lineSeries(coordinate,strong) {{
      context.strokeStyle=style.colors[coordinate]; context.lineWidth=strong?1.8:1;
      context.globalAlpha=strong?1:.2; context.beginPath();
      let started=false;
      for(let index=0;index<values.length;index++) {{
        if(strong && DATA.times[index]>current) break;
        const px=x(DATA.times[index]), py=y(values[index][coordinate]);
        if(!started) {{ context.moveTo(px,py); started=true; }} else context.lineTo(px,py);
      }}
      context.stroke(); context.globalAlpha=1;
    }}
    coordinates.forEach((_,index)=>lineSeries(index,false));
    coordinates.forEach((_,index)=>lineSeries(index,true));

    const playheadX=x(current); context.strokeStyle=style.playhead; context.lineWidth=2;
    context.beginPath(); context.moveTo(playheadX,panelTop); context.lineTo(playheadX,panelTop+panelHeight); context.stroke();
    context.fillStyle=style.text; context.textAlign='left'; context.textBaseline='bottom'; context.font='600 13px system-ui';
    context.fillText(side==='left'?'Left':'Right',left,panelTop-7);
  }});

  const styleNow=theme(); context.font='12px system-ui'; context.textBaseline='middle';
  coordinates.forEach((label,index)=>{{
    const legendX=left+index*50; context.fillStyle=styleNow.colors[index]; context.fillRect(legendX,height-14,14,2);
    context.fillStyle=styleNow.text; context.textAlign='left'; context.fillText(label,legendX+19,height-13);
  }});
  context.fillStyle=styleNow.muted; context.textAlign='right'; context.fillText('Time (seconds)',width-right,height-13);
}}

function update() {{
  const time=video.currentTime || 0;
  const segment=activeSegment(time);
  const trainingLabel=trainingLabelAt(time);
  activeAnnotation.textContent=segment.label;
  activeTrainingLabelElement.textContent=trainingLabel
    ? `${{labelName(trainingLabel)}} (${{trainingLabel.source==='automatic_text_rule'?'automatic':'manual edit'}})`
    : 'No label for this time';
  timeDisplay.textContent=`${{time.toFixed(2)}} / ${{DATA.episode.duration.toFixed(2)}}s`;
  annotationRows.forEach(row=>{{
    const active=Number(row.dataset.index)===segment.segment_index;
    row.classList.toggle('active',active);
    if(active) row.setAttribute('aria-current','true'); else row.removeAttribute('aria-current');
  }});
  draw();
}}

function animate() {{ update(); if(!video.paused && !video.ended) animationFrame=requestAnimationFrame(animate); }}
video.addEventListener('play',()=>{{ cancelAnimationFrame(animationFrame); animate(); }});
['pause','seeked','timeupdate','loadedmetadata'].forEach(event=>video.addEventListener(event,update));
poseSelect.addEventListener('change',draw);
canvas.addEventListener('click',event=>{{
  const bounds=canvas.getBoundingClientRect(); const left=58, right=14;
  const fraction=Math.max(0,Math.min(1,(event.clientX-bounds.left-left)/(bounds.width-left-right)));
  video.currentTime=fraction*DATA.episode.duration; update();
}});
annotationRows.forEach(row=>row.addEventListener('click',()=>{{ video.currentTime=Number(row.dataset.start); update(); }}));
labelClass.addEventListener('change',updateCustomLabelState);

if(DATA.editable) {{
  labelEditor.classList.remove('hidden');
  labelDocument.class_labels.forEach(name=>{{
    const option=document.createElement('option'); option.value=name; option.textContent=name; labelClass.append(option);
  }});
  document.getElementById('use-start').addEventListener('click',()=>{{ labelStart.value=(video.currentTime || 0).toFixed(3); }});
  document.getElementById('use-end').addEventListener('click',()=>{{ labelEnd.value=(video.currentTime || 0).toFixed(3); }});
  document.getElementById('new-label').addEventListener('click',()=>{{
    selectedLabelId=null;
    draftId=`manual-${{crypto.randomUUID()}}`;
    const start=Math.min(video.currentTime || 0,DATA.episode.duration-0.034);
    labelStart.value=start.toFixed(3);
    labelEnd.value=Math.min(start+1,DATA.episode.duration).toFixed(3);
    labelClass.value='OTHER'; customLabel.value=''; handLeft.checked=false; handRight.checked=false;
    labelUncertain.checked=true; updateCustomLabelState(); renderLabelList(); saveStatus.textContent='New unsaved interval';
  }});
  document.getElementById('save-label').addEventListener('click',async()=>{{
    const start=Number(labelStart.value), end=Number(labelEnd.value);
    if(!Number.isFinite(start) || !Number.isFinite(end) || end<=start) {{ saveStatus.textContent='End must be after start'; return; }}
    const existing=labelDocument.labels.find(label=>label.id===selectedLabelId);
    const id=existing?.id || draftId || `manual-${{crypto.randomUUID()}}`;
    const activeHands=[]; if(handLeft.checked) activeHands.push('left'); if(handRight.checked) activeHands.push('right');
    const updated={{
      ...(existing || {{}}), id, start_seconds:start, end_seconds:end, label:labelClass.value,
      custom_label:labelClass.value==='OTHER'?customLabel.value.trim():'', active_hands:activeHands,
      uncertain:labelUncertain.checked, accepted:true,
      source:existing?.source==='automatic_text_rule'?'manual_edit':(existing?.source || 'manual')
    }};
    const next=structuredClone(labelDocument);
    const index=next.labels.findIndex(label=>label.id===id);
    if(index>=0) next.labels[index]=updated; else next.labels.push(updated);
    try {{
      await persistLabels(next,'Saved to Git-tracked JSON');
      selectedLabelId=id; draftId=null;
      populateEditor(labelDocument.labels.find(label=>label.id===id));
      update();
    }} catch(error) {{ saveStatus.textContent=error.message; }}
  }});
  document.getElementById('delete-label').addEventListener('click',async()=>{{
    if(!selectedLabelId) {{ saveStatus.textContent='Select a saved label first'; return; }}
    const selected=labelDocument.labels.find(label=>label.id===selectedLabelId);
    if(!window.confirm(`Delete ${{labelName(selected)}} at ${{selected.start_seconds.toFixed(2)}}s?`)) return;
    const next=structuredClone(labelDocument); next.labels=next.labels.filter(label=>label.id!==selectedLabelId);
    try {{
      await persistLabels(next,'Label deleted');
      selectedLabelId=labelDocument.labels[0]?.id || null;
      if(selectedLabelId) populateEditor(labelDocument.labels[0]);
      update();
    }} catch(error) {{ saveStatus.textContent=error.message; }}
  }});
}} else {{
  labelSummary.textContent='Automatic text-rule labels are accepted by default; launch the labeling server to edit them.';
}}

new ResizeObserver(draw).observe(canvas.parentElement);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
renderLabelList();
if(DATA.editable && selectedLabelId) populateEditor(labelDocument.labels[0]);
update();
</script>
</body>
</html>
"""
    return document


def _write_viewer(data: dict, destination: Path) -> None:
    destination.write_text(render_viewer(data), encoding="utf-8")


def inspect_episode(
    episode_dir: Path, output_dir: Path | None, open_report: bool
) -> Path:
    required = ("rgb.mp4", "measurements.npz", "annotations.json", "metadata.json")
    missing = [name for name in required if not (episode_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {episode_dir}: {', '.join(missing)}")

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    annotation_document = json.loads((episode_dir / "annotations.json").read_text())
    annotations = annotation_document["segments"]
    with np.load(episode_dir / "measurements.npz") as archive:
        measurements = {key: archive[key] for key in archive.files}

    saved_labels_path = (
        Path(__file__).parent
        / "labels"
        / "accepted"
        / f"{metadata['episode_hash']}.json"
    )
    label_document = None
    if saved_labels_path.is_file():
        label_document = validate_label_document(
            json.loads(saved_labels_path.read_text()), metadata
        )

    destination_dir = output_dir or episode_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    report_path = destination_dir / ("index.html" if output_dir else "episode_report.html")
    if destination_dir.resolve() != episode_dir.resolve():
        shutil.copy2(episode_dir / "rgb.mp4", destination_dir / "rgb.mp4")
    _write_viewer(
        _viewer_data(
            metadata,
            annotations,
            measurements,
            label_document=label_document,
        ),
        report_path,
    )

    print(f"Episode: {metadata['episode_hash']}")
    print(
        f"Video: {metadata['frames']} frames, {metadata['fps']:g} FPS, "
        f"{metadata['duration_seconds']:.2f} seconds"
    )
    print(f"Annotations: {len(annotations)} timestamped segments")
    print(f"Viewer: {report_path.resolve()}")

    if open_report:
        webbrowser.open(report_path.resolve().as_uri())
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synchronized viewer for one prepared EgoVerse episode."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="Create a portable demo in this directory."
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the generated viewer in a browser."
    )
    arguments = parser.parse_args()
    inspect_episode(arguments.episode_dir, arguments.output_dir, arguments.open)


if __name__ == "__main__":
    main()
