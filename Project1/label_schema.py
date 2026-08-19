from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2
CLASS_LABELS = ("PICK_UP", "PLACE", "MOVE", "FOLD", "SMOOTH", "OTHER")

LABEL_PATTERNS = {
    "PICK_UP": (
        re.compile(r"\bpick(?:ed|ing)?\s+up\b", re.IGNORECASE),
        re.compile(r"\blift(?:ed|ing)?\b", re.IGNORECASE),
        re.compile(r"\bremove(?:d|ing)?\b", re.IGNORECASE),
    ),
    "PLACE": (
        re.compile(r"\bplace(?:d|ing)?\b", re.IGNORECASE),
        re.compile(r"\bput(?:ting)?\b", re.IGNORECASE),
        re.compile(r"\bset\s+down\b", re.IGNORECASE),
    ),
    "FOLD": (re.compile(r"\bfold(?:s|ing)?\b", re.IGNORECASE),),
    "SMOOTH": (
        re.compile(r"\bsmooth(?:en|ened|ening|ed|ing)?\b", re.IGNORECASE),
        re.compile(r"\bflatten(?:ed|ing)?\b", re.IGNORECASE),
        re.compile(r"\bspread(?:ing)?\b", re.IGNORECASE),
    ),
    "MOVE": (
        re.compile(r"\bmove(?:d|s|ing)?\b", re.IGNORECASE),
        re.compile(r"\badjust(?:ed|ing)?\b", re.IGNORECASE),
        re.compile(r"\brotate(?:d|s|ing)?\b", re.IGNORECASE),
        re.compile(r"\bflip(?:ped|ping|s)?\b", re.IGNORECASE),
        re.compile(r"\breposition(?:ed|ing)?\b", re.IGNORECASE),
    ),
}

OTHER_HINTS = (
    ("UNFOLD", re.compile(r"\bunfold", re.IGNORECASE)),
    ("CLEAN", re.compile(r"\bclean", re.IGNORECASE)),
    ("OPEN", re.compile(r"\bopen", re.IGNORECASE)),
    ("SEAL", re.compile(r"\bseal|\bclose", re.IGNORECASE)),
    ("PACK", re.compile(r"\bpack", re.IGNORECASE)),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classify_phrase(text: str) -> dict:
    matches = []
    matched_terms = []
    for label, patterns in LABEL_PATTERNS.items():
        terms = [match.group(0) for pattern in patterns if (match := pattern.search(text))]
        if terms:
            matches.append(label)
            matched_terms.extend(terms)

    label = matches[0] if matches else "OTHER"
    custom_label = ""
    if label == "OTHER":
        custom_label = next(
            (name for name, pattern in OTHER_HINTS if pattern.search(text)), ""
        )
    return {
        "label": label,
        "custom_label": custom_label,
        "matched_labels": matches,
        "matched_terms": matched_terms,
        "uncertain": len(matches) != 1 or label == "OTHER",
    }


def split_annotation(text: str) -> list[str]:
    parts = [part.strip(" .") for part in re.split(r"\s*,\s*|\s+then\s+", text)]
    return [part for part in parts if part] or [text.strip()]


def _frame_range(start: float, end: float, fps: float, total_frames: int) -> tuple[int, int]:
    start_frame = max(0, min(total_frames - 1, int(round(start * fps))))
    end_frame = max(start_frame + 1, min(total_frames, int(round(end * fps))))
    return start_frame, end_frame


def generate_label_document(metadata: dict, annotations: list[dict]) -> dict:
    fps = float(metadata["fps"])
    total_frames = int(metadata["frames"])
    labels = []
    for annotation in annotations:
        parts = split_annotation(annotation["label"])
        start = float(annotation["start_seconds"])
        end = float(annotation["end_seconds"])
        part_duration = (end - start) / len(parts)
        for part_index, part in enumerate(parts):
            part_start = start + part_index * part_duration
            part_end = end if part_index == len(parts) - 1 else start + (part_index + 1) * part_duration
            classification = classify_phrase(part)
            start_frame, end_frame = _frame_range(
                part_start, part_end, fps, total_frames
            )
            labels.append(
                {
                    "id": f"auto-{annotation['segment_index']}-{part_index}",
                    "start_seconds": round(part_start, 3),
                    "end_seconds": round(part_end, 3),
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "label": classification["label"],
                    "custom_label": classification["custom_label"],
                    "active_hands": [],
                    "uncertain": classification["uncertain"] or len(parts) > 1,
                    "accepted": True,
                    "source": "automatic_text_rule",
                    "raw_annotation_index": annotation["segment_index"],
                    "raw_annotation": annotation["label"],
                    "suggestion_text": part,
                    "matched_terms": classification["matched_terms"],
                }
            )

    timestamp = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_hash": metadata["episode_hash"],
        "fps": fps,
        "frames": total_frames,
        "duration_seconds": float(metadata["duration_seconds"]),
        "class_labels": list(CLASS_LABELS),
        "automatic_method": "annotation_text_rules_v1",
        "created_at": timestamp,
        "updated_at": timestamp,
        "labels": labels,
    }


def validate_label_document(document: dict, metadata: dict) -> dict:
    if document.get("episode_hash") != metadata["episode_hash"]:
        raise ValueError("Label document episode_hash does not match this episode")
    if not isinstance(document.get("labels"), list) or len(document["labels"]) > 1000:
        raise ValueError("labels must be a list containing at most 1000 items")

    fps = float(metadata["fps"])
    total_frames = int(metadata["frames"])
    duration = float(metadata["duration_seconds"])
    seen_ids = set()
    validated_labels = []
    for item in document["labels"]:
        label_id = str(item.get("id", "")).strip()[:100]
        if not label_id or label_id in seen_ids:
            raise ValueError("Every label needs a unique non-empty id")
        seen_ids.add(label_id)

        start = round(float(item["start_seconds"]), 3)
        end = round(float(item["end_seconds"]), 3)
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError(f"Invalid time range for {label_id}: {start}–{end}")

        label = str(item.get("label", "")).upper()
        if label not in CLASS_LABELS:
            raise ValueError(f"Unknown class label for {label_id}: {label}")
        active_hands = list(dict.fromkeys(item.get("active_hands", [])))
        if any(hand not in ("left", "right") for hand in active_hands):
            raise ValueError(f"active_hands for {label_id} must contain left/right only")

        start_frame, end_frame = _frame_range(start, end, fps, total_frames)
        validated_labels.append(
            {
                "id": label_id,
                "start_seconds": start,
                "end_seconds": end,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "label": label,
                "custom_label": str(item.get("custom_label", "")).strip()[:80],
                "active_hands": active_hands,
                "uncertain": bool(item.get("uncertain", False)),
                "accepted": bool(item.get("accepted", True)),
                "source": str(item.get("source", "manual"))[:40],
                "raw_annotation_index": item.get("raw_annotation_index"),
                "raw_annotation": str(item.get("raw_annotation", ""))[:500],
                "suggestion_text": str(item.get("suggestion_text", ""))[:500],
                "matched_terms": [str(term)[:80] for term in item.get("matched_terms", [])[:20]],
            }
        )

    validated_labels.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    for previous, current in zip(validated_labels, validated_labels[1:]):
        if current["start_seconds"] < previous["end_seconds"] - 0.001:
            raise ValueError(
                f"Labels {previous['id']} and {current['id']} overlap; "
                "the first model uses one scene-level label at a time"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_hash": metadata["episode_hash"],
        "fps": fps,
        "frames": total_frames,
        "duration_seconds": duration,
        "class_labels": list(CLASS_LABELS),
        "automatic_method": document.get(
            "automatic_method", "annotation_text_rules_v1"
        ),
        "created_at": document.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "labels": validated_labels,
    }


def load_or_create_label_document(
    path: Path, metadata: dict, annotations: list[dict]
) -> dict:
    if path.is_file():
        saved_document = json.loads(path.read_text())
        document = validate_label_document(saved_document, metadata)
        needs_migration = saved_document.get("schema_version") != SCHEMA_VERSION or any(
            "accepted" not in label for label in saved_document.get("labels", [])
        )
        if needs_migration:
            save_label_document(path, document)
        return document
    document = generate_label_document(metadata, annotations)
    save_label_document(path, document)
    return document


def save_label_document(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        json.dump(document, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
