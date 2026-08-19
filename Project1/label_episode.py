from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from inspect_episode import _viewer_data, render_viewer
from label_schema import (
    load_or_create_label_document,
    save_label_document,
    validate_label_document,
)


MAX_REQUEST_BYTES = 1_000_000


def load_episode(episode_dir: Path) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    required = ("rgb.mp4", "measurements.npz", "annotations.json", "metadata.json")
    missing = [name for name in required if not (episode_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {episode_dir}: {', '.join(missing)}")

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    annotations = json.loads((episode_dir / "annotations.json").read_text())[
        "segments"
    ]
    with np.load(episode_dir / "measurements.npz") as archive:
        measurements = {key: archive[key] for key in archive.files}
    return metadata, annotations, measurements


class LabelingServer(ThreadingHTTPServer):
    metadata: dict
    annotations: list[dict]
    measurements: dict[str, np.ndarray]
    video_path: Path
    labels_path: Path
    save_lock: threading.Lock


class LabelingRequestHandler(BaseHTTPRequestHandler):
    server: LabelingServer

    def log_message(self, format_string: str, *args) -> None:
        if args and str(args[1]).startswith("4"):
            super().log_message(format_string, *args)

    def _json_response(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _current_document(self) -> dict:
        return validate_label_document(
            json.loads(self.server.labels_path.read_text()), self.server.metadata
        )

    def _serve_page(self) -> None:
        document = self._current_document()
        data = _viewer_data(
            self.server.metadata,
            self.server.annotations,
            self.server.measurements,
            label_document=document,
            editable=True,
        )
        body = render_viewer(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self) -> None:
        file_size = self.server.video_path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match:
                self.send_error(416, "Invalid byte range")
                return
            if not match.group(1) and match.group(2):
                suffix_length = min(int(match.group(2)), file_size)
                start, end = file_size - suffix_length, file_size - 1
            elif match.group(1):
                start = int(match.group(1))
            if match.group(1) and match.group(2):
                end = min(int(match.group(2)), file_size - 1)
            if start > end or start >= file_size:
                self.send_error(416, "Byte range is outside the video")
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with self.server.video_path.open("rb") as video:
            video.seek(start)
            remaining = length
            while remaining:
                chunk = video.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_page()
        elif path == "/rgb.mp4":
            self._serve_video()
        elif path == "/api/labels":
            self._json_response(200, self._current_document())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/labels":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body must be between 1 byte and 1 MB")
            submitted = json.loads(self.rfile.read(length))
            validated = validate_label_document(submitted, self.server.metadata)
            with self.server.save_lock:
                save_label_document(self.server.labels_path, validated)
            self._json_response(200, validated)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._json_response(400, {"error": str(error)})


def create_server(
    episode_dir: Path, labels_dir: Path, host: str, port: int
) -> LabelingServer:
    metadata, annotations, measurements = load_episode(episode_dir)
    labels_path = labels_dir / f"{metadata['episode_hash']}.json"
    load_or_create_label_document(labels_path, metadata, annotations)

    server = LabelingServer((host, port), LabelingRequestHandler)
    server.metadata = metadata
    server.annotations = annotations
    server.measurements = measurements
    server.video_path = episode_dir / "rgb.mp4"
    server.labels_path = labels_path
    server.save_lock = threading.Lock()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect accepted EgoVerse labels and save manual edits."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=Path("labels/accepted"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    arguments = parser.parse_args()

    server = create_server(
        arguments.episode_dir.resolve(),
        arguments.labels_dir.resolve(),
        arguments.host,
        arguments.port,
    )
    url = f"http://{arguments.host}:{server.server_port}"
    document = json.loads(server.labels_path.read_text())
    other_count = sum(label["label"] == "OTHER" for label in document["labels"])
    print(f"Accepted labels: {len(document['labels'])} ({other_count} OTHER)")
    print(f"Saved labels: {server.labels_path}")
    print(f"Labeling viewer: {url}")
    if not arguments.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
