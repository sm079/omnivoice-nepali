"""Tiny local web server for the segment viewer (stdlib only, supports HTTP Range for audio seeking)."""

from __future__ import annotations

import json
import mimetypes
import re
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from nepvoice.workspace import Workspace, find_workspaces

from .data import build_payload, playable_audio

STATIC = Path(__file__).parent / "static"
mimetypes.add_type("audio/webm", ".webm")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("text/javascript", ".js")


def _named(works: list[Path]) -> dict[str, Path]:
    """Work directories by a unique URL-safe name (their folder name, numbered on a clash)."""
    out: dict[str, Path] = {}
    for work in works:
        name = base = work.resolve().name or "work"
        n = 2
        while name in out:
            name, n = f"{base}-{n}", n + 1
        out[name] = work.resolve()
    return out


def make_handler(works: Path | list[Path]):
    """Serve the recordings of one or more work directories.

    A recording is addressed as ``<work name>/<recording id>``, and its files under
    ``/media/<work name>/``, so recordings with the same id in two work dirs stay apart.
    """
    named = _named([works] if isinstance(works, Path) else list(works))

    def ready() -> dict[str, tuple[Path, Workspace]]:
        return {
            f"{name}/{rid}": (work, ws)
            for name, work in named.items()
            for rid, ws in find_workspaces(work).items()
            if ws.segments.exists() and ws.diar.exists() and ws.osd.exists()
        }

    @lru_cache(maxsize=8)
    def payload(key: str, version: int) -> bytes:  # version: segments.jsonl mtime, so re-selection shows up
        work, ws = ready()[key]
        media = f"/media/{key.split('/', 1)[0]}/"
        return json.dumps(build_payload(ws, work, key, media), ensure_ascii=False).encode()

    def recordings() -> list[dict]:
        order = {name: i for i, name in enumerate(named)}
        out = []
        for key, (_, ws) in ready().items():
            info = json.loads(ws.summary.read_text(encoding="utf-8")) if ws.summary.exists() else {}
            meta = ws.meta()
            out.append({
                "key": key,
                "work": key.split("/", 1)[0],
                "recording_id": ws.id,
                "title": meta.get("title"),
                **{k: info.get(k) for k in ("audio_seconds", "kept_seconds", "num_segments")},
            })
        return sorted(out, key=lambda r: (order[r["work"]], r["recording_id"]))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet apart from errors
            if not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
                super().log_message(fmt, *args)

        def _send_json(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = size - int(m.group(2))
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 16, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the browser cancels range requests while seeking

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/api/recordings":
                return self._send_json(json.dumps(recordings()).encode())
            if path.startswith("/api/recording/"):
                key = path.removeprefix("/api/recording/")
                found = ready().get(key)
                if found is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._send_json(payload(key, found[1].segments.stat().st_mtime_ns))
            if path.startswith("/api/audio/"):
                found = ready().get(path.removeprefix("/api/audio/"))
                if found is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._send_file(playable_audio(found[1]))
            if path.startswith("/media/"):
                name, _, rest = path.removeprefix("/media/").partition("/")
                work = named.get(name)
                if work is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                target = (work / rest).resolve()
                if work not in target.parents:
                    return self.send_error(HTTPStatus.FORBIDDEN)
                return self._send_file(target)
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / name).resolve()
            if STATIC.resolve() not in target.parents:
                return self.send_error(HTTPStatus.FORBIDDEN)
            return self._send_file(target)

    return Handler


def serve(works: list[Path], host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(works))
    url = f"http://{host}:{port}/"
    print(f"Segment viewer at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
