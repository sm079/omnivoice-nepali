"""Segment viewer API on a processed synthetic recording (stand-in model outputs)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

from nepvoice import config
from nepvoice.ingest import ingest
from nepvoice.prep import process
from nepvoice.workspace import find_workspaces
from tools.segment_viewer.server import make_handler


def processed_work(work: Path, inbox: Path, make_recording, fake_models, title: str) -> Path:
    make_recording(inbox, "ep 7", amp=0.3)
    (inbox / "ep 7.json").write_text(json.dumps({"title": title}))
    ingest(inbox, work)
    cfg = config.load([], ["select.min_speaker_seconds=5"])
    ws = find_workspaces(work)["ep_7"]
    fake_models(ws)
    process.select(ws, cfg.select)
    process.export(ws, cfg.export)
    return work


@pytest.fixture()
def server(tmp_path: Path, make_recording, fake_models):
    # two work dirs holding a recording with the same id
    works = [processed_work(tmp_path / name, tmp_path / f"in-{name}", make_recording, fake_models, title)
             for name, title in (("work", "Talk"), ("work-b", "Other talk"))]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(works))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url: str, **headers) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, b"", {}


def test_recording_list_and_payload(server):
    status, body, _ = get(f"{server}/api/recordings")
    recordings = json.loads(body)
    assert status == 200 and [r["key"] for r in recordings] == ["work/ep_7", "work-b/ep_7"]
    assert recordings[0]["recording_id"] == "ep_7"
    assert [r["title"] for r in recordings] == ["Talk", "Other talk"]
    for r in recordings:
        status, body, _ = get(f"{server}/api/recording/{quote(r['key'], safe='')}")
        data = json.loads(body)
        assert data["meta"]["title"] == r["title"] and data["segments"] and data["speakers"][0]["target"]
        assert data["segments"][0]["clip_url"].startswith(f"/media/{r['work']}/")
        assert get(server + data["segments"][0]["clip_url"])[0] == 200
        assert get(server + data["audio_url"])[0] == 200


def test_audio_supports_range_requests(server):
    status, body, headers = get(f"{server}/api/audio/work-b%2Fep_7", Range="bytes=0-99")
    assert status == 206 and len(body) == 100 and headers["Content-Range"].startswith("bytes 0-99/")


def test_unknown_and_outside_paths_are_refused(server):
    assert get(f"{server}/api/recording/nope")[0] == 404
    assert get(f"{server}/api/recording/ep_7")[0] == 404  # needs its work dir
    assert get(f"{server}/media/..%2F..%2Fsecret.txt")[0] in (403, 404)
    assert get(f"{server}/media/work/..%2F..%2Fsecret.txt")[0] in (403, 404)
    assert get(f"{server}/index.html")[0] == 200
