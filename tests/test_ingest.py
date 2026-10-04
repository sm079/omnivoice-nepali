"""Registering an input folder: pairing, ids, layout, metadata and cache invalidation."""

import json
import os
from pathlib import Path

import pytest

from nepvoice.ingest import ingest, recording_id, scan
from nepvoice.workspace import find_workspaces


def touch(path: Path, data: bytes | str = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


@pytest.fixture()
def inbox(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    touch(root / "ep1.webm")
    touch(root / "ep1.json3", '{"events": []}')
    touch(root / "ep1.json", json.dumps({"title": "Ep 1"}))
    touch(root / "sub" / "clip 2.wav")
    touch(root / "sub" / "clip 2.txt", "नमस्ते")
    touch(root / "sub" / "clip 2.srt", "")  # timed beats plain when both exist
    touch(root / "orphan.mp3")
    touch(root / "notes.txt", "not a transcript of anything")
    return root


def test_scan_pairs_audio_with_transcripts(inbox: Path):
    found, missing = scan(inbox)
    assert [r.id for r in found] == ["ep1", "clip_2"]
    assert [r.subdir for r in found] == [Path(), Path("sub")]
    assert found[0].meta == {"title": "Ep 1"}
    assert found[1].transcript.suffix == ".srt"
    assert [p.name for p in missing] == ["orphan.mp3"]


def test_recording_ids_are_folder_safe(tmp_path: Path):
    assert recording_id(tmp_path / "a" / "b" / "talk 1?.mp3") == "talk_1"


def test_file_names_must_be_unique_across_subfolders(inbox: Path):
    touch(inbox / "other" / "ep1.webm")
    touch(inbox / "other" / "ep1.json3", '{"events": []}')
    with pytest.raises(ValueError, match="same recording id 'ep1'"):
        scan(inbox)


def test_workspaces_mirror_the_input_folders_and_follow_moves(inbox: Path, tmp_path: Path):
    work = tmp_path / "work"
    ingest(inbox, work)
    assert find_workspaces(work)["clip_2"].dir == work / "recordings" / "sub" / "clip_2"
    touch(find_workspaces(work)["clip_2"].diar)

    moved = tmp_path / "moved"  # the whole input folder moves, and one recording changes subfolder
    inbox.rename(moved)
    (moved / "other").mkdir()
    for f in (moved / "sub").iterdir():
        f.rename(moved / "other" / f.name)
    report = ingest(moved, work)
    assert report.updated == ["ep1", "clip_2"] and not report.not_in_input
    ws = find_workspaces(work)["clip_2"]
    assert ws.dir == work / "recordings" / "other" / "clip_2"
    assert ws.diar.exists() and ws.audio_path == (moved / "other" / "clip 2.wav").resolve()
    assert not (work / "recordings" / "sub").exists()


def test_ingest_is_idempotent_and_resets_stale_caches(inbox: Path, tmp_path: Path):
    work = tmp_path / "work"
    report = ingest(inbox, work)
    assert report.added == ["ep1", "clip_2"] and report.no_transcript
    ws = find_workspaces(work)["ep1"]
    assert ws.timed and ws.meta()["title"] == "Ep 1"
    assert ws.transcript_path.read_text() == '{"events": []}'

    touch(ws.diar)  # audio-level cache
    touch(ws.segments)  # transcript-level output
    assert ingest(inbox, work).unchanged == ["ep1", "clip_2"]
    assert ws.diar.exists() and ws.segments.exists()

    touch(inbox / "ep1.json3", '{"events": [ ]}')  # new transcript: keep model outputs
    assert ingest(inbox, work).updated == ["ep1"]
    assert ws.diar.exists() and not ws.segments.exists()

    touch(ws.segments)
    audio = inbox / "ep1.webm"
    touch(audio, b"different audio")
    os.utime(audio, ns=(1, 1))
    assert ingest(inbox, work).updated == ["ep1"]
    assert not ws.diar.exists() and not ws.segments.exists()


def test_recordings_removed_from_the_input_are_reported(inbox: Path, tmp_path: Path):
    work = tmp_path / "work"
    ingest(inbox, work)
    (inbox / "ep1.webm").unlink()
    assert ingest(inbox, work).not_in_input == ["ep1"]


def test_a_changed_recording_keeps_a_workspace_nested_in_its_folder(tmp_path: Path):
    inbox, work = tmp_path / "in", tmp_path / "work"
    touch(inbox / "a.webm")
    touch(inbox / "a.json3", '{"events": []}')
    touch(inbox / "a" / "b.webm")  # its workspace is recordings/a/b/, inside a's
    touch(inbox / "a" / "b.json3", '{"events": []}')
    ingest(inbox, work)
    b = find_workspaces(work)["b"]
    touch(b.diar)
    audio = inbox / "a.webm"
    touch(audio, b"different audio")
    os.utime(audio, ns=(1, 1))
    assert ingest(inbox, work).updated == ["a"]
    assert b.diar.exists() and set(find_workspaces(work)) == {"a", "b"}
