"""Local web UI for the transcription pipeline.

    python app.py        # then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from transcriber.config import ConfigError, Settings
from transcriber.models import Segment
from transcriber.pipeline import run
from transcriber.preprocess import SUPPORTED, AudioError
from transcriber.qa import SUGGESTED, CallQA
from transcriber.retry import APIError

ROOT = Path(__file__).parent
RUNS = ROOT / "out" / "web"
SAMPLE = ROOT / "tests" / "fixtures" / "sample_call.mp3"
KEEPALIVE_S = 15

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
for noisy in ("httpx", "httpcore", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="Hinglish Transcriber")

# Set APP_PASSWORD when hosting publicly, otherwise anyone with the URL spends your API credits.
# Stripped because dashboard textareas (e.g. Render) easily save a trailing space/newline.
APP_USER = os.getenv("APP_USER", "admin").strip().lower()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not APP_PASSWORD or request.url.path == "/healthz":
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:
            user = pwd = ""
        user, pwd = user.strip().lower(), pwd.strip()
        if secrets.compare_digest(user.encode(), APP_USER.encode()) and \
                secrets.compare_digest(pwd.encode(), APP_PASSWORD.encode()):
            return await call_next(request)
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="transcriber"'})


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile | None = File(None),
    use_sample: bool = Form(False),
    script: str = Form("roman"),
    cleanup_model: str = Form(""),
    cleanup: bool = Form(True),
    speakers: int | None = Form(None),
    keyterms: str = Form(""),
    analysis: bool = Form(True),
    context: str = Form(""),
    stream: bool = Form(False),
):
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    if use_sample:
        src = run_dir / SAMPLE.name
        shutil.copy2(SAMPLE, src)
    elif file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file type {suffix!r}")
        stem = re.sub(r"[^\w\-]+", "_", Path(file.filename).stem)[:60] or "audio"
        src = run_dir / f"{stem}{suffix}"
        with src.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        raise HTTPException(400, "Upload an audio file or choose the sample")

    s = Settings()
    s.hindi_script = script
    s.cleanup = cleanup
    if cleanup_model:
        s.cleanup_model = cleanup_model
    if speakers:
        s.speakers_expected = speakers
    if keyterms.strip():
        s.keyterms = [t.strip() for t in keyterms.split(",") if t.strip()]
    s.analysis = analysis
    if context.strip():
        s.analysis_context = context.strip()

    reference = SAMPLE.with_suffix(".reference.txt") if use_sample else None

    def slim(data: dict) -> dict:
        for seg in data["segments"]:
            seg.pop("words", None)  # keep the payload small; full data is in the JSON download
        return data | {"run_id": run_id, "audio_url": f"/api/runs/{run_id}/audio"}

    async def process(on_event=None) -> dict:
        res = await run(src, run_dir, s, reference=reference, on_event=on_event)
        data = slim(json.loads(res.paths["json"].read_text(encoding="utf-8")))
        data["downloads"] = {k: f"/api/runs/{run_id}/file/{k}" for k in res.paths}
        return data

    if not stream:
        try:
            return await process()
        except (ConfigError, AudioError, APIError, FileNotFoundError) as exc:
            raise HTTPException(502, str(exc)) from exc

    # Streaming: one JSON object per line. The browser sees each stage start, gets the transcript as soon as
    # it is ready (before the analysis), and the connection never sits silent long enough for a proxy to drop it.
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def on_event(name: str, data: dict) -> None:
        queue.put_nowait({"event": name, **(slim(data) if name == "transcript" else data)})

    async def work() -> None:
        try:
            queue.put_nowait({"event": "result", **await process(on_event)})
        except (ConfigError, AudioError, APIError, FileNotFoundError) as exc:
            queue.put_nowait({"event": "error", "detail": str(exc)})
        except Exception:  # noqa: BLE001 - the response has already started, so report it in-band
            logging.exception("run %s crashed", run_id)
            queue.put_nowait({"event": "error", "detail": "Unexpected server error - check the server log."})
        finally:
            queue.put_nowait(None)

    async def lines():
        task = asyncio.create_task(work())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except asyncio.TimeoutError:
                    item = {"event": "ping"}
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            task.cancel()  # browser went away: stop spending API credits on a result nobody will see

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", run_id) or not (RUNS / run_id).is_dir():
        raise HTTPException(404, "Unknown run")
    return RUNS / run_id


def _transcript_path(d: Path) -> Path | None:
    return next(d.glob("*.transcript.json"), None)


def _load(run_id: str) -> tuple[Path, dict]:
    path = _transcript_path(_run_dir(run_id))
    if not path:
        raise HTTPException(404, "This run has no transcript (it may have failed)")
    return path, json.loads(path.read_text(encoding="utf-8"))


_summaries: dict[Path, tuple[float, dict]] = {}  # transcript path -> (mtime, summary): the list stays fast


def _summary(run_id: str, path: Path) -> dict:
    mtime = path.stat().st_mtime
    hit = _summaries.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    a = data.get("analysis") or {}
    req, ap, m = a.get("requirements") or {}, a.get("agent_performance") or {}, (data.get("meta") or {}).get("metrics") or {}
    src = next((p.name for p in path.parent.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED), path.name)
    out = {
        "run_id": run_id, "file": src, "created": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
        "duration": data.get("duration") or 0, "segments": len(data.get("segments") or []),
        "outcome": a.get("outcome"), "outcome_label": a.get("outcome_label"),
        "lead_temperature": a.get("lead_temperature"), "lead_score": a.get("lead_score"),
        "confidence": a.get("confidence"), "needs_review": bool(a.get("needs_review")),
        "agent_score": ap.get("score"), "agent_grade": ap.get("grade"),
        "buyer_type": (a.get("behaviour") or {}).get("buyer_type_label"),
        "follow_up_when": a.get("follow_up_when") or "", "next_action": a.get("next_action") or "",
        "summary": a.get("summary") or "", "budget": req.get("budget") or "", "location": req.get("location") or "",
        "bhk": req.get("bhk") or "", "agent_talk_pct": m.get("agent_talk_pct"),
        "cost_usd": (data.get("cost") or {}).get("total_usd"),
    }
    _summaries[path] = (mtime, out)
    return out


def _all_summaries() -> list[dict]:
    if not RUNS.is_dir():
        return []
    rows = []
    for d in RUNS.iterdir():
        path = _transcript_path(d) if d.is_dir() and re.fullmatch(r"[0-9a-f]{10}", d.name) else None
        if path:
            try:
                rows.append(_summary(d.name, path))
            except (OSError, ValueError) as exc:  # half-written or hand-edited file: skip, don't break the list
                logging.warning("skipping run %s: %s", d.name, exc)
    return sorted(rows, key=lambda r: r["created"], reverse=True)


@app.get("/api/runs")
def list_runs() -> dict:
    """Every saved call, newest first, with the fields the dashboard needs."""
    return {"runs": _all_summaries()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """A saved call in the same shape /api/transcribe returns, so the UI can reopen it."""
    path, data = _load(run_id)
    for seg in data.get("segments") or []:
        seg.pop("words", None)
    kinds = {"json": "transcript.json", "md": "transcript.md", "txt": "transcript.txt", "review": "review.md",
             "analysis": "analysis.json", "analysis_md": "analysis.md"}
    have = {k for k, ext in kinds.items() if next(path.parent.glob(f"*.{ext}"), None)}
    return data | {"run_id": run_id, "audio_url": f"/api/runs/{run_id}/audio",
                   "file": _summary(run_id, path)["file"],
                   "downloads": {k: f"/api/runs/{run_id}/file/{k}" for k in kinds if k in have}}


@app.get("/api/ask/suggestions")
def ask_suggestions() -> dict:
    return {"questions": SUGGESTED}


@app.post("/api/runs/{run_id}/ask")
async def ask(run_id: str, body: dict = Body(...)):
    """Answer a question about one call from its transcript, with checked quotes."""
    _, data = _load(run_id)
    segs = [Segment(id=s["id"], speaker=s["speaker"], start=s["start"], end=s["end"], text=s["text"])
            for s in data.get("segments") or []]
    history = body.get("history") if isinstance(body.get("history"), list) else []
    try:
        return await CallQA(Settings()).ask(segs, str(body.get("question") or ""), history)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (ConfigError, APIError) as exc:
        raise HTTPException(502, str(exc)) from exc


CSV_FIELDS = ("created", "file", "outcome_label", "lead_temperature", "lead_score", "agent_score", "agent_grade",
              "buyer_type", "budget", "location", "bhk", "follow_up_when", "next_action", "needs_review",
              "duration", "summary", "run_id")


@app.get("/api/leads.csv")
def leads_csv():
    """All analysed calls as a spreadsheet for the sales team (opens in Excel / Google Sheets)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in _all_summaries():
        if r["outcome"]:
            w.writerow(r | {"duration": round(r["duration"]), "created": r["created"][:16].replace("T", " ")})
    # BOM so Excel reads Hindi / rupee text as UTF-8
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="leads.csv"'})


@app.get("/api/runs/{run_id}/audio")
def audio(run_id: str):
    d = _run_dir(run_id)
    src = next((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED), None)
    if not src:
        raise HTTPException(404, "Audio not found")
    return FileResponse(src)


@app.get("/api/runs/{run_id}/file/{kind}")
def download(run_id: str, kind: str):
    d = _run_dir(run_id)
    ext = {"json": "transcript.json", "md": "transcript.md", "txt": "transcript.txt", "review": "review.md",
           "analysis": "analysis.json", "analysis_md": "analysis.md"}.get(kind)
    match = next(d.glob(f"*.{ext}"), None) if ext else None
    if not match:
        raise HTTPException(404, "File not found")
    return FileResponse(match, filename=match.name)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
