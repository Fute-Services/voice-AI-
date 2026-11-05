"""Local web UI for the transcription pipeline.

    python app.py        # then open http://localhost:8000
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from transcriber.config import ConfigError, Settings
from transcriber.pipeline import run
from transcriber.preprocess import SUPPORTED, AudioError
from transcriber.retry import APIError

ROOT = Path(__file__).parent
RUNS = ROOT / "out" / "web"
SAMPLE = ROOT / "tests" / "fixtures" / "sample_call.mp3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
for noisy in ("httpx", "httpcore", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="Hinglish Transcriber")

# Set APP_PASSWORD when hosting publicly, otherwise anyone with the URL spends your API credits.
APP_USER = os.getenv("APP_USER", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")


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
        if secrets.compare_digest(user, APP_USER) and secrets.compare_digest(pwd, APP_PASSWORD):
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

    try:
        res = await run(src, run_dir, s, reference=SAMPLE.with_suffix(".reference.txt") if use_sample else None)
    except (ConfigError, AudioError, APIError, FileNotFoundError) as exc:
        raise HTTPException(502, str(exc)) from exc

    data = json.loads(res.paths["json"].read_text(encoding="utf-8"))
    for seg in data["segments"]:
        seg.pop("words", None)  # keep the payload small; full data is in the JSON download
    data["run_id"] = run_id
    data["audio_url"] = f"/api/runs/{run_id}/audio"
    data["downloads"] = {k: f"/api/runs/{run_id}/file/{k}" for k in res.paths}
    return data


def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", run_id) or not (RUNS / run_id).is_dir():
        raise HTTPException(404, "Unknown run")
    return RUNS / run_id


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
