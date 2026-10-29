"""ffmpeg helpers: normalise input audio and cut chunks."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

SUPPORTED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm", ".mp4"}

# highpass/lowpass: drop rumble and hiss outside the speech band.
# afftdn: light FFT denoise (nr=10 dB is gentle; strong denoise hurts STT).
# loudnorm: EBU R128 loudness normalisation so quiet speakers are audible.
DENOISE_CHAIN = "highpass=f=80,lowpass=f=7800,afftdn=nr=10:nf=-40"
NORMALIZE = "loudnorm=I=-16:TP=-1.5:LRA=11"


class AudioError(RuntimeError):
    pass


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise AudioError(f"{tool} not found on PATH. Install ffmpeg (see README).")


async def _run(*args: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise AudioError(f"{args[0]} failed: {err.decode(errors='replace')[-800:]}")
    return out


async def probe_duration(path: Path) -> float:
    out = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path))
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise AudioError(f"Could not read duration of {path} - is it a valid audio file?") from exc


async def preprocess(src: Path, work_dir: Path, denoise: bool = True) -> tuple[Path, Path]:
    """Return (16 kHz mono WAV, compact FLAC copy for upload). Original is untouched."""
    if src.suffix.lower() not in SUPPORTED:
        raise AudioError(f"Unsupported file type {src.suffix!r}; expected one of {sorted(SUPPORTED)}")
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / f"{src.stem}.16k.wav"
    flac = work_dir / f"{src.stem}.16k.flac"
    chain = f"{DENOISE_CHAIN},{NORMALIZE}" if denoise else NORMALIZE
    await _run("ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
               "-af", chain, "-c:a", "pcm_s16le", str(wav))
    await _run("ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "flac", str(flac))
    return wav, flac


async def cut_clip(src: Path, start: float, end: float, dst: Path) -> Path:
    """Cut [start, end) seconds to a small Opus file (for sending to the LLM)."""
    await _run("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(start, 0):.3f}", "-to", f"{end:.3f}",
               "-i", str(src), "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(dst))
    return dst
