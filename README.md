# voice-AI: Hinglish call transcription

Audio file in → verbatim transcript with speaker labels and timestamps out
(`.json`, `.md`, `.txt`), plus a review list of spots a human should check.

```
audio ──ffmpeg──▶ 16 kHz mono WAV ──STT──▶ raw segments ──Gemini cleanup──▶ transcript + review list
        (normalise, light denoise)  (AssemblyAI      (listens to audio, fixes
                                     or Gemini)        misheard words; verbatim)
```

## Setup (Windows, macOS, Linux)

1. **Python 3.11+** and **ffmpeg** on PATH
   - Windows: `winget install Gyan.FFmpeg` · macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg`
   - Check: `ffmpeg -version`
2. Install:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Keys: `cp .env.example .env` (Windows: `copy .env.example .env`) and fill in
   - `ASSEMBLYAI_API_KEY`: primary STT, https://www.assemblyai.com/dashboard
   - `GEMINI_API_KEY`: cleanup pass, https://aistudio.google.com/apikey

   With only a Gemini key, STT falls back to `gemini-3.5-transcribe` automatically.

## Run

```bash
python transcribe.py path/to/call.mp3 --out out/
```

Useful flags:

| flag | meaning |
|---|---|
| `--script devanagari` | Hindi in Devanagari instead of Roman (English always stays Latin) |
| `--speakers 2` | expected speaker count (AssemblyAI) |
| `--keyterms "Sharma Logistics,IFSC,UTR"` | names/jargon to bias STT |
| `--provider gemini` | force an STT provider (`assemblyai` / `gemini`) |
| `--cleanup-model gemini-4-argon` | best cleanup model (needs a billing-enabled key) |
| `--no-cleanup` / `--no-denoise` | skip a stage |
| `--reference ref.txt` | compute word error rate against a reference |

Every setting also has an env var; see `.env.example`.

### Web UI

```bash
python app.py        # then open http://localhost:8000
```

Upload a file (or try the sample call), pick the script and cleanup model, and read the transcript.
Click any line to play that part of the audio, and download the JSON/MD/TXT/review files.
Runs are saved under `out/web/`.

### Outputs (in `--out`)

- `<name>.transcript.json`: `segments[]` with `id, speaker, start, end, text, confidence`, plus
  `raw_text` (STT before cleanup), `words[]` (word timestamps from STT), `flags`, `review`, `cost`
- `<name>.transcript.md` / `.txt`: readable `[hh:mm:ss] Speaker N: text`
- `<name>.review.md`: segments with `[unclear]`, low STT confidence, low-confidence words, or a rejected/failed cleanup
- `<name>.work/`: original copy, 16 kHz WAV, FLAC, audio chunks

## How accuracy is protected

- **STT**: AssemblyAI Universal-3.5 Pro with `language_codes=["en","hi"]`, which is AssemblyAI's documented
  English↔Hindi code-switching mode. Universal-2 is the fallback. Diarization, punctuation and word timestamps are on.
- **Cleanup**: Gemini gets the audio *and* the STT segments and may only fix what the audio supports.
  Unintelligible parts become `[unclear]`. It returns text per segment id, so timestamps and speaker
  turns can't be changed. If a cleaned segment's word count drifts more than 0.6×–1.5× from the raw one
  (a sign of summarising or invention), the raw text is kept and the segment is flagged.
- **Long files** (> 20 min): 10-min chunks with 15 s overlap. Each segment belongs to exactly one chunk.
  The overlap is sent only as read-only context, so the merge cannot duplicate lines.
- **Retries**: exponential backoff on 429/5xx/timeouts. If a chunk keeps failing, it moves to
  `CLEANUP_FALLBACK_MODEL`, and only then keeps the raw text (flagged for review).

## Notes and limits

- Word timestamps in `words[]` come from STT. After cleanup the segment text can differ slightly
  from those tokens, but segment start/end times are always the STT ones.
- Gemini STT gives no confidence scores, so the review list relies on `[unclear]` and cleanup flags.
  AssemblyAI adds per-word confidence.
- Gemini STT with diarization accepts at most 30 min per request. Longer files are split into 25-min pieces,
  and speaker numbers may not match across pieces. Use AssemblyAI for long calls.
- Free-tier Gemini keys: Pro models (`gemini-4-argon`, `gemini-3.1-pro-preview`) have zero quota, and
  Google may use free-tier inputs to improve its products. Use a billing-enabled key for real customer calls.
- Prices in `transcriber/cost.py` were checked on 2026-10-29. Several rise on 2026-12-01 / 2027-01-01.

## Tests

```bash
pytest -m "not integration"   # offline unit tests
pytest                        # + end-to-end on tests/fixtures/sample_call.mp3 (needs GEMINI_API_KEY)
```

`tests/fixtures/sample_call.mp3` is a 72 s synthetic two-speaker Hinglish phone call
(Gemini TTS → 300–3400 Hz band, 8 kHz, background noise). `make_sample.py` regenerates it,
and `sample_call.reference.txt` is its exact script.
