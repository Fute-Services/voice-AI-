# voice-AI: Hinglish call transcription

Audio file in → verbatim transcript with speaker labels and timestamps out
(`.json`, `.md`, `.txt`), plus a review list of spots a human should check.

```
audio ──ffmpeg──▶ 16 kHz mono WAV ──STT──▶ raw segments ──Gemini cleanup──▶ transcript ──Gemini──▶ customer decision
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
| `--no-analysis` | skip the customer decision step |
| `--context "Projects: Green Valley, Sector 150"` | project/location/agent names to help the decision |
| `--reference ref.txt` | compute word error rate against a reference |

Every setting also has an env var; see `.env.example`.

### Customer decision (real-estate calls)

After transcription, Gemini reads the call and decides what the **customer** said:

| outcome | meaning |
|---|---|
| `site_visit_scheduled` | agreed to a visit with a day/date fixed |
| `interested` | wants details / brochure / visit, nothing fixed yet |
| `follow_up` | no yes/no yet: "sochke batata hoon", "family se baat karke", "next week call karo" |
| `not_interested` | declined, already bought, budget mismatch, don't call |
| `wrong_number` | wrong person / not a lead / not a sales call |
| `unclear` | too short, cut off, or can't tell |

It also returns lead temperature (hot/warm/cold), confidence, a short summary, the next action,
the follow-up day if given, requirements (budget, location, BHK, property type, purpose, timeline),
objections, and 1-3 **evidence quotes** from the customer. Each quote is checked against the transcript.
Quotes that aren't found are dropped. If nothing supports the decision, it is set to low confidence and
marked `needs_review`.

**Customer mindset and sales coaching.** Each analysis also includes:

- `mindset`: buying motive, urgency, decision maker, financing, main concern, competing projects, and a
  suggested pitch angle for the next call. Anything the call did not reveal stays `unknown`.
- `buying_signals` / `risk_signals`: each backed by a customer quote that is checked against the transcript
  (unbacked signals are removed).
- `discovery`: the state of each of the 11 discovery topics: `covered` (with what the customer said),
  `dodged` (the agent asked, the customer did not answer) or `missed` (never came up).
- `next_questions`: up to 5 questions for the next conversation, written for this customer in Hinglish and
  English with a one-line reason each. They build on what the customer said, never re-ask something already
  answered, and fit the stage of the lead (no booking questions for a busy or cold customer, none for a
  wrong number). If the model writes too few, the fixed bank (`DISCOVERY` in `transcriber/analysis.py`)
  tops the list up to 3; those are marked `"tailored": false`.
- `lead_score` (0-100): computed in code from the outcome, urgency, decision maker, financing, and whether
  budget, location and motive are known.

**Agent coaching.** The same analysis call also judges the **sales agent**:

- `agent_performance.skills`: 7 skills (opening, rapport, need discovery, relevant pitch, objection handling,
  closing, professionalism), each 1-5 or 0 = "did not come up", with a one-line reason and an agent quote.
- `agent_performance.score` (0-100) and `grade` (A-D) are computed in code from the skills that applied.
  Wrong-number calls get no score, so they don't drag an agent's average down.
- `strengths`, `improvements` (each with the agent's line and a `better_line`: what to say instead, in
  Hinglish) and `compliance` (pressure tactics, false promises, made-up offers).
- `key_moments`: up to 6 moments that mattered (buying signal, objection, price talk, commitment...),
  pinned to real segments so the UI can play them.
- `sentiment_arc`: the customer's mood at the start, middle and end of the call.

**Call metrics (free, no model call).** `meta.metrics` in the transcript JSON is measured from timestamps
(`transcriber/metrics.py`): talk time and talk ratio per speaker, speaking pace, longest monologue, questions
asked, interruptions, average reply time, silence and long pauses, plus rule-based hints such as "Agent talked
78% of the call".

Re-run only the decision on existing transcripts (no new STT cost):

```bash
python analyze.py out/call.transcript.json
python analyze.py "out/*.transcript.json" --context "Projects: Green Valley (Sector 150, Noida)"
```

### Web UI

```bash
python app.py        # then open http://localhost:8000
```

Upload a file (or try the sample call), pick the script and cleanup model, and read the transcript.
Click any line to play that part of the audio, and download the JSON/MD/TXT/review files.
Runs are saved under `out/web/`.

- **Agent coaching** tab: score ring, skill scorecard, coaching points with better lines, talk-time split and
  call-health hints.
- **Ask AI** tab: ask anything about the call in English or Hinglish ("Customer ka budget kya hai?"). Answers
  come only from the transcript and cite the lines they used; click a cited line to play it.
- **Dashboard** (top bar, or press `D`): KPIs across every saved call (hot leads, site visits, average lead and
  agent score), outcome breakdown, upcoming follow-ups, and a searchable, sortable table of all calls.
  Click a row to reopen that call. **Export leads (CSV)** downloads the table for Excel / Google Sheets.
- Links are shareable: `/#dashboard`, `/#run/<id>`, `/#run/<id>/coaching`.

| endpoint | what it returns |
|---|---|
| `GET /api/runs` | every saved call, newest first, with outcome, scores and follow-up |
| `GET /api/runs/{id}` | one saved call in the same shape as `/api/transcribe` |
| `POST /api/runs/{id}/ask` | `{"question": "...", "history": [...]}` → answer + checked citations |
| `GET /api/leads.csv` | all analysed calls as CSV |

### Outputs (in `--out`)

- `<name>.transcript.json`: `segments[]` with `id, speaker, start, end, text, confidence`, plus
  `raw_text` (STT before cleanup), `words[]` (word timestamps from STT), `flags`, `review`, `cost`
- `<name>.transcript.md` / `.txt`: readable `[hh:mm:ss] Speaker N: text`
- `<name>.analysis.json` / `.analysis.md`: the customer decision (also under `analysis` in the transcript JSON)
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
