# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AutoRSR automates scoring of the **Redmond Sentence Recall (RSR)** test: a child listens to and repeats sentences, and the tool transcribes the audio, aligns each spoken attempt to its target sentence, counts word-level errors, and produces a pass/fail screening decision against age/percentile norms. There is no build system, test suite, or linter — it is a small set of Python scripts driven either as a library or a Flask API.

## Two parallel implementations

The repo contains the **same pipeline implemented twice**. They do not share code — pick the right directory before editing.

- **`AutoRsr-NoBert/`** — the current, recommended, documented version. Alignment is pure-Python (fuzzy sentence matching + DP), error analysis is A* edit-distance. Has a Flask REST API. **Default to working here** unless the task is specifically about BertAlign.
- **`AutoRsR_Bert/`** — older/heavier version that uses an external **BertAlign** clone for alignment and a modified Levenshtein + BFS-transposition algorithm for errors. More accurate alignment, much more setup, rougher code (e.g. `batch_process.py` relies on names from `bert.py`'s namespace and is not standalone-runnable). Treat as legacy/research code.

Behavior differs in ways that matter:
- **Edit-operation taxonomy differs.** NoBert emits `Insertions / Deletions / Substitutions / Swaps`; Bert emits `Repetition / Deletion / Insertion / Substitution / Transposition` (Bert additionally detects repetitions and `birdhouse`↔`bird house` fusions in `preprocess()`).
- **`english.json` path differs** (see Gotchas).

## The pipeline (shared mental model)

Both versions are a 5-stage flow. In NoBert the canonical orchestrator is `auto_rsr.run_full_rsr_analysis()`:

1. **Transcribe** — WhisperX `large-v3` on CUDA (`whispherx.py` in Bert, `transcribe()` in NoBert's `auto_rsr.py`).
2. **Standardize/normalize** — `standarize()`: ftfy unicode fix → strip filler tokens (`uh/um/ah`) → `normalize()`, which runs Whisper's `EnglishTextNormalizer` seeded from `english.json` (a British→American spelling + abbreviation map).
3. **Align** — match each ground-truth sentence to the best snippet of the transcription. NoBert: `fuzzy_align_response_to_gt()` (difflib similarity, then word-level DP refinement when a candidate is >1.3× the target length). Bert: `bert.align()` via `Bertalign`, then `clean_bert()`/`find_lines_and_next()` cleanup.
4. **Score errors** — count edits between aligned response and ground truth. NoBert: `score_rsr_errors()` (A* over insert/delete/substitute/swap). Bert: `modified_levenshtein_distance()` + `transpose()`/`transpose2()`.
5. **Decision** — convert per-sentence error counts to points, sum, then threshold.
   - **Points (`score_rsr` / `score`):** 0 errors → 2 pts, 1–3 errors → 1 pt, ≥4 errors → 0 pts.
   - **Pass/fail (`rsr_pass_fail` / `calculate_result`):** hardcoded thresholds for **ages 5–9 only** (out of range → `"N/A"`). **Age is always in MONTHS** at every API boundary; the function internally splits into years/months. A score *above* the norm threshold is `"Pass"`.

## Commands

Each version has its own pinned `requirements.txt`; use a separate conda/venv per version (CUDA GPU + `ffmpeg` on PATH required for WhisperX).

```bash
# Install (run inside the chosen version's directory)
pip install -r requirements.txt
```

**NoBert — Flask API.** Note: the README says `python app.py`, but there is no `app.py`; the server is `api_interface.py`:
```bash
cd AutoRsr-NoBert
python api_interface.py        # serves http://127.0.0.1:5000 (debug=True)
```
Endpoints: `POST /transcribe`, `/align`, `/edit_score`, `/decision`, and the full pipeline `/analyze` (multipart `wav_file` + `Age` + optional `Percentile`, `Ground Truth`). Per-endpoint I/O formats are documented in the comment block at the top of `api_interface.py` and in `AutoRsr-NoBert/README.md`.

**NoBert — library:**
```python
from auto_rsr import run_full_rsr_analysis
result = run_full_rsr_analysis("file.wav", age_in_months=65, percentile=5, ground_truth_text=None)
```
If `ground_truth_text` is omitted, a default 16-sentence RSR set hardcoded in `align_transcription_to_ground_truth()` is used.

**Bert — batch:** there is no CLI entry point; follow the usage sample in the docstring at the bottom of `AutoRsR_Bert/bert.py` (transcribe → `align` → `auto_rsr.prepare` → `auto_rsr.batch`). `batch_process.py` shows the directory-iteration pattern but is not runnable as-is.

There is no single test/lint command. To smoke-test a change, exercise the relevant pipeline stage directly (each standalone script has a usage sample at the bottom).

## Environment gotchas (these will bite)

- **`english.json` path is hardcoded and inconsistent.** NoBert opens `"english.json"` (relative — **you must run from inside `AutoRsr-NoBert/`** or it `FileNotFoundError`s). Bert opens `"/english.json"` (filesystem root). Bert's `batch_process.py` also writes to `/output.txt` (root). Fix the path for the local environment rather than copying files to `/`.
- **WhisperX `asr.py` patch.** This pinned `whisperx==3.1.1` may raise `TypeError: <lambda>() missing 3 required positional arguments: 'max_new_tokens', 'clip_timestamps', 'hallucination_silence_threshold'`. Fix: in the installed `whisperx/asr.py` (~line 322), after `"suppress_numerals": False,` add `"max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None,`. Documented in `AutoRsr-NoBert/README.md`.
- **GPU is assumed.** `device="cuda"` and `compute_type="float16"` are hardcoded in the transcribe functions; change both for CPU/low-memory runs.
- **Bert requires manual external setup.** Clone [BertAlign](https://github.com/bfsujason/bertalign), add a `return_sents()` method to its `aligner.py` (snippet in `AutoRsR_Bert/readme.txt`), then point `bert.py` line 6 (`/directory_of_bert/__init__.py`) at the clone. `bert.py` loads it via `SourceFileLoader`.
