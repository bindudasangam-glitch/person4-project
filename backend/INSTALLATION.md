# Installation Guide

This guide walks through setting up the **Hallucination Detection System** (Person 1 module) on a completely fresh machine.

---

## Prerequisites

- **Python 3.11** (the project is developed and tested against this version)
- `pip` (bundled with Python)
- Internet access for the initial dependency + spaCy model download

Verify your Python version:

```bash
python3 --version
# Python 3.11.x
```

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-root>/backend
```

All commands below are run from the `backend/` directory unless noted otherwise.

---

## 2. Create a Virtual Environment

```bash
python3.11 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

Your shell prompt should now be prefixed with `(venv)`.

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, spaCy, NLTK, Pydantic, and pytest, along with their transitive dependencies.

---

## 4. Install the spaCy Language Model

The claim extraction and hallucination detection services require `en_core_web_sm`:

```bash
python -m spacy download en_core_web_sm
```

Verify it installed correctly:

```bash
python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"
```

Expected output: `spaCy model OK`

---

## 5. NLTK Setup

`app/utils/tokenizer.py` calls `nltk.download("punkt", quiet=True)` automatically the first time `app.utils` is imported (which happens indirectly whenever `ClaimExtractor` is imported, via `text_cleaner`). **No manual NLTK download step is required** — it happens transparently on first run, provided the machine has internet access at that moment.

If you're deploying to an offline/air-gapped environment, pre-download it manually instead:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 6. Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

You should see log output ending in something like:

```
INFO | app | Starting up Hallucination Detection System v1.0.0...
INFO | app | NLP pipelines pre-warmed successfully.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

The first startup takes a few seconds longer than subsequent restarts — this is the spaCy model being pre-warmed via the app's lifespan hook.

---

## 7. Open the Swagger UI

With the server running, open:

```
http://127.0.0.1:8000/docs
```

You should see interactive API documentation with the `POST /api/v1/analyze` endpoint listed under **Analysis**, plus `GET /` and `GET /health` under **Meta**.

Quick sanity check without the browser:

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}
```

---

## 8. Run the Test Suite

In a separate terminal (with the same virtual environment activated):

```bash
pytest
```

Expected output:

```
59 passed in ~2-13s
```

(Exact timing varies — the two test files that load the real spaCy pipeline are slower than the two that use fakes.)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | spaCy model not downloaded | Run `python -m spacy download en_core_web_sm` (Step 4) |
| `ModuleNotFoundError: No module named 'app'` | Running commands from the wrong directory, or venv not activated | Ensure you're in `backend/` and `(venv)` shows in your prompt |
| `pytest` collection errors mentioning circular/partial imports | Stale `__pycache__` from an older version of the code | `find . -name "__pycache__" -exec rm -rf {} +` then re-run |
| Server starts but `/analyze` returns `500` on first request only | spaCy model failed to pre-warm at startup (check logs for a warning) but loaded lazily instead | Check network/model install; the endpoint self-heals on the next request once the model loads |
| `nltk` download hangs or fails silently on first import | No internet access at import time | Pre-download manually per Step 5, or ignore — nothing in the active pipeline currently calls `Tokenizer` |
| `pip install` fails on a specific package version | Python version mismatch | Confirm `python3 --version` reports 3.11.x — some pinned versions require it |

---

## Expected End-to-End Output

A successful `POST /api/v1/analyze` request with body `{"response_text": "Paris is the capital of France."}` should return `200 OK` with a JSON body containing `"verdict"`, `"scores"`, `"claim_summary"`, and a `"claims"` array — see the main README's **Example Response** section for the full shape.