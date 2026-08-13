# Hallucination Detection System

A Retrieval-Augmented pipeline for **detecting and mitigating hallucinations in Large Language Model responses**. The system ingests reference documents into a vector store, retrieves semantically relevant evidence for a claim or query, and cross-checks LLM-generated text against that evidence to produce a structured, explainable trust verdict.

Given a raw LLM-generated response, the system:

1. Extracts discrete, independently checkable factual claims from the response text.
2. Retrieves semantically relevant evidence for those claims from a document corpus indexed in a vector store.
3. Compares each claim against the retrieved evidence to detect unsupported, fabricated, or contradicted statements.
4. Aggregates per-claim outcomes into response-level trust, reliability, and confidence metrics.
5. Produces a final, structured JSON verdict, exposed over a FastAPI REST API.

It also exposes the document-ingestion, evidence-retrieval, and claim-verification stages as standalone endpoints, so each capability can be used independently of the end-to-end `/analyze` pipeline.

---

## Features

**Document Ingestion**
- Accepts PDF, DOCX, TXT, and Markdown source documents via a single upload endpoint.
- Validates file type, extracts and cleans text, and tracks each document's status through the pipeline (`text_extracted` → `cleaned` → `indexed`).
- Persists document-level metadata (filename, size, page count, author, title) in a lightweight JSON-file-backed registry so it survives process restarts.

**Chunking**
- Splits cleaned document text into overlapping chunks sized for embedding, preserving traceability back to the source document and page.

**Embedding & Vector Storage**
- Generates dense vector embeddings for each chunk using a Sentence-Transformers model.
- Persists embeddings in a local ChromaDB collection for semantic similarity search.

**Evidence Retrieval**
- Given a natural-language query, retrieves the top-k most similar chunks from ChromaDB, applies a similarity-score threshold, deduplicates results, and returns each result with full source attribution (document, filename, page number).
- Supports restricting retrieval to a single document.

**Claim Verification**
- Given one or more claim texts, retrieves relevant evidence for each and computes per-evidence support and contradiction scores.
- Returns an overall verification status (`supported` / `contradicted` / `insufficient_evidence` / `unverified`), a confidence score, and a human-readable explanation.
- Supports both single-claim and batch verification.

**Claim Extraction**
- spaCy-based sentence segmentation and Named Entity Recognition.
- Heuristic filtering of questions, greetings, instructions, and hedged/opinion statements.
- Claim classification into `NUMERIC`, `TEMPORAL`, `ENTITY_CENTRIC`, `FACTUAL`, `OPINION`, and `UNVERIFIABLE` types.

**Hallucination Detection**
- Retrieves evidence via a pluggable `EvidenceSource` interface. By default this is a dependency-free lexical-overlap implementation; it can be backed by the system's own semantic `Retriever` (ChromaDB-based) so claims are checked against real, embedding-retrieved evidence instead of an empty in-memory corpus.
- Computes support and entity-agreement scores between each claim and its best-matching evidence.
- Flags polarity mismatches (negation) between a claim and its evidence using spaCy's dependency parse.

**Confidence Scoring**
- Computes a weighted Trust Score, Reliability Score, Hallucination Probability, and overall Confidence Score.
- Produces a coarse `LOW` / `MEDIUM` / `HIGH` risk level, weighted by each claim's extraction confidence.

**Response Analysis**
- Orchestrates claim extraction, hallucination detection, and confidence scoring end-to-end.
- Derives a final human-facing Verdict (`reliable`, `mostly_reliable`, `questionable`, `unreliable`, `no_verifiable_claims`) with an explicit, human-readable reason.

**REST API**
- FastAPI application exposing all of the above as versioned, documented endpoints, with a pre-warmed NLP pipeline at startup and structured global exception handling.

**Test Suite**
- pytest tests covering claim extraction, hallucination detection, confidence scoring, and response analysis (normal, edge, and invalid-input cases), using constructor-injected fakes for the orchestrator so most tests don't require loading spaCy.

---

## Complete Backend Architecture

```
                     ┌────────────────────────┐
                     │   Client / API Caller   │
                     └────────────┬─────────────┘
                                  │
                        FastAPI Application
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        │                         │                              │
 ┌──────▼──────┐         ┌────────▼────────┐            ┌────────▼────────┐
 │  /documents  │         │   /retrieval    │            │  /verification   │
 │   (upload,   │         │  (query for     │            │  (verify one or  │
 │  list, get,  │         │   evidence)     │            │  many claims)    │
 │   delete)    │         └────────┬────────┘            └────────┬─────────┘
 └──────┬───────┘                  │                              │
        │                          │                              │
 ┌──────▼───────┐                  │                              │
 │  Ingestion    │                 │                              │
 │  Pipeline:    │                 │                              │
 │  validate →   │                 │                              │
 │  extract →    │                 │                              │
 │  clean →      │                 │                              │
 │  chunk        │                 │                              │
 └──────┬───────┘                  │                              │
        │                          │                              │
 ┌──────▼───────┐          ┌───────▼────────┐            ┌────────▼────────┐
 │  Embedding    │          │   Retriever    │◄───────────┤  FactVerifier    │
 │  Service      │──────────┤ (similarity     │            │ (support /       │
 │ (Sentence-    │  embeds  │  search over    │  evidence  │  contradiction   │
 │ Transformers) │  chunks  │  ChromaDB)      │            │  scoring)        │
 └──────┬───────┘          └───────┬────────┘            └────────┬────────┘
        │                          │                              │
 ┌──────▼──────────────────────────▼──────┐                       │
 │             ChromaDB Vector Store        │                       │
 └───────────────────────────────────────┘                       │
                                                                    │
                     ┌──────────────────────────────────┐          │
                     │              /analyze              │          │
                     └────────────────┬───────────────────┘          │
                                       │                              │
                            ┌──────────▼───────────┐                  │
                            │   ClaimExtractor      │                  │
                            │  (spaCy NER + rules)  │                  │
                            └──────────┬───────────┘                  │
                                       │                              │
                            ┌──────────▼────────────┐                 │
                            │ HallucinationDetector  │◄────────────────┘
                            │ (EvidenceSource-backed,│  optional: evidence
                            │ lexical + entity +     │  sourced via Retriever
                            │ negation checks)       │
                            └──────────┬────────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  ConfidenceScorer     │
                            │ (trust/reliability/   │
                            │  hallucination score) │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   ResponseAnalyzer     │
                            │  (final Verdict + JSON)│
                            └────────────────────────┘
```

---

## Updated Folder Structure

```
backend/
├── app/
│   ├── main.py                        # FastAPI app, router mounting, exception handlers
│   ├── core/
│   │   ├── config.py                  # Pydantic settings (shared by every module)
│   │   ├── exceptions.py              # Shared application exception hierarchy
│   │   └── logging.py                 # Logger configuration
│   ├── api/
│   │   ├── analysis.py                # POST /analyze router
│   │   └── routes/
│   │       ├── documents.py           # /documents router
│   │       ├── retrieval.py           # /retrieval router
│   │       └── verification.py        # /verification router
│   ├── ingestion/
│   │   ├── document_processor.py      # Orchestrates validation + text extraction
│   │   ├── file_validator.py          # File type / size validation
│   │   ├── pdf_loader.py              # PDF text extraction (pypdf)
│   │   ├── docx_loader.py             # DOCX text extraction (python-docx)
│   │   ├── markdown_loader.py         # Markdown text extraction
│   │   └── txt_loader.py              # Plain text extraction
│   ├── processing/
│   │   ├── chunker.py                 # TextChunker
│   │   └── text_cleaner.py            # Document text cleaning
│   ├── embeddings/
│   │   └── embedding_service.py       # Sentence-Transformers embedding generation
│   ├── vectorstore/
│   │   └── chroma_client.py           # ChromaDB collection management
│   ├── retrieval/
│   │   └── retriever.py               # Retriever (similarity search + attribution)
│   ├── verification/
│   │   └── fact_verifier.py           # FactVerifier (support/contradiction scoring)
│   ├── services/
│   │   ├── claim_extractor.py         # ClaimExtractor
│   │   ├── hallucination_detector.py  # HallucinationDetector, EvidenceSource
│   │   ├── confidence_scorer.py       # ConfidenceScorer
│   │   └── response_analyzer.py       # ResponseAnalyzer (pipeline orchestrator)
│   ├── models/
│   │   ├── claim_model.py             # ClaimModel, ClaimType, VerificationStatus, Entity
│   │   ├── document.py                # Document, DocumentFileType, DocumentStatus
│   │   ├── chunk.py                   # Chunk model
│   │   ├── evidence.py                # Evidence, EvidenceBundle, SourceAttribution
│   │   └── verification.py            # Claim, VerificationResult, EvidenceAssessment
│   ├── schemas/
│   │   ├── document_schema.py         # /documents request/response models
│   │   ├── retrieval_schema.py        # /retrieval request/response models
│   │   └── verification_schema.py     # /verification request/response models
│   └── utils/
│       ├── text_cleaner.py
│       ├── tokenizer.py
│       └── validators.py
├── tests/
│   ├── conftest.py                    # Shared fixtures, markers, sys.path setup
│   ├── test_claim_extractor.py
│   ├── test_hallucination_detector.py
│   ├── test_confidence_scorer.py
│   └── test_response_analyzer.py
├── requirements.txt
├── pytest.ini
├── Dockerfile
└── .env.example
```

---

## Technologies Used

| Technology | Role |
|---|---|
| **Python 3.11** | Core language |
| **FastAPI** | REST API layer |
| **Uvicorn** | ASGI server |
| **spaCy** (`en_core_web_sm`) | Sentence segmentation, NER, dependency-parse-based negation detection |
| **ChromaDB** | Vector store for chunk embeddings and semantic similarity search |
| **Sentence Transformers** | Embedding generation for document chunks and queries |
| **pypdf** | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **Pydantic v2** | Request/response validation and settings management |
| **pytest** | Test framework |
| **Docker** | Multi-stage Dockerfile for containerized deployment |
| **Python logging** | Structured application logging throughout (no `print()` statements) |

---

## Installation

```bash
git clone <repository-url>
cd <repository-root>/backend
```

## Virtual Environment Setup

```bash
python3.11 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

## Dependency Installation

```bash
pip install -r requirements.txt
```

The document-ingestion, embedding, and vector-store modules additionally require the following packages, which should be installed alongside `requirements.txt`:

```bash
pip install chromadb sentence-transformers pypdf python-docx
```

## Required spaCy Model Installation

Claim extraction and hallucination detection require the `en_core_web_sm` model:

```bash
python -m spacy download en_core_web_sm
```

## Running FastAPI

From the `backend/` directory:

```bash
uvicorn app.main:app --reload
```

- API docs (Swagger UI): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Running Tests

From the `backend/` directory:

```bash
pytest
```

---

## API Endpoints

### `POST /api/v1/analyze`
Runs the full hallucination-detection pipeline on a single LLM response and returns a structured verdict.

**Request body**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `response_text` | string | Yes | 1–20,000 characters |

**Response body**

| Field | Type | Description |
|---|---|---|
| `verdict` | string | `reliable` / `mostly_reliable` / `questionable` / `unreliable` / `no_verifiable_claims` |
| `verdict_reason` | string | Human-readable explanation of the verdict |
| `analyzed_at` | string (ISO 8601) | UTC timestamp of the analysis |
| `scores` | object \| null | `trust_score`, `reliability_score`, `hallucination_probability`, `confidence_score`, `risk_level` (null if no claims were found) |
| `claim_summary` | object | `total_claims`, `supported`, `contradicted`, `insufficient_evidence` |
| `claims` | array | Per-claim detail: `id`, `text`, `claim_type`, `entities`, `extraction_confidence`, `verification_status`, `evidence`, `source`, `verified`, `created_at` |

**Example request**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
        "response_text": "The Eiffel Tower is located in Paris, France, and was completed in 1889."
      }'
```

**Example response**

```json
{
  "verdict": "reliable",
  "verdict_reason": "Trust score 1.00, hallucination probability 0.00, reliability 1.00 across 2 claim(s) (2 supported, 0 contradicted, 0 insufficient evidence).",
  "analyzed_at": "2026-08-02T10:15:42.123456+00:00",
  "scores": {
    "trust_score": 1.0,
    "reliability_score": 1.0,
    "hallucination_probability": 0.0,
    "confidence_score": 0.95,
    "risk_level": "low"
  },
  "claim_summary": {
    "total_claims": 2,
    "supported": 2,
    "contradicted": 0,
    "insufficient_evidence": 0
  },
  "claims": [
    {
      "id": 1,
      "text": "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
      "claim_type": "entity_centric",
      "entities": [
        {"text": "The Eiffel Tower", "label": "ORG", "start_char": 0, "end_char": 16},
        {"text": "Paris", "label": "GPE", "start_char": 30, "end_char": 35},
        {"text": "France", "label": "GPE", "start_char": 37, "end_char": 43}
      ],
      "extraction_confidence": 0.92,
      "verification_status": "supported",
      "evidence": ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."],
      "source": null,
      "verified": true,
      "created_at": "2026-08-02T10:15:42.001000+00:00"
    }
  ]
}
```

Status codes: `200 OK` on success, `422 Unprocessable Entity` for invalid/empty input, `500 Internal Server Error` for unexpected failures.

---

### `POST /documents/upload`
Uploads a document (PDF, DOCX, TXT, or Markdown) and runs it through the full ingestion pipeline: validation, text extraction, cleaning, chunking, embedding, and storage in ChromaDB.

**Request:** `multipart/form-data` with a `file` field.

**Example request**

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@company_policy.pdf"
```

**Example response** (`201 Created`)

```json
{
  "document_id": "3f1b2c9e-4a2d-4e8a-9c3a-1a2b3c4d5e6f",
  "source_name": "company_policy.pdf",
  "file_type": "pdf",
  "status": "indexed",
  "file_size_bytes": 154213,
  "page_count": 8,
  "author": null,
  "title": null,
  "created_at": "2026-08-02T09:10:00Z"
}
```

Status codes: `201 Created` on success, `400 Bad Request` if the file fails validation or extraction, `500 Internal Server Error` for unexpected failures.

---

### `GET /documents`
Lists all previously ingested documents.

**Example response**

```json
{
  "total": 1,
  "documents": [
    {
      "document_id": "3f1b2c9e-4a2d-4e8a-9c3a-1a2b3c4d5e6f",
      "source_name": "company_policy.pdf",
      "file_type": "pdf",
      "status": "indexed",
      "file_size_bytes": 154213
    }
  ]
}
```

---

### `GET /documents/{document_id}`
Retrieves full details for a single ingested document.

Status codes: `200 OK`, `404 Not Found` if the document ID does not exist.

---

### `DELETE /documents/{document_id}`
Deletes a document's registry record and removes all of its chunks from ChromaDB.

**Example response**

```json
{
  "document_id": "3f1b2c9e-4a2d-4e8a-9c3a-1a2b3c4d5e6f",
  "deleted": true
}
```

Status codes: `200 OK`, `404 Not Found` if the document ID does not exist.

---

### `POST /retrieval/query`
Retrieves ranked, deduplicated, threshold-filtered evidence for a natural-language query.

**Request body**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query` | string | Yes | 1–5000 characters |
| `top_k` | integer | No | 1–100 (server default if omitted) |
| `similarity_threshold` | float | No | 0.0–1.0 (server default if omitted) |
| `document_id` | string | No | Restrict retrieval to a single document |

**Example request**

```bash
curl -X POST "http://localhost:8000/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the company remote work policy?",
        "top_k": 3
      }'
```

**Example response**

```json
{
  "query": "What is the company remote work policy?",
  "total_results": 1,
  "results": [
    {
      "chunk_id": "3f1b2c9e-chunk-4",
      "text": "Employees may work remotely up to three days per week with manager approval.",
      "similarity_score": 0.87,
      "document_id": "3f1b2c9e-4a2d-4e8a-9c3a-1a2b3c4d5e6f",
      "source_name": "company_policy.pdf",
      "page_number": 3
    }
  ]
}
```

Status codes: `200 OK`, `404 Not Found` if no documents have been ingested yet, `400 Bad Request` for other application-level errors.

---

### `POST /verification/verify`
Verifies a single factual claim by retrieving relevant evidence and comparing it against the claim.

**Request body**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `claim` | string | Yes | 1–5000 characters |
| `source_response_id` | string | No | Identifier of the source LLM response, if known |
| `top_k` | integer | No | 1–100 (server default if omitted) |
| `document_id` | string | No | Restrict evidence retrieval to a single document |

**Example request**

```bash
curl -X POST "http://localhost:8000/verification/verify" \
  -H "Content-Type: application/json" \
  -d '{
        "claim": "Employees can work remotely up to three days per week."
      }'
```

**Example response**

```json
{
  "claim_id": "c1a2b3d4-...",
  "claim_text": "Employees can work remotely up to three days per week.",
  "status": "supported",
  "confidence": 0.91,
  "explanation": "Retrieved evidence directly supports this claim with high similarity.",
  "evidence": [
    {
      "chunk_id": "3f1b2c9e-chunk-4",
      "text": "Employees may work remotely up to three days per week with manager approval.",
      "source_name": "company_policy.pdf",
      "support_score": 0.91,
      "contradiction_score": 0.02
    }
  ]
}
```

Status codes: `200 OK`, `404 Not Found` if no documents have been ingested yet, `400 Bad Request` for other application-level errors.

---

### `POST /verification/verify/batch`
Verifies multiple factual claims in a single call, each independently retrieving its own evidence.

**Request body**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `claims` | array of strings | Yes | 1–50 claims |
| `source_response_id` | string | No | Applied to every claim in the batch |
| `top_k` | integer | No | 1–100 (server default if omitted) |
| `document_id` | string | No | Restrict evidence retrieval to a single document, for every claim |

**Example response**

```json
{
  "total": 2,
  "results": [
    { "claim_id": "...", "claim_text": "...", "status": "supported", "confidence": 0.91, "explanation": "...", "evidence": [] },
    { "claim_id": "...", "claim_text": "...", "status": "insufficient_evidence", "confidence": 0.40, "explanation": "...", "evidence": [] }
  ]
}
```

---

### `GET /`
Returns basic service metadata (name, version, docs URL).

### `GET /health`
Liveness/readiness probe endpoint.

```json
{ "status": "healthy" }
```

---

## Project Workflow

```
Document Upload
      │
      ▼
Validation & Text Extraction  (PDF / DOCX / TXT / Markdown)
      │
      ▼
Cleaning
      │
      ▼
Chunking
      │
      ▼
Embedding Generation  (Sentence Transformers)
      │
      ▼
ChromaDB  (vector storage)
      │
      ▼
Retrieval  (semantic similarity search + source attribution)
      │
      ▼
Verification  (per-evidence support / contradiction scoring)
      │
      ▼
Hallucination Analysis  (claim extraction → detection → confidence scoring)
      │
      ▼
Final Structured Verdict (JSON)
```

---

## Known Limitations

- **`/analyze`'s default evidence source is lexical, not semantic.** Out of the box, `HallucinationDetector` uses a dependency-free, stopword-filtered Jaccard token-overlap `EvidenceSource` with no built-in corpus. It can be backed by the system's ChromaDB-based `Retriever` for real semantic evidence without any change to the detector itself, via constructor injection on `ResponseAnalyzer`.
- **Contradiction detection is a negation heuristic, not full entailment.** It flags polarity mismatches (e.g. "was" vs. "was not") using spaCy's dependency parse plus a small cue-word list; it does not perform natural language inference and can miss semantically contradictory claims that don't use explicit negation.
- **`en_core_web_sm`** is spaCy's small, general-purpose English model. Entity recognition and dependency parsing accuracy are lower than larger or transformer-based models, particularly on domain-specific or unusual named entities.
- **Document metadata persistence is file-based**, not a database — a JSON registry tracks document-level bookkeeping, while chunk vectors are persisted by ChromaDB.
- **No authentication or rate limiting** is implemented on any endpoint.

---

## Contributors

| Contributor | Area of Contribution |
|---|---|
| [Ronika08](https://github.com/Ronika08) | Claim Extraction, Hallucination Detection, Confidence Scoring, Response Analyzer |
| [adhyashreepm](https://github.com/adhyashreepm) | Document Ingestion, Chunking, Embeddings, ChromaDB, Retrieval, Fact Verification |
| [Talluri Kavyashri](https://github.com/Talluri-Kavyashri) | *(update with actual area of contribution)* |
| [bindudasangam-glitch](https://github.com/bindudasangam-glitch) | *(update with actual area of contribution)* |

> Note: the GitHub link for **Talluri Kavyashri** was built from the name as given (`Talluri-Kavyashri`), since GitHub usernames can't contain spaces — please confirm this matches the actual username, or send it over and I'll correct the link.

---

## License

This project is provided as-is for educational and demonstration purposes as part of a final-year engineering project. No license has been formally published for this repository; contact the repository owner for reuse or distribution terms.
