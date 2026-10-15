# RAG on Edge

End-to-end Retrieval-Augmented Generation running fully on-device using
[Qdrant Edge](https://qdrant.tech/) for vector search and
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) for embeddings and
generation — no network calls at inference time.

The pipeline indexes the **Gemma 4 technical report** PDF and answers questions
about it using EmbeddingGemma2 (retrieval) and Gemma 4 E2B (generation).

## Components

| File | Purpose |
|------|---------|
| `download_models.py` | Downloads the two `.litertlm` model files from Hugging Face. |
| `index.py` | Parses the PDF, chunks it, embeds the chunks, and stores them in a local Qdrant Edge shard (`db_gemma`). |
| `app.py` | Loads the shard + models and answers a set of queries via retrieval + Gemma 4 generation. |
| `quiet_native.py` | Silences noisy native (C++) stderr output from the runtime. |

## Models

- **Embedding:** `litert-community/embeddinggemma-2-740m-litert-lm` → `models/embeddinggemma2/embeddinggemma-2-740m.litertlm` (768-dim)
- **Generation:** `litert-community/gemma-4-E2B-it-litert-lm` → `models/gemma4/gemma-4-E2B-it.litertlm`

Only the single required `.litertlm` file is pulled per model.

## Setup

```bash
# install dependencies
pip install -e .

# download the embedding + generation models
python download_models.py          # or: runedge-download-model
```

## Usage

```bash
# 1. build the vector index from the PDF (../sample-data/gemma4_technical_report.pdf)
python index.py                    # or: runedge-index

# 2. run retrieval + generation over the sample queries
python app.py
```

`index.py` creates/loads the shard at `db_gemma/`; if the shard already exists it
is reused rather than rebuilt. Both scripts cache runtime artifacts in `.cache/`.

## Requirements

- Python >= 3.11
- A GPU-capable backend (both the embedding and generation engines use `Backend.GPU()`)
