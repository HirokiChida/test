# RAG Example

This repository provides a minimal example of Retrieval‑Augmented Generation (RAG).
The script under `examples/` shows how to index local text files with FAISS and
use a language model to answer queries based on those documents.

## Usage

1. Put your `.txt` documents under a `data/` directory.
2. Build the index:

```bash
python examples/rag_simple.py --build
```

3. Ask a question using the index:

```bash
python examples/rag_simple.py "Your question here"
```

The default language model is `gpt2`. Ensure you have the necessary models
available locally or via Hugging Face before running the script.
