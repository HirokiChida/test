import glob
import pickle

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_index(data_path="data/*.txt", index_path="index.faiss", docs_path="docs.pkl"):
    """Load text files, create embeddings, and build a FAISS index."""
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    docs = []
    for fname in glob.glob(data_path):
        with open(fname, "r", encoding="utf-8") as f:
            docs.append(f.read())
    if not docs:
        raise ValueError("No documents found. Place .txt files under 'data/'")
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)


def load_index(index_path="index.faiss", docs_path="docs.pkl"):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, docs, embedder


def answer(query, llm_model="gpt2", top_k=3):
    index, docs, embedder = load_index()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, top_k)
    context = "\n".join(docs[i] for i in I[0])
    prompt = f"Use the following context to answer.\n{context}\n\nQuestion: {query}\nAnswer:"
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple RAG example")
    parser.add_argument("--build", action="store_true", help="Build index from data/")
    parser.add_argument("query", nargs="*", help="Query to answer")
    args = parser.parse_args()

    if args.build:
        build_index()
    else:
        if not args.query:
            parser.error("Provide a query or use --build to build index")
        response = answer(" ".join(args.query))
        print(response)
