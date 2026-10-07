from functools import lru_cache
import os
import sys

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8
MODEL = os.getenv("LLM_MODEL", "").strip()

SYSTEM_PROMPT = """You are a careful research assistant.
Answer the question using ONLY the numbered context excerpts provided.
Cite the excerpts you used in square brackets, like [1] or [2][3].
If the context does not contain the answer, reply exactly:
"I can't find this in the provided papers."
Do not use outside knowledge. Answer in the language of the question."""

@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)


def retrieve(collection, question, k=TOP_K):
    """Zoek de k meest relevante stukjes tekst bij de vraag."""
    results = collection.query(query_texts=[question], n_results=k)
    return results["documents"][0], results["metadatas"][0]


def build_context(docs, metas):
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        parts.append(f"[{i}] (source: {meta['source']})\n{doc}")
    return "\n\n".join(parts)


def ask(question):
    collection = get_collection()
    docs, metas = retrieve(collection, question)
    context = build_context(docs, metas)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    answer = response.content[0].text
    sources = sorted({m["source"] for m in metas})
    return answer, sources


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Vraag: ")
    answer, sources = ask(question)
    print("\nAntwoord:\n" + answer)
    print("\nBronnen:")
    for s in sources:
        print(f"- {s}")