import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

PAPERS_DIR = Path("data/papers")
DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 150
OVERLAP_WORDS = 30


def extract_text(pdf_path):
    """Haal alle tekst uit een pdf, pagina voor pagina."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


HEADING = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\.?\s+)?(references|bibliography)\s*\n",
    flags=re.IGNORECASE,
)


def strip_references(text):
    """Knip de referentielijst af; die bevat veel trefwoorden maar geen inhoud."""
    matches = list(HEADING.finditer(text))
    for match in reversed(matches):
        if match.start() > len(text) * 0.25:  # negeer een kop in een inhoudsopgave
            return text[: match.start()]
    return text


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Knip tekst in stukjes van 150 woorden met 30 woorden overlap."""
    words = text.split()
    chunks = []
    start = 0
    step = chunk_words - overlap
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def build_index():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Elke run begint schoon, zodat je geen dubbele chunks krijgt
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedder
    )

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Geen pdf's gevonden in {PAPERS_DIR}. Check je map!")
        return

    total_chunks = 0
    for pdf in pdfs:
        full_text = extract_text(pdf)
        text = strip_references(full_text)
        removed = len(full_text.split()) - len(text.split())
        chunks = chunk_text(text)
        if not chunks:
            print(f"WAARSCHUWING: geen tekst gevonden in {pdf.name}")
            continue

        collection.add(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": pdf.name, "chunk": i} for i in range(len(chunks))],
        )
        total_chunks += len(chunks)
        print(f"{pdf.name}: {len(chunks)} chunks ({removed} woorden referenties verwijderd)")

    print(f"\nKlaar: {len(pdfs)} pdf's, {total_chunks} chunks in '{DB_DIR}'")


if __name__ == "__main__":
    build_index()