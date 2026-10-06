from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

PAPERS_DIR = Path("data/papers")
DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_WORDS = 150  # ~200 tokens: past binnen de 256-tokengrens van het embeddingmodel
OVERLAP_WORDS = 30


def extract_text(pdf_path: Path) -> str:
    """Lees alle pagina's van een PDF en geef de tekst terug als één string."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return " ".join(pages)


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Knip tekst in stukken van `chunk_words` woorden, met `overlap` woorden overlap."""
    if overlap >= chunk_words:
        raise ValueError("overlap moet kleiner zijn dan chunk_words")
    words = text.split()
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def build_index() -> None:
    client = chromadb.PersistentClient(path=DB_DIR)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Begin altijd schoon, zodat je niet per ongeluk dubbele stukken opslaat.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"Geen PDF's gevonden in {PAPERS_DIR}/")

    total_chunks = 0
    for pdf_path in pdf_files:
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            print(f"Overgeslagen (geen tekst): {pdf_path.name}")
            continue
        collection.add(
            ids=[f"{pdf_path.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": pdf_path.name, "chunk": i} for i in range(len(chunks))],
        )
        total_chunks += len(chunks)
        print(f"{pdf_path.name}: {len(chunks)} stukken")

    print(f"Klaar: {total_chunks} stukken opgeslagen in {DB_DIR}/")


if __name__ == "__main__":
    build_index()