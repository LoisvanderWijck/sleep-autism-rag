# Sleep & Autism RAG

A retrieval-augmented generation (RAG) system that answers questions about 13
scientific papers on sleep, sensorimotor gating and autism. It answers only from
the papers and says so when it cannot find the answer.

## How it works

1. `src/ingest.py` extracts text from the PDFs, removes reference lists, splits
   it into 150-word chunks (30-word overlap) and stores embeddings
   (all-MiniLM-L6-v2) in a Chroma vector database.
2. `src/ask.py` retrieves the most relevant chunks for a question and asks
   Claude to answer using only those chunks, with citations.
3. `src/evaluate.py` measures retrieval and refusal behaviour on a small test set.

## Evaluation

- 15 in-scope questions (one or two per paper) and 8 out-of-scope questions.
- Retrieval: [fill in hit rate after re-ingest]
- Refusal on out-of-scope questions: [fill in]
- I also checked answers manually against the papers.

## Limitations

- **Reference lists polluted retrieval.** A manual check found a false refusal:
  the answer was in the abstract of a paper, but reference-list chunks ranked
  higher. My first evaluation counted those chunks as hits, so it looked better
  than it was. I now strip reference lists before indexing. [add result]
- The evaluation only checks whether the right paper is retrieved, not whether
  the retrieved chunk contains the answer.
- The test set is small and written by me, so it is easy.
- Questions with two parts retrieve worse than single questions.

## Setup

[fill in later: install, create .env from .env.example, run ingest, run ask]

## Data

The papers are not included in this repository (copyright). See the source
table below. [source table comes later]