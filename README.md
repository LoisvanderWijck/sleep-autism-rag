# Sleep & Autism RAG

A small retrieval-augmented generation (RAG) system that answers questions about 13 scientific papers on sleep and sensorimotor gating in autism. Answers use only the retrieved passages, cite them, and refuse when the papers do not contain the answer.

I built it to learn the full LLM workflow on a topic I know from my neuroscience research: ingestion, embeddings, retrieval, grounded generation, evaluation, tests, an API and a container.

## Results

| Item | Result |
|---|---|
| Papers in the index | 13 PDFs |
| Chunks | 885 after removing reference lists (1,368 before) |
| Test questions | 15 about the papers, 8 outside the papers |
| Correct paper among top 5 retrieved chunks | 15 of 15 |
| Correct paper ranked first | 15 of 15 |
| Correct refusals outside the papers | 8 of 8 (including 3 plausible trap questions) |
| Automated tests (pytest) | 16 passed |
| API | FastAPI: `GET /health`, `POST /ask` |
| Docker | Image of 2.48 GB (CPU-only PyTorch); the API answers `POST /ask` from the container |

These numbers come from a small question set I wrote myself, so read them as a sanity check and not as a benchmark. See [Limitations](#limitations).

## How it works

```
PDFs -> extract text -> strip reference lists -> chunk (150 words, 30 overlap)
     -> embed (all-MiniLM-L6-v2) -> store in Chroma

question -> embed -> retrieve top 5 chunks -> prompt Claude with numbered context
         -> answer with [1], [2] citations, or "I can't find this in the provided papers."
```

- **Ingestion** (`src/ingest.py`): `pypdf` extracts text, the reference list is cut off, the text is split into overlapping 150-word chunks, and each chunk is embedded with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions) and stored in a persistent Chroma collection.
- **Answering** (`src/ask.py`): the question is embedded, the 5 nearest chunks are retrieved and numbered, and the Anthropic API generates an answer. The system prompt says to answer only from the context, cite chunk numbers, answer in the language of the question, and use a fixed refusal sentence when the context is not enough.
- **API** (`src/api.py`): FastAPI with Pydantic validation (question of 3 to 500 characters) returning `answer` and `sources`.

## Project structure

```
sleep-autism-rag/
  src/
    ingest.py            build the index from data/papers
    ask.py               retrieve + generate (CLI: python src/ask.py "question")
    evaluate.py          retrieval and refusal evaluation
    api.py               FastAPI app
    debug_retrieval.py   print the top 8 retrieved chunks for a question
    check_refs.py        find where a PDF's reference list starts
  eval/questions.json    15 in-scope and 8 out-of-scope questions
  tests/                 pytest tests (chunking, context, eval data, API)
  Dockerfile  .dockerignore  requirements.txt  pytest.ini  .env.example
```

## Setup

Requires Python 3.12 (newer versions broke some dependencies for me).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add your own key
```

In `.env`:

```
ANTHROPIC_API_KEY=your-key-here
LLM_MODEL=claude-haiku-4-5-20251001
```

Keep `.env` out of Git (it is in `.gitignore`) and do not leave spaces after values.

The PDFs are not included because of copyright. Put your own papers in `data/papers/`, then build the index:

```bash
python src/ingest.py
```

## Usage

Ask a question from the command line:

```bash
python src/ask.py "What is prepulse inhibition?"
```

Run the API locally and open `http://127.0.0.1:8000/docs`:

```bash
uvicorn api:app --app-dir src
```

Run it in Docker (the index and key are mounted at run time, not baked into the image):

```bash
docker build -t sleep-autism-rag .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/chroma_db:/app/chroma_db" sleep-autism-rag
```

Run the evaluation and the tests:

```bash
python src/evaluate.py
pytest
```

## Evaluation

`eval/questions.json` holds 15 questions with the paper that should answer each, and 8 out-of-scope questions (5 nonsense, 3 plausible but not covered, such as a risperidone dose for children with autism).

`src/evaluate.py` reports two things:

1. **Retrieval:** is the expected paper among the 5 retrieved chunks, and is it the first one?
2. **Refusal:** does the answer to an out-of-scope question contain the refusal sentence?

I also checked answers by hand, which found the one real bug so far.

### A bug the evaluation helped find

A manual check of a question about the Baker 2015 paper returned "I can't find this", although the answer was in the abstract. Printing the retrieved chunks with `src/debug_retrieval.py` showed that 6 of 8 were reference lists, which match author and journal names very well. Removing reference lists before chunking fixed it and reduced the index from 1,368 to 885 chunks. One paper (ancoli2003) used a heading like "9.0 REFERENCES" at 36% of the text, which my first pattern missed, so I allowed optional numbering and lowered the position threshold to 25%. The cleanup is covered by unit tests.

## Limitations

- **Small, self-written test set.** The questions are fairly direct, and I wrote them knowing the papers, so 15 of 15 is likely optimistic.
- **Paper-level retrieval metric.** A hit means the right paper was retrieved, not that the right passage was. Passage-level labels would be stricter.
- **Refusal check is a string match** on the fixed refusal sentence, so it will miss answers that refuse in other words.
- **Two-part questions work worse**, because one query embedding has to match both parts.
- **`sources` is not the same as citations.** The API lists the papers the retrieved chunks came from, not only the ones the answer cites. The `[1]`, `[2]` numbers in the text refer to the retrieved chunks, not to the `sources` list.
- **Cutting at the reference heading can remove appendices or tables** that come after it, for example in ancoli2003.
- **Docker image size:** the first image was 10.2 GB because pip installed the full CUDA build of PyTorch. The Dockerfile now installs the CPU build, and the image is 2.48 GB.
- **The API has no authentication.** Run it locally and do not expose it to the internet.
- **Retrieval uses embeddings only.** No re-ranking and no hybrid keyword search yet.

## Design choices

- **Small chunks with overlap (150 words, 30 overlap):** the embedding model reads at most 256 tokens, so longer chunks would be partly ignored.
- **Local embeddings, hosted LLM:** embeddings are free and need no key, and only the final answer calls the Anthropic API.
- **Grounded prompt with a fixed refusal sentence:** it makes refusals testable.
- **Chroma persistent store:** one folder, no server to run.
- **Tests that mock the LLM:** the API tests replace `ask` so they run without a key or network.

## Data and sources

The 13 papers are PDFs I collected for my own research on sleep, sensorimotor gating (prepulse inhibition) and autism. They are not redistributed here because of copyright. Reference details below were taken from each paper's first page.

| File | Reference | DOI |
|---|---|---|
| Ahmari et al 2012 | Ahmari SE, Risbrough VB, Geyer MA, Simpson HB (2012). Impaired sensorimotor gating in unmedicated adults with obsessive-compulsive disorder. *Neuropsychopharmacology* 37:1216-1223. | [10.1038/npp.2011.308](https://doi.org/10.1038/npp.2011.308) |
| ancoli2003 | Ancoli-Israel S, Cole R, Alessi C, Chambers M, Moorcroft W, Pollak CP (2003). The role of actigraphy in the study of sleep and circadian rhythms. American Academy of Sleep Medicine review paper. *Sleep* 26(3):342-392. | [Publisher page](https://academic.oup.com/sleep/article/26/3/342/2708388) |
| antsheletal2019 | Antshel KM, Russo N (2019). Autism spectrum disorders and ADHD: overlapping phenomenology, diagnostic issues, and treatment considerations. *Current Psychiatry Reports* 21:34. | [10.1007/s11920-019-1020-5](https://doi.org/10.1007/s11920-019-1020-5) |
| Baker2015 | Baker EK, Richdale AL (2015). Sleep patterns in adults with a diagnosis of high-functioning autism spectrum disorder. *Sleep* 38(11):1765-1774. | [10.5665/sleep.5160](https://doi.org/10.5665/sleep.5160) |
| Boele2023 | Boele HJ et al. (2023). Accessible and reliable neurometric testing in humans using a smartphone platform. *Scientific Reports* 13:22871. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10739701/)) | [10.1038/s41598-023-49568-2](https://doi.org/10.1038/s41598-023-49568-2) |
| Braff2001. PPI | Braff DL, Geyer MA, Swerdlow NR (2001). Human studies of prepulse inhibition of startle: normal subjects, patient groups, and pharmacological studies. *Psychopharmacology* 156:234-258. | [10.1007/s002130100810](https://doi.org/10.1007/s002130100810) |
| Buysse | Buysse DJ (2014). Sleep health: can we define it? Does it matter? *Sleep* 37(1):9-17. | [10.5665/sleep.3298](https://doi.org/10.5665/sleep.3298) |
| cheng2018 | Cheng C-H, Chan P-YS, Hsu S-C, Liu C-Y (2018). Meta-analysis of sensorimotor gating in patients with autism spectrum disorders. *Psychiatry Research* 262:413-419. | [10.1016/j.psychres.2017.09.016](https://doi.org/10.1016/j.psychres.2017.09.016) |
| cheng2020 | Cheng Y-C, Huang Y-C, Huang W-L (2020). Heart rate variability in individuals with autism spectrum disorders: a meta-analysis. *Neuroscience & Biobehavioral Reviews* 118:463-471. | [10.1016/j.neubiorev.2020.08.007](https://doi.org/10.1016/j.neubiorev.2020.08.007) |
| Dodds_HRV | Dodds KL, Miller CB, Kyle SD, Marshall NS, Gordon CJ (2017). Heart rate variability in insomnia patients: a critical review of the literature. *Sleep Medicine Reviews* 33:88-100. | [10.1016/j.smrv.2016.06.004](https://doi.org/10.1016/j.smrv.2016.06.004) |
| drake | Drake CL, Hays RD, Morlock R, Wang F, Shikiar R, Frank L, Downey R, Roth T (2014). Development and evaluation of a measure to assess restorative sleep. *Journal of Clinical Sleep Medicine* 10(7):733-741. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4067436/)) | [10.5664/jcsm.3860](https://doi.org/10.5664/jcsm.3860) |
| Frau. SD disrupts PPI in rats et al 2008 | Frau R, Orru M, Puligheddu M, Gessa GL, Mereu G, Marrosu F, Bortolato M (2008). Sleep deprivation disrupts prepulse inhibition of the startle reflex: reversal by antipsychotic drugs. *International Journal of Neuropsychopharmacology* 11(7):947-955. | [10.1017/S1461145708008900](https://doi.org/10.1017/S1461145708008900) |
| Goldman.Defining the Sleep Phenotype in Children With Autism | Goldman SE, Surdyka K, Cuevas R, Adkins K, Wang L, Malow BA (2009). Defining the sleep phenotype in children with autism. *Developmental Neuropsychology* 34(5):560-573. | [10.1080/87565640903133509](https://doi.org/10.1080/87565640903133509) |

## Privacy and cost

- No participant data, internal company data or unpublished work is used.
- Each question sends the question and 5 retrieved text chunks to the Anthropic API.
- Cost per question is small with a Haiku-class model, but it is not zero.
- The answers are for learning and demonstration, not medical advice.

## What I would do next

- Passage-level evaluation with labelled chunks.
- Hybrid search (keyword plus embeddings) and re-ranking.
- Cite only the sources the answer actually used.
- Continuous integration that runs the tests on every push.
