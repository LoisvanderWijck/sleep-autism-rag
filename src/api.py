from fastapi import FastAPI
from pydantic import BaseModel, Field

from ask import ask

app = FastAPI(title="Sleep & Autism RAG")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask_endpoint(body: Question):
    answer, sources = ask(body.question)
    return Answer(answer=answer, sources=sources)