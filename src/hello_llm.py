import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # leest .env en zet de waarden in os.environ

MODEL = os.getenv("LLM_MODEL")
if not MODEL:
    raise SystemExit("Zet LLM_MODEL in je .env (zie .env.example).")

client = anthropic.Anthropic()  # pakt ANTHROPIC_API_KEY automatisch uit de omgeving

SYSTEM_PROMPT = (
    "You are a careful research assistant. "
    'Answer ONLY with valid JSON in exactly this shape: '
    '{"answer": "<one sentence>", "confidence": "low|medium|high"}. '
    "No text outside the JSON."
)


def extract_json(text: str) -> dict:
    """Haal het JSON-object uit de modeloutput, ook als er ```json-hekjes omheen staan."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Geen JSON gevonden in: {text!r}")
    return json.loads(text[start : end + 1])


def main() -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Why is sleep important for memory?"}],
    )
    raw = response.content[0].text
    data = extract_json(raw)
    print("Antwoord:  ", data["answer"])
    print("Zekerheid: ", data["confidence"])


if __name__ == "__main__":
    main()