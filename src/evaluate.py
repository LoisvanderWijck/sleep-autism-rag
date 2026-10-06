import json
from pathlib import Path

from ask import ask, get_collection, retrieve

QUESTIONS_FILE = Path("eval/questions.json")
REFUSAL = "can't find this in the provided papers"


def main():
    data = json.loads(QUESTIONS_FILE.read_text())
    collection = get_collection()

    # Deel 1: vindt de zoekstap het juiste paper?
    hits = 0
    top1 = 0
    for item in data["in_scope"]:
        docs, metas = retrieve(collection, item["question"])
        sources = [m["source"] for m in metas]
        hit = item["source"] in sources
        hits += hit
        top1 += sources[0] == item["source"]
        print(f"{'OK  ' if hit else 'MISS'} {item['question']}")
        if not hit:
            print(f"     verwacht: {item['source']}")
            print(f"     kreeg:    {sorted(set(sources))}")
    n = len(data["in_scope"])
    print(f"\nRetrieval hit rate: {hits}/{n} = {hits / n:.0%}\n")
    print(f"Juiste paper als eerste: {top1}/{n} = {top1 / n:.0%}\n")

    # Deel 2: weigert het systeem bij vragen buiten de papers?
    refusals = 0
    for question in data["out_of_scope"]:
        answer, _ = ask(question)
        normalized = answer.lower().replace("\u2019", "'")
        ok = REFUSAL in normalized
        refusals += ok
        print(f"{'OK  ' if ok else 'FAIL'} {question}")
        if not ok:
            print(f"     antwoord: {answer[:150]}")
    m = len(data["out_of_scope"])
    print(f"\nCorrecte weigering: {refusals}/{m} = {refusals / m:.0%}")


if __name__ == "__main__":
    main()