import re

from ingest import extract_text

text = extract_text("data/papers/ancoli2003.pdf")
for match in re.finditer(r"references", text, flags=re.IGNORECASE):
    start = max(0, match.start() - 40)
    print(round(match.start() / len(text), 2), repr(text[start : match.end() + 40]))