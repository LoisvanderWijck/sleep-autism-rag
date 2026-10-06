from ingest import chunk_text, strip_references


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_short_text_gives_one_chunk():
    assert chunk_text("one two three") == ["one two three"]


def test_chunks_respect_max_size():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_words=150, overlap=30)
    assert all(len(c.split()) <= 150 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, chunk_words=100, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_no_words_are_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_words=100, overlap=20)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_strip_references_removes_bibliography():
    text = "Intro text. " * 50 + "\nREFERENCES\n1. Smith J. Something. 2001."
    assert "Smith" not in strip_references(text)


def test_strip_references_handles_numbered_heading():
    text = "Intro text. " * 50 + "\n9.0 REFERENCES\n1. Smith J. Something. 2001."
    assert "Smith" not in strip_references(text)


def test_strip_references_ignores_heading_at_start():
    text = "\nReferences\n" + "Body text. " * 200
    assert strip_references(text) == text


def test_strip_references_keeps_text_without_heading():
    text = "Intro text. " * 50
    assert strip_references(text) == text