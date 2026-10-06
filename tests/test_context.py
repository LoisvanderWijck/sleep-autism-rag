from ask import build_context


def test_build_context_numbers_chunks_and_names_sources():
    docs = ["alpha", "beta"]
    metas = [{"source": "a.pdf"}, {"source": "b.pdf"}]
    context = build_context(docs, metas)
    assert "[1] (source: a.pdf)" in context
    assert "[2] (source: b.pdf)" in context