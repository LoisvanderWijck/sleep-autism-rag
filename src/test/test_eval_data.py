import json
from pathlib import Path

import pytest

QUESTIONS = Path("eval/questions.json")
PAPERS = Path("data/papers")


def load():
    return json.loads(QUESTIONS.read_text())


def test_questions_file_has_both_sections():
    data = load()
    assert len(data["in_scope"]) >= 10
    assert len(data["out_of_scope"]) >= 5


def test_every_in_scope_item_is_well_formed():
    for item in load()["in_scope"]:
        assert item["question"].strip().endswith("?")
        assert item["source"].endswith(".pdf")


@pytest.mark.skipif(not PAPERS.exists(), reason="papers are not in the repo")
def test_expected_sources_exist():
    for item in load()["in_scope"]:
        assert (PAPERS / item["source"]).exists(), item["source"]