import pytest

from src.dataset import clean

BODY = "Filler sentence to clear the length floor. " * 20


def c(text, **kw):
    kw.setdefault("min_chars", 0)
    return clean(text, **kw)


def test_heading_markers_go_but_the_break_stays():
    assert c("## Career\n\nHe played football.") == "Career\n\nHe played football."


def test_emphasis_stripped():
    assert c("**Bold** and *italic* and ***both***.") == "Bold and italic and both."


def test_bullets_become_real_characters():
    assert c("Intro:\n\n* first\n* second") == "Intro:\n\n• first\n• second"


def test_tables_are_dropped_not_salvaged():
    assert c("Roster\n\n| No. | Player |\n| --- | --- |\n| 1 | Ross |") is None


@pytest.mark.parametrize("raw,want", [
    ("a\xa0b", "a b"),            # NFKC folds the non-breaking space
    ("unknown.¹⁰⁹", "unknown."),  # superscript refs
    ("Simone Records [fr] released", "Simone Records released"),
    ("a  \t b", "a b"),
    ("a   \nb", "a\nb"),          # trailing whitespace before a newline
])
def test_normalisation(raw, want):
    assert c(raw) == want


def test_blank_run_collapses_to_exactly_one():
    assert c("Title\n\n\n\n\nBody") == "Title\n\nBody"


def test_short_documents_are_dropped():
    assert clean("Too short.") is None
    assert clean(BODY) is not None


def test_output_contract_holds_on_a_messy_document():
    doc = c("# T\n\n\n\n**A**\xa0 b.\n\n\n## S\n\n* x  \n* y\n" + BODY)
    assert "\n\n\n" not in doc
    assert "  " not in doc
    assert not any(line != line.rstrip() for line in doc.splitlines())
    assert doc == doc.strip()
