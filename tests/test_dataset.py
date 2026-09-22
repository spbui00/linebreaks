import json
import itertools
import pytest

from src.dataset import clean, LABEL_ID, clean_docs, corrupt, label, render

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


def test_label_render_round_trip():
    doc = "A Heading\n\nFirst para.\n• one\n• two\n\nSecond para."
    toks, labs = label(doc)
    assert len(toks) == len(labs)
    assert render(toks, labs) == doc


def test_corrupt_preserves_the_words_and_the_labels():
    toks, labs = label("Heading\n\n" + "word " * 200)
    toks2, labs2, text = corrupt(toks, labs, width=60, keep_para=0)
    assert "".join(toks2) == "".join(toks)          # JOIN splits, never loses, characters
    assert render(toks2, labs2) == render(toks, labs)
    assert "\n" in text and text.count("\n\n") == 0  # wraps are single newlines


DOC = "Heading\n\nBody sentence here.\n\u2022 one\n\u2022 two\n\nAnother paragraph follows."


def test_keep_para_zero_destroys_the_original_structure():
    toks, labs = label(DOC)
    _, _, text = corrupt(toks, labs, width=1000, hard_split=0, keep_para=0)
    assert "\n" not in text


def test_keep_para_one_leaves_paragraph_breaks_but_not_line_breaks():
    toks, labs = label(DOC)
    _, _, text = corrupt(toks, labs, width=1000, hard_split=0, keep_para=1)
    assert text.count("\n\n") == DOC.count("\n\n")
    assert "one \u2022 two" in text  # single newlines are still flattened


def test_unwrapped_documents_occur():
    toks, labs = label(DOC * 20)
    widths = {corrupt(toks, labs, hard_split=0, keep_para=0)[2].count("\n") for _ in range(200)}
    assert 0 in widths  # the no-wrap case fires
    assert len(widths) > 5  # and the wrapped ones vary


def test_round_trip_holds_on_the_real_corpus():
    for doc in itertools.islice(clean_docs(), 300):
        toks, labs = label(doc)
        assert render(toks, labs) == doc
        toks2, labs2, _ = corrupt(toks, labs)
        assert render(toks2, labs2) == doc


def test_splits_are_disjoint_by_document():
    from src.dataset import DATA, build

    build(seed=0)
    docs = {name: {json.loads(l)["text"] for l in (DATA / f"{name}.jsonl").open()}
            for name in ("train", "val", "test")}
    assert docs["train"] & docs["val"] == set()
    assert docs["train"] & docs["test"] == set()
    assert docs["val"] & docs["test"] == set()


def test_seeded_examples_are_reproducible_unseeded_are_not():
    from src.dataset import examples

    a = [e["text"] for e in itertools.islice(examples("val", seed=1), 5)]
    b = [e["text"] for e in itertools.islice(examples("val", seed=1), 5)]
    c = [e["text"] for e in itertools.islice(examples("val"), 5)]
    assert a == b
    assert a != c  # train-side augmentation resamples the corruption


def test_examples_stay_labelled_correctly():
    from src.dataset import examples

    for ex in itertools.islice(examples("val", seed=0), 20):
        assert len(ex["tokens"]) == len(ex["labels"])
        assert "".join(ex["tokens"]) == "".join(ex["text"].split())
