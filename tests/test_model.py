import itertools

import pytest
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.dataset import LABELS, examples
from src.model import CHECKPOINT, IGNORE, encode, predictor, to_words


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CHECKPOINT)


def test_markers_carry_no_label_and_real_words_keep_theirs(tok):
    ex = next(iter(examples("val", seed=0)))

    words, is_marker = to_words(ex["text"])
    assert [w for w, m in zip(words, is_marker) if not m] == ex["tokens"]

    enc, word_ids, _ = encode(tok, ex["text"], ex["labels"])
    for w, ids in enumerate(word_ids):
        seen = set()
        for pos, wid in enumerate(ids):
            lab = int(enc["labels"][w][pos])
            if wid is None or wid in seen:
                assert lab == IGNORE  # padding, specials, continuation subwords
            elif is_marker[wid]:
                assert lab == IGNORE  # markers are input only
            else:
                assert lab != IGNORE
            if wid is not None:
                seen.add(wid)


def test_markers_can_be_switched_off(tok):
    ex = next(iter(examples("val", seed=0)))
    words, is_marker = to_words(ex["text"], markers=False)
    assert words == ex["tokens"]
    assert not any(is_marker)


def test_long_documents_produce_overlapping_windows(tok):
    text = " ".join(["word"] * 3000)
    enc, word_ids, _ = encode(tok, text)
    assert len(enc["input_ids"]) > 1
    first, second = {w for w in word_ids[0] if w is not None}, {w for w in word_ids[1] if w is not None}
    assert first & second  # consecutive windows share context, so seams keep both sides


def test_predictor_returns_one_label_per_token(tok):
    model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT, num_labels=len(LABELS))
    predict = predictor(model.eval(), tok)
    for ex in itertools.islice(examples("val", seed=0), 3):
        assert len(predict(ex["tokens"], ex["text"])) == len(ex["labels"])
