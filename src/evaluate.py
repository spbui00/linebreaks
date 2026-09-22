import itertools
import re
from collections.abc import Callable, Iterator

from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import LABEL_ID, LABELS, examples, render

SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
BULLET_START = re.compile(r"^[•\-*–]")

Predictor = Callable[[list[str], str], list[int]]


def predict_majority(tokens: list[str], text: str) -> list[int]:
    """gap = space"""
    return [LABEL_ID["SPACE"]] * len(tokens)


def predict_heuristic(tokens: list[str], text: str) -> list[int]:
    """Punctuation + capitalisation rules"""
    out = []
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if BULLET_START.match(nxt):
            out.append(LABEL_ID["NEWLINE"])
        elif SENTENCE_END.search(token) and nxt[:1].isupper():
            out.append(LABEL_ID["PARA"])
        else:
            out.append(LABEL_ID["SPACE"])
    return out


def evaluate(predict: Predictor, split: str = "val", limit: int | None = None,
             seed: int = 0) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for ex in itertools.islice(examples(split, seed=seed), limit):
        y_true.extend(ex["labels"])
        y_pred.extend(predict(ex["tokens"], ex["text"]))
    return y_true, y_pred


def report(name: str, y_true: list[int], y_pred: list[int]) -> None:
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=range(len(LABELS)),
                                target_names=LABELS, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    print("confusion (rows = true, cols = predicted)")
    print(f"{'':9}" + "".join(f"{n:>10}" for n in LABELS))
    for name_, row in zip(LABELS, cm):
        print(f"{name_:9}" + "".join(f"{v:>10,}" for v in row))


if __name__ == "__main__":
    import sys

    preds: list[tuple[str, Predictor]] = [
        ("majority", predict_majority),
        ("heuristic", predict_heuristic),
    ]
    if "--model" in sys.argv:
        from src.model import load, predictor

        preds.append(("model", predictor(*load())))

    for name, fn in preds:
        report(name, *evaluate(fn, "val", limit=100))
