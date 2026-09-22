from datasets import load_dataset
import itertools
from collections import Counter
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator
import random

DATA = Path("data")
RAW = DATA / "raw.jsonl"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

SUPERSCRIPT = re.compile(r"[²³¹⁰-₟]+")
LANG_MARKER = re.compile(r"\[[a-z]{2,3}\]")
HEADING = re.compile(r"^#+[ \t]*", re.M)
BULLET = re.compile(r"^[ \t]*[*\-+][ \t]+", re.M)
EMPHASIS = re.compile(r"[*_]{1,3}")
SPACES = re.compile(r"[ \t]+")
TRAILING = re.compile(r"[ \t]+$", re.M)
LEADING = re.compile(r"^[ \t]+", re.M)
BLANKS = re.compile(r"\n{3,}")

TOKEN = re.compile(r"(\S+)([ \t]*\n*[ \t]*)")
LABELS = ["JOIN", "SPACE", "NEWLINE", "PARA"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
SEPS = {"SPACE": " ", "NEWLINE": "\n", "PARA": "\n\n", "JOIN": ""}


def fetch(n: int = 5000) -> None:
    """Fetch n documents from the Clean-Wikipedia-English-Articles dataset."""
    if RAW.exists():
        return
    ds = load_dataset("OVHaiLLM/Clean-Wikipedia-English-Articles", split="train", streaming=True)
    RAW.parent.mkdir(exist_ok=True)
    with RAW.open("w") as f:
        for doc in itertools.islice(ds, n):
            f.write(json.dumps({"text": doc["text"]}) + "\n")


def clean(text: str, min_chars: int = 500) -> str | None:
    """Markdown Wikipedia -> plain text whose newlines are trustworthy labels.

    Returns None for documents we refuse to label (tables, too short).
    """
    text = SUPERSCRIPT.sub("", text)  # before NFKC, which folds them into plain digits
    text = unicodedata.normalize("NFKC", text)
    if "|" in text:  # table markup: newlines there are structural, not semantic
        return None
    text = LANG_MARKER.sub("", text)
    text = HEADING.sub("", text)
    text = BULLET.sub("• ", text)
    text = EMPHASIS.sub("", text)
    text = SPACES.sub(" ", text)
    text = TRAILING.sub("", text)
    text = LEADING.sub("", text)
    text = BLANKS.sub("\n\n", text)
    text = text.strip()
    return text if len(text) >= min_chars else None


def clean_docs() -> Iterator[str]:
    for line in RAW.open():
        doc = clean(json.loads(line)["text"])
        if doc:
            yield doc


def label(doc: str) -> tuple[list[str], list[int]]:
    """Clean doc -> (tokens, labels), where labels[i] is the separator AFTER tokens[i]."""
    tokens, labels = [], []
    for token, sep in TOKEN.findall(doc):
        tokens.append(token)
        labels.append(LABEL_ID["PARA" if sep.count("\n") > 1 else "NEWLINE" if "\n" in sep else "SPACE"])
    return tokens, labels


def render(tokens: list[str], labels: list[int]) -> str:
    return "".join(t + SEPS[LABELS[l]] for t, l in zip(tokens, labels)).strip()


def corrupt(
    tokens: list[str],
    labels: list[int],
    width: int | None = None,
    hard_split: float = 0.1,
    keep_para: float = 0.3,
) -> tuple[list[str], list[int], str]:
    if width is None:
        width = 10 ** 9 if random.random() < 0.15 else random.randint(40, 160)
    out_tokens, out_labels, seps, line = [], [], [], 0
    for token, lab in zip(tokens, labels):
        if line and line + 1 + len(token) > width:
            if len(token) > 4 and random.random() < hard_split:
                at = random.randint(2, len(token) - 2)
                out_tokens.append(token[:at])
                out_labels.append(LABEL_ID["JOIN"])
                seps.append("\n")
                token, line = token[at:], 0
            else:
                seps[-1] = "\n"
                line = 0
        out_tokens.append(token)
        out_labels.append(lab)
        if lab == LABEL_ID["PARA"] and random.random() < keep_para:
            seps.append("\n\n")  # the extract kept this one
            line = 0
        else:
            seps.append(" ")
            line += len(token) + 1
    text = "".join(t + s for t, s in zip(out_tokens, seps)).strip()
    return out_tokens, out_labels, text


def build(seed: int = 0) -> dict[str, int]:
    docs = list(clean_docs())
    random.Random(seed).shuffle(docs)
    counts, start = {}, 0
    for name, frac in SPLITS.items():
        end = start + round(frac * len(docs)) if name != "test" else len(docs)
        with (DATA / f"{name}.jsonl").open("w") as f:
            for doc in docs[start:end]:
                f.write(json.dumps({"text": doc}) + "\n")
        counts[name] = end - start
        start = end
    return counts


def examples(split: str, seed: int | None = None, **kw) -> Iterator[dict]:
    rng_seed = seed
    for i, line in enumerate((DATA / f"{split}.jsonl").open()):
        if rng_seed is not None:
            random.seed(rng_seed + i)
        tokens, labels = label(json.loads(line)["text"])
        tokens, labels, text = corrupt(tokens, labels, **kw)
        yield {"text": text, "tokens": tokens, "labels": labels}


if __name__ == "__main__":
    fetch()
    print("split sizes:", build())

    dist = Counter()
    for ex in examples("train"):
        dist.update(LABELS[l] for l in ex["labels"])
    total = sum(dist.values())
    print(f"{total:,} train gaps")
    for name, n in dist.most_common():
        print(f"  {name:8} {n:9,}  {n / total:6.2%}")
