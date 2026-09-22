from datasets import load_dataset
import itertools
import json
import re
import unicodedata
from pathlib import Path

RAW = Path("data/raw.jsonl")

SUPERSCRIPT = re.compile(r"[²³¹⁰-₟]+")
LANG_MARKER = re.compile(r"\[[a-z]{2,3}\]")
HEADING = re.compile(r"^#+[ \t]*", re.M)
BULLET = re.compile(r"^[ \t]*[*\-+][ \t]+", re.M)
EMPHASIS = re.compile(r"[*_]{1,3}")
SPACES = re.compile(r"[ \t]+")
TRAILING = re.compile(r"[ \t]+$", re.M)
BLANKS = re.compile(r"\n{3,}")


def fetch(n=5000) -> None:
    """Fetch n documents from the Clean-Wikipedia-English-Articles dataset."""
    if RAW.exists():
        return
    ds = load_dataset("OVHaiLLM/Clean-Wikipedia-English-Articles", split="train", streaming=True)
    RAW.parent.mkdir(exist_ok=True)
    with RAW.open("w") as f:
        for doc in itertools.islice(ds, n):
            f.write(json.dumps({"text": doc["text"]}) + "\n")


def clean(text, min_chars=500) -> str | None:
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
    text = BLANKS.sub("\n\n", text)
    text = text.strip()
    return text if len(text) >= min_chars else None


def clean_docs():
    for line in RAW.open():
        doc = clean(json.loads(line)["text"])
        if doc:
            yield doc


if __name__ == "__main__":
    fetch()
    kept = list(clean_docs())
    total = sum(1 for _ in RAW.open())
    print(f"kept {len(kept)}/{total} docs")
    for doc in kept[:2]:
        print(repr(doc[:600]), "\n---")

    for doc in kept:
        assert "\n\n\n" not in doc
        assert not TRAILING.search(doc)
        assert "  " not in doc
    print("ok")
