import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.dataset import LABELS, examples

CHECKPOINT = "distilbert-base-cased"
CKPT_DIR = Path("data/model")
MAX_LEN = 512
STRIDE = 128
NL = "⏎"
IGNORE = -100

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def to_words(text: str, markers: bool = True) -> tuple[list[str], list[bool]]:
    words, marker = [], []
    for i, line in enumerate(text.split("\n")):
        if i and markers:
            words.append(NL)
            marker.append(True)
        for w in line.split():
            words.append(w)
            marker.append(False)
    return words, marker


def encode(tok, text: str, labels: list[int] | None = None, markers: bool = True):
    """Tokenize into overlapping windows, with labels on the first subword of each word.

    Overlap means a word near a window edge still gets predicted from a window where it has context on both sides
    """
    words, is_marker = to_words(text, markers)
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, padding="max_length",
              return_tensors="pt")
    word_ids = [enc.word_ids(i) for i in range(len(enc["input_ids"]))]

    if labels is None:
        return enc, word_ids, is_marker

    label_of_word, it = [], iter(labels)
    for m in is_marker:
        label_of_word.append(IGNORE if m else next(it))

    aligned = []
    for ids in word_ids:
        seen, row = set(), []
        for wid in ids:
            if wid is None or wid in seen:  # padding/specials, and later subwords
                row.append(IGNORE)
            else:
                seen.add(wid)
                row.append(label_of_word[wid])
        aligned.append(row)
    enc["labels"] = torch.tensor(aligned)
    return enc, word_ids, is_marker


def windows(split: str, tok, limit: int | None = None, seed: int | None = None,
            markers: bool = True):
    """Yield {input_ids, attention_mask, labels} for each window of each example."""
    for ex in itertools.islice(examples(split, seed=seed), limit):
        enc, _, _ = encode(tok, ex["text"], ex["labels"], markers)
        for i in range(len(enc["input_ids"])):
            yield {k: enc[k][i] for k in ("input_ids", "attention_mask", "labels")}


def train(steps: int = 500, batch_size: int = 8, lr: float = 5e-5, docs: int = 200,
          markers: bool = True):
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={n: i for i, n in enumerate(LABELS)},
    ).to(DEVICE)

    data = list(windows("train", tok, limit=docs))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step:5d}  loss {loss.item():.4f}", flush=True)
            if step >= steps:
                break

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tok.save_pretrained(CKPT_DIR)
    return model, tok


def load(path: Path = CKPT_DIR):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path).to(DEVICE).eval()
    return model, tok


def predictor(model, tok, markers: bool = True):
    device = next(model.parameters()).device 

    @torch.no_grad()
    def predict(tokens: list[str], text: str) -> list[int]:
        enc, word_ids, is_marker = encode(tok, text, None, markers)
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits.cpu()

        # a word can appear in two overlapping windows, keep the copy furthest from an edge
        best: dict[int, tuple[int, int]] = {}
        for w, ids in enumerate(word_ids):
            n = len(ids)
            for pos, wid in enumerate(ids):
                if wid is None:
                    continue
                margin = min(pos, n - 1 - pos)
                if wid not in best or margin > best[wid][0]:
                    best[wid] = (margin, int(logits[w, pos].argmax()))

        out = [best.get(i, (0, 1))[1] for i, m in enumerate(is_marker) if not m]
        return out[:len(tokens)] + [1] * max(0, len(tokens) - len(out))

    return predict


if __name__ == "__main__":
    print("device:", DEVICE)
    model, tok = train(steps=int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 200)
