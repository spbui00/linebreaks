import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.dataset import LABELS, examples
from sklearn.metrics import f1_score


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


def encode(tok, text: str, labels: list[int] | None = None, markers: bool = True, pad: bool = True):
    """Tokenize into overlapping windows, with labels on the first subword of each word.

    Overlap means a word near a window edge still gets predicted from a window where it has context on both sides
    """
    words, is_marker = to_words(text, markers)
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True,
              **({"padding": "max_length", "return_tensors": "pt"} if pad else {}))
    word_ids = [enc.word_ids(i) for i in range(len(enc["input_ids"]))]

    if labels is None:
        return enc, word_ids, is_marker

    label_of_word, j = [], 0
    for m in is_marker:
        if m:
            label_of_word.append(IGNORE)
        else:
            label_of_word.append(labels[j])
            j += 1

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
    enc["labels"] = torch.tensor(aligned) if pad else aligned
    return enc, word_ids, is_marker


def windows(split: str, tok, limit: int | None = None, seed: int | None = None,
            markers: bool = True, pad: bool = False):
    """Yield {input_ids, attention_mask, labels} for each window of each example."""
    for ex in itertools.islice(examples(split, seed=seed), limit):
        enc, _, _ = encode(tok, ex["text"], ex["labels"], markers, pad=pad)
        for i in range(len(enc["input_ids"])):
            yield {k: enc[k][i] for k in ("input_ids", "attention_mask", "labels")}


def val_macro_f1(model, tok, docs: int = 40, markers: bool = True) -> float:
    predict = predictor(model, tok, markers)
    y_true, y_pred = [], []
    for ex in itertools.islice(examples("val", seed=0), docs):
        y_true.extend(ex["labels"])
        y_pred.extend(predict(ex["tokens"], ex["text"]))
    return f1_score(y_true, y_pred, average="macro", labels=range(len(LABELS)),
                    zero_division=0)


def train(steps: int = 4000, batch_size: int = 16, lr: float = 5e-5,
          docs: int | None = None, markers: bool = True, eval_every: int = 500,
          warmup: float = 0.06, out: Path = CKPT_DIR):
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={n: i for i, n in enumerate(LABELS)},
    ).to(DEVICE)

    data = list(windows("train", tok, limit=docs, markers=markers))
    collate = DataCollatorForTokenClassification(tok)  # pad per batch, not to MAX_LEN
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    print(f"{len(data)} windows, device {DEVICE}", flush=True)

    best, step = -1.0, 0
    while step < steps:
        for batch in loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1

            if step % eval_every == 0 or step == steps:
                model.eval()
                f1 = val_macro_f1(model, tok, markers=markers)
                flag = ""
                if f1 > best:
                    best, flag = f1, "  *"
                    out.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
                print(f"step {step:5d}  loss {loss.item():.4f}  val macro-F1 {f1:.4f}{flag}",
                      flush=True)
            if step >= steps:
                break

    print(f"best val macro-F1 {best:.4f}  ->  {out}")
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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--docs", type=int, default=None, help="limit training documents")
    ap.add_argument("--no-markers", action="store_true", help="ablation: hide input newlines")
    ap.add_argument("--out", type=Path, default=CKPT_DIR)
    a = ap.parse_args()
    train(steps=a.steps, batch_size=a.batch_size, lr=a.lr, docs=a.docs,
          markers=not a.no_markers, out=a.out)
