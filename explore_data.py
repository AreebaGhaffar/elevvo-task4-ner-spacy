"""Explore the CoNLL-2003 data: split sizes, entity-type counts, sentence
lengths, and example sentences with their gold entities.

Run: python explore_data.py  (writes results/data_exploration.md)
"""
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"


def read_conll(path):
    """Return a list of (tokens, tags) sentences from a classic CoNLL file."""
    sentences, tokens, tags = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("-DOCSTART-"):
            continue
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        parts = line.split()
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def entity_spans(tokens, tags):
    """Extract (text, label) entity spans from IOB2 tags."""
    spans, cur, cur_label = [], [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur:
                spans.append((" ".join(cur), cur_label))
            cur, cur_label = [tok], tag[2:]
        elif tag.startswith("I-") and cur and tag[2:] == cur_label:
            cur.append(tok)
        else:
            if cur:
                spans.append((" ".join(cur), cur_label))
            cur, cur_label = [], None
    if cur:
        spans.append((" ".join(cur), cur_label))
    return spans


def main():
    rows = []
    examples = {}
    for split, fname in [("train", "eng.train"), ("validation", "eng.testa"), ("test", "eng.testb")]:
        sents = read_conll(DATA / fname)
        type_counts = Counter()
        lengths = []
        for tokens, tags in sents:
            lengths.append(len(tokens))
            for _, label in entity_spans(tokens, tags):
                type_counts[label] += 1
        n_tokens = sum(lengths)
        rows.append({
            "split": split,
            "sentences": len(sents),
            "tokens": n_tokens,
            "entities": sum(type_counts.values()),
            "PER": type_counts["PER"],
            "ORG": type_counts["ORG"],
            "LOC": type_counts["LOC"],
            "MISC": type_counts["MISC"],
            "avg sentence length (tokens)": round(n_tokens / len(sents), 1),
            "max sentence length (tokens)": max(lengths),
        })
        # Keep the first 2 sentences with >= 2 entities as examples.
        ex = [s for s in sents if len(entity_spans(*s)) >= 2][:2]
        examples[split] = ex

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    for split, ex in examples.items():
        print(f"--- example sentences ({split}) ---")
        for tokens, tags in ex:
            print(" ".join(tokens))
            print("entities:", entity_spans(tokens, tags))
        print()

    out = ROOT / "results" / "data_exploration.md"
    out.parent.mkdir(exist_ok=True)
    lines = ["# Data exploration — CoNLL-2003 (English)\n",
             df.to_markdown(index=False), ""]
    for split, ex in examples.items():
        lines.append(f"## Example sentences ({split})\n")
        for tokens, tags in ex:
            lines.append("```")
            lines.append(" ".join(tokens))
            lines.append(f"entities: {entity_spans(tokens, tags)}")
            lines.append("```\n")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
