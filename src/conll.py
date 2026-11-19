"""Helpers for reading CoNLL-2003 files and converting between tag schemes."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw"

# spaCy (OntoNotes) labels -> CoNLL-2003 labels. Labels with no CoNLL
# counterpart (dates, money, quantities, ...) are dropped, not forced into
# MISC, so they do not count as false positives.
SPACY_TO_CONLL = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "NORP": "MISC",
    "FAC": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "PRODUCT": "MISC",
}


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


def tags_to_spans(tags):
    """Convert IOB2 tags to a list of (start, end, label) token-index spans."""
    spans, start, label = [], None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == label:
            continue
        else:
            if start is not None:
                spans.append((start, i, label))
            start, label = None, None
    if start is not None:
        spans.append((start, len(tags), label))
    return spans


def spans_to_tags(spans, n_tokens):
    """Convert (start, end, label) spans back to IOB2 tags."""
    tags = ["O"] * n_tokens
    for start, end, label in spans:
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return tags


def load_split(split, n_sentences=None):
    """Load a split ('train', 'validation', 'test'); optionally take the
    first n_sentences (the fixed evaluation slice)."""
    fname = {"train": "eng.train", "validation": "eng.testa", "test": "eng.testb"}[split]
    sentences = read_conll(DATA / fname)
    if n_sentences is not None:
        sentences = sentences[:n_sentences]
    return sentences
