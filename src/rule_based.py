"""Rule-based NER with spaCy's EntityRuler.

Two kinds of hand-written rules:
1. Gazetteer patterns: the most frequent entity surface forms per type in
   the CoNLL-2003 training split (min. 3 occurrences). This is the classic
   gazetteer approach — rules built from training data, applied to new text.
2. Structural patterns: titles ("Mr. X", "President X") mark a person;
   organisation suffixes ("X Inc.", "X Corp.", "X Group") mark an org.

The ruler runs on a blank English pipeline (no statistical model), so every
entity it finds comes from a rule.
"""
from collections import Counter
from pathlib import Path

import spacy

from conll import DATA, read_conll, tags_to_spans

MIN_GAZETTEER_COUNT = 3
MAX_GAZETTEER_PER_TYPE = 1500

ORG_SUFFIXES = [
    "inc", "corp", "ltd", "co", "company", "group", "bank", "airlines",
    "airways", "holdings", "industries", "brothers", "partners", "fund",
]
TITLES = [
    "mr", "mrs", "ms", "dr", "president", "prime", "minister", "king",
    "queen", "pope", "senator", "governor", "coach", "captain", "general",
    "chairman", "secretary",
]


def gazetteer_patterns():
    """Most frequent (surface form -> type) pairs from the training split."""
    counts = Counter()
    for tokens, tags in read_conll(DATA / "eng.train"):
        for start, end, label in tags_to_spans(tags):
            counts[(" ".join(tokens[start:end]), label)] += 1
    patterns = []
    by_type = Counter()
    for (text, label), count in counts.most_common():
        if count < MIN_GAZETTEER_COUNT:
            continue
        if by_type[label] >= MAX_GAZETTEER_PER_TYPE:
            continue
        by_type[label] += 1
        patterns.append({"label": label, "pattern": text})
    return patterns, dict(by_type)


def structural_patterns():
    """Title + organisation-suffix token patterns."""
    patterns = []
    for title in TITLES:
        patterns.append({
            "label": "PER",
            "pattern": [{"LOWER": title}, {"IS_TITLE": True, "OP": "+"}],
        })
    for suffix in ORG_SUFFIXES:
        patterns.append({
            "label": "ORG",
            "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": suffix}],
        })
    return patterns


def build_ruler_nlp():
    """Blank English pipeline with only the EntityRuler."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    gazetteer, by_type = gazetteer_patterns()
    # Longest patterns first so "New York Times" wins over "New York".
    gazetteer.sort(key=lambda p: len(p["pattern"].split()), reverse=True)
    structural = structural_patterns()
    ruler.add_patterns(gazetteer + structural)
    nlp.meta["rule_patterns"] = {
        "gazetteer": by_type,
        "structural": len(structural),
        "total": len(gazetteer) + len(structural),
    }
    return nlp


if __name__ == "__main__":
    nlp = build_ruler_nlp()
    print("patterns:", nlp.meta["rule_patterns"])
    for text in [
        "President Barack Obama visited Paris and met officials from Microsoft Corp.",
        "Mr. John Smith of the United Nations spoke in Berlin.",
    ]:
        doc = nlp(text)
        print(text)
        print(" ->", [(e.text, e.label_) for e in doc.ents])
