"""Predictors: model-based (spaCy), rule-based (EntityRuler) and trained.

All predictors take a list of token lists (one per sentence, CoNLL
tokenization) and return, per sentence, a list of (start, end, label) spans
with CoNLL labels (PER, ORG, LOC, MISC). Sentences are fed to spaCy as
pre-tokenized Docs so predictions line up exactly with the gold tokens.
"""
from pathlib import Path

import spacy
from spacy.tokens import Doc

from conll import SPACY_TO_CONLL

ROOT = Path(__file__).parent.parent

_NLP_CACHE = {}


def load_spacy(model_name):
    if model_name not in _NLP_CACHE:
        _NLP_CACHE[model_name] = spacy.load(model_name)
    return _NLP_CACHE[model_name]


def predict_spacy(sentences, model_name):
    """Run a pre-trained spaCy model over pre-tokenized sentences."""
    nlp = load_spacy(model_name)
    docs = [Doc(nlp.vocab, words=tokens) for tokens, _ in sentences]
    out = []
    for doc in nlp.pipe(docs, batch_size=64):
        spans = []
        for ent in doc.ents:
            label = SPACY_TO_CONLL.get(ent.label_)
            if label is not None:
                spans.append((ent.start, ent.end, label))
        out.append(spans)
    return out


def predict_rule(sentences):
    """Rule-based NER: EntityRuler with gazetteer + title/suffix patterns."""
    from rule_based import build_ruler_nlp

    nlp = build_ruler_nlp()
    docs = [Doc(nlp.vocab, words=tokens) for tokens, _ in sentences]
    out = []
    for doc in nlp.pipe(docs, batch_size=64):
        spans = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
        out.append(spans)
    return out


def predict_trained(sentences, model_path):
    """Run the locally trained demo model over pre-tokenized sentences."""
    nlp = load_spacy(str(model_path))
    docs = [Doc(nlp.vocab, words=tokens) for tokens, _ in sentences]
    out = []
    for doc in nlp.pipe(docs, batch_size=64):
        spans = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
        out.append(spans)
    return out
