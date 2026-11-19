"""Train a small spaCy NER model on a subset of CoNLL-2003 (TRAINING DEMO).

This is a CPU-friendly demo, not a full fine-tune: a blank English pipeline
(tok2vec + ner) trained on the first N_TRAIN sentences of eng.train for a
few epochs. It shows the full training loop end to end; it is not expected
to match the pre-trained spaCy models, which were trained on much larger
corpora (OntoNotes).

Run: python train_demo.py  (saves the model to models/ner_demo)
"""
import json
import random
from pathlib import Path

import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conll import load_split, tags_to_spans

ROOT = Path(__file__).parent
N_TRAIN = 1500
EPOCHS = 10
DROPOUT = 0.2
SEED = 42
OUT = ROOT / "models" / "ner_demo"


def make_examples(nlp, sentences):
    """Build spaCy Examples from (tokens, IOB tags) with exact char offsets."""
    examples = []
    for tokens, tags in sentences:
        text = " ".join(tokens)
        doc = Doc(nlp.vocab, words=tokens)
        entities = []
        for start, end, label in tags_to_spans(tags):
            start_char = sum(len(t) + 1 for t in tokens[:start])
            end_char = start_char + sum(len(t) for t in tokens[start:end]) + (end - start - 1)
            entities.append((start_char, end_char, label))
        examples.append(Example.from_dict(doc, {"entities": entities}))
    return examples


def main():
    fix_random_seed(SEED)
    random.seed(SEED)
    sentences = load_split("train", N_TRAIN)
    print(f"Training demo: blank NER on first {N_TRAIN} train sentences, "
          f"{EPOCHS} epochs, dropout {DROPOUT}")

    nlp = spacy.blank("en")
    nlp.add_pipe("tok2vec")
    nlp.add_pipe("ner")
    train_examples = make_examples(nlp, sentences)
    nlp.initialize(lambda: train_examples)

    losses_history = []
    for epoch in range(EPOCHS):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=8):
            nlp.update(batch, drop=DROPOUT, losses=losses)
        losses_history.append({k: float(round(v, 2)) for k, v in losses.items()})
        print(f"epoch {epoch + 1}/{EPOCHS} losses={losses_history[-1]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUT)
    (ROOT / "results" / "training_demo.json").write_text(json.dumps({
        "n_train_sentences": N_TRAIN,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
        "seed": SEED,
        "losses_per_epoch": losses_history,
        "model_dir": "models/ner_demo",
    }, indent=2))
    print(f"Saved demo model to {OUT}")


if __name__ == "__main__":
    main()
