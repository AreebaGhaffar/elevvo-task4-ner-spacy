"""Evaluate an NER approach on a FIXED slice of the CoNLL-2003 test set.

Usage:
    python evaluate.py --approach sm          # en_core_web_sm
    python evaluate.py --approach lg          # en_core_web_lg
    python evaluate.py --approach rule        # rule-based EntityRuler
    python evaluate.py --approach trained     # locally trained demo model
    python evaluate.py --approach all         # every approach
    python evaluate.py --approach sm --n 500 --split test

The fixed slice is the first N sentences of the test split (default 500).
Scores are strict entity-level precision/recall/F1 from seqeval, reported
per entity type (PER/ORG/LOC/MISC) and overall. Every score is reported
with the slice size it was measured on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from seqeval.metrics import classification_report

from conll import load_split, spans_to_tags
from predict import predict_rule, predict_spacy, predict_trained

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
TRAINED_MODEL = ROOT / "models" / "ner_demo"

APPROACHES = {
    "sm": ("spaCy en_core_web_sm (model-based)",
           lambda s: predict_spacy(s, "en_core_web_sm")),
    "lg": ("spaCy en_core_web_lg (model-based)",
           lambda s: predict_spacy(s, "en_core_web_lg")),
    "rule": ("Rule-based EntityRuler (gazetteer + patterns)", predict_rule),
    "trained": ("Trained spaCy NER demo (on CoNLL-2003 train subset)",
                lambda s: predict_trained(s, TRAINED_MODEL)),
}


def to_native(obj):
    """Recursively convert numpy scalars to plain Python types for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def evaluate(predict_fn, sentences):
    """Run a predictor over gold sentences; return (report dict, gold, pred)."""
    gold_tags = [tags for _, tags in sentences]
    preds = predict_fn(sentences)
    pred_tags = [spans_to_tags(p, len(tokens)) for (tokens, _), p in zip(sentences, preds)]
    report = classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0)
    return report, gold_tags, pred_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="sm",
                        choices=list(APPROACHES) + ["all"])
    parser.add_argument("--n", type=int, default=500,
                        help="fixed evaluation slice: N test sentences")
    parser.add_argument("--start", type=int, default=0,
                        help="first sentence index of the slice (default 0)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    sentences = load_split(args.split, args.n, args.start)
    n_entities = sum(1 for _, tags in sentences for t in tags if t.startswith("B-"))
    print(f"Evaluation slice: sentences {args.start}-{args.start + args.n - 1} "
          f"of CoNLL-2003 {args.split} ({args.n} sentences, "
          f"{n_entities} gold entities)")

    names = list(APPROACHES) if args.approach == "all" else [args.approach]
    all_reports = {}
    for key in names:
        label, fn = APPROACHES[key]
        print(f"\n=== {label} ===")
        report, gold_tags, pred_tags = evaluate(fn, sentences)
        all_reports[key] = report
        print(classification_report(gold_tags, pred_tags, zero_division=0))

    out = RESULTS / (f"evaluation_{args.approach}_slice{args.start}-{args.start + args.n}.json"
                     if args.start else f"evaluation_{args.approach}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(to_native({
        "slice": {"split": args.split, "start": args.start,
                  "n_sentences": args.n, "gold_entities": n_entities},
        "reports": all_reports,
    }), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
