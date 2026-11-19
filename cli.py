"""Command-line NER: extract and categorize entities from text or a file.

Usage:
    python cli.py "Barack Obama visited Paris last week."
    python cli.py --file examples/news_sample.txt
    python cli.py --model lg "Microsoft Corp. is based in Redmond."
    python cli.py --model rule --format json "Mr. John Smith spoke in Berlin."

Models: sm (en_core_web_sm, default), lg (en_core_web_lg),
        rule (rule-based EntityRuler), trained (local demo model).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import spacy

from predict import load_spacy
from rule_based import build_ruler_nlp

ROOT = Path(__file__).parent


def load_model(name):
    if name == "rule":
        return build_ruler_nlp()
    if name == "trained":
        return load_spacy(str(ROOT / "models" / "ner_demo"))
    return load_spacy({"sm": "en_core_web_sm", "lg": "en_core_web_lg"}[name])


def extract(nlp, text):
    doc = nlp(text)
    grouped = defaultdict(list)
    for ent in doc.ents:
        grouped[ent.label_].append(ent.text)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Named Entity Recognition CLI")
    parser.add_argument("text", nargs="?", help="text to analyse")
    parser.add_argument("--file", help="read text from a file instead")
    parser.add_argument("--model", default="sm",
                        choices=["sm", "lg", "rule", "trained"],
                        help="NER model to use (default: sm)")
    parser.add_argument("--format", default="table", choices=["table", "json"])
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("give TEXT or --file")

    nlp = load_model(args.model)
    grouped = extract(nlp, text)

    if args.format == "json":
        print(json.dumps({k: sorted(set(v)) for k, v in grouped.items()},
                         indent=2, ensure_ascii=False))
        return

    if not grouped:
        print("No named entities found.")
        return
    total = sum(len(v) for v in grouped.values())
    print(f"Found {total} entit{'y' if total == 1 else 'ies'} "
          f"(model: {args.model}):")
    for label in sorted(grouped):
        uniq = sorted(set(grouped[label]))
        print(f"\n{label} ({len(grouped[label])}):")
        for item in uniq:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
