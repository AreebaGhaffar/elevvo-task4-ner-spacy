"""Visualize extracted entities with displaCy (bonus).

Renders sample news sentences as highlighted HTML and saves the pages to
outputs/ so they can be opened in a browser.

Run: python visualize.py
"""
from pathlib import Path

import spacy
from spacy import displacy

ROOT = Path(__file__).parent

SAMPLES = [
    "Barack Obama visited Paris last week to meet officials from the "
    "European Commission and Microsoft Corp.",
    "The United Nations said on Thursday that the World Bank will lend "
    "India $ 500 million for schools in New Delhi.",
    "Roger Federer beat Pete Sampras in the Wimbledon final in London.",
]


def main():
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    for model_name in ["en_core_web_sm", "en_core_web_lg"]:
        nlp = spacy.load(model_name)
        for i, text in enumerate(SAMPLES, 1):
            doc = nlp(text)
            html = displacy.render(doc, style="ent", page=True)
            path = out_dir / f"displacy_{model_name}_sample{i}.html"
            path.write_text(html, encoding="utf-8")
            print(f"{model_name} sample {i}: "
                  f"{[(e.text, e.label_) for e in doc.ents]} -> {path.name}")
    print(f"\nOpen the files in {out_dir} in a browser to see the highlights.")


if __name__ == "__main__":
    main()
