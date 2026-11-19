"""Download the spaCy models used in this project.

Run: python download_models.py
Models:
  - en_core_web_sm (12 MB): baseline model-based NER
  - en_core_web_lg (780 MB): second model for the bonus comparison
Models are installed into the Python environment, not committed to git.
"""
import subprocess
import sys

MODELS = ["en_core_web_sm", "en_core_web_lg"]


def main():
    for model in MODELS:
        print(f"downloading {model} ...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model], check=True
        )
    print("All models downloaded.")


if __name__ == "__main__":
    main()
