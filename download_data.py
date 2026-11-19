"""Download CoNLL-2003 (English) from Hugging Face Hub and export the classic
4-column files (token, POS, chunk, NER) used by the original shared task.

Kaggle substitution note: the brief recommends CoNLL-2003 from Kaggle
(dataset slug "conll2003-dataset", files conll2003/eng.train, eng.testa,
eng.testb). The Kaggle API (api.kaggle.com) is unreachable from this machine,
so the identical official files are fetched from Hugging Face Hub instead:
BramVanroy/conll2003 — an exact duplicate of eriktks/conll2003, which mirrors
the original CoNLL-2003 shared-task files (Reuters news wire, annotated at
the University of Antwerp). The files are downloaded directly from the Hub
with curl because the `datasets` library's HTTP layer fails in this
environment (an httpx decoder error); the source files are the same.

Run: python download_data.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = "BramVanroy/conll2003"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
OUT = Path(__file__).parent / "data" / "raw"

FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
OUT_NAMES = {"train": "eng.train", "validation": "eng.testa", "test": "eng.testb"}

# Integer tag ids used by the CoNLL-2003 files (IOB2 scheme). The label names
# are read from the parquet file's Hugging Face metadata when available.
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def label_names(table, column, fallback):
    md = table.schema.metadata
    if md and b"huggingface" in md:
        feats = json.loads(md[b"huggingface"])["info"]["features"]
        return feats[column]["feature"]["names"]
    return fallback


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} ...")
    subprocess.run(
        ["curl", "-sSL", "--max-time", "300", "-o", str(dest), url],
        check=True,
    )
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")


def write_conll(rows, path: Path) -> None:
    """Write one split in the classic CoNLL format: token POS chunk NER."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("-DOCSTART- -X- -X- O\n\n")
        for tokens, pos_tags, chunk_tags, ner_tags in rows:
            for tok, pos, chunk, ner in zip(tokens, pos_tags, chunk_tags, ner_tags):
                f.write(f"{tok} {pos} {chunk} {ner}\n")
            f.write("\n")


def main() -> None:
    stats = {}
    for split, rel in FILES.items():
        parquet_path = OUT / Path(rel).name
        if not parquet_path.exists():
            download(f"{BASE}/{rel}", parquet_path)
        table = pq.read_table(parquet_path)
        pos_names = label_names(
            table, "pos_tags",
            ['"', "''", "#", "$", "(", ")", ",", ".", ":", "``", "CC", "CD",
             "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN",
             "NNP", "NNPS", "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$",
             "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD",
             "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"],
        )
        chunk_names = label_names(
            table, "chunk_tags",
            ["O", "B-ADJP", "I-ADJP", "B-ADVP", "I-ADVP", "B-CONJP",
             "I-CONJP", "B-INTJ", "I-INTJ", "B-LST", "I-LST", "B-NP",
             "I-NP", "B-PP", "I-PP", "B-PRT", "I-PRT", "B-SBAR", "I-SBAR",
             "B-UCP", "I-UCP", "B-VP", "I-VP"],
        )
        rows = [
            (r["tokens"], r["pos_tags"], r["chunk_tags"], r["ner_tags"])
            for r in table.to_pylist()
        ]
        # Tags are integer ids in this parquet dump; map them to labels.
        rows = [
            (
                t,
                [pos_names[i] for i in p],
                [chunk_names[i] for i in c],
                [NER_LABELS[i] for i in n],
            )
            for t, p, c, n in rows
        ]
        n_tokens = sum(len(t) for t, _, _, _ in rows)
        n_entities = sum(
            1 for _, _, _, tags in rows for t in tags if t.startswith("B-")
        )
        out_path = OUT / OUT_NAMES[split]
        write_conll(rows, out_path)
        stats[split] = {
            "sentences": len(rows),
            "tokens": n_tokens,
            "entities": n_entities,
            "file": OUT_NAMES[split],
            "source": f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}",
        }
        print(
            f"{split}: {len(rows)} sentences, {n_tokens} tokens, "
            f"{n_entities} entities -> {OUT_NAMES[split]}"
        )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
