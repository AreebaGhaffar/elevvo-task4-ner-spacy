# Elevvo NLP Internship — Task 4: Named Entity Recognition (NER) from News Articles

Identify named entities (people, locations, organizations, miscellaneous) in
news article text, using rule-based and model-based NER. Entities are
highlighted and categorized in the text.

## Dataset

**CoNLL-2003 (English)** — Reuters news-wire articles, annotated with four
entity types: PER, ORG, LOC, MISC.

- Recommended source (brief): Kaggle, dataset slug `conll2003-dataset`
  (files `conll2003/eng.train`, `eng.testa`, `eng.testb`).
- **Substitution:** the Kaggle API (`api.kaggle.com`) is unreachable from
  this machine, so the identical official files were fetched from Hugging
  Face Hub instead: [`BramVanroy/conll2003`](https://huggingface.co/datasets/BramVanroy/conll2003)
  (an exact duplicate of `eriktks/conll2003`, which mirrors the original
  shared-task files). Files downloaded:
  `data/train-00000-of-00001.parquet` (1,221,000 bytes),
  `data/validation-00000-of-00001.parquet` (310,076 bytes),
  `data/test-00000-of-00001.parquet` (281,499 bytes).
- `download_data.py` re-downloads them and exports the classic 4-column
  files (`token POS chunk NER`) to `data/raw/eng.train`, `eng.testa`,
  `eng.testb`. Raw data is git-ignored; run the script to reproduce it.

Verified counts (match the official CoNLL-2003 paper, Table 1/2):

| split | sentences | tokens | entities | PER | ORG | LOC | MISC |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 14,041 | 203,621 | 23,499 | 6,600 | 6,321 | 7,140 | 3,438 |
| validation | 3,250 | 51,362 | 5,942 | 1,842 | 1,341 | 1,837 | 922 |
| test | 3,453 | 46,435 | 5,648 | 1,617 | 1,661 | 1,668 | 702 |

## Approaches

1. **Model-based (baseline):** pre-trained spaCy `en_core_web_sm`.
   spaCy's OntoNotes labels are mapped to CoNLL labels
   (PERSON→PER, ORG→ORG, GPE→LOC, NORP/FAC/EVENT/WORK_OF_ART/LAW/LANGUAGE/
   PRODUCT→MISC; DATE/TIME/MONEY/quantities are dropped, not forced into
   MISC). See `src/conll.py`.
2. **Rule-based:** spaCy `EntityRuler` on a blank pipeline — a gazetteer of
   the most frequent entity surface forms per type in the training split
   (min. 3 occurrences; 1,986 patterns) plus hand-written title patterns
   ("Mr. X", "President X" → PER) and organisation-suffix patterns
   ("X Inc.", "X Corp." → ORG). 2,017 patterns in total.
   See `src/rule_based.py`.
3. **Model comparison (bonus):** pre-trained spaCy `en_core_web_lg`
   (same mapping, same slice).
4. **Training demo:** a blank spaCy pipeline (tok2vec + ner) trained on the
   first 1,500 training sentences for 10 epochs (dropout 0.2, seed 42).
   **This is a demo, not a full fine-tune** — full training was not run
   because the point is the end-to-end loop on CPU. Loss fell from
   3,689 to ~350. See `train_demo.py`; the model is saved to
   `models/ner_demo` (git-ignored, reproducible with the script).

## Evaluation

Strict entity-level precision / recall / F1 (`seqeval`), per type and
micro-averaged. All approaches are scored on the **same fixed slice** so
the numbers are comparable. Sentences are fed to spaCy pre-tokenized, so
predictions line up exactly with the gold tokens.

### Slice 1 — first 500 sentences of the test split (983 gold entities)

| Approach | P | R | F1 | PER F1 | ORG F1 | LOC F1 | MISC F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| en_core_web_sm | 0.66 | 0.55 | **0.60** | 0.69 | 0.27 | 0.74 | 0.39 |
| en_core_web_lg | 0.69 | 0.63 | **0.66** | 0.79 | 0.31 | 0.74 | 0.39 |
| Rule-based (EntityRuler) | 0.69 | 0.31 | **0.42** | 0.10 | 0.20 | 0.79 | 0.49 |
| Trained demo (1,500 sents) | 0.75 | 0.72 | **0.73** | 0.80 | 0.50 | 0.82 | 0.56 |

### Slice 2 — test sentences 1000–1499 (500 sentences, 778 gold entities)

| Approach | P | R | F1 | PER F1 | ORG F1 | LOC F1 | MISC F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| en_core_web_sm | 0.67 | 0.57 | **0.62** | 0.65 | 0.47 | 0.70 | 0.65 |
| en_core_web_lg | 0.71 | 0.64 | **0.67** | 0.69 | 0.51 | 0.77 | 0.67 |
| Rule-based (EntityRuler) | 0.78 | 0.42 | **0.55** | 0.12 | 0.38 | 0.73 | 0.66 |
| Trained demo (1,500 sents) | 0.54 | 0.53 | **0.54** | 0.44 | 0.35 | 0.67 | 0.56 |

### What the numbers say

- `en_core_web_lg` beats `en_core_web_sm` on both slices (F1 0.66–0.67 vs
  0.60–0.62), mostly on PER and ORG. The pre-trained models are stable
  across slices.
- Rule-based NER is precise (P 0.69–0.78) but low-recall (R 0.31–0.42):
  the gazetteer finds locations and organisations it has seen before
  (LOC F1 0.73–0.79) but misses unseen person names (PER recall 0.06–0.12).
  It scores better on slice 2, where more entities repeat from training.
- The trained demo looks best on slice 1 (F1 0.73) but drops to 0.54 on
  slice 2 — a small model trained on only 1,500 sentences does not
  generalise evenly. That is the honest result for a demo-sized training
  set; a full training run on all 14,041 sentences would be the next step.
- ORG is the hardest type for the pre-trained models here: CoNLL labels
  sports teams as ORG and capitalised headlines make the mapping noisy.

Full per-type reports are in `results/evaluation_*.json`.

## displaCy visualization (bonus)

`python visualize.py` renders three sample news sentences with both spaCy
models and saves highlighted HTML pages to `outputs/` (e.g.
`outputs/displacy_en_core_web_sm_sample1.html` — open in a browser).

## CLI

```bash
python cli.py "Barack Obama visited Paris last week."
python cli.py --file examples/news_sample.txt
python cli.py --model lg "Microsoft Corp. is based in Redmond."
python cli.py --model rule --format json "Mr. John Smith spoke in Berlin."
```

Models: `sm` (en_core_web_sm, default), `lg` (en_core_web_lg),
`rule` (rule-based EntityRuler), `trained` (local demo model).

## Streamlit app

```bash
streamlit run app.py
```

Paste news text, pick a model, and see entities highlighted with displaCy
plus a table of entities by type. Verified to boot and answer (HTTP 200).

## Setup

```bash
pip install -r requirements.txt
python download_data.py      # CoNLL-2003 from Hugging Face Hub
python download_models.py    # en_core_web_sm + en_core_web_lg
python explore_data.py       # dataset statistics
python evaluate.py --approach all --n 500
python train_demo.py         # training demo (CPU, ~10 min)
```

## Environment notes

- Python 3.12, spaCy 3.8.18, seqeval 1.2.2, Streamlit 1.67. The machine
  runs Debian with a system Python, so user-site installs need
  `--break-system-packages`; NumPy was upgraded to 2.x (and pandas/scipy
  with it) because `pyarrow` requires NumPy ≥ 2.
- `en_core_web_lg` (~780 MB installed) failed to download at first with
  "No space left on device": pip's temporary directory is a 512 MB tmpfs,
  so `download_models.py` is run with `TMPDIR` under the home directory.
- The `datasets` library's HTTP layer fails in this environment (an httpx
  decoder error), so `download_data.py` downloads the parquet files from
  the Hub directly with `curl` and reads them with `pyarrow`.

## Project structure

```
download_data.py      CoNLL-2003 download (HF Hub) + classic-format export
download_models.py    spaCy model downloads (sm + lg)
explore_data.py       dataset statistics -> results/data_exploration.md
train_demo.py         small spaCy NER training demo
evaluate.py           fixed-slice evaluation harness (seqeval)
cli.py                command-line NER
app.py                Streamlit app
visualize.py          displaCy HTML examples -> outputs/
src/conll.py          CoNLL reading, span/tag conversion, label mapping
src/predict.py        model / rule / trained predictors
src/rule_based.py     EntityRuler gazetteer + structural patterns
examples/news_sample.txt
results/              evaluation JSONs + data exploration
outputs/              displaCy example pages
data/                 raw data (git-ignored, re-downloadable)
models/               trained demo model (git-ignored, reproducible)
```
