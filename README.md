# Do Term Extractors Learn Termhood? — data and code

Data, code and results for the paper *Do Term Extractors Learn Termhood? Evidence from Unseen
Terms in English and Spanish* (under review). This copy is anonymised for review.

> **Status:** all results reported in the paper are included. The BiLSTM-CRF+char runs at the
> other reservation levels will be added during the review period.

## Contents

| Folder | What it contains |
|---|---|
| `data/annotation/` | Token-level `BIO` annotation of the English and Spanish corpora (terms `B-T`/`I-T`, entities `B-E`/`I-E`, `O`), with document, sentence and token identifiers and the glossary entry of each annotated token. **The words themselves are not included**, because the source texts are under copyright. |
| `data/glossary/` | The 261-entry bilingual glossary (152 terms, 109 entities) used to normalise and project the annotation. |
| `data/sources/` | Bibliographic list of the 100 source publications (authors, year, title, source, text type). |
| `data/partitions/` | For every corpus, reservation level (10/20/30/50 %) and draw (seeds 0–9), the glossary entries reserved for the test set and those used for validation. |
| `code/` | All the code (`corpus_preparation/`: PDF extraction and cleaning of the source texts): annotation projection, feature extraction, the four model families, the evaluation protocol, statistical tests and figures (`run_all.py` runs everything). Code comments are in Spanish. |
| `results/` | All metrics in CSV: one row per run (precision, recall and F1 per label, term-level, span-level, seen/reserved), summaries, paired tests, bootstrap, ablation tests, error analyses, per-type results (see `results/LEEME.txt`); also the composition sensitivity analysis (15–16), BiLSTM-CRF+char recall on terms with and without unknown words (17), the unsupervised C-value/NC-value baseline under type reservation (18) and the CRF with separate entity labels (19). |

## Reproducing the experiments

The annotation can be reconstructed from the source publications (listed in `data/sources/`)
with `code/paso1_generar_bio.py`, which applies the spaCy transformer pipelines and projects the
glossary forms. Once the token files are in place:

```
pip install -r requirements.txt
python code/run_all.py            # all experiments, then exports every result to CSV
python code/paso14_exportar.py     # only the CSV export
```

ACTER (Terryn et al., 2020) is not redistributed here: download it from its official repository
and prepare it with `code/acter_preprocess.py`.

## Licence

Code: MIT. Data (annotation, glossary, partitions, results): CC BY 4.0. *(To be confirmed.)*
