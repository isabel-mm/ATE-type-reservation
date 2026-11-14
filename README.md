# Do Term Extractors Learn Termhood? — data and code

Data, code and results for the paper *Do Term Extractors Learn Termhood? Evidence from Unseen
Terms in English and Spanish* (under review). This copy is anonymised for review.

> **Status:** all results reported in the paper are included.

## Contents

| Folder | What it contains |
|---|---|
| `data/annotation/` | Token-level `BIO` annotation of the English and Spanish corpora (terms `B-T`/`I-T`, entities `B-E`/`I-E`, `O`), with document, sentence and token identifiers and the glossary entry of each annotated token. **The words themselves are not included**, because the source texts are under copyright. |
| `data/glossary/` | The 261-entry bilingual glossary (152 terms, 109 entities) used to normalise and project the annotation. |
| `data/sources/` | Bibliographic list of the 100 source publications (authors, year, title, source, text type). |
| `data/partitions/` | For every corpus, reservation level (10/20/30/50%) and draw (seeds 0–9), the glossary entries reserved for the test set and those used for validation. |
| `code/` | All the code (see below). |
| `results/` | All metrics in CSV (see `results/README.txt`). |
| `figures/` | Every figure of the paper in PDF (see below). |

## Figures

| File | Figure |
|---|---|
| `fig_distribucion.pdf` | 1. Rank–frequency distribution and terminological saturation |
| `fig_flujo.pdf` | 2. Study workflow |
| `fig_bio.pdf` | 3. The `BIO` scheme, with an example of a partial error |
| `fig_protocolo.pdf` | 4. The type-reservation protocol |
| `fig_reserva.pdf` | 5. Term F1 under type reservation |
| `fig_recall.pdf` | 6. Recall of seen and reserved terms |
| `fig_oov.pdf` | 7. Recall of reserved terms with and without unknown words |
| `fig_ablacion.pdf` | 8. CRF feature ablation |
| `fig_acter.pdf` | B1. ACTER, leave-one-domain-out |

Figures 5–8 and B1 are drawn by `code/make_figures.py` from `results/`. Figures 2–4 are TikZ
diagrams whose sources are in `figures/diagrams/`; each one compiles on its own with
`pdflatex -jobname=fig_flujo "\def\fig{fig_flujo.tex}\input{standalone}"`. The TikZ diagrams
were drafted with the assistance of an LLM and revised by the authors.

## Code

| Script | What it does |
|---|---|
| `corpus_preparation/` | PDF-to-text conversion and cleaning of the source texts. |
| `glossary_match.py`, `build_bio_annotation.py` | Mark the glossary forms in the texts and build the token-level `BIO` files with spaCy features. |
| `crf_base.py`, `train_crf.py` | Feature-based CRF (27 features in 9 families): type reservation, separate entity labels, feature ablation, cross-validation. |
| `train_token_classifiers.py` | Token classifiers with the same features. |
| `train_bilstm_crf.py`, `train_bilstm_crf_char.py` | BiLSTM-CRF, without and with a character-level encoder. |
| `train_xlmr.py` | XLM-RoBERTa fine-tuned for token classification. |
| `acter_preprocess.py`, `acter_experiments.py` | ACTER: preparation and leave-one-domain-out evaluation. |
| `metrics.py` | Token-, term-, and span-level metrics shared by all models. |
| `export_results.py` | Collects every run into the CSV files of `results/`, with statistical tests and error analyses. |
| `baseline_memorisation.py`, `baseline_cvalue.py` | Memorisation baseline and unsupervised C-value/NC-value baseline under type reservation. |
| `char_oov_analysis.py`, `oov_subword_analysis.py` | Reserved-term recall with and without unknown words (BiLSTM-CRF+char; XLM-R subword coverage). |
| `source_composition.py`, `source_sensitivity.py` | Sensitivity of the results to the composition of the corpus. |
| `make_figures.py` | Result figures of the paper, from `results/`. |
| `run_all.py` | Runs every experiment in order and exports the results. |

Comments and messages are in English. Variable names, command-line options (`reserva`,
`--semillas`, `--niveles`…) and the column names of the result files are in Spanish;
`results/README.txt` explains every column.

## Reproducing the experiments

The annotation can be reconstructed from the source publications (listed in `data/sources/`)
with `code/build_bio_annotation.py`, which applies the spaCy transformer pipelines and marks the
glossary forms. Once the token files are in place:

```
pip install -r requirements.txt
python code/run_all.py --quick    # quick check of the pipeline
python code/run_all.py            # all experiments, then exports every result to CSV
python code/export_results.py     # only the CSV export
python code/make_figures.py       # figures
```

ACTER (Terryn et al., 2020) is not redistributed here: download it from its official repository
and prepare it with `code/acter_preprocess.py`.

## Licence

Code: MIT. Data (annotation, glossary, partitions, results): CC BY 4.0. *(To be confirmed.)*
