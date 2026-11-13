Results of the paper (all in CSV; .csv.gz = compressed CSV). Column names are in Spanish;
this file explains them.

Common keys: modelo (CRF, BiLSTM-CRF, XLM-R or a token classifier), exp (reserva = type
reservation, cv = cross-validation, lodo = leave-one-domain-out), lang (en, es, acter), pct
(percentage of reserved types), seed (partition 0-9), fold (cross-validation fold), dominio
(ACTER domain, LODO).

Metric prefixes: tp_/fp_/fn_/tn_ counts; p_/r_/f1_ precision, recall, F1.
Suffixes: _bt (B-T), _it (I-T), _o (O), _termino (B-T+I-T against O; the main metric, term F1),
_macro, _global (weighted), _span_exacto / _span_parcial (exact / partial span matching).
res_*: the same metrics on the tokens of reserved (never seen) types only; vis_*: on the rest.
In LODO, res_ = types absent from the training domains.

07/08, analysis per run: res_/vis_prop_exacta|por_defecto|por_exceso|solapamiento|omitida
(outcome of each occurrence: exact, under-extended, over-extended, overlapping, missed),
*_mono/_poli_prop_exacta (exact delimitation of single- and multi-word terms),
res_recall_token_sin_oov|con_oov (without / with words absent from training),
res_recall_token_<band> (frequency band of the type), err_* (error typology on reserved terms
plus false positives), errsc_* (the same without the type "corpus").
OOV: some word of the term does not occur in training (validation set excluded).
Type frequency: annotated occurrences in the whole corpus.

Statistical tests (the difference is always modelo_b - modelo_a):
05  pairs of models, partition by partition (reservation, 10 partitions) or fold by fold (CV):
    mean difference, 95% CI (t), paired Cohen's d, rank-biserial correlation, wins, Wilcoxon
    and paired t-test; Holm's correction per protocol, pair of models and metric over the 8
    language-level comparisons (EN+ES) or the 4 levels (ACTER).
05b mean, standard deviation, standard error and 95% CI of each metric by model, corpus and level.
05c ACTER LODO: each neural model (5 seeds) against the CRF (deterministic) per domain,
    one-sample t-test with Holm's correction.
05d CRF ablation: one-sample t-test and Wilcoxon over the 10 differences of each family,
    language and level (Δ term F1 and Δ reserved recall), Holm over the 72 tests.
05e/05f paired sentence-level bootstrap within each partition (1000 resamples): difference in
    term F1 and in reserved recall, 95% CI and p; 05f counts in how many partitions it is
    significant in each direction.
11b Spearman correlation between the frequency of each reserved type and its recall.
07 also: tipo_* = type-level evaluation, as in TermEval (tipo_recall: distinct test terms with
at least one occurrence recognised in full; tipo_precision_test/_corpus: distinct proposed forms
that are annotated forms in the test set / in the whole corpus); *_por_defecto_prop_termino_anotado:
share of under-extensions that propose an annotated term themselves (e.g. "corpus" in "corpus
linguistics"), to assess the effect of annotating only the longest sequence; *_hasta2019 /
*_desde2020: F1 and reserved recall by the publication year of the document (XLM-R cannot have
seen texts published from 2020 onwards during pretraining).

Additional files:
15  CRF results broken down by groups of documents (text type, dominant source, longest
    documents) and averaged over documents.
16  CRF trained on a training set balanced by document (equilibrado) or reduced at random to
    the same size (aleatorio).
17  BiLSTM-CRF+char, BiLSTM-CRF, CRF and XLM-R at every reservation level (the draws run for
    the character-level variant): token recall of reserved terms with (con_oov) and without
    (sin_oov) a word absent from training.
18  Unsupervised baseline: candidates ranked by C-value or NC-value (medida), with a list as
    long as the CRF's (k_como = CRF) or as another model's; tipo_recall, tipo_recall_reservados,
    tipo_precision_test and tipo_f1_test as in 07.
19  CRF with separate labels for terms and entities: term F1 and entity F1 per run.
20  Memorisation baseline (labels the sequences annotated as terms in training): term F1,
    precision and recall under cross-validation and type reservation, and token recall of
    reserved (res_) and seen (vis_) terms.
21  Reserved occurrences with an unknown word, split by whether all its XLM-R subwords occur in
    training (oov_subpalabras_vistas) or not (oov_alguna_subpalabra_nueva): token recall of
    each model.
