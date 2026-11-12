from __future__ import annotations
"""
crf_base.py
───────────
Feature-based CRF: data loading, the 27 features in 9 families, the type-reservation
split and training. Corpus statistics and PMI are computed on the training partition
only, so that nothing leaks from the test set.

  · paths are set with DATOS_GENERADOS and RESULTADOS;
  · the unit of reservation is the glossary entry (column "entrada"), not the string;
  · metrics come from metrics.py (counts, P/R/F1 per label, term, macro/weighted, spans);
  · the full annotation (terms and entities) can be loaded to give entities their
    own labels.
"""
import os

import re
import sys
import json
import time
import random
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import sklearn_crfsuite
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import metricas_completas, guardar_predicciones  # noqa: E402
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BIO_DIR = Path(os.environ.get("DATOS_GENERADOS", HERE.parent / "datos_generados"))
OUT_DIR = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados"))
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
HOLDOUT_LEVELS = [10, 20, 30, 50]
ABLATION_LEVEL = 20  # reservation level of the family ablation

# ─────────────────────────────────────────────────────────────────────────
# Definitional patterns
# ─────────────────────────────────────────────────────────────────────────

DEF_PATTERNS_ES = [
    r"\bse define(?:n)? como\b", r"\bse conoce(?:n)? como\b",
    r"\bdenominad[oa]s?\b", r"\bconsiste(?:n)? en\b",
    r"\bes un tipo de\b", r"\bes una tipo de\b",
    r"\bhace referencia a\b", r"\bentendid[oa]s? como\b",
    r"\bse entiende(?:n)? como\b",
]
DEF_PATTERNS_EN = [
    r"\bis defined as\b", r"\bare defined as\b", r"\brefers? to\b",
    r"\bis known as\b", r"\bare known as\b", r"\bis a type of\b",
    r"\bconsists? of\b", r"\bdenotes?\b",
]
DEF_RE = {
    "en": [re.compile(p, re.IGNORECASE) for p in DEF_PATTERNS_EN],
    "es": [re.compile(p, re.IGNORECASE) for p in DEF_PATTERNS_ES],
}
DEF_RE["acter"] = DEF_RE["en"]  # ACTER (English), external validation (acter_experiments.py)

ACRONYM_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ]{2,6}$")


def norm(s: str) -> str:
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.replace(" - ", "-").replace("- ", "-").lower().strip()


# ─────────────────────────────────────────────────────────────────────────
# Loading and preparing the data
# ─────────────────────────────────────────────────────────────────────────

COLS = [
    "doc_id", "sentence_id", "token_id", "token", "token_lower", "lemma",
    "pos", "tag", "dep", "shape", "label_bio",
    "is_stop", "is_punct", "is_alpha", "is_propn", "is_noun", "is_verb", "is_adj",
    "is_in_noun_phrase", "is_in_noun_chunk_head", "has_adjective_modifier", "has_pre_modifier",
    "has_digit", "freq_global", "document_frequency", "contextual_diversity", "tf_idf",
    "lemma_pos_noun_prob", "lemma_pos_verb_prob", "lemma_pos_adj_prob", "lemma_pos_propn_prob",
    "entrada",
]


def load_corpus(lang: str, completa: bool = False) -> pd.DataFrame:
    """completa=False: terms only (entities = O). completa=True: terms (B-T/I-T) and
    entities (B-E/I-E), to give entities their own labels."""
    fname = f"{lang}_bio_tokens.csv" if completa else f"{lang}_bio_tokens_terms_only.csv"
    ruta = BIO_DIR / fname
    if not ruta.exists() and Path(str(ruta) + ".gz").exists():  # compressed data
        ruta = Path(str(ruta) + ".gz")
    df = pd.read_csv(ruta, usecols=COLS, low_memory=False)
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    df["token_lower"] = df["token_lower"].fillna("").astype(str)
    df["token"] = df["token"].fillna("").astype(str)
    df["lemma"] = df["lemma"].fillna("").astype(str)
    df["label_bio"] = df["label_bio"].fillna("O").astype(str)
    df["entrada"] = df["entrada"].fillna("").astype(str)
    return df


def build_sentences(df: pd.DataFrame):
    """Returns a list of sentence dicts with token-aligned lists."""
    sentences = []
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sentence_id"], sort=False):
        g = g.reset_index(drop=True)
        sentences.append({
            "doc_id": doc_id,
            "sentence_id": sent_id,
            "rows": g.to_dict("records"),
        })
    return sentences


def sentence_surface_types(sent) -> set[str]:
    """Types (normalised surface forms) of the B-T/I-T spans of a sentence."""
    rows = sent["rows"]
    types = set()
    cur = []
    for r in rows:
        if r["label_bio"] == "B-T":
            if cur:
                types.add(norm(" ".join(cur)))
            cur = [r["token_lower"]]
        elif r["label_bio"] == "I-T":
            cur.append(r["token_lower"])
        else:
            if cur:
                types.add(norm(" ".join(cur)))
            cur = []
    if cur:
        types.add(norm(" ".join(cur)))
    return types


def attach_span_text(df: pd.DataFrame, solo_terminos: bool = True) -> pd.DataFrame:
    """Unit of type reservation: the glossary ENTRY each span belongs to (column
    "entrada"). All the forms of a concept (keyword / keywords, lema / lemas, POS
    tagging / part-of-speech tagging) thus always fall in the same partition, and
    nothing leaks between variants. With solo_terminos=True, entities are not drawn
    (when entities have their own labels, only terms are reserved)."""
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    es_t = df["label_bio"].isin(["B-T", "I-T"]) if solo_terminos else df["label_bio"] != "O"
    df["_span_text"] = np.where(es_t & (df["entrada"] != ""), df["entrada"], None)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Cohesion: PMI between adjacent words
# ─────────────────────────────────────────────────────────────────────────

def compute_pmi_lookup(sentences):
    """PMI(w_i, w_{i+1}) for every pair of adjacent words within a sentence."""
    unigram = Counter()
    bigram = Counter()
    total_tokens = 0
    for sent in sentences:
        toks = [r["token_lower"] for r in sent["rows"]]
        total_tokens += len(toks)
        for w in toks:
            unigram[w] += 1
        for i in range(len(toks) - 1):
            bigram[(toks[i], toks[i + 1])] += 1
    total_bigrams = sum(bigram.values())

    def pmi(w1, w2):
        c12 = bigram.get((w1, w2), 0)
        if c12 == 0:
            return 0.0
        p12 = c12 / total_bigrams
        p1 = unigram[w1] / total_tokens
        p2 = unigram[w2] / total_tokens
        val = np.log2(p12 / (p1 * p2))
        return max(val, 0.0)  # descartamos asociaciones negativas (ruido)

    return pmi


def pmi_quartile_bounds(sentences, pmi_fn):
    vals = []
    for sent in sentences:
        toks = [r["token_lower"] for r in sent["rows"]]
        for i in range(len(toks)):
            if i > 0:
                v = pmi_fn(toks[i - 1], toks[i])
                if v > 0:
                    vals.append(v)
            if i < len(toks) - 1:
                v = pmi_fn(toks[i], toks[i + 1])
                if v > 0:
                    vals.append(v)
    vals = np.array(vals) if vals else np.array([0.0])
    return np.percentile(vals, [25, 50, 75])


def pmi_q(val, bounds):
    if val <= 0:
        return "Q0"
    if val <= bounds[0]:
        return "Q1"
    if val <= bounds[1]:
        return "Q2"
    if val <= bounds[2]:
        return "Q3"
    return "Q4"


# ─────────────────────────────────────────────────────────────────────────
# Definitional context and acronym pattern (computed once per sentence)
# ─────────────────────────────────────────────────────────────────────────

def sentence_has_definition(sent, lang) -> bool:
    text = " ".join(r["token_lower"] for r in sent["rows"])
    return any(p.search(text) for p in DEF_RE[lang])


def acronym_flags(sent) -> set[int]:
    """Positions of the tokens involved in an acronym pattern:
    'X ( ABC )' or 'ABC ( X Y )'."""
    rows = sent["rows"]
    toks = [r["token"] for r in rows]
    flags = set()
    n = len(toks)
    for i in range(n):
        if toks[i] == "(" and i + 2 < n and toks[i + 2] == ")":
            candidate = toks[i + 1]
            if ACRONYM_RE.match(candidate):
                flags.add(i + 1)  # the acronym itself
                if i > 0:
                    flags.add(i - 1)  # the token right before the parenthesis
    return flags


# ─────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────

FAMILIAS = {
    "morfologia":            ["suffix3", "suffix4"],
    "forma":                 ["shape", "is_alpha", "has_digit", "is_punct", "is_stop"],
    "morfosintaxis":         ["pos", "tag", "is_noun", "is_verb", "is_adj", "is_propn", "dom_pos"],
    "sintaxis":               ["dep", "in_np", "np_head", "has_amod", "has_premod"],
    "estadisticos":          ["freq_q", "df_q", "tfidf_q", "cd_q"],
    "contexto":              ["ctx"],  # marker: all prevN/nextN/BOS/EOS features are removed together
    "cohesion_pmi":          ["pmi_prev_q", "pmi_next_q"],
    "contexto_definitorio":  ["in_definition_context"],
    "patron_sigla":          ["acronym_pattern"],
}


def _dominant_pos(row) -> str:
    probs = {
        "NOUN": float(row.get("lemma_pos_noun_prob", 0) or 0),
        "VERB": float(row.get("lemma_pos_verb_prob", 0) or 0),
        "ADJ": float(row.get("lemma_pos_adj_prob", 0) or 0),
        "PROPN": float(row.get("lemma_pos_propn_prob", 0) or 0),
    }
    return max(probs, key=probs.get)


def recompute_freq_stats(train_sents):
    """Recomputes freq_global, document_frequency, contextual_diversity and TF-IDF
    from the training sentences ONLY. The precomputed columns of the CSV are not
    used, because they were computed on the whole corpus and would leak
    information from the test set. Contextual diversity is computed per sentence
    (finer than document frequency, which is per document)."""
    lemma_freq = Counter()
    lemma_docs = defaultdict(set)
    lemma_sents = defaultdict(set)
    doc_lemma_count = defaultdict(Counter)
    doc_total_tokens = Counter()
    all_docs = set()

    for sent in train_sents:
        doc_id, sent_id = sent["doc_id"], sent["sentence_id"]
        all_docs.add(doc_id)
        for r in sent["rows"]:
            lemma = r["lemma"] or r["token_lower"]
            lemma_freq[lemma] += 1
            lemma_docs[lemma].add(doc_id)
            lemma_sents[lemma].add((doc_id, sent_id))
            doc_lemma_count[doc_id][lemma] += 1
            doc_total_tokens[doc_id] += 1

    n_docs = max(len(all_docs), 1)

    def get(lemma, doc_id):
        freq_global = lemma_freq.get(lemma, 0)
        df_ = len(lemma_docs.get(lemma, ()))
        cd = len(lemma_sents.get(lemma, ()))
        tf = doc_lemma_count.get(doc_id, {}).get(lemma, 0) / max(doc_total_tokens.get(doc_id, 1), 1)
        idf = np.log2((n_docs + 1) / (df_ + 1))
        tfidf = tf * idf
        return freq_global, df_, cd, tfidf

    # distributions for the quartiles (over the training tokens themselves)
    fg_vals, df_vals, cd_vals, tfidf_vals = [], [], [], []
    for sent in train_sents:
        doc_id = sent["doc_id"]
        for r in sent["rows"]:
            lemma = r["lemma"] or r["token_lower"]
            fg, df_, cd, tfidf = get(lemma, doc_id)
            fg_vals.append(fg); df_vals.append(df_); cd_vals.append(cd); tfidf_vals.append(tfidf)

    def bounds(vals):
        arr = np.array([v for v in vals if v > 0])
        return np.percentile(arr, [25, 50, 75]) if len(arr) else np.array([0, 0, 0])

    stat_bounds = {
        "freq_global": bounds(fg_vals),
        "document_frequency": bounds(df_vals),
        "tf_idf": bounds(tfidf_vals),
        "contextual_diversity": bounds(cd_vals),
    }
    return get, stat_bounds


def quartile_bounds(df, col):
    vals = df[col].values
    vals = vals[vals > 0]
    if len(vals) == 0:
        return [0, 0, 0]
    return np.percentile(vals, [25, 50, 75])


def quartile(val, bounds):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Q0"
    if v <= 0:
        return "Q0"
    if v <= bounds[0]:
        return "Q1"
    if v <= bounds[1]:
        return "Q2"
    if v <= bounds[2]:
        return "Q3"
    return "Q4"


def build_features_for_sentences(sentences, stats_get, stat_bounds, pmi_fn, pmi_bounds, lang, exclude_family=None):
    sequences, labels = [], []
    for sent in sentences:
        rows = sent["rows"]
        n = len(rows)
        doc_id = sent["doc_id"]
        toks_lower = [r["token_lower"] for r in rows]
        has_def = sentence_has_definition(sent, lang) if exclude_family != "contexto_definitorio" else False
        acro_idx = acronym_flags(sent) if exclude_family != "patron_sigla" else set()

        sent_feats, sent_labels = [], []
        for i, row in enumerate(rows):
            lemma = str(row["lemma"])
            feats = {}

            if exclude_family != "morfologia":
                feats["suffix3"] = lemma[-3:] if len(lemma) >= 3 else lemma
                feats["suffix4"] = lemma[-4:] if len(lemma) >= 4 else lemma

            if exclude_family != "forma":
                feats["shape"] = str(row["shape"])
                feats["is_alpha"] = bool(row["is_alpha"])
                feats["has_digit"] = bool(row["has_digit"])
                feats["is_punct"] = bool(row["is_punct"])
                feats["is_stop"] = bool(row["is_stop"])

            if exclude_family != "morfosintaxis":
                feats["pos"] = str(row["pos"])
                feats["tag"] = str(row["tag"])
                feats["is_noun"] = bool(row["is_noun"])
                feats["is_verb"] = bool(row["is_verb"])
                feats["is_adj"] = bool(row["is_adj"])
                feats["is_propn"] = bool(row["is_propn"])
                feats["dom_pos"] = _dominant_pos(row)

            if exclude_family != "sintaxis":
                feats["dep"] = str(row["dep"])
                feats["in_np"] = bool(row["is_in_noun_phrase"])
                feats["np_head"] = bool(row["is_in_noun_chunk_head"])
                feats["has_amod"] = bool(row["has_adjective_modifier"])
                feats["has_premod"] = bool(row["has_pre_modifier"])

            if exclude_family != "estadisticos":
                lemma_key = lemma or row["token_lower"]
                fg, df_, cd, tfidf = stats_get(lemma_key, doc_id)
                feats["freq_q"] = quartile(fg, stat_bounds["freq_global"])
                feats["df_q"] = quartile(df_, stat_bounds["document_frequency"])
                feats["tfidf_q"] = quartile(tfidf, stat_bounds["tf_idf"])
                feats["cd_q"] = quartile(cd, stat_bounds["contextual_diversity"])

            if exclude_family != "contexto":
                for offset, prefix in [(-1, "prev1"), (-2, "prev2"), (1, "next1"), (2, "next2")]:
                    idx = i + offset
                    if 0 <= idx < n:
                        r2 = rows[idx]
                        rl = str(r2["lemma"])
                        feats[f"{prefix}:pos"] = str(r2["pos"])
                        feats[f"{prefix}:suffix3"] = rl[-3:] if len(rl) >= 3 else rl
                        feats[f"{prefix}:is_noun"] = bool(r2["is_noun"])
                        feats[f"{prefix}:is_adj"] = bool(r2["is_adj"])
                        feats[f"{prefix}:in_np"] = bool(r2["is_in_noun_phrase"])
                    else:
                        feats[f"{prefix}:BOS/EOS"] = True
                if i == 0:
                    feats["BOS"] = True
                if i == n - 1:
                    feats["EOS"] = True

            if exclude_family != "cohesion_pmi":
                pv = pmi_fn(toks_lower[i - 1], toks_lower[i]) if i > 0 else 0.0
                nv = pmi_fn(toks_lower[i], toks_lower[i + 1]) if i < n - 1 else 0.0
                feats["pmi_prev_q"] = pmi_q(pv, pmi_bounds)
                feats["pmi_next_q"] = pmi_q(nv, pmi_bounds)

            if exclude_family != "contexto_definitorio":
                feats["in_definition_context"] = bool(has_def)

            if exclude_family != "patron_sigla":
                feats["acronym_pattern"] = i in acro_idx

            sent_feats.append(feats)
            sent_labels.append(str(row["label_bio"]))

        sequences.append(sent_feats)
        labels.append(sent_labels)

    return sequences, labels


# ─────────────────────────────────────────────────────────────────────────
# Type reservation
# ─────────────────────────────────────────────────────────────────────────

def split_by_type_holdout(sentences, span_types, pct, seed=SEED):
    """Type reservation: pd.Series.sample(random_state=seed) over the list of span
    types as they appear in the corpus, with n_holdout truncated (int, not round),
    so that every model gets exactly the same reserved types for a level and seed.
    Every sentence containing a reserved type goes to the test set."""
    n_holdout = max(1, int(len(span_types) * pct / 100))
    held_out = set(pd.Series(span_types).sample(n=n_holdout, random_state=seed).values)

    train, test = [], []
    for sent in sentences:
        stypes = {r["_span_text"] for r in sent["rows"] if pd.notna(r.get("_span_text"))}
        if stypes & held_out:
            test.append(sent)
        else:
            train.append(sent)
    return train, test, held_out


def train_crf(X_train, y_train):
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.01, max_iterations=200)
    crf.fit(X_train, y_train)
    return crf


