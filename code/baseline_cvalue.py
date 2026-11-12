"""
Unsupervised baseline: C-value and NC-value (Frantzi et al., 2000) on the same
type-reservation partitions as the supervised models.

In each partition, candidates are extracted from the test sentences with part-of-speech
patterns and ranked by C-value or NC-value, and a list as long as the CRF's in that
partition is kept. We measure the recall of glossary entries (all of them and the reserved
ones only) and the precision of the proposed forms, as in export_results.py.

Usage: python baseline_cvalue.py
"""
import os
import sys
from collections import Counter, defaultdict
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[2]
os.environ.setdefault("DATOS_GENERADOS", str(Path(__file__).resolve().parent.parent / "datos_generados"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import train_bilstm_crf as B  # noqa: E402
from extractor.linguistic_filter import extract_candidates  # noqa: E402

GRAMMAR = AQUI / "extractor" / "grammar.yaml"
EJEC = AQUI.parent / "results" / "07_analisis_por_ejecucion.csv"
MIN_FREQ = 2
REF = 0.05   # NC-value reference terms: the top 5% of the C-value list


def spans(labels):   # same as export_results.spans
    out, ini = [], None
    for i, l in enumerate(list(labels) + ["O"]):
        if l == "B-T" or l == "O" or (l == "I-T" and ini is None):
            if ini is not None:
                out.append((ini, i))
                ini = None
            if l in ("B-T", "I-T"):
                ini = i
    return out


def pos_lemas(lang):
    """(doc_id, sentence_id) -> (lemmas, POS), in the same order as B.cargar."""
    ruta = B.DATOS / f"{lang}_bio_tokens_terms_only.csv.gz"
    df = pd.read_csv(ruta, usecols=["doc_id", "sentence_id", "token_id", "lemma", "pos"], low_memory=False)
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    df["lemma"] = df["lemma"].fillna("").astype(str)
    df["pos"] = df["pos"].fillna("X").astype(str)
    return {(d, s): (g["lemma"].tolist(), g["pos"].tolist())
            for (d, s), g in df.groupby(["doc_id", "sentence_id"], sort=False)}


def cvalue_lista(test, info, lang):
    """Test candidates ranked by C-value and NC-value (highest first)."""
    docs = defaultdict(list)
    for o in test:
        lemas, pos = info[(o["doc_id"], o["sentence_id"])]
        docs[o["doc_id"]].append([{"text": t, "lemma": l, "pos": p} for t, l, p in zip(o["tokens"], lemas, pos)])
    documentos = [{"doc_id": d, "sentences": s} for d, s in docs.items()]
    cands = extract_candidates(documentos, language=lang, grammar_path=GRAMMAR)
    claves = {tuple(c["term"].split()) for c in cands}
    max_len = max(len(k) for k in claves)
    freq = Counter()
    for o in test:
        t = o["tokens"]
        for n in range(1, max_len + 1):
            for i in range(len(t) - n + 1):
                k = tuple(t[i:i + n])
                if k in claves:
                    freq[k] += 1
    claves = {k for k in claves if freq[k] >= MIN_FREQ}
    contenedores = defaultdict(set)
    for b in claves:
        for n in range(1, len(b)):
            for i in range(len(b) - n + 1):
                if b[i:i + n] in claves:
                    contenedores[b[i:i + n]].add(b)
    cv = {}
    for a in claves:
        tb = contenedores.get(a)
        cv[a] = (1.0 if len(a) == 1 else log2(len(a))) * (freq[a] - (sum(freq[b] for b in tb) / len(tb) if tb else 0))
    orden_c = sorted(claves, key=lambda k: (-cv[k], k))
    # NC-value (Frantzi et al. 2000): 0.8·C-value + 0.2·Σ_b f_a(b)·weight(b), where b are the nouns,
    # adjectives and verbs (lemmas) within ±2 tokens, weight(b) = t(b)/n and t(b) is the number of
    # reference terms b occurs with; the reference terms are the top REF of the C-value list
    ctx = defaultdict(Counter)
    for c in cands:
        k = tuple(c["term"].split())
        if k in claves:
            ctx[k].update(c["context_words"])
    ref = orden_c[:max(1, int(len(orden_c) * REF))]
    t = Counter(b for k in ref for b in ctx[k])
    nc = {a: 0.8 * cv[a] + 0.2 * sum(f * t[b] / len(ref) for b, f in ctx[a].items()) for a in claves}
    orden_nc = sorted(claves, key=lambda k: (-nc[k], k))
    return {"C-value": [" ".join(k) for k in orden_c], "NC-value": [" ".join(k) for k in orden_nc]}


def main():
    ejec = pd.read_csv(EJEC)
    filas = []
    for lang in ("en", "es"):
        oraciones, entradas = B.cargar(lang)
        info = pos_lemas(lang)
        for pct in (10, 20, 30, 50):
            for seed in range(10):
                _, test, reservadas = B.particion_reserva(oraciones, entradas, pct, seed)
                formas_de = defaultdict(set)     # entry -> forms annotated in the test set
                formas_test = set()
                for o in test:
                    for a, b in spans(o["labels"]):
                        f = " ".join(o["tokens"][a:b])
                        formas_test.add(f)
                        if o["entradas"][a]:
                            formas_de[o["entradas"][a]].add(f)
                tipos_test = set(formas_de)
                tipos_res = tipos_test & reservadas
                ordenes = cvalue_lista(test, info, lang)
                for (medida, orden), ref in [(mo, r) for mo in ordenes.items() for r in ("CRF", "BiLSTM-CRF", "XLM-R", "gold")]:
                    if ref == "gold":
                        k = len(formas_test)
                    else:
                        m = ejec[(ejec.lang == lang) & (ejec.pct == pct) & (ejec.seed == seed) & (ejec.modelo == ref)]
                        if m.empty:
                            continue
                        k = int(m["tipo_n_formas_propuestas"].iloc[0])
                    lista = set(orden[:k])
                    ok = {e for e in tipos_test if formas_de[e] & lista}
                    p = len(lista & formas_test) / len(lista) if lista else np.nan
                    r = len(ok) / len(tipos_test)
                    filas.append({"medida": medida, "lang": lang, "pct": pct, "seed": seed, "k_como": ref, "k": k,
                                  "n_candidatos": len(orden),
                                  "tipo_recall": r, "tipo_recall_reservados": len(ok & tipos_res) / len(tipos_res),
                                  "tipo_precision_test": p, "tipo_f1_test": 2 * p * r / (p + r) if p + r else 0.0})
                print(lang, pct, seed, "candidates:", len(ordenes["C-value"]), flush=True)
    df = pd.DataFrame(filas)
    salida = AQUI.parent / "results" / "18_unsupervised_cvalue_ncvalue.csv"
    df.to_csv(salida, index=False)
    print(df[df.k_como == "CRF"].groupby(["medida", "lang", "pct"])[["tipo_recall", "tipo_recall_reservados", "tipo_precision_test",
                                                 "tipo_f1_test"]].mean().round(3).to_string())
    print("Saved to", salida)


if __name__ == "__main__":
    main()
