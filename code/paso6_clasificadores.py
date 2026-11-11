"""
paso6_clasificadores.py
───────────────────────
Clasificadores clásicos (sin secuencia) con la anotación actual: sustituye a la
tabla de julio de 2026 (anotación anterior) del Apéndice V.

Cada token se clasifica por separado como término (B-T o I-T) o no término, con
los mismos 27 rasgos del CRF (crf_base.build_features_for_sentences; estadísticas
recalculadas solo sobre el entrenamiento de cada partición). Dos protocolos:

  cv       validación cruzada aleatoria de 5 pliegues (los mismos que el CRF)
  reserva  reserva de tipos (por defecto 10 y 20 %), semillas 0-9 (las mismas
           particiones que el CRF, el BiLSTM-CRF y XLM-R)

En cada ejecución se guardan P/R/F1 de término, AUC, métricas por unidad
completa (spans, regla de reconstrucción), delimitación exacta de monoléxicos y
poliléxicos y recuperación de los tokens reservados y vistos.

Clasificadores: regresión logística, SVM lineal, Naive Bayes (Bernoulli), árbol
de decisión y Random Forest. Se omiten los que no escalan a ~1,4 millones de
tokens con miles de rasgos dispersos (SVM con núcleo RBF, k-NN, gradient boosting).

Uso:    python paso6_clasificadores.py [--lang en] [--niveles 10,20] [--semillas 0,...,9]
Salida: resultados/clasificadores/clasificadores_todas.csv (una fila por ejecución y clasificador)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crf_base as base  # noqa: E402
from metricas import metricas_span, spans_termino  # noqa: E402

OUT = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados")) / "clasificadores"
OUT.mkdir(parents=True, exist_ok=True)
DEST = OUT / "clasificadores_todas.csv"
NJOBS = max(1, (os.cpu_count() or 2) - 2)  # deja núcleos libres para alimentar la GPU


SPANS = True  # siempre se calculan las métricas por unidad completa


def modelos():
    return {
        "Regresión logística": LogisticRegression(solver="liblinear", max_iter=1000),
        "SVM lineal": LinearSVC(dual=False),
        "Naive Bayes": BernoulliNB(),
        "Árbol de decisión": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=NJOBS, random_state=42),
    }


def matriz(train_sents, test_sents, lang):
    stats_get, stat_bounds = base.recompute_freq_stats(train_sents)
    pmi_fn = base.compute_pmi_lookup(train_sents)
    pmi_bounds = base.pmi_quartile_bounds(train_sents, pmi_fn)
    Xtr, ytr = base.build_features_for_sentences(train_sents, stats_get, stat_bounds, pmi_fn, pmi_bounds, lang)
    Xte, yte = base.build_features_for_sentences(test_sents, stats_get, stat_bounds, pmi_fn, pmi_bounds, lang)
    dv = DictVectorizer(sparse=True)
    Xtr = dv.fit_transform([f for s in Xtr for f in s])
    Xte = dv.transform([f for s in Xte for f in s])
    gold = yte
    ytr = np.array([lab != "O" for s in ytr for lab in s], dtype=int)
    yte = np.array([lab != "O" for s in yte for lab in s], dtype=int)
    return Xtr, ytr, Xte, yte, gold


def a_bio(pred, gold):
    """Predicción binaria por token -> secuencias BIO (cada tramo continuo de términos es una unidad)."""
    out, k = [], 0
    for g in gold:
        seq, prev = [], 0
        for _ in g:
            v = pred[k]; k += 1
            seq.append("O" if not v else ("I-T" if prev else "B-T")); prev = v
        out.append(seq)
    return out


def por_longitud(gold, pred_bio):
    r = {}
    for nombre, cond in (("mono", lambda a, b: b - a == 1), ("poli", lambda a, b: b - a > 1)):
        tot = ok = 0
        for g, p in zip(gold, pred_bio):
            ps = set(spans_termino(p))
            for a, b in spans_termino(g):
                if cond(a, b):
                    tot += 1; ok += (a, b) in ps
        r[f"r_span_exacto_{nombre}"] = ok / tot if tot else np.nan
    return r


def evaluar(Xtr, ytr, Xte, yte, reservado, base_fila, gold=None):
    filas = []
    for nombre, clf in modelos().items():
        t0 = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        punt = clf.decision_function(Xte) if hasattr(clf, "decision_function") else clf.predict_proba(Xte)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        fila = dict(base_fila, clasificador=nombre, p_termino=p, r_termino=r, f1_termino=f,
                    roc_auc=roc_auc_score(yte, punt), segundos=round(time.time() - t0, 1))
        if SPANS and gold is not None:
            bio = a_bio(pred, gold)
            fila.update({k: v for k, v in metricas_span(gold, bio).items() if "span" in k})
            fila.update(por_longitud(gold, bio))
        if reservado is not None:
            m = reservado & (yte == 1)
            fila["res_r_termino"] = pred[m].mean() if m.any() else np.nan
            v = (~reservado) & (yte == 1)
            fila["vis_r_termino"] = pred[v].mean() if v.any() else np.nan
        filas.append(fila)
        print(f"    {nombre}: F1={f:.3f} ({fila['segundos']:.0f}s)", flush=True)
    return filas


def main():
    langs = [sys.argv[sys.argv.index("--lang") + 1]] if "--lang" in sys.argv else ["en", "es"]
    hechas = pd.read_csv(DEST) if DEST.exists() else pd.DataFrame()
    for lang in langs:
        df = base.attach_span_text(base.load_corpus(lang), solo_terminos=True)
        span_types = np.array(sorted(df["_span_text"].dropna().unique()))
        sentences = base.build_sentences(df)
        y_strat = np.array([any(r["label_bio"] != "O" for r in s["rows"]) for s in sentences], dtype=int)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(sentences)), y_strat))
        niveles = [int(x) for x in sys.argv[sys.argv.index("--niveles") + 1].split(",")] if "--niveles" in sys.argv else [20, 10]
        semillas = [int(x) for x in sys.argv[sys.argv.index("--semillas") + 1].split(",")] if "--semillas" in sys.argv else list(range(10))
        tareas = [("cv", k + 1, None) for k in range(5)] + [("reserva", s, p) for p in niveles for s in semillas]
        for exp, k, pct in tareas:
            rid = f"cv_{lang}_f{k}" if exp == "cv" else f"reserva_{lang}_p{pct}_s{k}"
            if len(hechas) and rid in set(hechas.get("rid", [])):
                continue
            print(f"[{rid}]", flush=True)
            if exp == "cv":
                tr, te = folds[k - 1]
                train, test, reservadas = [sentences[i] for i in tr], [sentences[i] for i in te], None
            else:
                train, test, reservadas = base.split_by_type_holdout(sentences, span_types, pct, seed=k)
            Xtr, ytr, Xte, yte, gold = matriz(train, test, lang)
            res = None
            if reservadas is not None:
                res = np.array([r.get("_span_text") in reservadas for s in test for r in s["rows"]])
            filas = evaluar(Xtr, ytr, Xte, yte, res, {"rid": rid, "exp": exp, "lang": lang, "pct": pct,
                                                       "seed": k if exp == "reserva" else None,
                                                       "fold": k if exp == "cv" else None}, gold)
            hechas = pd.concat([hechas, pd.DataFrame(filas)], ignore_index=True)
            hechas.to_csv(DEST, index=False)


if __name__ == "__main__":
    main()
