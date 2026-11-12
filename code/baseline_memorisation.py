"""
Memorisation baseline: labels as a term any test sequence that matches a form annotated
as a term in training (longest match first). It learns nothing but the list of seen terms,
so it measures how far the overlap between training and test alone can go.

Same partitions as the models: five-fold cross-validation (StratifiedKFold, seed 42) and
type reservation (10/20/30/50%, seeds 0-9).

Usage: python baseline_memorisation.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("DATOS_GENERADOS", str(Path(__file__).resolve().parent.parent / "datos_generados"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_bilstm_crf as B  # noqa: E402
from metrics import metricas_completas  # noqa: E402

AQUI = Path(__file__).resolve().parent


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


def etiquetar(test, lista):
    largo = max(len(k) for k in lista)
    pred = []
    for o in test:
        t, y, i = o["tokens"], ["O"] * len(o["tokens"]), 0
        while i < len(t):
            for n in range(min(largo, len(t) - i), 0, -1):
                if tuple(t[i:i + n]) in lista:
                    y[i:i + n] = ["B-T"] + ["I-T"] * (n - 1)
                    i += n
                    break
            else:
                i += 1
        pred.append(y)
    return pred


def evaluar(train, test, reservadas):
    lista = {tuple(o["tokens"][a:b]) for o in train for a, b in spans(o["labels"])}
    pred = etiquetar(test, lista)
    m = metricas_completas([o["labels"] for o in test], pred)
    fila = {"f1_termino": m["f1_termino"], "p_termino": m["p_termino"], "r_termino": m["r_termino"]}
    if reservadas is not None:
        n = ok = nv = okv = 0
        for o, y in zip(test, pred):
            for a, b in spans(o["labels"]):
                aciertos = sum(y[k] != "O" for k in range(a, b))
                if o["entradas"][a] in reservadas:
                    n, ok = n + b - a, ok + aciertos
                else:
                    nv, okv = nv + b - a, okv + aciertos
        fila.update(res_recall_token=ok / n if n else np.nan, vis_recall_token=okv / nv if nv else np.nan)
    return fila


filas = []
for lang in ("en", "es"):
    oraciones, entradas = B.cargar(lang)
    from sklearn.model_selection import StratifiedKFold
    y = np.array([any(l != "O" for l in o["labels"]) for o in oraciones], dtype=int)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y), 1):
        filas.append({"exp": "cv", "lang": lang, "fold": fold,
                      **evaluar([oraciones[i] for i in tr], [oraciones[i] for i in te], None)})
    for pct in (10, 20, 30, 50):
        for seed in range(10):
            train, test, reservadas = B.particion_reserva(oraciones, entradas, pct, seed)
            filas.append({"exp": "reserva", "lang": lang, "pct": pct, "seed": seed, **evaluar(train, test, reservadas)})
    print(lang, "done", flush=True)

df = pd.DataFrame(filas)
df.to_csv(AQUI.parent / "results" / "20_memorisation_baseline.csv", index=False)
print(df.groupby(["exp", "lang", "pct"], dropna=False)[["f1_termino", "p_termino", "r_termino", "res_recall_token",
                                                          "vis_recall_token"]].mean().round(3).to_string())
