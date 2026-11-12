"""
char_oov_analysis.py
────────────────────
BiLSTM-CRF+char compared with the BiLSTM-CRF, the CRF and XLM-R at 20% reservation
(seeds 0-4): token recall of reserved terms, separating the occurrences that contain a
word absent from training (OOV, as defined in export_results.py) from those whose words
are all known.

Usage:  python char_oov_analysis.py RESULTS_FOLDER
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("DATOS_GENERADOS", str(Path(__file__).resolve().parent.parent / "datos_generados"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_bilstm_crf as B  # noqa: E402

RES = Path(sys.argv[1])
MODELOS = {"CRF": RES / "crf" / "pred", "BiLSTM-CRF": RES / "bilstm" / "pred",
           "BiLSTM-CRF+char": RES / "bilstm_char" / "pred",
           "XLM-R": RES / "transformer" / "xlm-roberta-base" / "pred"}


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


filas = []
for lang in ("en", "es"):
    oraciones, entradas = B.cargar(lang)
    for seed in range(5):
        train, test, reservadas = B.particion_reserva(oraciones, entradas, 20, seed)
        tr, _ = B.separar_validacion(train, seed, por_entradas=True)
        vocab = {t for o in tr for t in o["tokens"]}
        for modelo, carpeta in MODELOS.items():
            f = carpeta / f"reserva_{lang}_p20_s{seed}.csv.gz"
            if not f.exists():
                f = carpeta / f"reserva_{lang}_p20_s{seed}_todas.csv.gz"   # CRF with all features
            if not f.exists():
                continue
            pred = pd.read_csv(f, usecols=["doc_id", "sentence_id", "i", "pred"], keep_default_na=False,
                               dtype={"doc_id": str, "sentence_id": str})
            pmap = defaultdict(dict)
            for d, s, i, p in pred.itertuples(index=False):
                pmap[(d, s)][int(i)] = p
            n = defaultdict(int)
            for o in test:
                pm = pmap.get((str(o["doc_id"]), str(o["sentence_id"])), {})
                yp = [pm.get(i, "O") for i in range(len(o["tokens"]))]
                for a, b in spans(o["labels"]):
                    if o["entradas"][a] not in reservadas:
                        continue
                    g = "con_oov" if any(t not in vocab for t in o["tokens"][a:b]) else "sin_oov"
                    n[g + "_n"] += b - a
                    n[g + "_ok"] += sum(yp[k] != "O" for k in range(a, b))
            filas.append({"lang": lang, "seed": seed, "modelo": modelo,
                          "recall_con_oov": n["con_oov_ok"] / n["con_oov_n"] if n["con_oov_n"] else np.nan,
                          "recall_sin_oov": n["sin_oov_ok"] / n["sin_oov_n"] if n["sin_oov_n"] else np.nan,
                          "tokens_con_oov": n["con_oov_n"], "tokens_sin_oov": n["sin_oov_n"]})

df = pd.DataFrame(filas)
salida = Path(__file__).resolve().parent.parent / "results" / "17_bilstm_char_oov.csv"
df.to_csv(salida, index=False)
print(df.groupby(["lang", "modelo"])[["recall_con_oov", "recall_sin_oov"]].agg(["mean", "count"]).round(3))
print("Saved to", salida)
