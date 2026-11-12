"""
Unknown words seen through XLM-R's subwords.

A word absent from training may be made of subwords that XLM-R did see during
fine-tuning (key + ness). For every reserved occurrence containing an unknown word, we
compute the share of the subwords of those words that occur in training, and compare the
token recall of XLM-R (and of the CRF and the BiLSTM-CRF for reference) by that coverage.
All levels and seeds.

Usage: python oov_subword_analysis.py RESULTS_FOLDER
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

os.environ.setdefault("DATOS_GENERADOS", str(Path(__file__).resolve().parent.parent / "datos_generados"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_bilstm_crf as B  # noqa: E402

RES = Path(sys.argv[1])
MODELOS = {"CRF": RES / "crf" / "pred", "BiLSTM-CRF": RES / "bilstm" / "pred",
           "XLM-R": RES / "transformer" / "xlm-roberta-base" / "pred"}
TOK = AutoTokenizer.from_pretrained("xlm-roberta-base")
_cache = {}


def piezas(palabra):
    if palabra not in _cache:
        _cache[palabra] = TOK.tokenize(palabra)
    return _cache[palabra]


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


def franja(c):
    return "sin_oov" if c is None else ("oov_subpalabras_vistas" if c == 1 else "oov_alguna_subpalabra_nueva")


filas = []
for lang in ("en", "es"):
    oraciones, entradas = B.cargar(lang)
    for pct in (10, 20, 30, 50):
        for seed in range(10):
            train, test, reservadas = B.particion_reserva(oraciones, entradas, pct, seed)
            tr, _ = B.separar_validacion(train, seed, por_entradas=True)
            vocab = {t for o in tr for t in o["tokens"]}
            subvocab = {p for w in vocab for p in piezas(w)}
            apariciones = []   # (sentence, start, end, group)
            for o in test:
                for a, b in spans(o["labels"]):
                    if o["entradas"][a] not in reservadas:
                        continue
                    nuevas = [w for w in o["tokens"][a:b] if w not in vocab]
                    if not nuevas:
                        c = None
                    else:
                        ps = [p for w in nuevas for p in piezas(w)]
                        c = sum(p in subvocab for p in ps) / len(ps)
                    apariciones.append((o, a, b, franja(c)))
            for modelo, carpeta in MODELOS.items():
                f = carpeta / f"reserva_{lang}_p{pct}_s{seed}.csv.gz"
                if not f.exists():
                    f = carpeta / f"reserva_{lang}_p{pct}_s{seed}_todas.csv.gz"
                if not f.exists():
                    continue
                pred = pd.read_csv(f, usecols=["doc_id", "sentence_id", "i", "pred"], keep_default_na=False,
                                   dtype={"doc_id": str, "sentence_id": str})
                pmap = defaultdict(dict)
                for d, s, i, p in pred.itertuples(index=False):
                    pmap[(d, s)][int(i)] = p
                n = defaultdict(int)
                for o, a, b, g in apariciones:
                    pm = pmap.get((str(o["doc_id"]), str(o["sentence_id"])), {})
                    n[g + "_n"] += b - a
                    n[g + "_ok"] += sum(pm.get(k, "O") != "O" for k in range(a, b))
                fila = {"lang": lang, "pct": pct, "seed": seed, "modelo": modelo}
                for g in ("sin_oov", "oov_subpalabras_vistas", "oov_alguna_subpalabra_nueva"):
                    fila[f"tokens_{g}"] = n[g + "_n"]
                    fila[f"recall_{g}"] = n[g + "_ok"] / n[g + "_n"] if n[g + "_n"] else np.nan
                filas.append(fila)
    print(lang, "done", flush=True)

df = pd.DataFrame(filas)
salida = Path(__file__).resolve().parent.parent / "results" / "21_oov_by_subword_coverage.csv"
df.to_csv(salida, index=False)
cols = [c for c in df.columns if c.startswith("recall_") or c.startswith("tokens_")]
print(df.groupby(["lang", "modelo"])[cols].mean().round(3).T.to_string())
print("Saved to", salida)
