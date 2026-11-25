"""
nested_credit.py
────────────────
Relaxed evaluation of under-extensions (§5.2 of the paper). Since the annotation keeps only the
longest term, an under-extension of a reserved term may still propose a valid nested term
("corpus" inside "corpus linguistics"). From results/09_apariciones_reservadas.csv.gz, for every
run (model, corpus, level, draw):

  prop_valid       share of under-extensions whose proposed forms are all annotated terms
  gain_all         points of exact recall of reserved units gained by crediting them
  prop_seen        share of the credited nested terms whose glossary entry was seen in training
  gain_reserved    points gained if only nested terms of reserved entries are credited

Requires the token files (DATOS_GENERADOS) to map every annotated form to its glossary entry.

Usage:  python nested_credit.py
Output: results/27_nested_credit.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("DATOS_GENERADOS", str(HERE.parent / "datos_generados"))
sys.path.insert(0, str(HERE))
import train_bilstm_crf as B  # noqa: E402
from metrics import spans_termino  # noqa: E402

RESULTS = HERE.parent / "results"


def formas_por_entrada(lang):
    """Every annotated form of the corpus (lowercased) -> glossary entries it realises."""
    oraciones, _ = B.cargar(lang)
    mapa = {}
    for o in oraciones:
        for a, b in spans_termino(o["labels"]):
            mapa.setdefault(" ".join(o["tokens"][a:b]), set()).add(o["entradas"][a])
    return mapa


def main():
    ap = pd.read_csv(RESULTS / "09_apariciones_reservadas.csv.gz", low_memory=False,
                     dtype={"forma_predicha": str}, keep_default_na=False)
    ap = ap[ap.lang.isin(["en", "es"])]
    part = pd.read_csv(HERE.parent / "data" / "partitions" / "reserved_and_validation_entries.csv")
    part = part[part.role.str.startswith("test")]
    reservadas = {k: set(g.glossary_entry) for k, g in part.groupby(["corpus", "reserved_pct", "seed"])}
    filas = []
    for lang in ("en", "es"):
        mapa = formas_por_entrada(lang)
        for (modelo, pct, seed), g in ap[ap.lang == lang].groupby(["modelo", "pct", "seed"]):
            res = reservadas[(lang, pct, seed)]
            n = len(g)
            defecto = g[g.resultado == "por_defecto"]
            validas = vistas = reservadas_ok = 0
            for forma in defecto.forma_predicha:
                partes = [p.strip() for p in str(forma).split("|")]
                if not all(p in mapa for p in partes):
                    continue
                validas += 1
                entradas = set().union(*(mapa[p] for p in partes))
                if entradas & res:
                    reservadas_ok += 1
                else:
                    vistas += 1
            filas.append({"modelo": modelo, "lang": lang, "pct": pct, "seed": seed, "n_reserved": n,
                          "n_under": len(defecto), "prop_valid": validas / len(defecto) if len(defecto) else float("nan"),
                          "gain_all": 100 * validas / n, "prop_seen": vistas / validas if validas else float("nan"),
                          "gain_reserved": 100 * reservadas_ok / n})
    d = pd.DataFrame(filas)
    d.to_csv(RESULTS / "27_nested_credit.csv", index=False)
    print(d.groupby(["lang", "modelo", "pct"])[["prop_valid", "gain_all", "prop_seen", "gain_reserved"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
