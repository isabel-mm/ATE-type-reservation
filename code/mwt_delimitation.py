"""
mwt_delimitation.py
───────────────────
Exact delimitation of single-word and multi-word terms (§5.2 of the paper): share of the gold
term spans of the test set (seen and reserved) that a model predicts with exactly the same
boundaries, at the reference level of 20% reservation (ten draws). The sequential models are
read from their stored predictions; the token classifiers already store the same measure
(r_span_exacto_mono / r_span_exacto_poli, computed by train_token_classifiers.py).

Usage:  python mwt_delimitation.py RESULTS_FOLDER
Output: results/26_mwt_delimitation.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import spans_termino  # noqa: E402

RES = Path(sys.argv[1])
SALIDA = Path(__file__).resolve().parent.parent / "results" / "26_mwt_delimitation.csv"
MODELOS = {"CRF": RES / "crf" / "pred", "BiLSTM-CRF": RES / "bilstm" / "pred",
           "XLM-R": RES / "transformer" / "xlm-roberta-base" / "pred"}


def por_longitud(p):
    tot = {"mono": 0, "poli": 0}
    ok = {"mono": 0, "poli": 0}
    for _, s in p.groupby(["doc_id", "sentence_id"], sort=False):
        s = s.sort_values("i")
        pred = set(spans_termino(list(s.pred)))
        for a, b in spans_termino(list(s.gold)):
            k = "mono" if b - a == 1 else "poli"
            tot[k] += 1
            ok[k] += (a, b) in pred
    return {f"r_span_exacto_{k}": ok[k] / tot[k] for k in tot}


def main():
    filas = []
    for modelo, carpeta in MODELOS.items():
        for lang in ("en", "es"):
            for seed in range(10):
                f = carpeta / f"reserva_{lang}_p20_s{seed}.csv.gz"
                if not f.exists():
                    f = carpeta / f"reserva_{lang}_p20_s{seed}_todas.csv.gz"   # CRF with all features
                p = pd.read_csv(f, usecols=["doc_id", "sentence_id", "i", "gold", "pred"],
                                keep_default_na=False, na_values=[""])
                filas.append({"modelo": modelo, "lang": lang, "pct": 20, "seed": seed, **por_longitud(p)})
    clas = pd.read_csv(RES / "clasificadores" / "clasificadores_todas.csv")
    clas = clas[(clas.exp == "reserva") & (clas.pct == 20)]
    for r in clas.itertuples(index=False):
        filas.append({"modelo": r.clasificador, "lang": r.lang, "pct": 20, "seed": int(r.seed),
                      "r_span_exacto_mono": r.r_span_exacto_mono, "r_span_exacto_poli": r.r_span_exacto_poli})
    d = pd.DataFrame(filas)
    d.to_csv(SALIDA, index=False)
    print(d.groupby(["lang", "modelo"])[["r_span_exacto_mono", "r_span_exacto_poli"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
