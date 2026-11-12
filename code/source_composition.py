"""
source_composition.py
───────────────────
Do the CRF results depend on the uneven composition of the corpus?
Breaks down the stored predictions (type reservation, 10 seeds, 4 levels) by groups of
documents: text type, dominant source (EN: ICAME; ES: 2022), the five longest documents
against the rest, and the mean over documents (macro).

Output: results/15_fuentes_desglose.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[2]
sys.path.insert(0, str(REPO / "colab_reanotacion" / "scripts"))
from metrics import metricas_completas  # noqa: E402

CRF = REPO / "colab_reanotacion" / "resultados" / "crf"
FUENTES = REPO.parent / "tesis-doctoral-datos" / "cap3_corpus"
SALIDA = AQUI.parent / "results" / "15_fuentes_desglose.csv"
TIPOS = {"artículo": "article", "capítulo de libro": "chapter", "libro": "book"}


def grupos_documentos(lang):
    d = pd.read_csv(FUENTES / f"documentos_{lang}.csv")
    d["doc_id"] = lang.upper() + "::" + d["documento"]
    top5 = set(d.nlargest(5, "tokens").doc_id)
    g = {}
    for _, r in d.iterrows():
        dom = (r["fuente"] == "ICAME") if lang == "en" else (r["año"] == 2022)
        g[r.doc_id] = {"tipo": TIPOS[r["tipo"]],
                       "dominante": ("ICAME" if lang == "en" else "2022") if dom else "rest",
                       "tamaño": "top5" if r.doc_id in top5 else "rest"}
    return g


def medir(sub, reservadas):
    m = metricas_completas(list(sub.gold), list(sub.pred))
    res = sub[sub.entrada.isin(reservadas) & (sub.gold != "O")]
    r_res = (res.pred != "O").mean() if len(res) else float("nan")
    return {"f1_termino": m["f1_termino"], "r_termino": m["r_termino"],
            "res_recall_token": r_res, "tokens": len(sub), "res_tokens": len(res)}


def main():
    filas = []
    for lang in ("en", "es"):
        g = grupos_documentos(lang)
        for pct in (10, 20, 30, 50):
            for seed in range(10):
                rid = f"reserva_{lang}_p{pct}_s{seed}_todas"
                p = pd.read_csv(CRF / "pred" / f"{rid}.csv.gz", keep_default_na=False, na_values=[""])
                reservadas = set(json.loads((CRF / "reservadas" / f"{rid}.json").read_text()))
                sin = set(p.doc_id) - set(g)
                if sin:
                    print("WARNING, documents without a group:", list(sin)[:3])
                base = {"lang": lang, "pct": pct, "seed": seed}
                filas.append({**base, "criterio": "all", "grupo": "all", **medir(p, reservadas)})
                for crit in ("tipo", "dominante", "tamaño"):
                    p["_g"] = p.doc_id.map(lambda x: g[x][crit])
                    for grupo, sub in p.groupby("_g"):
                        filas.append({**base, "criterio": crit, "grupo": grupo, **medir(sub, reservadas)})
                # mean over documents (every document weighs the same)
                por_doc = [medir(sub, reservadas) for _, sub in p.groupby("doc_id") if (sub.gold != "O").any()]
                pdoc = pd.DataFrame(por_doc)
                filas.append({**base, "criterio": "macro_doc", "grupo": "mean_doc",
                              "f1_termino": pdoc.f1_termino.mean(), "r_termino": pdoc.r_termino.mean(),
                              "res_recall_token": pdoc.res_recall_token.mean(), "tokens": len(p),
                              "res_tokens": len(pdoc)})
            print(lang, pct, "done", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA, index=False)
    res = (df.groupby(["lang", "criterio", "grupo", "pct"])[["f1_termino", "res_recall_token"]]
             .agg(["mean", "std"]).round(3))
    print(res.to_string())


if __name__ == "__main__":
    main()
