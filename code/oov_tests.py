"""
oov_tests.py
────────────
Statistical tests on reserved terms with unknown words (§5.2 of the paper), from
results/07_analisis_por_ejecucion.csv:

  1. loss of reserved recall when a term contains a word absent from training
     (recall without OOV minus recall with OOV), per model: paired Wilcoxon test over the
     40 runs (4 levels x 10 draws), and the difference in that loss between each neural
     model and the CRF on the same runs;
  2. Spearman correlation, per model and language, between the share of reserved
     occurrences with an unknown word and reserved recall. Both variables are centred on
     the mean of their level, so that the correlation is measured within each level.

Output: results/22_oov_tests.csv
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RESULTS = Path(__file__).resolve().parent.parent / "results"
SHARE, RECALL = "prop_apariciones_reservadas_con_oov", "res_recall_token_todos"


def main():
    d = pd.read_csv(RESULTS / "07_analisis_por_ejecucion.csv", low_memory=False)
    d = d[d.lang.isin(["en", "es"])].copy()
    d["perdida"] = d.res_recall_token_sin_oov - d.res_recall_token_con_oov

    filas = []
    for (lang, modelo), g in d.groupby(["lang", "modelo"]):
        filas.append({"test": "wilcoxon_loss", "lang": lang, "comparison": modelo, "n": len(g),
                      "statistic": g.perdida.mean(), "p": wilcoxon(g.perdida).pvalue})
    p = d.pivot_table(index=["lang", "pct", "seed"], columns="modelo", values="perdida")
    for lang in ("en", "es"):
        for modelo in ("BiLSTM-CRF", "XLM-R"):
            x = (p.loc[lang][modelo] - p.loc[lang]["CRF"]).dropna()
            filas.append({"test": "wilcoxon_loss", "lang": lang, "comparison": f"{modelo} - CRF",
                          "n": len(x), "statistic": x.mean(), "p": wilcoxon(x).pvalue})

    for (lang, modelo), g in d.groupby(["lang", "modelo"]):
        x = g[SHARE] - g.groupby("pct")[SHARE].transform("mean")
        y = g[RECALL] - g.groupby("pct")[RECALL].transform("mean")
        rho, pv = spearmanr(x, y)
        filas.append({"test": "spearman_oov_share", "lang": lang, "comparison": modelo, "n": len(g),
                      "statistic": rho, "p": pv})

    r = pd.DataFrame(filas)
    r.to_csv(RESULTS / "22_oov_tests.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
