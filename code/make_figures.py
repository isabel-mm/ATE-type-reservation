"""
make_figures.py
───────────────
Draws the result figures of the paper from the CSV files in results/ (vector PDF, written
next to this script, or to the folder given as first argument):

  fig_reserva.pdf    term F1 under type reservation, by level
  fig_recall.pdf     recall of seen and reserved terms (B-T / I-T; by token and by type)
  fig_oov.pdf        recall of reserved terms with all words seen vs. some word unseen
  fig_ablacion.pdf   CRF feature ablation at 20% reservation (Δ term F1 and Δ reserved recall)
  fig_acter.pdf      ACTER, leave-one-domain-out

The rank-frequency and saturation figure is not included: it is computed from the source
texts, which cannot be redistributed.

Colours follow the Okabe-Ito palette (safe for colour blindness): CRF blue, BiLSTM-CRF
orange, XLM-R green; English blue and Spanish vermilion.

Usage:  python make_figures.py [OUTPUT_FOLDER]
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)

COL = {"CRF": "#0072B2", "BiLSTM-CRF": "#E69F00", "XLM-R": "#009E73"}
MARK = {"CRF": "o", "BiLSTM-CRF": "s", "XLM-R": "^"}
MODELS = list(COL)
INK, INK2, GRID = "#222222", "#666666", "#DDDDDD"
LANG = {"en": "#0072B2", "es": "#D55E00"}
LEVELS = [10, 20, 30, 50]
COL1 = 3.2  # column width in inches

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  ", name)


def reservation_runs():
    """One row per type-reservation run (EN/ES, no ablation), all models."""
    d = pd.read_csv(RESULTS / "01_ejecuciones_todas.csv", low_memory=False)
    return d[(d.exp == "reserva") & d.familia.isna() & d.lang.isin(["en", "es"]) & d.modelo.isin(MODELS)]


def fig_reserva(runs):
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), sharey=True)
    shift = dict(zip(MODELS, np.linspace(-0.9, 0.9, len(MODELS))))
    for a, lang in zip(ax, ("en", "es")):
        for m in MODELS:
            g = runs[(runs.modelo == m) & (runs.lang == lang)].groupby("pct")["f1_termino"].agg(["mean", "std", "count"])
            a.errorbar(np.array(g.index) + shift[m], g["mean"], yerr=g["std"], color=COL[m], marker=MARK[m], ms=4,
                       lw=1.6, capsize=2, elinewidth=0.8, label=f"{m} (n={int(g['count'].min())})")
        a.set_title(lang.upper()); a.set_xticks(LEVELS); a.set_xlabel("Reserved types (%)"); a.set_ylim(0.35, 0.85)
    ax[0].set_ylabel("Term F$_1$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09),
               handlelength=1.4, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_reserva")


def fig_recall(runs):
    an = pd.read_csv(RESULTS / "07_analisis_por_ejecucion.csv", low_memory=False)
    an = an[an.lang.isin(["en", "es"])]
    fig, ax = plt.subplots(2, 2, figsize=(COL1, 3.1), sharey="row")
    # top: seen terms against reserved B-T and I-T tokens, mean of the four levels
    series = (("vis_r_termino", "Seen", "#56B4E9"), ("res_r_bt", "Reserved B-T", "#D55E00"),
              ("res_r_it", "Reserved I-T", "#CC79A7"))
    w = 0.26
    for a, lang in zip(ax[0], ("en", "es")):
        g = runs[runs.lang == lang].groupby("modelo")[[s[0] for s in series]].mean().reindex(MODELS)
        x = np.arange(len(MODELS))
        for k, (col, label, colour) in enumerate(series):
            a.bar(x + (k - 1) * w, g[col].values, width=w - 0.03, color=colour, label=label, edgecolor="white", lw=0.5)
        a.set_xticks(x); a.set_xticklabels(["CRF", "BiLSTM", "XLM-R"]); a.set_title(lang.upper()); a.set_ylim(0, 1)
    ax[0, 0].set_ylabel("Recall (mean of levels)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03),
               handlelength=0.9, fontsize=6.3, columnspacing=0.8)
    # bottom: reserved terms by token and by type, per level
    for a, lang in zip(ax[1], ("en", "es")):
        for m in MODELS:
            g = an[(an.modelo == m) & (an.lang == lang)].groupby("pct")[["res_recall_token_todos", "tipo_recall_reservados"]].mean()
            a.plot(g.index, g["res_recall_token_todos"], color=COL[m], marker=MARK[m], ms=3.2, lw=1.5)
            a.plot(g.index, g["tipo_recall_reservados"], color=COL[m], marker=MARK[m], ms=3.2, lw=1.5, ls="--",
                   markerfacecolor="white")
        a.set_xticks(LEVELS); a.set_xlabel("Reserved types (%)"); a.set_ylim(0, 0.6)
    ax[1, 0].set_ylabel("Recall of reserved terms")
    handles = [Line2D([], [], color=COL[m], marker=MARK[m], lw=1.5, label=m) for m in MODELS]
    handles += [Line2D([], [], color=INK2, lw=1.3, label="by token"), Line2D([], [], color=INK2, lw=1.3, ls="--", label="by type")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), handlelength=1.4,
               columnspacing=0.6, fontsize=6.3)
    fig.tight_layout(h_pad=1.0, w_pad=0.5, rect=(0, 0.04, 1, 0.96))
    save(fig, "fig_recall")


def fig_oov():
    d = pd.read_csv(RESULTS / "07_analisis_por_ejecucion.csv", low_memory=False)
    d = d[d.lang.isin(["en", "es"])]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), sharey=True)
    w = 0.36
    for a, lang in zip(ax, ("en", "es")):
        g = d[d.lang == lang].groupby("modelo")[["res_recall_token_sin_oov", "res_recall_token_con_oov"]].mean().reindex(MODELS)
        for k, m in enumerate(MODELS):
            a.bar(k - w / 2, g.loc[m, "res_recall_token_sin_oov"], width=w - 0.03, color=COL[m], edgecolor=COL[m], lw=0.8)
            a.bar(k + w / 2, g.loc[m, "res_recall_token_con_oov"], width=w - 0.03, color="white", edgecolor=COL[m],
                  hatch="////", lw=0.8)
        a.set_xticks(range(len(MODELS))); a.set_xticklabels(["CRF", "BiLSTM", "XLM-R"])
        a.set_title(lang.upper()); a.set_ylim(0, 0.6); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("Recall of reserved terms")
    handles = [Patch(facecolor="#777777", edgecolor="#777777", label="all words seen"),
               Patch(facecolor="white", edgecolor="#777777", hatch="////", label="some word OOV")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_oov")


FAMILIES = {"morfologia": "Morphology", "contexto": "Context (±2)", "cohesion_pmi": "Cohesion (PMI)",
            "morfosintaxis": "Part of speech", "sintaxis": "Syntax", "forma": "Form",
            "estadisticos": "Statistics", "contexto_definitorio": "Definitional context", "patron_sigla": "Acronym"}


def fig_ablacion():
    d = pd.read_csv(RESULTS / "05d_pruebas_ablacion.csv")
    d = d[(d.protocolo == "reserva") & (d.pct == 20)]
    order = list(FAMILIES)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.3), sharey=True)
    for a, (measure, title) in zip(ax, (("delta_f1_termino", "$\\Delta$ term F$_1$"),
                                        ("delta_recall_reservadas", "$\\Delta$ reserved recall"))):
        for lang, dy, marker in (("en", 0.15, "o"), ("es", -0.15, "s")):
            g = d[(d.lang == lang) & (d.medida == measure)].set_index("familia").loc[order]
            ci = 1.96 * g["dt"] / np.sqrt(g["n"])
            a.errorbar(g["media"], np.arange(len(order)) + dy, xerr=ci, fmt=marker, color=LANG[lang], ms=3.5,
                       elinewidth=0.9, capsize=1.5, label=lang.upper())
        a.axvline(0, color=INK2, lw=0.7)
        a.set_title(title); a.set_yticks(np.arange(len(order))); a.set_yticklabels([FAMILIES[f] for f in order])
        a.grid(axis="y", visible=False)
    ax[0].legend(loc="lower left", handlelength=0.8)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_ablacion")


DOMAINS = {"corp": "Corruption", "equi": "Dressage", "htfl": "Heart\nfailure", "wind": "Wind\nenergy", "MEDIA": "Mean"}


def fig_acter():
    d = pd.read_csv(RESULTS / "04_resumen_lodo.csv").set_index(["modelo", "dominio"])["f1_termino_mean"]
    fig, a = plt.subplots(figsize=(COL1, 1.9))
    w = 0.26
    x = np.arange(len(DOMAINS))
    for k, m in enumerate(MODELS):
        a.bar(x + (k - 1) * w, [d.get((m, dom), np.nan) for dom in DOMAINS], width=w - 0.03, color=COL[m],
              label=m, edgecolor="white", lw=0.5)
    a.set_xticks(x); a.set_xticklabels(list(DOMAINS.values()), fontsize=6.5)
    a.set_ylabel("Term F$_1$ (unseen domain)"); a.set_ylim(0, 0.7); a.grid(axis="x", visible=False)
    a.legend(loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.8, fontsize=6.5)
    save(fig, "fig_acter")


if __name__ == "__main__":
    runs = reservation_runs()
    fig_reserva(runs)
    fig_recall(runs)
    fig_oov()
    fig_ablacion()
    fig_acter()
