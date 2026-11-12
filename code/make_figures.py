"""
Figuras del artículo (PDF vectorial, en figuras/).

Fuentes:
  tesis/reanotacion_2026-10/                zipf_*.csv, saturacion_*.csv, zipf_exponentes.csv
  colab_reanotacion/resultados/crf/         crf_reserva_todas.csv, crf_ablacion_resumen.csv
  colab_reanotacion/resultados/bilstm/      bilstm_reserva_todas.csv
  colab_reanotacion/resultados/transformer/xlm-roberta-base/transformer_reserva_todas.csv (si existe)
  Tablas de la tesis (cap. 6) para palabras desconocidas y errores (tres particiones comunes).

Paleta Okabe-Ito (segura para daltonismo): CRF azul, BiLSTM-CRF naranja, XLM-R verde;
EN azul y ES bermellón; reservado/error en bermellón.
Uso: python generar_figuras.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[3]
# Datos CON bibliografías: son los mismos con los que se entrenan y evalúan los modelos
# (tesis/reanotacion_2026-10 se regeneró el 7-nov sin bibliografías; la versión anterior está aquí).
DIST = RAIZ / "tesis" / "sin_bibliografia_2026-11" / "material_complementario" / "cap5_distribucion"   # mismas cifras que el cap. 5 de la tesis
RES = RAIZ / "colab_reanotacion" / "resultados"   # ablación del CRF (ejecutada en el Mac)
CSV_FINAL = AQUI.parent / "resultados_2026-11-11" / "csv"   # exportación con todos los resultados (all runs)

COL = {"CRF": "#0072B2", "BiLSTM-CRF": "#E69F00", "XLM-R": "#009E73"}  # paleta Okabe-Ito
MARK = {"CRF": "o", "BiLSTM-CRF": "s", "XLM-R": "^"}
TINTA, TINTA2, REJILLA = "#222222", "#666666", "#DDDDDD"
LENG = {"en": "#0072B2", "es": "#D55E00"}  # EN azul, ES bermellón (Okabe-Ito), en todas las figuras
NIVELES = [10, 20, 30, 50]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.labelcolor": TINTA, "text.color": TINTA, "axes.grid": True,
    "grid.color": REJILLA, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL1, COL2 = 3.2, 6.6  # ancho de columna y de página (pulgadas)


def guardar(fig, nombre):
    fig.savefig(AQUI / f"{nombre}.pdf")
    plt.close(fig)
    print("  ", nombre)


def resultados_modelos():
    """Una fila por ejecución de reserva (EN/ES, sin ablación), todos los modelos."""
    d = pd.read_csv(CSV_FINAL / "01_ejecuciones_todas.csv", low_memory=False)
    return d[(d.exp == "reserva") & d.familia.isna() & d.lang.isin(["en", "es"])]


# ── Figura 1: Zipf y saturación ─────────────────────────────────────
def fig_distribucion():
    exps = pd.read_csv(DIST / "zipf_exponentes.csv").set_index("Lang")
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.85))
    for i, (lang, tit) in enumerate((("en", "EN"), ("es", "ES"))):
        gen = pd.read_csv(DIST / f"zipf_general_{lang}.csv")
        ter = pd.read_csv(DIST / f"zipf_termino_{lang}.csv")
        a = ax[i]
        a.loglog(gen["rango"], gen["frecuencia"] / gen["frecuencia"].sum(), color="#AAAAAA", lw=1.5,
                 label=f"All words ($\\alpha$={exps.loc[tit, 'Exponente_general']:.2f})")
        a.loglog(ter["rango"], ter["frecuencia"] / ter["frecuencia"].sum(), color=LENG[lang], lw=0, marker="o",
                 ms=2.6, label=f"Term types ($\\alpha$={exps.loc[tit, 'Exponente_termino']:.2f})")
        a.set_title(f"Rank–frequency ({tit})")
        a.set_xlabel("Rank")
        a.set_ylabel("Relative frequency" if i == 0 else "")
        a.legend(loc="upper right", handlelength=1.2)
    a = ax[2]
    for lang, tit, ls in (("en", "EN", "-"), ("es", "ES", "--")):
        s = pd.read_csv(DIST / f"saturacion_{lang}.csv")
        a.fill_between(s["pct_tokens"], s["pct_tipos_p05"], s["pct_tipos_p95"], color=LENG[lang], alpha=0.15, lw=0)
        a.plot(s["pct_tokens"], s["pct_tipos_media"], color=LENG[lang], ls=ls, lw=1.6, label=tit)
    a.axhline(80, color=TINTA2, lw=0.6, ls=":")
    a.text(98, 76, "80%", ha="right", va="top", color=TINTA2, fontsize=6.5)
    a.set_xlim(0, 100); a.set_ylim(0, 102)
    a.set_title("Terminological saturation")
    a.set_xlabel("% of corpus tokens")
    a.set_ylabel("% of term types observed")
    a.legend(loc="lower right")
    fig.tight_layout(w_pad=1.2)
    guardar(fig, "fig_distribucion")


# ── Figura 2: F1 de término por nivel ───────────────────────────────
def fig_reserva(df):
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), sharey=True)
    modelos = [m for m in COL if m in df.modelo.unique()]
    desp = dict(zip(modelos, np.linspace(-0.9, 0.9, len(modelos))))
    for i, lang in enumerate(("en", "es")):
        a = ax[i]
        for m in modelos:
            g = df[(df.modelo == m) & (df.lang == lang)].groupby("pct")["f1_termino"].agg(["mean", "std", "count"])
            x = np.array(g.index) + desp[m]
            a.errorbar(x, g["mean"], yerr=g["std"], color=COL[m], marker=MARK[m], ms=4, lw=1.6,
                       capsize=2, elinewidth=0.8, label=f"{m} (n={int(g['count'].min())})")
        a.set_title(lang.upper())
        a.set_xticks(NIVELES)
        a.set_xlabel("Reserved types (%)")
        a.set_ylim(0.35, 0.85)
    ax[0].set_ylabel("Term F$_1$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09),
               handlelength=1.4, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "fig_reserva")


# ── Figura 3: recuperación de vistos y reservados (CRF) ─────────────
def fig_recall_tipo(df):
    c = df[df.modelo == "CRF"]
    series = (("vis_r_termino", "Seen terms", "#0072B2"),
              ("res_r_bt", "Reserved, B-T", "#D55E00"),
              ("res_r_it", "Reserved, I-T", "#CC79A7"))
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.9), sharey=True)
    w = 0.26
    for i, lang in enumerate(("en", "es")):
        a = ax[i]
        g = c[c.lang == lang].groupby("pct")[[s[0] for s in series]].mean()
        x = np.arange(len(NIVELES))
        for k, (col, lab, color) in enumerate(series):
            a.bar(x + (k - 1) * w, g[col].values, width=w - 0.03, color=color, label=lab, edgecolor="white", lw=0.5)
        a.set_xticks(x); a.set_xticklabels([f"{n}%" for n in NIVELES])
        a.set_title(lang.upper()); a.set_xlabel("Reserved types")
        a.set_ylim(0, 1)
    ax[0].set_ylabel("Recall (CRF)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "fig_recall_tipo")


# ── Figura 4: palabras fuera de vocabulario (tabla de la tesis, 3 particiones) ──
OOV = {  # (CRF conocidas, BiLSTM conocidas, CRF con OOV, BiLSTM con OOV)
    "en": [(.220, .454, .156, .006), (.288, .408, .143, .007), (.263, .286, .238, .003), (.220, .354, .106, .022)],
    "es": [(.451, .419, .030, .014), (.466, .519, .158, .011), (.384, .343, .089, .006), (.327, .281, .341, .319)],
}


def fig_oov():
    """Recuperación de reservados con todas las palabras vistas frente a alguna desconocida (media de los niveles)."""
    d = pd.read_csv(CSV_FINAL / "07_analisis_por_ejecucion.csv", low_memory=False)
    d = d[d.lang.isin(["en", "es"])]
    mods = ["CRF", "BiLSTM-CRF", "XLM-R"]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), sharey=True)
    w = 0.36
    for i, lang in enumerate(("en", "es")):
        a = ax[i]
        g = d[d.lang == lang].groupby("modelo")[["res_recall_token_sin_oov", "res_recall_token_con_oov"]].mean().reindex(mods)
        x = np.arange(len(mods))
        for k, m in enumerate(mods):
            a.bar(x[k] - w / 2, g.loc[m, "res_recall_token_sin_oov"], width=w - 0.03, color=COL[m], edgecolor=COL[m], lw=0.8)
            a.bar(x[k] + w / 2, g.loc[m, "res_recall_token_con_oov"], width=w - 0.03, color="white", edgecolor=COL[m],
                  hatch="////", lw=0.8)
        a.set_xticks(x); a.set_xticklabels(["CRF", "BiLSTM", "XLM-R"])
        a.set_title(lang.upper()); a.set_ylim(0, 0.6); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("Recall of reserved terms")
    from matplotlib.patches import Patch
    h = [Patch(facecolor="#777777", edgecolor="#777777", label="all words seen"),
         Patch(facecolor="white", edgecolor="#777777", hatch="////", label="some word OOV")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "fig_oov")


# ── Figura 5: ablación (IC 95 %, diez particiones, reserva del 20 %) ─
FAM = {"morfologia": "Morphology", "contexto": "Context (±2)", "cohesion_pmi": "Cohesion (PMI)",
       "morfosintaxis": "Part of speech", "sintaxis": "Syntax", "forma": "Form",
       "estadisticos": "Statistics", "contexto_definitorio": "Definitional context", "patron_sigla": "Acronym"}


def fig_ablacion():
    d = pd.read_csv(RES / "crf" / "crf_ablacion_resumen.csv")
    d = d[(d.pct == 20) & (d.familia != "BASELINE")]
    orden = list(FAM)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.3), sharey=True)
    for i, (col, tit) in enumerate((("delta_f1_termino", "$\\Delta$ term F$_1$"),
                                     ("delta_recall_reservadas", "$\\Delta$ reserved recall"))):
        a = ax[i]
        for lang, dy, mk, color in (("en", 0.15, "o", LENG["en"]), ("es", -0.15, "s", LENG["es"])):
            g = d[d.lang == lang].set_index("familia").loc[orden]
            ci = 1.96 * g[f"{col}_std"] / np.sqrt(g[f"{col}_count"])
            y = np.arange(len(orden)) + dy
            a.errorbar(g[f"{col}_mean"], y, xerr=ci, fmt=mk, color=color, ms=3.5, elinewidth=0.9,
                       capsize=1.5, label=lang.upper())
        a.axvline(0, color=TINTA2, lw=0.7)
        a.set_title(tit)
        a.set_yticks(np.arange(len(orden))); a.set_yticklabels([FAM[f] for f in orden])
        a.grid(axis="y", visible=False)
    ax[0].legend(loc="lower left", handlelength=0.8)
    fig.tight_layout(w_pad=0.5)
    guardar(fig, "fig_ablacion")


# ── Figura 6: tipología de errores (tabla de la tesis) ──────────────
ERR_TIPOS = ["Missed SWT", "Missed MWT", "Under-extension", "Over-extension", "Partial overlap", "False positive"]
ERR_COLS = ["err_pct_omision_mono", "err_pct_omision_poli", "err_pct_por_defecto", "err_pct_por_exceso",
            "err_pct_solapamiento", "err_pct_falso_positivo"]


def errores_finales():
    """% de los errores de cada modelo sobre términos reservados; media de los cuatro niveles y de las particiones."""
    d = pd.read_csv(CSV_FINAL / "07_analisis_por_ejecucion.csv", low_memory=False)
    d = d[d.lang.isin(["en", "es"])]
    g = d.groupby(["lang", "modelo"])[ERR_COLS].mean()
    out = {}
    for lang in ("en", "es"):
        for m in ("CRF", "BiLSTM-CRF", "XLM-R"):
            v = g.loc[(lang, m)].values
            out[(lang.upper(), m)] = list(100 * v / v.sum()) if v.sum() > 1.5 else list(v / v.sum() * 100)
    return out


# omisiones (corales), errores de límite (berenjenas), falsos positivos (gris)
ERR_COL = ["#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#CC79A7", "#999999"]


def fig_errores():
    ERR = errores_finales()
    fig, a = plt.subplots(figsize=(COL1, 2.0))
    filas = list(ERR)
    for r, clave in enumerate(filas):
        izq = 0
        for v, color, lab in zip(ERR[clave], ERR_COL, ERR_TIPOS):
            a.barh(r, v, left=izq, color=color, edgecolor="white", lw=0.8, height=0.7, label=lab if r == 0 else None)
            if v >= 8:
                a.text(izq + v / 2, r, f"{v:.0f}", ha="center", va="center", fontsize=6,
                       color="white" if color in (ERR_COL[0], ERR_COL[2], ERR_COL[5]) else TINTA)
            izq += v
    a.set_yticks(range(len(filas))); a.set_yticklabels([f"{l} {m}" for l, m in filas])
    a.invert_yaxis(); a.set_xlim(0, 100); a.set_xlabel("% of errors")
    a.grid(axis="y", visible=False)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=1.0, columnspacing=0.6, fontsize=6.5)
    guardar(fig, "fig_errores")


# ── Figura nueva: validación cruzada frente a reserva (pendientes) ──
NOMBRES_CLAS = {"Árbol de decisión": "Decision tree", "Regresión logística": "Logistic regression",
                "SVM lineal": "Linear SVM", "Random Forest": "Random forest", "Naive Bayes": "Naive Bayes"}


def gap_final():
    """F1 de término: CV y reserva (clasificadores al 20 %, modelos secuenciales al 10 %)."""
    d = pd.read_csv(CSV_FINAL / "01_ejecuciones_todas.csv", low_memory=False)
    d = d[d.lang.isin(["en", "es"]) & d.familia.isna()]
    d["modelo"] = d.modelo.replace(NOMBRES_CLAS)
    # Clasificadores: su CSV propio (las ejecuciones nuevas, «_p20_s» y «_f», sustituyen a las antiguas)
    c = pd.read_csv(RES / "clasificadores" / "clasificadores_todas.csv")
    nuevo = c.rid.str.contains("_p20_s|_f[0-9]$", regex=True)
    c = pd.concat([g[nuevo[g.index]] if nuevo[g.index].any() else g
                   for _, g in c.groupby(["lang", "exp"])])
    c = c.assign(modelo=c.clasificador.replace(NOMBRES_CLAS), pct=20, familia=np.nan)
    d = pd.concat([d[~d.modelo.isin(NOMBRES_CLAS.values())], c[["modelo", "exp", "lang", "pct", "familia", "f1_termino"]]])
    gap = {}
    for m in list(NOMBRES_CLAS.values()) + ["CRF", "BiLSTM-CRF", "XLM-R"]:
        nivel = 10 if m in COL else 20
        par = {}
        for l in ("en", "es"):
            cv = d[(d.modelo == m) & (d.lang == l) & (d.exp == "cv")].f1_termino.mean()
            rs = d[(d.modelo == m) & (d.lang == l) & (d.exp == "reserva") & (d.pct == nivel)].f1_termino.mean()
            par[l] = (cv, rs)
        if m == "XLM-R" and not np.isfinite(par["en"][0]):   # CV de XLM-R: registro de Kaggle (10-nov)
            par = {"en": (0.993, par["en"][1]), "es": (0.984, par["es"][1])}
        if all(np.isfinite(v).all() for v in par.values()):
            gap[m] = par
    return gap


def fig_cv_gap():
    GAP = gap_final()
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.2), sharey=True)
    for i, lang in enumerate(("en", "es")):
        a = ax[i]
        for m, v in GAP.items():
            cv, rs = v[lang]
            color = COL.get(m, "#B0B0B0")
            fuerte = m in COL
            a.plot([0, 1], [cv, rs], color=color, lw=2.0 if fuerte else 0.9, marker=MARK.get(m, "o"),
                   ms=4 if fuerte else 2.5, zorder=3 if fuerte else 2, label=m if fuerte else None)
            if fuerte:
                a.text(1.06, rs, f"{rs:.2f}", va="center", fontsize=6, color=color)
                a.text(-0.06, cv, f"{cv:.2f}", va="center", ha="right", fontsize=6, color=color)
        a.set_xticks([0, 1]); a.set_xticklabels(["Random CV", "Type\nreservation"])
        a.set_xlim(-0.35, 1.35); a.set_title(lang.upper()); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("Term F$_1$")
    ax[0].plot([], [], color="#B0B0B0", lw=0.9, label="Token classifiers")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08),
               handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(w_pad=0.8)
    guardar(fig, "fig_cv_gap")


# ── Figura nueva: recuperación de reservados por token y por tipo ──
CSV_ANALISIS = Path(__file__).resolve().parent.parent / "resultados_provisionales" / "csv" / "07_analisis_por_ejecucion.csv"


def fig_tipo_token(ruta=CSV_ANALISIS):
    if not ruta.exists():
        print("   (sin 07_analisis_por_ejecucion.csv: se omite fig_tipo_token)")
        return
    d = pd.read_csv(ruta)
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.0), sharey=True)
    for i, lang in enumerate(("en", "es")):
        a = ax[i]
        for m in [m for m in COL if m in set(d.modelo)]:
            g = d[(d.modelo == m) & (d.lang == lang)].groupby("pct")[["res_recall_token_todos", "tipo_recall_reservados"]].mean()
            a.plot(g.index, g["res_recall_token_todos"], color=COL[m], marker=MARK[m], ms=3.5, lw=1.6, ls="-")
            a.plot(g.index, g["tipo_recall_reservados"], color=COL[m], marker=MARK[m], ms=3.5, lw=1.6, ls="--",
                   markerfacecolor="white")
        a.set_title(lang.upper()); a.set_xticks(NIVELES); a.set_xlabel("Reserved types (%)")
        a.set_ylim(0, 0.6)
    ax[0].set_ylabel("Recall of reserved terms")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[m], marker=MARK[m], lw=1.6, label=m) for m in COL if m in set(d.modelo)]
    h += [Line2D([], [], color=TINTA2, lw=1.4, ls="-", label="by token"),
          Line2D([], [], color=TINTA2, lw=1.4, ls="--", label="by type")]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "fig_tipo_token")


# ── Figura de recuperación (una columna): datos completos del PC del trabajo ──
MODELOS = ["CRF", "BiLSTM-CRF", "XLM-R"]
from matplotlib.lines import Line2D  # noqa: E402


def fig_recall(CSV=CSV_FINAL):
    """Vistos frente a reservados (B-T, I-T) y reservados por token frente a por tipo, tres modelos."""
    runs = pd.read_csv(CSV / "01_ejecuciones_todas.csv", low_memory=False)
    runs = runs[(runs.exp == "reserva") & runs.familia.isna() & runs.lang.isin(["en", "es"])]
    an = pd.read_csv(CSV / "07_analisis_por_ejecucion.csv", low_memory=False)
    an = an[an.lang.isin(["en", "es"])]
    fig, ax = plt.subplots(2, 2, figsize=(COL1, 3.1), sharey="row")
    # (a) vistos frente a reservados (B-T, I-T), media de los cuatro niveles, por modelo
    series = (("vis_r_termino", "Seen", "#56B4E9"), ("res_r_bt", "Reserved B-T", "#D55E00"),
              ("res_r_it", "Reserved I-T", "#CC79A7"))
    w = 0.26
    for i, lang in enumerate(("en", "es")):
        a = ax[0, i]
        g = runs[runs.lang == lang].groupby("modelo")[[s[0] for s in series]].mean().reindex(MODELOS)
        x = np.arange(len(MODELOS))
        for k, (col, lab, color) in enumerate(series):
            a.bar(x + (k - 1) * w, g[col].values, width=w - 0.03, color=color, label=lab, edgecolor="white", lw=0.5)
        a.set_xticks(x); a.set_xticklabels(["CRF", "BiLSTM", "XLM-R"])
        a.set_title(lang.upper())
        a.set_ylim(0, 1)
    ax[0, 0].set_ylabel("Recall (mean of levels)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03),
               handlelength=0.9, fontsize=6.3, columnspacing=0.8)
    # (b) reservados por token y por tipo, por nivel
    for i, lang in enumerate(("en", "es")):
        a = ax[1, i]
        for m in MODELOS:
            g = an[(an.modelo == m) & (an.lang == lang)].groupby("pct")[["res_recall_token_todos", "tipo_recall_reservados"]].mean()
            a.plot(g.index, g["res_recall_token_todos"], color=COL[m], marker=MARK[m], ms=3.2, lw=1.5)
            a.plot(g.index, g["tipo_recall_reservados"], color=COL[m], marker=MARK[m], ms=3.2, lw=1.5, ls="--",
                   markerfacecolor="white")
        pass
        a.set_xticks(NIVELES); a.set_xlabel("Reserved types (%)"); a.set_ylim(0, 0.6)
    ax[1, 0].set_ylabel("Recall of reserved terms")
    h = [Line2D([], [], color=COL[m], marker=MARK[m], lw=1.5, label=m) for m in MODELOS]
    h += [Line2D([], [], color=TINTA2, lw=1.3, label="by token"), Line2D([], [], color=TINTA2, lw=1.3, ls="--", label="by type")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), handlelength=1.4,
               columnspacing=0.6, fontsize=6.3)
    fig.tight_layout(h_pad=1.0, w_pad=0.5, rect=(0, 0.04, 1, 0.96))
    guardar(fig, "fig_recall")


# ── Figura nueva: ACTER, transferencia a un dominio no visto (LODO) ──
LODO_CRF = {"corp": 0.374, "equi": 0.478, "htfl": 0.137, "wind": 0.528}  # comprobado: reproduce la tesis
DOMINIOS = {"corp": "Corruption", "equi": "Dressage", "htfl": "Heart\nfailure", "wind": "Wind\nenergy"}


# XLM-R: archivos de la sesión 2 de Kaggle (11-nov); BiLSTM-CRF: registro (determinista, idéntico en las dos sesiones)
LODO_PROV = {"XLM-R": {"corp": 0.036, "equi": 0.427, "htfl": 0.539, "wind": 0.490},   # lodo_todas.csv, Kaggle sesión 2 (11-nov)
             "BiLSTM-CRF": {"corp": 0.097, "equi": 0.169, "htfl": 0.103, "wind": 0.105}}


def fig_acter():
    vals = {"CRF": dict(LODO_CRF)}
    vals.update({m: dict(v) for m, v in LODO_PROV.items()})
    for m, sub in (("BiLSTM-CRF", "bilstm"), ("XLM-R", "transformer/xlm-roberta-base")):
        f = RES / "acter_lodo" / sub / "lodo_todas.csv"
        if f.exists():
            vals[m] = pd.read_csv(f).groupby("dominio").f1_termino.mean().to_dict()
    doms = list(DOMINIOS) + ["media"]
    for m in vals:
        vals[m]["media"] = np.mean([vals[m][k] for k in DOMINIOS]) if all(k in vals[m] for k in DOMINIOS) else np.nan
    fig, a = plt.subplots(figsize=(COL1, 1.9))
    modelos = list(COL)
    w = 0.26
    x = np.arange(len(doms))
    for k, m in enumerate(modelos):
        if m in vals:
            a.bar(x + (k - 1) * w, [vals[m].get(dm, np.nan) for dm in doms], width=w - 0.03, color=COL[m],
                  label=m, edgecolor="white", lw=0.5)
        else:
            a.bar(x + (k - 1) * w, [0.02] * len(doms), width=w - 0.03, color="white", edgecolor=COL[m],
                  hatch="....", lw=0.6, label=f"{m} (pending)")
    a.set_xticks(x); a.set_xticklabels([DOMINIOS.get(dm, "Mean") for dm in doms], fontsize=6.5)
    a.set_ylabel("Term F$_1$ (unseen domain)"); a.set_ylim(0, 0.7); a.grid(axis="x", visible=False)
    a.legend(loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.8, fontsize=6.5)
    guardar(fig, "fig_acter")


if __name__ == "__main__":
    df = resultados_modelos()
    fig_distribucion()
    fig_reserva(df)
    fig_recall()  # sustituye a fig_recall_tipo y fig_tipo_token (10-nov)
    fig_oov()
    fig_ablacion()
    fig_errores()
    fig_cv_gap()
    fig_acter()
