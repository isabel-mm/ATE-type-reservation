"""
paso14_exportar.py
──────────────────
Reúne TODOS los resultados en CSV, en <carpeta>/resultados_csv/. No entrena
nada: lee los JSON de cada ejecución, las curvas por época y las predicciones
token a token, y recalcula las particiones (mismo sorteo que los modelos) para
los análisis de errores. Se puede ejecutar en cualquier momento, también con
los experimentos a medias: exporta lo que haya.

Salidas (resultados_csv/):

  01_ejecuciones_todas.csv        una fila por ejecución y modelo (CRF, BiLSTM-CRF,
                                  XLM-R, clasificadores; EN, ES y ACTER; reserva,
                                  validación cruzada y LODO), con TODAS las métricas:
                                  TP/FP/FN/TN, P/R/F1 por etiqueta (B-T, I-T, O),
                                  término (B-T+I-T) frente a no término, macro y
                                  ponderada, por unidad exacta y parcial, y las
                                  mismas restringidas a tipos reservados (res_) y
                                  vistos (vis_); época elegida, segundos, etc.
  02_resumen_reserva.csv          media, desviación típica y n de cada métrica
                                  por modelo, corpus y nivel de reserva
  03_resumen_cv.csv               ídem, validación cruzada
  04_resumen_lodo.csv             ídem, ACTER leave-one-domain-out, por dominio
  05_pruebas_estadisticas.csv     comparación de cada par de modelos, partición a
                                  partición: diferencia, IC 95 %, tamaño del efecto,
                                  Wilcoxon y t pareada, con corrección de Holm
  05b..05f                        intervalos de confianza, LODO, ablación y
                                  bootstrap por oraciones (ver LEEME.txt)
  06_curvas_epocas.csv            pérdida y métricas de validación por época
  07_analisis_por_ejecucion.csv   por ejecución de reserva: recuperación de tokens
                                  reservados con y sin palabras fuera de
                                  vocabulario, por franja de frecuencia, resultado
                                  por aparición (exacta, por defecto, por exceso,
                                  solapamiento, omitida) para reservadas y vistas,
                                  monoléxicos y poliléxicos, tipología de errores
                                  (con y sin el tipo «corpus»)
  08_resumen_analisis.csv         media, desviación típica y n de 07
  09_apariciones_reservadas.csv.gz una fila por aparición de término reservado en
                                  la prueba, por ejecución: forma esperada, forma
                                  predicha, resultado, longitud, frecuencia, OOV
  10_falsos_positivos.csv.gz      una fila por unidad propuesta que no solapa
                                  ningún término
  11_por_tipo_reservado.csv       por tipo reservado y ejecución: apariciones,
                                  reconocidas enteras, recuperación por token
  12_inventario_frecuencias.csv   frecuencia de cada tipo y franja; reparto de
                                  tipos y apariciones por franja
  13_epoca_y_tiempos.csv          época elegida, épocas entrenadas y minutos
  previos/                        resultados ya obtenidos antes (ablación,
                                  configuración de tres clases, sensibilidad,
                                  convergencia del CRF) y datos de distribución
                                  (Zipf, saturación)
  LEEME.txt                       descripción de cada fichero y columna clave

Uso:  python paso14_exportar.py
"""
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paso3_bilstm as B  # noqa: E402  (cargar, particion_reserva, separar_validacion)

RAIZ = HERE.parent
RES = Path(os.environ.get("RESULTADOS", RAIZ / "resultados"))
OUT = Path(os.environ.get("SALIDA_CSV", RAIZ / "resultados_csv"))
PREVIOS = RAIZ / "resultados_previos"
OUT.mkdir(parents=True, exist_ok=True)
NIVELES = [10, 20, 30, 50]
BANDAS = [("raro (<=10)", 0, 10), ("medio (11-100)", 11, 100), ("frecuente (>100)", 101, 10 ** 12)]
METRICAS_TEST = ["f1_termino", "p_termino", "r_termino", "res_r_termino", "vis_r_termino",
                 "f1_span_exacto", "f1_span_parcial", "f1_bt", "f1_it"]


def banda(n):
    for nombre, lo, hi in BANDAS:
        if lo <= n <= hi:
            return nombre
    return BANDAS[-1][0]


# ─────────────────────────── 1. ejecuciones ───────────────────────────

def leer_runs(carpeta: Path, modelo: str, extra=None):
    filas = []
    for p in sorted((carpeta / "runs").glob("*.json")) if (carpeta / "runs").exists() else []:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        d["modelo"] = modelo
        d["id_ejecucion"] = p.stem
        d.update(extra or {})
        filas.append(d)
    return filas


def ejecuciones():
    filas = []
    crf = leer_runs(RES / "crf", "CRF")
    filas += [f for f in crf if f.get("exp") in ("reserva", "cv", "dos")]
    filas += leer_runs(RES / "bilstm", "BiLSTM-CRF")
    for sub in sorted((RES / "transformer").glob("*")) if (RES / "transformer").exists() else []:
        filas += leer_runs(sub, "XLM-R" if "xlm" in sub.name.lower() else sub.name)
    lodo = RES / "acter_lodo"
    filas += leer_runs(lodo / "crf", "CRF", {"lang": "acter"})
    filas += leer_runs(lodo / "bilstm", "BiLSTM-CRF", {"lang": "acter"})
    for sub in sorted((lodo / "transformer").glob("*")) if (lodo / "transformer").exists() else []:
        filas += leer_runs(sub, "XLM-R" if "xlm" in sub.name.lower() else sub.name, {"lang": "acter"})
    df = pd.DataFrame(filas)
    clas = RES / "clasificadores" / "clasificadores_todas.csv"
    if clas.exists():
        c = pd.read_csv(clas)
        c["modelo"] = c.pop("clasificador")
        c["id_ejecucion"] = c["rid"]
        df = pd.concat([df, c], ignore_index=True)
    if df.empty:
        return df
    df["exp"] = df["exp"].fillna("")
    primero = ["modelo", "exp", "lang", "pct", "seed", "fold", "dominio", "id_ejecucion"]
    df = df[[c for c in primero if c in df] + [c for c in df.columns if c not in primero]]
    df.to_csv(OUT / "01_ejecuciones_todas.csv", index=False)
    print(f"  01: {len(df)} ejecuciones")
    return df


def resumenes(df):
    if df.empty:
        return
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("pct", "seed", "fold")]
    for exp, nombre, claves in (("reserva", "02_resumen_reserva.csv", ["modelo", "lang", "pct"]),
                                ("cv", "03_resumen_cv.csv", ["modelo", "lang"]),
                                ("lodo", "04_resumen_lodo.csv", ["modelo", "dominio"])):
        g = df[df.exp == exp]
        if g.empty:
            continue
        r = g.groupby(claves)[num].agg(["mean", "std", "count"])
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r = r.loc[:, [c for c in r.columns if not r[c].isna().all()]]
        if exp == "lodo":  # media de los cuatro dominios
            m = g.groupby(["modelo", "dominio"])[num].mean().groupby("modelo").mean()
            m.columns = [f"{c}_mean" for c in m.columns]
            m["dominio"] = "MEDIA"
            r = pd.concat([r.reset_index(), m.reset_index()], ignore_index=True)
        else:
            r = r.reset_index()
        r.to_csv(OUT / nombre, index=False)
        print(f"  {nombre[:2]}: {len(r)} filas")


def holm(p):
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acum = 0.0
    for k, i in enumerate(orden):
        acum = max(acum, (m - k) * p[i])
        adj[i] = min(1.0, acum)
    return adj


def _par(x, y):
    """Estadísticos de una comparación emparejada (y - x)."""
    from scipy.stats import t as tdist, ttest_rel, wilcoxon
    d = (y - x)
    n = len(d)
    out = {"n_particiones": n, "media_a": x.mean(), "media_b": y.mean(), "diferencia_b_menos_a": d.mean(),
           "dt_diferencia": d.std(ddof=1) if n > 1 else np.nan,
           "victorias_b": int((d > 0).sum()), "victorias_a": int((d < 0).sum()), "empates": int((d == 0).sum())}
    if n > 1 and d.std(ddof=1) > 0:
        se = d.std(ddof=1) / np.sqrt(n)
        q = tdist.ppf(0.975, n - 1)
        out.update({"ic95_inf": d.mean() - q * se, "ic95_sup": d.mean() + q * se,
                    "d_cohen_pareada": d.mean() / d.std(ddof=1), "p_t_pareada": ttest_rel(y, x).pvalue})
    else:
        out.update({"ic95_inf": np.nan, "ic95_sup": np.nan, "d_cohen_pareada": np.nan, "p_t_pareada": np.nan})
    try:
        out["p_wilcoxon"] = wilcoxon(d).pvalue if np.any(d != 0) and n > 1 else np.nan
    except ValueError:
        out["p_wilcoxon"] = np.nan
    # tamaño del efecto no paramétrico: correlación biserial de rangos emparejada
    nz = d[d != 0]
    if len(nz):
        from scipy.stats import rankdata
        r = rankdata(np.abs(nz))
        out["r_biserial_rangos"] = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    else:
        out["r_biserial_rangos"] = np.nan
    return out


def _holm_por(r, claves, cols=("p_wilcoxon", "p_t_pareada")):
    for col in cols:
        r[col + "_holm"] = np.nan
        for _, idx in r.groupby(claves).groups.items():
            v = r.loc[idx, col]
            ok = v.notna()
            if ok.any():
                r.loc[v[ok].index, col + "_holm"] = holm(v[ok].values)
    return r


def pruebas(df):
    """Comparación de cada par de modelos, partición a partición (reserva) o pliegue a pliegue (CV).
    Holm: por par de modelos y métrica, sobre las 8 comparaciones lengua-nivel (EN+ES) o los 4 niveles (ACTER)."""
    filas = []
    for exp, clave_rep, niveles in (("reserva", "seed", NIVELES), ("cv", "fold", [None])):
        g = df[(df.exp == exp) & df[clave_rep].notna()] if clave_rep in df else pd.DataFrame()
        if g.empty:
            continue
        for a, b in itertools.combinations(sorted(g.modelo.unique()), 2):
            for met in METRICAS_TEST:
                if met not in g:
                    continue
                for lang in sorted(g.lang.dropna().unique()):
                    for pct in niveles:
                        sel = (g.lang == lang) & ((g.pct == pct) if pct is not None else True)
                        x = g[sel & (g.modelo == a)].set_index(clave_rep)[met].dropna()
                        y = g[sel & (g.modelo == b)].set_index(clave_rep)[met].dropna()
                        comun = x.index.intersection(y.index)
                        if len(comun) < 2:
                            continue
                        fila = {"protocolo": exp, "modelo_a": a, "modelo_b": b, "metrica": met, "lang": lang,
                                "pct": pct, **_par(x.loc[comun].values, y.loc[comun].values)}
                        filas.append(fila)
    r = pd.DataFrame(filas)
    if r.empty:
        return
    r["grupo_holm"] = np.where(r.lang == "acter", "acter", "en+es")
    r = _holm_por(r, ["protocolo", "modelo_a", "modelo_b", "metrica", "grupo_holm"])
    r.to_csv(OUT / "05_pruebas_estadisticas.csv", index=False)
    print(f"  05: {len(r)} comparaciones")


def intervalos(df):
    """Media, desviación típica, error típico e IC del 95 % (t de Student) de las métricas principales."""
    from scipy.stats import t as tdist
    filas = []
    for (modelo, exp, lang, pct), g in df.groupby(["modelo", "exp", "lang", df["pct"].fillna(-1)], dropna=False):
        for met in METRICAS_TEST + ["f1_global", "f1_macro", "f1_o"]:
            if met not in g:
                continue
            v = g[met].dropna().values
            if len(v) == 0:
                continue
            n = len(v)
            se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            q = tdist.ppf(0.975, n - 1) if n > 1 else np.nan
            filas.append({"modelo": modelo, "exp": exp, "lang": lang, "pct": None if pct == -1 else pct, "metrica": met,
                          "n": n, "media": v.mean(), "dt": v.std(ddof=1) if n > 1 else np.nan, "error_tipico": se,
                          "ic95_inf": v.mean() - q * se if n > 1 else np.nan,
                          "ic95_sup": v.mean() + q * se if n > 1 else np.nan, "min": v.min(), "max": v.max()})
    if filas:
        pd.DataFrame(filas).to_csv(OUT / "05b_intervalos_confianza.csv", index=False)
        print(f"  05b: {len(filas)} intervalos")


def pruebas_lodo(df):
    """ACTER LODO: modelos neuronales (5 semillas por dominio) frente al CRF (determinista):
    t de una muestra contra el valor del CRF en cada dominio, y comparación de las medias por dominio."""
    from scipy.stats import ttest_1samp
    g = df[df.exp == "lodo"]
    if g.empty or "CRF" not in set(g.modelo):
        return
    filas = []
    crf = g[g.modelo == "CRF"].set_index("dominio")
    for modelo in sorted(set(g.modelo) - {"CRF"}):
        for met in ("f1_termino", "p_termino", "r_termino", "res_r_termino", "f1_span_exacto"):
            if met not in g:
                continue
            for dom, gg in g[g.modelo == modelo].groupby("dominio"):
                if dom not in crf.index:
                    continue
                v = gg[met].dropna().values
                ref = crf.loc[dom, met]
                p = ttest_1samp(v, ref).pvalue if len(v) > 1 and v.std() > 0 else np.nan
                filas.append({"modelo": modelo, "metrica": met, "dominio": dom, "n_semillas": len(v),
                              "media_modelo": v.mean(), "dt_modelo": v.std(ddof=1) if len(v) > 1 else np.nan,
                              "crf": ref, "diferencia_modelo_menos_crf": v.mean() - ref,
                              "semillas_por_encima_crf": int((v > ref).sum()), "p_t_una_muestra": p})
    r = pd.DataFrame(filas)
    if r.empty:
        return
    r = _holm_por(r, ["modelo", "metrica"], cols=("p_t_una_muestra",))
    r.to_csv(OUT / "05c_pruebas_lodo.csv", index=False)
    print(f"  05c: {len(r)} comparaciones LODO")


def pruebas_ablacion():
    """Ablación del CRF (resultados previos): para cada familia, lengua y nivel, t de una muestra y
    Wilcoxon sobre las diez diferencias (ΔF1 de término y Δ recuperación de reservados), con Holm
    sobre los 72 contrastes (9 familias x 2 lenguas x 4 niveles) de cada medida; y la versión con CV."""
    from scipy.stats import ttest_1samp, wilcoxon
    filas = []
    for fichero, clave in (("crf_ablacion_todas.csv", "pct"), ("crf_ablacv_todas.csv", None)):
        f = PREVIOS / "crf" / fichero
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.familia.notna() & (d.familia != "BASELINE")]
        grupos = ["familia", "lang"] + ([clave] if clave else [])
        for keys, g in d.groupby(grupos):
            keys = dict(zip(grupos, keys if isinstance(keys, tuple) else (keys,)))
            for med in ("delta_f1_termino", "delta_recall_reservadas"):
                if med not in g:
                    continue
                v = g[med].dropna().values
                if len(v) < 2:
                    continue
                try:
                    pw = wilcoxon(v).pvalue if np.any(v != 0) else np.nan
                except ValueError:
                    pw = np.nan
                filas.append({"protocolo": "reserva" if clave else "cv", **keys, "medida": med, "n": len(v),
                              "media": v.mean(), "dt": v.std(ddof=1), "negativos": int((v < 0).sum()),
                              "positivos": int((v > 0).sum()),
                              "p_t_una_muestra": ttest_1samp(v, 0).pvalue if v.std() > 0 else np.nan,
                              "p_wilcoxon": pw})
    r = pd.DataFrame(filas)
    if r.empty:
        return
    r = _holm_por(r, ["protocolo", "medida"], cols=("p_t_una_muestra", "p_wilcoxon"))
    r.to_csv(OUT / "05d_pruebas_ablacion.csv", index=False)
    print(f"  05d: {len(r)} contrastes de ablación")


def curvas():
    trozos = []
    for modelo, carpeta in [("BiLSTM-CRF", RES / "bilstm" / "epocas"), ("BiLSTM-CRF", RES / "acter_lodo" / "bilstm" / "epocas")] + \
            [("XLM-R", p / "epocas") for p in (RES / "transformer").glob("*")] + \
            [("XLM-R", p / "epocas") for p in (RES / "acter_lodo" / "transformer").glob("*")]:
        for f in sorted(carpeta.glob("*.csv")) if carpeta.exists() else []:
            d = pd.read_csv(f)
            d.insert(0, "id_ejecucion", f.stem)
            d.insert(0, "modelo", modelo)
            trozos.append(d)
    if trozos:
        pd.concat(trozos, ignore_index=True).to_csv(OUT / "06_curvas_epocas.csv", index=False)
        print(f"  06: {len(trozos)} curvas")


# ─────────────────────────── 2. análisis de errores ───────────────────────────

def spans(labels):
    out, ini = [], None
    for i, l in enumerate(list(labels) + ["O"]):
        if l == "B-T" or l == "O" or (l == "I-T" and ini is None):
            if ini is not None:
                out.append((ini, i))
                ini = None
            if l in ("B-T", "I-T"):
                ini = i
    return out


def archivos_pred(lang, pct, seed):
    """Predicciones de cada modelo para una partición de reserva."""
    cand = {"CRF": RES / "crf" / "pred" / f"reserva_{lang}_p{pct}_s{seed}_todas.csv.gz",
            "BiLSTM-CRF": RES / "bilstm" / "pred" / f"reserva_{lang}_p{pct}_s{seed}.csv.gz"}
    for sub in (RES / "transformer").glob("*") if (RES / "transformer").exists() else []:
        cand["XLM-R" if "xlm" in sub.name.lower() else sub.name] = sub / "pred" / f"reserva_{lang}_p{pct}_s{seed}.csv.gz"
    return {m: p for m, p in cand.items() if p.exists()}


B_BOOT = int(os.environ.get("BOOTSTRAP", 1000))


def bootstrap_particion(conteos, lang, pct, seed):
    """Bootstrap pareado por oraciones (mismas oraciones remuestreadas para los dos modelos):
    diferencia de F1 de término y de recuperación de reservados, IC del 95 % y p bilateral."""
    filas = []
    modelos = sorted(conteos)
    if len(modelos) < 2:
        return filas
    n = len(next(iter(conteos.values())))
    rng = np.random.default_rng(12345 + 1000 * pct + seed)
    sumas = {m: [] for m in modelos}
    for ini in range(0, B_BOOT, 100):
        idx = rng.integers(0, n, size=(min(100, B_BOOT - ini), n))
        for m in modelos:
            sumas[m].append(conteos[m][idx].sum(axis=1))
    sumas = {m: np.vstack(v) for m, v in sumas.items()}

    def f1(c):
        return np.where(2 * c[..., 0] + c[..., 1] + c[..., 2] > 0,
                        2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1), 0.0)

    def rres(c):
        return np.where(c[..., 3] > 0, c[..., 4] / np.maximum(c[..., 3], 1), np.nan)

    for a, b in itertools.combinations(modelos, 2):
        for nombre, fn in (("f1_termino", f1), ("res_r_termino", rres)):
            obs = fn(conteos[b].sum(0)) - fn(conteos[a].sum(0))
            dist = fn(sumas[b]) - fn(sumas[a])
            dist = dist[~np.isnan(dist)]
            if not len(dist):
                continue
            p = min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))
            filas.append({"modelo_a": a, "modelo_b": b, "metrica": nombre, "lang": lang, "pct": pct, "seed": seed,
                          "diferencia_b_menos_a": float(obs), "ic95_inf": float(np.percentile(dist, 2.5)),
                          "ic95_sup": float(np.percentile(dist, 97.5)), "p_bootstrap": p, "remuestras": len(dist)})
    return filas


def spearman_frecuencia(tipos):
    """Correlación de Spearman entre la frecuencia de cada tipo reservado y su recuperación
    (media de las particiones en que aparece), por modelo, corpus y nivel."""
    from scipy.stats import spearmanr
    if tipos.empty:
        return
    filas = []
    for (modelo, lang, pct), g in tipos.groupby(["modelo", "lang", "pct"]):
        t = g.groupby("entrada").agg(freq=("frecuencia_tipo", "first"), recall=("recall_token", "mean"),
                                     exactas=("prop_exactas", "mean"))
        if len(t) < 5:
            continue
        for med in ("recall", "exactas"):
            rho, p = spearmanr(t["freq"], t[med])
            filas.append({"modelo": modelo, "lang": lang, "pct": pct, "medida": med, "n_tipos": len(t),
                          "rho_spearman": rho, "p": p})
    if filas:
        pd.DataFrame(filas).to_csv(OUT / "11b_spearman_frecuencia.csv", index=False)


def analisis_errores():
    filas_run, filas_ap, filas_fp, filas_tipo, filas_inv, filas_boot = [], [], [], [], [], []
    langs = [l for l in ("en", "es", "acter")
             if any((B.DATOS / f"{l}_bio_tokens_terms_only.csv{ext}").exists() for ext in ("", ".gz"))]
    for lang in langs:
        tareas = [(p, s) for p in NIVELES for s in range(10) if archivos_pred(lang, p, s)]
        if not tareas:
            continue
        oraciones, entradas = B.cargar(lang)
        por_clave = {(str(o["doc_id"]), str(o["sentence_id"])): o for o in oraciones}
        # frecuencia de cada tipo = número de apariciones anotadas en todo el corpus
        freq = Counter()
        for o in oraciones:
            for a, b in spans(o["labels"]):
                if o["entradas"][a]:
                    freq[o["entradas"][a]] += 1
        tot = sum(freq.values())
        # todas las formas anotadas en el corpus (para la precisión por tipo y los anidados válidos)
        formas_corpus = {" ".join(o["tokens"][a:b]) for o in oraciones for a, b in spans(o["labels"])}
        for e, n in freq.items():
            filas_inv.append({"lang": lang, "entrada": e, "apariciones": n, "franja": banda(n)})
        print(f"  {lang}: {len(tareas)} particiones con predicciones")
        for pct, seed in tareas:
            train, test, reservadas = B.particion_reserva(oraciones, entradas, pct, seed)
            tr, _ = B.separar_validacion(train, seed, por_entradas=True)
            vocab = {t for o in tr for t in o["tokens"]}
            conteos = {}   # modelo -> matriz (oraciones x [tp, fp, fn, tokens_res, aciertos_res])
            # año de publicación del documento (en el identificador: "EN::2023_...") para el control de contaminación
            anyo = np.array([int(m.group(1)) if (m := re.search(r"::(\d{4})_", str(o["doc_id"]))) else 0 for o in test])
            tipos_test = {o["entradas"][a] for o in test for a, b in spans(o["labels"]) if o["entradas"][a]}
            formas_test = {" ".join(o["tokens"][a:b]) for o in test for a, b in spans(o["labels"])}
            for modelo, ruta in archivos_pred(lang, pct, seed).items():
                cuentas = np.zeros((len(test), 5), dtype=np.int64)
                pred = pd.read_csv(ruta, usecols=["doc_id", "sentence_id", "i", "pred"], keep_default_na=False,
                                   dtype={"doc_id": str, "sentence_id": str})
                pmap = defaultdict(dict)
                for d, s, i, p in pred.itertuples(index=False):
                    pmap[(d, s)][int(i)] = p
                c = Counter()
                tipos_ok, formas_pred = set(), set()
                tok = Counter()          # recuperación por token reservado: (grupo, acierto)
                por_tipo = defaultdict(lambda: Counter())
                for k_or, o in enumerate(test):
                    clave = (str(o["doc_id"]), str(o["sentence_id"]))
                    pm = pmap.get(clave, {})
                    yp = [pm.get(i, "O") for i in range(len(o["tokens"]))]
                    for gl, pr, en in zip(o["labels"], yp, o["entradas"]):
                        gt, pt_ = gl != "O", pr != "O"
                        cuentas[k_or, 0] += gt and pt_
                        cuentas[k_or, 1] += pt_ and not gt
                        cuentas[k_or, 2] += gt and not pt_
                        if gt and en in reservadas:
                            cuentas[k_or, 3] += 1
                            cuentas[k_or, 4] += pt_
                    gs, ps = spans(o["labels"]), spans(yp)
                    ps_set = set(ps)
                    formas_pred.update(" ".join(o["tokens"][x:y]) for x, y in ps)
                    for a, b in gs:
                        e = o["entradas"][a]
                        res = e in reservadas
                        grupo = "res" if res else "vis"
                        forma = "mono" if b - a == 1 else "poli"
                        solapes = [(x, y) for x, y in ps if x < b and y > a]
                        if (a, b) in ps_set:
                            resultado = "exacta"
                        elif not solapes:
                            resultado = "omitida"
                        elif all(x >= a and y <= b for x, y in solapes):
                            resultado = "por_defecto"
                        elif any(x <= a and y >= b for x, y in solapes):
                            resultado = "por_exceso"
                        else:
                            resultado = "solapamiento"
                        c[f"{grupo}_n"] += 1
                        c[f"{grupo}_{resultado}"] += 1
                        if resultado == "exacta" and e:
                            tipos_ok.add(e)
                        if resultado == "por_defecto" and all(" ".join(o["tokens"][x:y]) in formas_corpus for x, y in solapes):
                            c[f"{grupo}_por_defecto_termino_anotado"] += 1   # p. ej. "corpus" en "corpus linguistics"
                        c[f"{grupo}_{forma}_n"] += 1
                        c[f"{grupo}_{forma}_exacta"] += resultado == "exacta"
                        if not res:
                            continue
                        oov = any(t not in vocab for t in o["tokens"][a:b])
                        fr = banda(freq.get(e, 0))
                        aciertos = sum(yp[k] != "O" for k in range(a, b))
                        for g2 in ("todos", "con_oov" if oov else "sin_oov", fr):
                            tok[(g2, "n")] += b - a
                            tok[(g2, "ok")] += aciertos
                        por_tipo[e]["apariciones"] += 1
                        por_tipo[e]["exactas"] += resultado == "exacta"
                        por_tipo[e]["tokens"] += b - a
                        por_tipo[e]["tokens_ok"] += aciertos
                        por_tipo[e]["con_oov"] += oov
                        if resultado != "exacta":
                            tipo = ("omision_" + forma) if resultado == "omitida" else resultado
                            c["err_" + tipo] += 1
                            if e != "corpus":
                                c["errsc_" + tipo] += 1
                        filas_ap.append({"modelo": modelo, "lang": lang, "pct": pct, "seed": seed,
                                         "doc_id": clave[0], "sentence_id": clave[1], "entrada": e,
                                         "forma_esperada": " ".join(o["tokens"][a:b]), "n_tokens": b - a,
                                         "forma_predicha": " | ".join(" ".join(o["tokens"][x:y]) for x, y in solapes),
                                         "resultado": resultado, "tokens_reconocidos": aciertos,
                                         "con_oov": oov, "frecuencia_tipo": freq.get(e, 0), "franja": fr})
                    for x, y in ps:
                        if not any(x < b and y > a for a, b in gs):
                            c["err_falso_positivo"] += 1
                            c["errsc_falso_positivo"] += 1
                            filas_fp.append({"modelo": modelo, "lang": lang, "pct": pct, "seed": seed,
                                             "doc_id": clave[0], "sentence_id": clave[1],
                                             "forma_predicha": " ".join(o["tokens"][x:y]), "n_tokens": y - x})
                fila = {"modelo": modelo, "lang": lang, "pct": pct, "seed": seed}
                for grupo in ("res", "vis"):
                    n = c[f"{grupo}_n"]
                    fila[f"{grupo}_apariciones"] = n
                    for r_ in ("exacta", "por_defecto", "por_exceso", "solapamiento", "omitida"):
                        fila[f"{grupo}_prop_{r_}"] = c[f"{grupo}_{r_}"] / n if n else np.nan
                    fila[f"{grupo}_prop_reconocida_en_parte_o_entera"] = 1 - fila[f"{grupo}_prop_omitida"] if n else np.nan
                    for forma in ("mono", "poli"):
                        m = c[f"{grupo}_{forma}_n"]
                        fila[f"{grupo}_{forma}_apariciones"] = m
                        fila[f"{grupo}_{forma}_prop_exacta"] = c[f"{grupo}_{forma}_exacta"] / m if m else np.nan
                for g2 in ["todos", "sin_oov", "con_oov"] + [b[0] for b in BANDAS]:
                    n = tok[(g2, "n")]
                    fila[f"res_tokens_{g2}"] = n
                    fila[f"res_recall_token_{g2}"] = tok[(g2, "ok")] / n if n else np.nan
                # ── por tipo (como TermEval): términos distintos encontrados y formas propuestas correctas
                tipos_res = tipos_test & reservadas
                fila["tipo_n_gold"] = len(tipos_test)
                fila["tipo_recall"] = len(tipos_ok & tipos_test) / len(tipos_test) if tipos_test else np.nan
                fila["tipo_n_gold_reservados"] = len(tipos_res)
                fila["tipo_recall_reservados"] = len(tipos_ok & tipos_res) / len(tipos_res) if tipos_res else np.nan
                fila["tipo_n_formas_propuestas"] = len(formas_pred)
                for ref, conj in (("test", formas_test), ("corpus", formas_corpus)):
                    pr_ = len(formas_pred & conj) / len(formas_pred) if formas_pred else np.nan
                    fila[f"tipo_precision_{ref}"] = pr_
                    rc_ = fila["tipo_recall"]
                    fila[f"tipo_f1_{ref}"] = 2 * pr_ * rc_ / (pr_ + rc_) if pr_ and rc_ else 0.0
                for grupo in ("res", "vis"):
                    n = c[f"{grupo}_por_defecto"]
                    fila[f"{grupo}_por_defecto_prop_termino_anotado"] = c[f"{grupo}_por_defecto_termino_anotado"] / n if n else np.nan
                # ── contaminación: textos hasta 2019 frente a textos desde 2020 (posteriores a XLM-R)
                for periodo, mascara in (("hasta2019", (anyo > 0) & (anyo < 2020)), ("desde2020", anyo >= 2020)):
                    cs = cuentas[mascara].sum(0)
                    fila[f"f1_termino_{periodo}"] = 2 * cs[0] / (2 * cs[0] + cs[1] + cs[2]) if (2 * cs[0] + cs[1] + cs[2]) else np.nan
                    fila[f"res_recall_token_{periodo}"] = cs[4] / cs[3] if cs[3] else np.nan
                    fila[f"res_tokens_{periodo}"] = int(cs[3])
                    fila[f"oraciones_{periodo}"] = int(mascara.sum())
                ap_res = c["res_n"]
                fila["prop_apariciones_reservadas_con_oov"] = (
                    sum(1 for f in filas_ap[-ap_res:] if f["con_oov"]) / ap_res if ap_res else np.nan)
                for pref in ("err", "errsc"):
                    tipos = ["omision_mono", "omision_poli", "por_defecto", "por_exceso", "solapamiento", "falso_positivo"]
                    total = sum(c[f"{pref}_{t}"] for t in tipos)
                    fila[f"{pref}_total"] = total
                    for t in tipos:
                        fila[f"{pref}_n_{t}"] = c[f"{pref}_{t}"]
                        fila[f"{pref}_pct_{t}"] = 100 * c[f"{pref}_{t}"] / total if total else np.nan
                filas_run.append(fila)
                conteos[modelo] = cuentas
                for e, cc in por_tipo.items():
                    filas_tipo.append({"modelo": modelo, "lang": lang, "pct": pct, "seed": seed, "entrada": e,
                                       "frecuencia_tipo": freq.get(e, 0), "franja": banda(freq.get(e, 0)), **cc,
                                       "prop_exactas": cc["exactas"] / cc["apariciones"],
                                       "recall_token": cc["tokens_ok"] / cc["tokens"]})
            filas_boot.extend(bootstrap_particion(conteos, lang, pct, seed))
    if not filas_run:
        print("  (sin predicciones de reserva todavía)")
        return
    df = pd.DataFrame(filas_run)
    df.to_csv(OUT / "07_analisis_por_ejecucion.csv", index=False)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("pct", "seed")]
    r = df.groupby(["modelo", "lang", "pct"])[num].agg(["mean", "std", "count"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    r.reset_index().to_csv(OUT / "08_resumen_analisis.csv", index=False)
    pd.DataFrame(filas_ap).to_csv(OUT / "09_apariciones_reservadas.csv.gz", index=False, compression="gzip")
    pd.DataFrame(filas_fp).to_csv(OUT / "10_falsos_positivos.csv.gz", index=False, compression="gzip")
    pd.DataFrame(filas_tipo).to_csv(OUT / "11_por_tipo_reservado.csv", index=False)
    if filas_boot:
        b = pd.DataFrame(filas_boot)
        b.to_csv(OUT / "05e_bootstrap_por_particion.csv", index=False)
        # resumen: en cuántas particiones la diferencia es significativa en cada sentido
        res = b.groupby(["modelo_a", "modelo_b", "metrica", "lang", "pct"]).agg(
            particiones=("seed", "count"), diferencia_media=("diferencia_b_menos_a", "mean"),
            signif_b_mejor=("p_bootstrap", lambda v: int(((v < 0.05) & (b.loc[v.index, "diferencia_b_menos_a"] > 0)).sum())),
            signif_a_mejor=("p_bootstrap", lambda v: int(((v < 0.05) & (b.loc[v.index, "diferencia_b_menos_a"] < 0)).sum())))
        res.reset_index().to_csv(OUT / "05f_bootstrap_resumen.csv", index=False)
    spearman_frecuencia(pd.DataFrame(filas_tipo))
    inv = pd.DataFrame(filas_inv)
    resumen = inv.groupby(["lang", "franja"]).agg(tipos=("entrada", "count"), apariciones=("apariciones", "sum"))
    resumen["prop_tipos"] = resumen["tipos"] / resumen.groupby("lang")["tipos"].transform("sum")
    resumen["prop_apariciones"] = resumen["apariciones"] / resumen.groupby("lang")["apariciones"].transform("sum")
    inv.to_csv(OUT / "12_inventario_frecuencias.csv", index=False)
    resumen.reset_index().to_csv(OUT / "12b_inventario_por_franja.csv", index=False)
    print(f"  07-12: {len(df)} análisis, {len(filas_ap)} apariciones reservadas, {len(filas_fp)} falsos positivos")


def tiempos(df):
    cols = [c for c in ("modelo", "exp", "lang", "pct", "seed", "fold", "dominio", "epoca_elegida",
                        "epocas_entrenadas", "val_f1_termino_mejor", "segundos") if c in df]
    if "segundos" in cols:
        t = df[cols].copy()
        t["minutos"] = t["segundos"] / 60
        t.to_csv(OUT / "13_epoca_y_tiempos.csv", index=False)


def previos():
    if PREVIOS.exists():
        dest = OUT / "previos"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(PREVIOS, dest)
        print(f"  previos: {sum(1 for _ in dest.rglob('*.csv'))} ficheros")


LEEME = """Resultados del artículo (todos en CSV; .csv.gz = CSV comprimido)

Claves comunes: modelo (CRF, BiLSTM-CRF, XLM-R o clasificador), exp (reserva, cv, lodo),
lang (en, es, acter), pct (porcentaje de tipos reservados), seed (partición 0-9), fold
(pliegue de la validación cruzada), dominio (ACTER LODO).

Prefijos de métricas: tp_/fp_/fn_/tn_ recuentos; p_/r_/f1_ precisión, recuperación, F1.
Sufijos: _bt (B-T), _it (I-T), _o (O), _termino (B-T+I-T frente a O; métrica principal),
_macro, _global (ponderada), _span_exacto / _span_parcial (por unidad completa).
res_*: las mismas métricas solo sobre los tokens de tipos reservados (nunca vistos);
vis_*: sobre el resto. En LODO, res_ = tipos ausentes de los dominios de entrenamiento.

07/08 análisis por ejecución: res_/vis_prop_exacta|por_defecto|por_exceso|solapamiento|omitida
(resultado de cada aparición), *_mono/_poli_prop_exacta (delimitación exacta de monoléxicos y
poliléxicos), res_recall_token_sin_oov|con_oov (palabras fuera del vocabulario de
entrenamiento), res_recall_token_<franja> (frecuencia del tipo), err_* (tipología de errores
sobre reservados + falsos positivos), errsc_* (ídem sin el tipo «corpus», como en la tesis).
OOV: alguna palabra del término no aparece en el entrenamiento (sin el conjunto de validación).
Frecuencia del tipo: apariciones anotadas en todo el corpus.

Pruebas estadísticas (diferencia siempre = modelo_b - modelo_a):
05  pares de modelos, partición a partición (reserva, 10 particiones) o pliegue a pliegue (CV):
    diferencia media, IC 95 % (t), d de Cohen pareada, correlación biserial de rangos, victorias,
    Wilcoxon y t pareada; Holm por protocolo, par de modelos y métrica sobre las 8 comparaciones
    lengua-nivel (EN+ES) o los 4 niveles (ACTER).
05b media, desviación típica, error típico e IC 95 % de cada métrica por modelo, corpus y nivel.
05c ACTER LODO: cada modelo neuronal (5 semillas) frente al CRF (determinista) por dominio,
    t de una muestra con Holm.
05d ablación del CRF: t de una muestra y Wilcoxon sobre las 10 diferencias de cada familia,
    lengua y nivel (ΔF1 de término y Δ recuperación de reservados), Holm sobre los 72 contrastes.
05e/05f bootstrap pareado por oraciones dentro de cada partición (1000 remuestras): diferencia de
    F1 de término y de recuperación de reservados, IC 95 % y p; 05f cuenta en cuántas particiones
    es significativa en cada sentido.
11b Spearman entre la frecuencia de cada tipo reservado y su recuperación.
07 además: tipo_* = evaluación por tipo, como en TermEval (tipo_recall: términos distintos de la
prueba con al menos una aparición reconocida entera; tipo_precision_test/_corpus: formas propuestas
distintas que son formas anotadas en la prueba / en todo el corpus); *_por_defecto_prop_termino_anotado:
de las delimitaciones por defecto, cuántas proponen a su vez un término anotado (p. ej. "corpus" en
"corpus linguistics"), para valorar el efecto de anotar solo la cadena más larga; *_hasta2019 /
*_desde2020: F1 y recuperación de reservados según el año del documento (XLM-R no pudo ver en su
preentrenamiento los textos publicados desde 2020).
"""


def main():
    print(f"Exportando a {OUT}")
    df = ejecuciones()
    resumenes(df)
    if not df.empty:
        pruebas(df)
        intervalos(df)
        pruebas_lodo(df)
        tiempos(df)
    pruebas_ablacion()
    curvas()
    analisis_errores()
    previos()
    (OUT / "LEEME.txt").write_text(LEEME, encoding="utf-8")
    print("Exportación terminada.")


if __name__ == "__main__":
    main()
