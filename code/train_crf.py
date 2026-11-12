"""
train_crf.py
────────────
All experiments with the feature-based CRF (27 features; corpus statistics and PMI
are computed on the training partition only):

  reserva      type reservation at 10/20/30/50%, 10 seeds (0-9), terms only
  dos          separate labels for terms and entities (B-T/I-T + B-E/I-E), same partitions
  ablacion     20% reservation: each of the 9 feature families removed in turn with seed 0,
               plus seeds 1-9 without morphology, part of speech and syntax. With
               --ablacion-completa, all 9 families with all 10 seeds; with
               --todos-niveles, at the four reservation levels.
  cv           standard five-fold cross-validation (random partitions)
  ablacv       family ablation under five-fold cross-validation, to contrast it
               with type reservation
  todo         reserva + dos + ablacion + cv

Each run writes, in $RESULTADOS/crf/:
  runs/<id>.json         all metrics (metrics.py) on the whole test set, and also
                         restricted to the tokens of reserved entries (res_*) and to
                         the rest (vis_*)
  pred/<id>.csv.gz       token-level predictions (gold/pred); not for the ablation
  reservadas/<id>.json   entries reserved in that partition
`python train_crf.py consolidar` merges the JSON files into CSV.

Runs that already have a JSON file are skipped.

Usage:
  python train_crf.py reserva [--jobs 4] [--lang en]
  python train_crf.py dos | ablacion | cv | ablacv | todo
  python train_crf.py consolidar
  python train_crf.py reserva --rapido      (quick check: one seed, 20% level)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crf_base as base  # noqa: E402
from metrics import guardar_predicciones, metricas_completas  # noqa: E402

OUT = base.OUT_DIR / "crf"
for d in ("runs", "pred", "reservadas"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

NIVELES = [10, 20, 30, 50]
SEMILLAS = list(range(10))
SEMILLAS_ABL = list(range(10))   # the same seeds as the reservation (0-9)
FAMILIAS_MULTI = ["morfologia", "morfosintaxis", "sintaxis"]
N_FOLDS = 5

DATA = {}   # (lang, full) -> (sentences, span_types); inherited by the child processes (fork)


def cargar(lang, completa):
    if (lang, completa) not in DATA:
        t0 = time.time()
        df = base.load_corpus(lang, completa=completa)
        df = base.attach_span_text(df, solo_terminos=True)
        span_types = np.array(sorted(df["_span_text"].dropna().unique()))
        DATA[(lang, completa)] = (base.build_sentences(df), span_types)
        print(f"  {lang} ({'terms and entities' if completa else 'terms only'}): {len(DATA[(lang, completa)][0])} sentences, "
              f"{len(span_types)} entries that can be reserved ({time.time()-t0:.0f}s)", flush=True)
    return DATA[(lang, completa)]


def run_id(t):
    fam = t.get("familia") or "todas"
    if t["exp"] == "cv":
        return f"cv_{t['lang']}_f{t['fold']}"
    if t["exp"] == "ablacv":
        return f"ablacv_{t['lang']}_f{t['fold']}_{fam}"
    return f"{t['exp']}_{t['lang']}_p{t['pct']}_s{t['seed']}_{fam}"


def entrenar_y_evaluar(train_sents, test_sents, lang, familia=None):
    stats_get, stat_bounds = base.recompute_freq_stats(train_sents)
    pmi_fn = base.compute_pmi_lookup(train_sents)
    pmi_bounds = base.pmi_quartile_bounds(train_sents, pmi_fn)
    X_tr, y_tr = base.build_features_for_sentences(train_sents, stats_get, stat_bounds, pmi_fn, pmi_bounds, lang,
                                                   exclude_family=familia)
    X_te, y_te = base.build_features_for_sentences(test_sents, stats_get, stat_bounds, pmi_fn, pmi_bounds, lang,
                                                   exclude_family=familia)
    crf = base.train_crf(X_tr, y_tr)
    return crf, X_te, y_te, crf.predict(X_te)


def metricas_restringidas(test_sents, y_te, y_pr, reservadas):
    """Metrics on the tokens of reserved (never seen) entries and on the rest."""
    res_t, res_p, vis_t, vis_p = [], [], [], []
    for s, yt, yp in zip(test_sents, y_te, y_pr):
        for r, a, b in zip(s["rows"], yt, yp):
            if r.get("_span_text") is not None and r.get("_span_text") in reservadas:
                res_t.append(a); res_p.append(b)
            else:
                vis_t.append(a); vis_p.append(b)
    out = {}
    for pref, a, b in (("res_", res_t, res_p), ("vis_", vis_t, vis_p)):
        if a:
            m = metricas_completas(a, b)
            out.update({pref + k: v for k, v in m.items()})
    return out


def ejecutar(t):
    rid = run_id(t)
    dest = OUT / "runs" / f"{rid}.json"
    if dest.exists():
        return rid, "ya hecho"
    t0 = time.time()
    lang = t["lang"]
    completa = t["exp"] == "dos"
    sentences, span_types = cargar(lang, completa)
    fila = {k: v for k, v in t.items()}

    if t["exp"] in ("cv", "ablacv"):
        from sklearn.model_selection import StratifiedKFold
        y_strat = np.array([any(r["label_bio"] != "O" for r in s["rows"]) for s in sentences], dtype=int)
        folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(sentences)), y_strat))
        tr_idx, te_idx = folds[t["fold"] - 1]
        train_sents = [sentences[i] for i in tr_idx]
        test_sents = [sentences[i] for i in te_idx]
        crf, X_te, y_te, y_pr = entrenar_y_evaluar(train_sents, test_sents, lang, t.get("familia"))
        fila.update(metricas_completas(y_te, y_pr))
        if t["exp"] == "ablacv":
            fila["segundos"] = round(time.time() - t0, 1)
            dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
            return rid, f"term F1={fila['f1_termino']:.3f} ({fila['segundos']:.0f}s)"
        marg = crf.predict_marginals(X_te)
        filas = [{"y_true": int(a != "O"), "p_term": m.get("B-T", 0.0) + m.get("I-T", 0.0)}
                 for yt, mm in zip(y_te, marg) for a, m in zip(yt, mm)]
        pd.DataFrame(filas).to_csv(OUT / "pred" / f"{rid}_marginales.csv.gz", index=False, compression="gzip")
        guardar_predicciones(OUT / "pred" / f"{rid}.csv.gz", test_sents, y_te, y_pr)
    else:
        train_sents, test_sents, reservadas = base.split_by_type_holdout(sentences, span_types, t["pct"], seed=t["seed"])
        fila.update({"n_entradas_sorteables": len(span_types), "n_reservadas": len(reservadas),
                     "n_oraciones_train": len(train_sents), "n_oraciones_test": len(test_sents)})
        crf, X_te, y_te, y_pr = entrenar_y_evaluar(train_sents, test_sents, lang, t.get("familia"))
        fila.update(metricas_completas(y_te, y_pr))
        fila.update(metricas_restringidas(test_sents, y_te, y_pr, reservadas))
        (OUT / "reservadas" / f"{rid}.json").write_text(json.dumps(sorted(reservadas), ensure_ascii=False))
        if t["exp"] != "ablacion" or t.get("familia") is None:
            guardar_predicciones(OUT / "pred" / f"{rid}.csv.gz", test_sents, y_te, y_pr)
    fila["segundos"] = round(time.time() - t0, 1)
    dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
    return rid, f"term F1={fila['f1_termino']:.3f} ({fila['segundos']:.0f}s)"


def tareas(exp, langs, rapido, abl_completa, todos_niveles=False):
    out = []
    for lang in langs:
        if exp in ("reserva", "dos", "todo"):
            for e in (["reserva", "dos"] if exp == "todo" else [exp]):
                for pct in ([20] if rapido else NIVELES):
                    for seed in (SEMILLAS[:1] if rapido else SEMILLAS):
                        out.append({"exp": e, "lang": lang, "pct": pct, "seed": seed, "familia": None})
        if exp in ("ablacion", "todo"):
            fams_all = list(base.FAMILIAS)
            for pct in (NIVELES if todos_niveles else [20]):
                for seed in (SEMILLAS_ABL[:1] if rapido else SEMILLAS_ABL):
                    fams = fams_all if (seed == SEMILLAS_ABL[0] or abl_completa) else FAMILIAS_MULTI
                    for fam in [None] + fams:
                        out.append({"exp": "ablacion", "lang": lang, "pct": pct, "seed": seed, "familia": fam})
        if exp == "ablacv":
            for fold in range(1, N_FOLDS + 1):
                for fam in [None] + list(base.FAMILIAS):
                    out.append({"exp": "ablacv", "lang": lang, "fold": fold, "pct": None, "seed": 42, "familia": fam})
        if exp in ("cv", "todo"):
            for fold in ([1] if rapido else range(1, N_FOLDS + 1)):
                out.append({"exp": "cv", "lang": lang, "fold": fold, "pct": None, "seed": 42, "familia": None})
    return out


def consolidar():
    filas = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    if not filas:
        print("No results yet.")
        return
    df = pd.DataFrame(filas)
    for exp, g in df.groupby("exp"):
        g.to_csv(OUT / f"crf_{exp}_todas.csv", index=False)
    # summaries: mean and standard deviation by language and level
    num = [c for c in df.columns if df[c].dtype != object and c not in ("pct", "seed", "fold")]
    for exp in ("reserva", "dos"):
        g = df[df.exp == exp]
        if len(g):
            r = g.groupby(["lang", "pct"])[num].agg(["mean", "std"])
            r.columns = [f"{a}_{b}" for a, b in r.columns]
            r.reset_index().to_csv(OUT / f"crf_{exp}_resumen.csv", index=False)
    g = df[df.exp == "ablacion"].copy()
    if len(g):
        g["familia"] = g["familia"].fillna("BASELINE")
        base_f1 = g[g.familia == "BASELINE"].set_index(["lang", "pct", "seed"])[["f1_termino", "res_r_termino"]]
        g = g.join(base_f1, on=["lang", "pct", "seed"], rsuffix="_base")
        g["delta_f1_termino"] = g["f1_termino"] - g["f1_termino_base"]
        if "res_r_termino" in g:
            g["delta_recall_reservadas"] = g["res_r_termino"] - g["res_r_termino_base"]
        g.to_csv(OUT / "crf_ablacion_todas.csv", index=False)
        r = g.groupby(["lang", "pct", "familia"])[["delta_f1_termino", "delta_recall_reservadas", "f1_termino"]].agg(["mean", "std", "count"])
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r.reset_index().to_csv(OUT / "crf_ablacion_resumen.csv", index=False)
    g = df[df.exp == "ablacv"].copy()
    if len(g):
        g["familia"] = g["familia"].fillna("BASELINE")
        base_f1 = g[g.familia == "BASELINE"].set_index(["lang", "fold"])[["f1_termino"]]
        g = g.join(base_f1, on=["lang", "fold"], rsuffix="_base")
        g["delta_f1_termino"] = g["f1_termino"] - g["f1_termino_base"]
        g.to_csv(OUT / "crf_ablacv_todas.csv", index=False)
        r = g.groupby(["lang", "familia"])[["delta_f1_termino", "f1_termino"]].agg(["mean", "std", "count"])
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r.reset_index().to_csv(OUT / "crf_ablacv_resumen.csv", index=False)
    print(f"Merged {len(df)} runs -> {OUT}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    exp = args[0]
    if exp == "consolidar":
        consolidar()
        return
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
    langs = [args[args.index("--lang") + 1]] if "--lang" in args else ["en", "es"]
    ts = tareas(exp, langs, "--rapido" in args, "--ablacion-completa" in args, "--todos-niveles" in args)
    pendientes = [t for t in ts if not (OUT / "runs" / f"{run_id(t)}.json").exists()]
    print(f"{len(ts)} runs ({len(ts) - len(pendientes)} already done), {jobs} in parallel", flush=True)
    for t in pendientes:                      # load the data before creating the processes
        cargar(t["lang"], t["exp"] == "dos")
    t0 = time.time()
    if jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs) as pool:
            for i, (rid, msg) in enumerate(pool.imap_unordered(ejecutar, pendientes), 1):
                print(f"[{i}/{len(pendientes)}] {rid}: {msg}  ·  {(time.time()-t0)/60:.0f} min", flush=True)
    else:
        for i, t in enumerate(pendientes, 1):
            rid, msg = ejecutar(t)
            print(f"[{i}/{len(pendientes)}] {rid}: {msg}  ·  {(time.time()-t0)/60:.0f} min", flush=True)
    consolidar()


if __name__ == "__main__":
    main()
