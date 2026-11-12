"""
source_sensitivity.py
───────────────────────
Sensitivity of the CRF to the uneven composition of the corpus (a few books hold more
than half of the tokens). With the same type-reservation partitions (10 seeds, 4 levels),
the test set stays the same and the training set is reduced in two ways, to the same
number of sentences:

  equilibrado   at most K sentences per document (K = median number of sentences per
                document), drawn at random, so that books no longer dominate
  aleatorio     the same total number of sentences, drawn at random regardless of the
                document (controls for having less data)

If equilibrado ≈ aleatorio, the composition of the corpus does not affect the results
beyond the amount of data.

Usage:   python source_sensitivity.py [--jobs 4] [--lang en]
Output:  resultados_fuentes/runs/*.json and results/16_sensibilidad_fuentes.csv
"""
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[2]
sys.path.insert(0, str(REPO / "colab_reanotacion" / "scripts"))
import crf_base as base  # noqa: E402
import train_crf as P  # noqa: E402
from metrics import metricas_completas  # noqa: E402

OUT = AQUI / "resultados_fuentes" / "runs"
OUT.mkdir(parents=True, exist_ok=True)
CSV = AQUI.parent / "results" / "16_sensibilidad_fuentes.csv"
K = {}   # lang -> maximum number of sentences per document


def tope(lang):
    if lang not in K:
        sentences, _ = P.cargar(lang, False)
        por_doc = pd.Series([s["doc_id"] for s in sentences]).value_counts()
        K[lang] = int(por_doc.median())
    return K[lang]


def reducir(train, modo, lang, seed):
    rng = random.Random(1000 + seed)
    k = tope(lang)
    por_doc = {}
    for s in train:
        por_doc.setdefault(s["doc_id"], []).append(s)
    equilibrado = []
    for doc in sorted(por_doc):
        ss = por_doc[doc]
        equilibrado += ss if len(ss) <= k else rng.sample(ss, k)
    if modo == "equilibrado":
        return equilibrado
    return random.Random(2000 + seed).sample(train, len(equilibrado))


def ejecutar(t):
    rid = f"{t['modo']}_{t['lang']}_p{t['pct']}_s{t['seed']}"
    dest = OUT / f"{rid}.json"
    if dest.exists():
        return rid, "ya hecho"
    t0 = time.time()
    sentences, span_types = P.cargar(t["lang"], False)
    train, test, reservadas = base.split_by_type_holdout(sentences, span_types, t["pct"], seed=t["seed"])
    train_r = reducir(train, t["modo"], t["lang"], t["seed"])
    crf, X_te, y_te, y_pr = P.entrenar_y_evaluar(train_r, test, t["lang"])
    docs = pd.Series([s["doc_id"] for s in train_r]).value_counts()
    fila = {**t, "k_por_documento": tope(t["lang"]), "oraciones_train_completo": len(train),
            "oraciones_train": len(train_r), "tokens_train": sum(len(s["rows"]) for s in train_r),
            "cuota_5_docs_mayores": round(docs.nlargest(5).sum() / docs.sum(), 3)}
    fila.update(metricas_completas(y_te, y_pr))
    fila.update(P.metricas_restringidas(test, y_te, y_pr, reservadas))
    fila["segundos"] = round(time.time() - t0, 1)
    dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
    return rid, f"F1 término={fila['f1_termino']:.3f}, rec. reservados={fila.get('res_r_termino', float('nan')):.3f} ({fila['segundos']:.0f}s)"


def consolidar():
    filas = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    df = pd.DataFrame(filas)
    # the full condition comes from the main reservation runs
    comp = []
    for f in (P.OUT / "runs").glob("reserva_*_todas.json"):
        r = json.loads(f.read_text())
        r["modo"] = "completo"
        comp.append(r)
    df = pd.concat([df, pd.DataFrame(comp)], ignore_index=True)
    df.to_csv(CSV, index=False)
    cols = [c for c in ("f1_termino", "res_r_termino", "vis_r_termino") if c in df.columns]
    print(df.groupby(["lang", "pct", "modo"])[cols].agg(["mean", "std"]).round(3).to_string())


def main():
    args = sys.argv[1:]
    if args and args[0] == "consolidar":
        consolidar()
        return
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
    langs = [args[args.index("--lang") + 1]] if "--lang" in args else ["en", "es"]
    ts = [{"modo": m, "lang": l, "pct": p, "seed": s}
          for l in langs for p in (20, 50, 10, 30) for s in range(10) for m in ("equilibrado", "aleatorio")]
    pend = [t for t in ts if not (OUT / f"{t['modo']}_{t['lang']}_p{t['pct']}_s{t['seed']}.json").exists()]
    for l in langs:
        print(f"{l}: at most {tope(l)} sentences per document", flush=True)
    print(f"{len(ts)} runs ({len(ts) - len(pend)} already done), {jobs} in parallel", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs) as pool:
        for i, (rid, msg) in enumerate(pool.imap_unordered(ejecutar, pend), 1):
            print(f"[{i}/{len(pend)}] {rid}: {msg}  ·  {(time.time()-t0)/60:.0f} min", flush=True)
    consolidar()


if __name__ == "__main__":
    main()
