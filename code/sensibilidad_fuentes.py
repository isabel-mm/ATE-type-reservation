"""
sensibilidad_fuentes.py
───────────────────────
Sensibilidad del CRF a la composición desigual del corpus (unos pocos libros
concentran más de la mitad de los tokens). Con las mismas particiones de la
reserva de tipos (10 semillas, 4 niveles), el conjunto de prueba no cambia y el
de entrenamiento se reduce de dos maneras con el mismo número de oraciones:

  equilibrado   como máximo K oraciones por documento (K = mediana de oraciones
                por documento del corpus), elegidas al azar: cada documento pesa
                lo mismo, como mucho, y los libros dejan de dominar
  aleatorio     el mismo número total de oraciones, elegidas al azar sin tener
                en cuenta el documento (controla el efecto de tener menos datos)

Si equilibrado ≈ aleatorio, la composición del corpus no condiciona los
resultados más allá del volumen de datos.

Uso:  python sensibilidad_fuentes.py [--jobs 4] [--lang en]
Salida: resultados_fuentes/runs/*.json y resultados_provisionales/csv/16_sensibilidad_fuentes.csv
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
import paso2_crf as P  # noqa: E402
from metricas import metricas_completas  # noqa: E402

OUT = AQUI / "resultados_fuentes" / "runs"
OUT.mkdir(parents=True, exist_ok=True)
CSV = AQUI / "resultados_provisionales" / "csv" / "16_sensibilidad_fuentes.csv"
K = {}   # lang -> tope de oraciones por documento


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
    # la condición completa sale de la reserva principal ya ejecutada
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
        print(f"{l}: tope de {tope(l)} oraciones por documento", flush=True)
    print(f"{len(ts)} ejecuciones ({len(ts) - len(pend)} ya hechas), {jobs} en paralelo", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs) as pool:
        for i, (rid, msg) in enumerate(pool.imap_unordered(ejecutar, pend), 1):
            print(f"[{i}/{len(pend)}] {rid}: {msg}  ·  {(time.time()-t0)/60:.0f} min", flush=True)
    consolidar()


if __name__ == "__main__":
    main()
