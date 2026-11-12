"""
acter_experiments.py
───────────────
External validation on ACTER (English, four domains, annotation without named
entities) with the three sequence models and the same protocol as our corpus:
statistics and PMI from the training partition only, epoch chosen on validation
(never on the test set), same metrics.

  preparar   converts data/acter/acter_en_bio_tokens.csv (spaCy, produced by
             acter_preprocess.py) into $DATOS_GENERADOS/acter_bio_tokens_terms_only.csv,
             with "entrada" = the lowercased term form (the unit of reservation).
  lodo       leave-one-domain-out: training on three domains, testing on the
             fourth. The CRF is deterministic (one run); the BiLSTM-CRF and the
             Transformer use 5 seeds (0-4) per domain.

Type reservation on ACTER needs no extra code (lang = acter):
  python train_crf.py reserva --lang acter
  python train_bilstm_crf.py reserva --lang acter --semillas 0,1,2,3,4,5,6,7,8,9
  python train_xlmr.py reserva --lang acter

Usage:  python acter_experiments.py preparar [--origen PATH.csv]
        python acter_experiments.py lodo [--modelos crf,bilstm,transformer] [--semillas 0,1,2,3,4]
        python acter_experiments.py consolidar
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATOS = Path(os.environ.get("DATOS_GENERADOS", HERE.parent / "datos_generados"))
RES = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados"))
OUT = RES / "acter_lodo"
DOMINIOS = ["corp", "equi", "htfl", "wind"]
ORIGEN = HERE.parent.parent / "data" / "acter" / "acter_en_bio_tokens.csv"


def dominio(doc_id: str) -> str:
    return str(doc_id).split("_")[0]


# ─────────────────────────── preparar ───────────────────────────

def preparar(origen: Path):
    df = pd.read_csv(origen, low_memory=False)
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    df["token_lower"] = df["token_lower"].fillna("").astype(str)
    # entry = lowercased span form, assigned to all its tokens
    entrada = [""] * len(df)
    labs, toks = df["label_bio"].tolist(), df["token_lower"].tolist()
    keys = list(zip(df["doc_id"], df["sentence_id"]))
    i = 0
    while i < len(df):
        if labs[i] == "B-T":
            j = i + 1
            while j < len(df) and labs[j] == "I-T" and keys[j] == keys[i]:
                j += 1
            forma = " ".join(toks[i:j]).strip()
            for k in range(i, j):
                entrada[k] = forma
            i = j
        else:
            i += 1
    df["entrada"] = entrada
    df["language"] = "en"
    dest = DATOS / "acter_bio_tokens_terms_only.csv"
    DATOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    n_tipos = len({e for e in entrada if e})
    print(f"{dest}: {len(df)} tokens, {df.groupby(['doc_id','sentence_id']).ngroups} sentences, {n_tipos} types")


# ─────────────────────────── LODO ───────────────────────────

def _metricas_res(rows_entr, yt, yp, reservadas):
    """Metrics on the tokens of types absent from training (res_) and on the rest (vis_)."""
    from metrics import metricas_completas
    res_t, res_p, vis_t, vis_p = [], [], [], []
    for ents, a_seq, b_seq in zip(rows_entr, yt, yp):
        for e, a, b in zip(ents, a_seq, b_seq):
            (res_t if e and e in reservadas else vis_t).append(a)
            (res_p if e and e in reservadas else vis_p).append(b)
    out = {}
    for pref, a, b in (("res_", res_t, res_p), ("vis_", vis_t, vis_p)):
        if a:
            out.update({pref + k: v for k, v in metricas_completas(a, b).items()})
    return out


def lodo_crf():
    import crf_base as base
    from metrics import guardar_predicciones, metricas_completas
    import train_crf as p2
    out = OUT / "crf"
    for d in ("runs", "pred"):
        (out / d).mkdir(parents=True, exist_ok=True)
    sentences, _ = p2.cargar("acter", False)
    for dom in DOMINIOS:
        rid = f"lodo_acter_{dom}"
        dest = out / "runs" / f"{rid}.json"
        if dest.exists():
            print(f"  {rid}: already done"); continue
        t0 = time.time()
        train = [s for s in sentences if dominio(s["doc_id"]) != dom]
        test = [s for s in sentences if dominio(s["doc_id"]) == dom]
        _, _, y_te, y_pr = p2.entrenar_y_evaluar(train, test, "acter")
        vistos = {r.get("_span_text") for s in train for r in s["rows"] if r.get("_span_text")}
        nuevos = {r.get("_span_text") for s in test for r in s["rows"] if r.get("_span_text")} - vistos
        fila = {"exp": "lodo", "modelo": "crf", "dominio": dom, "seed": None,
                "n_oraciones_train": len(train), "n_oraciones_test": len(test), "n_tipos_nuevos": len(nuevos),
                **metricas_completas(y_te, y_pr),
                **_metricas_res([[r.get("_span_text") or "" for r in s["rows"]] for s in test], y_te, y_pr, nuevos),
                "segundos": round(time.time() - t0, 1)}
        guardar_predicciones(out / "pred" / f"{rid}.csv.gz", test, y_te, y_pr)
        dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
        print(f"[{rid}] CRF term F1={fila['f1_termino']:.3f}", flush=True)


def lodo_neuronal(modelo, semillas):
    """BiLSTM-CRF or Transformer, validated on reserved entries of the training domains."""
    if modelo == "bilstm":
        import train_bilstm_crf as m
        out = OUT / "bilstm"
    else:
        import train_xlmr as m
        out = OUT / "transformer" / m.MODELO.replace("/", "__")
    for d in ("runs", "pred", "epocas"):
        (out / d).mkdir(parents=True, exist_ok=True)
    m.OUT = out  # each module writes its output to m.OUT
    oraciones, _ = m.cargar("acter")
    for dom in DOMINIOS:
        train = [o for o in oraciones if dominio(o["doc_id"]) != dom]
        test = [o for o in oraciones if dominio(o["doc_id"]) == dom]
        vistos = {e for o in train for e in o["entradas"] if e}
        nuevos = {e for o in test for e in o["entradas"] if e} - vistos
        for seed in semillas:
            rid = f"lodo_acter_{dom}_s{seed}"
            if (out / "runs" / f"{rid}.json").exists():
                print(f"  {rid}: already done"); continue
            meta = {"exp": "lodo", "modelo": modelo, "dominio": dom, "seed": seed, "n_tipos_nuevos": len(nuevos)}
            if modelo == "bilstm":
                t0 = time.time()
                tr, va = m.separar_validacion(train, seed, por_entradas=True)
                net, vocab, info = m.entrenar(tr, va, seed, rid, m.MAX_EPOCAS)
                met = m.evaluar(net, vocab, test, nuevos, rid)
                fila = {**meta, "n_oraciones_train": len(tr), "n_oraciones_val": len(va), "n_oraciones_test": len(test),
                        **info, **met, "segundos": round(time.time() - t0, 1)}
                (out / "runs" / f"{rid}.json").write_text(json.dumps(fila, ensure_ascii=False, default=float))
                print(f"[{rid}] BiLSTM term F1={met['f1_termino']:.3f}", flush=True)
            else:
                m.ejecutar(rid, train, test, nuevos, seed, True, m.MAX_EPOCAS, meta, out)


def consolidar():
    for sub in sorted(OUT.rglob("runs")):
        filas = [json.loads(p.read_text()) for p in sorted(sub.glob("*.json"))]
        if filas:
            df = pd.DataFrame(filas)
            df.to_csv(sub.parent / "lodo_todas.csv", index=False)
            r = df.groupby("dominio")[["f1_termino", "p_termino", "r_termino"]].agg(["mean", "std", "count"])
            r.columns = [f"{a}_{b}" for a, b in r.columns]
            r.reset_index().to_csv(sub.parent / "lodo_resumen.csv", index=False)
            print(sub.parent.relative_to(OUT), "\n", r.round(3))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    if args[0] == "preparar":
        hecho = [p for p in (DATOS / "acter_bio_tokens_terms_only.csv", DATOS / "acter_bio_tokens_terms_only.csv.gz") if p.exists()]
        if hecho and "--origen" not in args:
            print(f"ACTER already prepared: {hecho[0].name}")
        else:
            preparar(Path(args[args.index("--origen") + 1]) if "--origen" in args else ORIGEN)
    elif args[0] == "lodo":
        modelos = args[args.index("--modelos") + 1].split(",") if "--modelos" in args else ["crf", "bilstm", "transformer"]
        semillas = [int(s) for s in args[args.index("--semillas") + 1].split(",")] if "--semillas" in args else [0, 1, 2, 3, 4]
        for mod in modelos:
            lodo_crf() if mod == "crf" else lodo_neuronal(mod, semillas)
        consolidar()
    elif args[0] == "consolidar":
        consolidar()


if __name__ == "__main__":
    main()
