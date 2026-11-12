"""
train_xlmr.py
─────────────────────
A pretrained Transformer (XLM-RoBERTa base by default) fine-tuned for BIO tagging,
with the same protocol as the CRF and the BiLSTM-CRF:

  · same partitions: reserved entries are drawn by train_bilstm_crf.particion_reserva
    (= crf_base.split_by_type_holdout);
  · same choice of epoch: validation on 10% of the term entries left in training
    (train_bilstm_crf.separar_validacion), early stopping, and a single evaluation
    of the test set;
  · same metrics (metrics.py), also restricted to the tokens of reserved entries
    (res_*) and to the rest (vis_*).

Fine-tuning follows the usual setup for token classification:
  · the label goes on the first subword of each word, the others get -100;
  · long sentences are split into chunks of TROZO words (no overlap) to stay under
    512 subwords, and the predictions are joined again;
  · AdamW, learning rate 3e-5, weight decay 0.01, 10% linear warm-up, batches of 16
    chunks grouped by length, gradient clipping at 1.0, mixed precision on CUDA;
  · at most MAX_EPOCAS epochs, patience PACIENCIA.

Environment variables: MODELO (e.g. xlm-roberta-base, roberta-base,
PlanTL-GOB-ES/roberta-base-bne), DISPOSITIVO, DATOS_GENERADOS, RESULTADOS, LR,
MAX_EPOCAS, PACIENCIA, SOBREMUESTREO (default 1: no oversampling).

Output in $RESULTADOS/transformer/<model>/: runs/<id>.json, pred/<id>.csv.gz and
epocas/<id>.csv. Runs that already have a JSON file are skipped.

Usage:  python train_xlmr.py reserva [--lang en] [--semillas 0,...,9] [--niveles 20,10,30,50]
        python train_xlmr.py cv [--lang en]
        python train_xlmr.py consolidar
        python train_xlmr.py reserva --rapido        (quick check: one level, one epoch, 300 sentences)
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import guardar_predicciones, metricas_completas  # noqa: E402
from train_bilstm_crf import particion_reserva, separar_validacion  # noqa: E402

MODELO = os.environ.get("MODELO", "xlm-roberta-base")
DATOS = Path(os.environ.get("DATOS_GENERADOS", HERE.parent / "datos_generados"))
OUT = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados")) / "transformer" / MODELO.replace("/", "__")
for d in ("runs", "pred", "epocas"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(os.environ.get("DISPOSITIVO") or ("cuda" if torch.cuda.is_available() else
                                                        "mps" if torch.backends.mps.is_available() else "cpu"))
LR = float(os.environ.get("LR", 3e-5))
MAX_EPOCAS = int(os.environ.get("MAX_EPOCAS", 6))
PACIENCIA = int(os.environ.get("PACIENCIA", 2))
SOBREMUESTREO = int(os.environ.get("SOBREMUESTREO", 1))
BATCH = int(os.environ.get("BATCH", 16))  # lower to 8 if the GPU runs out of memory
TROZO = 120
MAX_SUB = 512
NIVELES = [10, 20, 30, 50]
LABELS = {"O": 0, "B-T": 1, "I-T": 2}
INV = {v: k for k, v in LABELS.items()}


# ─────────────────────────── datos ───────────────────────────

def cargar(lang, path=None):
    """As train_bilstm_crf.cargar, but keeps the original case (the tokenizer uses it)."""
    cols = ["doc_id", "sentence_id", "token_id", "token", "label_bio", "entrada"]
    ruta = Path(path) if path else DATOS / f"{lang}_bio_tokens_terms_only.csv"
    if not ruta.exists() and Path(str(ruta) + ".gz").exists():  # datos comprimidos
        ruta = Path(str(ruta) + ".gz")
    df = pd.read_csv(ruta, usecols=cols, low_memory=False)
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    df["token"] = df["token"].fillna("").astype(str)
    df["entrada"] = df["entrada"].fillna("").astype(str)
    oraciones = []
    for (doc, sid), g in df.groupby(["doc_id", "sentence_id"], sort=False):
        toks = [t if t.strip() else "_" for t in g["token"].tolist()]  # empty or whitespace tokens
        oraciones.append({"doc_id": doc, "sentence_id": sid, "tokens": toks,
                          "labels": g["label_bio"].tolist(), "entradas": g["entrada"].tolist()})
    entradas = sorted({e for o in oraciones for e, l in zip(o["entradas"], o["labels"]) if e and l != "O"})
    return oraciones, np.array(entradas)


def trocear(oraciones):
    """List of chunks (sentence index, start, end) of at most TROZO words."""
    trozos = []
    for i, o in enumerate(oraciones):
        n = len(o["tokens"])
        for a in range(0, max(n, 1), TROZO):
            trozos.append((i, a, min(a + TROZO, n)))
    return trozos


def codificar(tok, oraciones, trozos, con_etiquetas=True):
    """Each chunk -> input_ids, aligned labels (-100 except on the first subword) and the
    position of the first subword of each word (-1 if it was truncated)."""
    datos = []
    truncadas = 0
    for (i, a, b) in trozos:
        palabras = oraciones[i]["tokens"][a:b]
        enc = tok(palabras, is_split_into_words=True, truncation=True, max_length=MAX_SUB)
        wids = enc.word_ids()
        primera = [-1] * len(palabras)
        etiq = [-100] * len(enc["input_ids"])
        prev = None
        for pos, w in enumerate(wids):
            if w is not None and w != prev:
                primera[w] = pos
                if con_etiquetas:
                    etiq[pos] = LABELS[oraciones[i]["labels"][a + w]]
            prev = w
        truncadas += sum(1 for p in primera if p < 0)
        datos.append({"ids": enc["input_ids"], "etiq": etiq, "primera": primera})
    return datos, truncadas


def lotes(datos, idx, pad_id):
    largo = max(len(datos[k]["ids"]) for k in idx)
    X = torch.full((len(idx), largo), pad_id, dtype=torch.long)
    Y = torch.full((len(idx), largo), -100, dtype=torch.long)
    M = torch.zeros((len(idx), largo), dtype=torch.long)
    for j, k in enumerate(idx):
        n = len(datos[k]["ids"])
        X[j, :n] = torch.tensor(datos[k]["ids"])
        Y[j, :n] = torch.tensor(datos[k]["etiq"])
        M[j, :n] = 1
    return X, Y, M


def orden_por_longitud(datos, idx, rng):
    """Groups chunks by length (less padding) and shuffles the batches."""
    idx = sorted(idx, key=lambda k: len(datos[k]["ids"]))
    bloques = [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]
    rng.shuffle(bloques)
    return bloques


# ─────────────────────────── model ───────────────────────────

def predecir(model, tok, oraciones):
    trozos = trocear(oraciones)
    datos, truncadas = codificar(tok, oraciones, trozos, con_etiquetas=False)
    preds = [["O"] * len(o["tokens"]) for o in oraciones]
    model.eval()
    idx = sorted(range(len(datos)), key=lambda k: len(datos[k]["ids"]))
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bloque = idx[s:s + 64]
            X, _, M = lotes(datos, bloque, tok.pad_token_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(input_ids=X.to(DEVICE), attention_mask=M.to(DEVICE)).logits
            arg = logits.argmax(-1).cpu().numpy()
            for j, k in enumerate(bloque):
                i, a, _ = trozos[k]
                for w, pos in enumerate(datos[k]["primera"]):
                    if pos >= 0:
                        preds[i][a + w] = INV[int(arg[j, pos])]
    return preds, truncadas


def entrenar(train, val, seed, rid, max_epocas, tok):
    from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        MODELO, num_labels=3, id2label=INV, label2id=LABELS).to(DEVICE)
    trozos = trocear(train)
    dtr, _ = codificar(tok, train, trozos)
    con_t = [k for k, d in enumerate(dtr) if any(e > 0 for e in d["etiq"])]
    base_idx = list(range(len(dtr))) + con_t * (SOBREMUESTREO - 1)
    pasos = math.ceil(len(base_idx) / BATCH) * max_epocas
    sin_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grupos = [{"params": [p for n, p in model.named_parameters() if not any(s in n for s in sin_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(s in n for s in sin_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(grupos, lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    mejor_f1, mejor_estado, mejor_epoca, sin_mejora, log = -1.0, None, 0, 0, []
    for epoca in range(1, max_epocas + 1):
        t0 = time.time()
        model.train()
        perdidas = []
        for bloque in orden_por_longitud(dtr, base_idx, rng):
            X, Y, M = lotes(dtr, bloque, tok.pad_token_id)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = model(input_ids=X.to(DEVICE), attention_mask=M.to(DEVICE), labels=Y.to(DEVICE)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            perdidas.append(loss.item())
        yp, _ = predecir(model, tok, val)
        mv = metricas_completas([o["labels"] for o in val], yp)
        log.append({"epoca": epoca, "perdida_media": float(np.mean(perdidas)),
                    **{f"val_{k}": v for k, v in mv.items()}, "segundos": round(time.time() - t0, 1)})
        pd.DataFrame(log).to_csv(OUT / "epocas" / f"{rid}.csv", index=False)
        print(f"    epoch {epoca}: loss={np.mean(perdidas):.3f}  val term F1={mv['f1_termino']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if mv["f1_termino"] > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = mv["f1_termino"], epoca, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    model.load_state_dict(mejor_estado)
    return model, {"epoca_elegida": mejor_epoca, "epocas_entrenadas": epoca, "val_f1_termino_mejor": mejor_f1}


def evaluar(model, tok, test, reservadas, rid, out=None):
    out = out or OUT
    yp, truncadas = predecir(model, tok, test)
    yt = [o["labels"] for o in test]
    m = metricas_completas(yt, yp)
    m["palabras_truncadas"] = truncadas
    if reservadas is not None:
        res_t, res_p, vis_t, vis_p = [], [], [], []
        for o, a_seq, b_seq in zip(test, yt, yp):
            for e, a, b in zip(o["entradas"], a_seq, b_seq):
                (res_t if e in reservadas else vis_t).append(a)
                (res_p if e in reservadas else vis_p).append(b)
        for pref, a, b in (("res_", res_t, res_p), ("vis_", vis_t, vis_p)):
            if a:
                m.update({pref + k: v for k, v in metricas_completas(a, b).items()})
    claves = [{"doc_id": o["doc_id"], "sentence_id": o["sentence_id"]} for o in test]
    guardar_predicciones(out / "pred" / f"{rid}.csv.gz", claves, yt, yp)
    return m


def ejecutar(rid, train, test, reservadas, seed, por_entradas, max_epocas, meta, out=None):
    """Trains with validation, evaluates the test set once and writes the JSON file."""
    from transformers import AutoTokenizer
    out = out or OUT
    dest = out / "runs" / f"{rid}.json"
    if dest.exists():
        print(f"  {rid}: already done", flush=True)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODELO, add_prefix_space=True) if "roberta" in MODELO.lower() and "xlm" not in MODELO.lower() \
        else AutoTokenizer.from_pretrained(MODELO)
    train, val = separar_validacion(train, seed, por_entradas=por_entradas)
    print(f"\n[{rid}] train={len(train)} val={len(val)} test={len(test)} sentences · {MODELO} · {DEVICE}", flush=True)
    model, info = entrenar(train, val, seed, rid, max_epocas, tok)
    m = evaluar(model, tok, test, reservadas, rid, out)
    fila = {**meta, "modelo": MODELO, "n_oraciones_train": len(train), "n_oraciones_val": len(val),
            "n_oraciones_test": len(test), "n_reservadas": len(reservadas) if reservadas else None,
            **info, **m, "segundos": round(time.time() - t0, 1)}
    dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{rid}] term F1={m['f1_termino']:.3f} (epoch {info['epoca_elegida']}) · {(time.time()-t0)/60:.1f} min", flush=True)


def run(lang, exp, datos, pct=None, seed=0, fold=None, max_epocas=MAX_EPOCAS):
    oraciones, entradas = datos
    if exp == "reserva":
        rid = f"reserva_{lang}_p{pct}_s{seed}"
        train, test, reservadas = particion_reserva(oraciones, entradas, pct, seed)
    else:
        from sklearn.model_selection import StratifiedKFold
        rid = f"cv_{lang}_f{fold}"
        y = np.array([any(l != "O" for l in o["labels"]) for o in oraciones], dtype=int)
        tr, te = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[fold - 1]
        train, test, reservadas = [oraciones[i] for i in tr], [oraciones[i] for i in te], None
    meta = {"exp": exp, "lang": lang, "pct": pct, "seed": seed, "fold": fold, "n_entradas_sorteables": len(entradas)}
    ejecutar(rid, train, test, reservadas, seed, exp == "reserva", max_epocas, meta)


def consolidar(out=None):
    out = out or OUT
    filas = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    if not filas:
        print("No results yet.")
        return
    df = pd.DataFrame(filas)
    for exp, g in df.groupby("exp"):
        g.to_csv(out / f"transformer_{exp}_todas.csv", index=False)
    g = df[df.exp == "reserva"]
    if len(g):
        num = [c for c in g.columns if g[c].dtype != object and c not in ("pct", "seed", "fold")]
        r = g.groupby(["lang", "pct"])[num].agg(["mean", "std", "count"])
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r.reset_index().to_csv(out / "transformer_reserva_resumen.csv", index=False)
    print(f"Merged {len(df)} runs -> {out}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "consolidar":
        consolidar()
        return
    exp = args[0]
    rapido = "--rapido" in args
    langs = [args[args.index("--lang") + 1]] if "--lang" in args else ["en", "es"]
    semillas = [int(s) for s in args[args.index("--semillas") + 1].split(",")] if "--semillas" in args else list(range(10))
    niveles = [int(s) for s in args[args.index("--niveles") + 1].split(",")] if "--niveles" in args else NIVELES
    max_ep = int(os.environ.get("EPOCAS_RAPIDO", 1)) if rapido else MAX_EPOCAS
    for lang in langs:
        datos = cargar(lang)
        if rapido:
            datos = (datos[0][:300] + [o for o in datos[0][300:] if any(l != "O" for l in o["labels"])][:300], datos[1])
        print(f"{lang}: {len(datos[0])} sentences, {len(datos[1])} entries that can be reserved", flush=True)
        if exp == "reserva":
            # Level by level, then seed by seed: if interrupted, every level has its lowest seeds done.
            for pct in ([20] if rapido else niveles):
                for seed in (semillas[:1] if rapido else semillas):
                    run(lang, "reserva", datos, pct=pct, seed=seed, max_epocas=max_ep)
        elif exp == "cv":
            for fold in ([1] if rapido else range(1, 6)):
                run(lang, "cv", datos, seed=42, fold=fold, max_epocas=max_ep)
    consolidar()


if __name__ == "__main__":
    main()
