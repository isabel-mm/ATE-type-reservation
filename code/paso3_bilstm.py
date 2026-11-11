"""
paso3_bilstm.py
───────────────
Modelo neuronal (BiLSTM-CRF) con la arquitectura de la tesis
(bilstm_crf_deploy/bilstm_crf_span_completo.py): embeddings de palabra de
128 dimensiones entrenados desde cero, BiLSTM de 256 unidades por dirección,
capa CRF, Adam (lr 0,001), lotes de 16 oraciones, sobremuestreo ×4 de las
oraciones con término y recorte de gradiente a 5,0.

Cambio de protocolo: la época ya no se elige mirando el conjunto de prueba ni
se toma la última sin más. Dentro del entrenamiento se reserva un 10 % de las
entradas de término (sus oraciones forman la validación, con términos que el
modelo no ve al entrenar, igual que en la prueba); tras cada época se mide el
F1 de término en validación, se guarda el mejor modelo y se para cuando lleva PACIENCIA épocas
sin mejorar (máximo 50). El conjunto de prueba se evalúa una sola vez, con el
mejor modelo según validación.

Experimentos:
  reserva   reserva de entradas 10/20/30/50 %, semillas de partición 0, 1, 2
            (las mismas particiones que las tres primeras semillas del CRF)
  cv        validación cruzada estándar de 5 pliegues (particiones aleatorias)

Guarda en $RESULTADOS/bilstm/: runs/<id>.json (todas las métricas, también
restringidas a entradas reservadas res_* y vistas vis_*), pred/<id>.csv.gz
(predicciones token a token) y epocas/<id>.csv (pérdida y métricas de
validación por época). Reanudable.

Uso:  python paso3_bilstm.py reserva [--lang en] [--semillas 0,1,2] [--niveles 20,10,30,50]
      python paso3_bilstm.py cv
      python paso3_bilstm.py consolidar
      python paso3_bilstm.py reserva --rapido      (prueba: 1 nivel, 2 épocas)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchcrf import CRF as TorchCRF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metricas import guardar_predicciones, metricas_completas  # noqa: E402

DATOS = Path(os.environ.get("DATOS_GENERADOS", HERE.parent / "datos_generados"))
OUT = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados")) / "bilstm"
for d in ("runs", "pred", "epocas"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# Dispositivo: CUDA (Colab) > MPS (Mac) > CPU; DISPOSITIVO=cpu|mps|cuda lo fuerza.
DEVICE = torch.device(os.environ.get("DISPOSITIVO") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
OVERSAMPLE = 4
BATCH = 16
MAX_EPOCAS = 50
PACIENCIA = 5
VAL_FRAC = 0.10
NIVELES = [10, 20, 30, 50]
LABELS = {"O": 0, "B-T": 1, "I-T": 2}
INV = {v: k for k, v in LABELS.items()}


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_tags)
        self.crf = TorchCRF(num_tags, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        # Las oraciones de un lote se rellenan hasta la más larga; se empaquetan para
        # que la LSTM (sobre todo en la dirección derecha-izquierda) ignore el relleno.
        largos = mask.sum(1).cpu()
        emb = nn.utils.rnn.pack_padded_sequence(self.embedding(x), largos, batch_first=True, enforce_sorted=False)
        salida, _ = self.lstm(emb)
        salida, _ = nn.utils.rnn.pad_packed_sequence(salida, batch_first=True, total_length=x.size(1))
        emissions = self.fc(salida)
        if tags is not None:
            # Pérdida media por oración (REDUCCION=sum recupera la suma de la versión anterior)
            return -self.crf(emissions, tags, mask=mask, reduction=os.environ.get("REDUCCION", "mean"))
        return self.crf.decode(emissions, mask=mask)


def cargar(lang):
    cols = ["doc_id", "sentence_id", "token_id", "token", "label_bio", "entrada"]
    ruta = DATOS / f"{lang}_bio_tokens_terms_only.csv"
    if not ruta.exists() and Path(str(ruta) + ".gz").exists():  # datos comprimidos
        ruta = Path(str(ruta) + ".gz")
    df = pd.read_csv(ruta, usecols=cols, low_memory=False)
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    df["token"] = df["token"].fillna("").astype(str)
    df["entrada"] = df["entrada"].fillna("").astype(str)
    oraciones = []
    for (doc, sid), g in df.groupby(["doc_id", "sentence_id"], sort=False):
        oraciones.append({"doc_id": doc, "sentence_id": sid,
                          "tokens": g["token"].str.lower().tolist(),
                          "labels": g["label_bio"].tolist(),
                          "entradas": g["entrada"].tolist()})
    entradas = sorted({e for o in oraciones for e, l in zip(o["entradas"], o["labels"]) if e and l != "O"})
    return oraciones, np.array(entradas)


def particion_reserva(oraciones, entradas, pct, seed):
    """Mismo sorteo que el CRF (crf_base.split_by_type_holdout): pd.Series.sample."""
    n = max(1, int(len(entradas) * pct / 100))
    reservadas = set(pd.Series(entradas).sample(n=n, random_state=seed).values)
    train, test = [], []
    for o in oraciones:
        (test if reservadas & set(o["entradas"]) else train).append(o)
    return train, test, reservadas


def codificar(oraciones, vocab):
    """Cada oración completa como dos listas de enteros (tokens y etiquetas), sin recortar."""
    return [(np.array([vocab.get(t, 1) for t in o["tokens"]], dtype=np.int64),
             np.array([LABELS[l] for l in o["labels"]], dtype=np.int64)) for o in oraciones]


def lote(datos, idx):
    """Rellena un lote hasta la longitud de su oración más larga."""
    largo = max(len(datos[i][0]) for i in idx)
    X = np.zeros((len(idx), largo), dtype=np.int64)
    Y = np.zeros((len(idx), largo), dtype=np.int64)
    M = np.zeros((len(idx), largo), dtype=bool)
    for k, i in enumerate(idx):
        n = len(datos[i][0])
        X[k, :n], Y[k, :n], M[k, :n] = datos[i][0], datos[i][1], True
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


def predecir(model, datos):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(datos), 64):
            X, _, M = lote(datos, range(i, min(i + 64, len(datos))))
            preds.extend(model(X.to(DEVICE), mask=M.to(DEVICE)))
    return preds


def secuencias(oraciones, preds):
    """Etiquetas gold y predichas por oración completa."""
    yt, yp = [], []
    for o, p in zip(oraciones, preds):
        yt.append(o["labels"])
        yp.append([INV[int(x)] for x in p])
    return yt, yp


def entrenar(train, val, seed, rid, max_epocas):
    torch.manual_seed(seed)
    np.random.seed(seed)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for o in train:
        for t in o["tokens"]:
            vocab.setdefault(t, len(vocab))
    dtr = codificar(train, vocab)
    dva = codificar(val, vocab)
    con_t = np.array([any(l != "O" for l in o["labels"]) for o in train])
    idx_t, idx_o = np.where(con_t)[0], np.where(~con_t)[0]

    model = BiLSTMCRF(len(vocab)).to(DEVICE)
    opt = Adam(model.parameters(), lr=0.001)
    mejor_f1, mejor_estado, mejor_epoca, sin_mejora = -1.0, None, 0, 0
    log = []
    for epoca in range(1, max_epocas + 1):
        t0 = time.time()
        model.train()
        orden = np.concatenate([idx_o] + [idx_t] * OVERSAMPLE)
        np.random.shuffle(orden)
        perdidas = []
        for i in range(0, len(orden), BATCH):
            X, Y, M = lote(dtr, orden[i:i + BATCH])
            opt.zero_grad()
            loss = model(X.to(DEVICE), Y.to(DEVICE), M.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            perdidas.append(loss.item())
        yt, yp = secuencias(val, predecir(model, dva))
        mv = metricas_completas(yt, yp)
        log.append({"epoca": epoca, "perdida_media": float(np.mean(perdidas)),
                    **{f"val_{k}": v for k, v in mv.items()}, "segundos": round(time.time() - t0, 1)})
        pd.DataFrame(log).to_csv(OUT / "epocas" / f"{rid}.csv", index=False)
        print(f"    época {epoca}: pérdida={np.mean(perdidas):.2f}  val F1 término={mv['f1_termino']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if mv["f1_termino"] > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = mv["f1_termino"], epoca, 0
            mejor_estado = copy.deepcopy(model.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    model.load_state_dict(mejor_estado)
    return model, vocab, {"epoca_elegida": mejor_epoca, "epocas_entrenadas": epoca, "val_f1_termino_mejor": mejor_f1}


def separar_validacion(train, seed, por_entradas):
    """Validación para elegir la época.
    · Reserva (por_entradas=True): se reserva a su vez un 10 % de las entradas de
      término presentes en el entrenamiento; sus oraciones forman la validación.
      Así la validación mide lo mismo que la prueba, el reconocimiento de términos
      no vistos, y la parada no premia la memorización.
    · Validación cruzada estándar: 10 % de oraciones al azar, como su prueba."""
    rng = np.random.RandomState(seed)
    if not por_entradas:
        idx = rng.permutation(len(train))
        n_val = max(1, int(len(train) * VAL_FRAC))
        return [train[i] for i in idx[n_val:]], [train[i] for i in idx[:n_val]]
    ents = sorted({e for o in train for e, l in zip(o["entradas"], o["labels"]) if e and l != "O"})
    n_val = max(1, int(len(ents) * VAL_FRAC))
    val_ents = set(rng.choice(ents, size=n_val, replace=False))
    tr, va = [], []
    for o in train:
        (va if val_ents & set(o["entradas"]) else tr).append(o)
    return tr, va


def evaluar(model, vocab, test, reservadas, rid):
    yt, yp = secuencias(test, predecir(model, codificar(test, vocab)))
    m = metricas_completas(yt, yp)
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
    guardar_predicciones(OUT / "pred" / f"{rid}.csv.gz", claves, yt, yp)
    return m


def run(lang, exp, pct=None, seed=0, fold=None, max_epocas=MAX_EPOCAS, datos=None):
    rid = f"reserva_{lang}_p{pct}_s{seed}" if exp == "reserva" else f"cv_{lang}_f{fold}"
    dest = OUT / "runs" / f"{rid}.json"
    if dest.exists():
        print(f"  {rid}: ya hecho", flush=True)
        return
    t0 = time.time()
    oraciones, entradas = datos
    if exp == "reserva":
        train, test, reservadas = particion_reserva(oraciones, entradas, pct, seed)
    else:
        from sklearn.model_selection import StratifiedKFold
        y = np.array([any(l != "O" for l in o["labels"]) for o in oraciones], dtype=int)
        tr, te = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[fold - 1]
        train, test, reservadas = [oraciones[i] for i in tr], [oraciones[i] for i in te], None
    train, val = separar_validacion(train, seed, por_entradas=(exp == "reserva"))
    print(f"\n[{rid}] train={len(train)} val={len(val)} test={len(test)} oraciones · {DEVICE}", flush=True)
    model, vocab, info = entrenar(train, val, seed, rid, max_epocas)
    m = evaluar(model, vocab, test, reservadas, rid)
    fila = {"exp": exp, "lang": lang, "pct": pct, "seed": seed, "fold": fold,
            "n_oraciones_train": len(train), "n_oraciones_val": len(val), "n_oraciones_test": len(test),
            "n_entradas_sorteables": len(entradas), "n_reservadas": len(reservadas) if reservadas else None,
            **info, **m, "segundos": round(time.time() - t0, 1)}
    dest.write_text(json.dumps(fila, ensure_ascii=False, default=float))
    # Libera la memoria del modelo antes de la siguiente ejecución (en MPS se acumulaba)
    import gc
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{rid}] F1 término={m['f1_termino']:.3f} (época {info['epoca_elegida']}) · "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


def consolidar():
    filas = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    if not filas:
        print("No hay resultados todavía.")
        return
    df = pd.DataFrame(filas)
    for exp, g in df.groupby("exp"):
        g.to_csv(OUT / f"bilstm_{exp}_todas.csv", index=False)
    g = df[df.exp == "reserva"]
    if len(g):
        num = [c for c in g.columns if g[c].dtype != object and c not in ("pct", "seed", "fold")]
        r = g.groupby(["lang", "pct"])[num].agg(["mean", "std", "count"])
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r.reset_index().to_csv(OUT / "bilstm_reserva_resumen.csv", index=False)
    print(f"Consolidado: {len(df)} ejecuciones -> {OUT}")


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
    semillas = [int(s) for s in args[args.index("--semillas") + 1].split(",")] if "--semillas" in args else [0, 1, 2]
    max_ep = int(args[args.index("--epocas") + 1]) if "--epocas" in args else (2 if rapido else MAX_EPOCAS)
    niveles = [int(x) for x in args[args.index("--niveles") + 1].split(",")] if "--niveles" in args else NIVELES
    for lang in langs:
        datos = cargar(lang)
        print(f"{lang}: {len(datos[0])} oraciones, {len(datos[1])} entradas de término sorteables", flush=True)
        if exp == "reserva":
            for seed in (semillas[:1] if rapido else semillas):
                for pct in ([20] if rapido else niveles):
                    run(lang, "reserva", pct=pct, seed=seed, max_epocas=max_ep, datos=datos)
        elif exp == "cv":
            for fold in ([1] if rapido else range(1, 6)):
                run(lang, "cv", seed=42, fold=fold, max_epocas=max_ep, datos=datos)
    consolidar()


if __name__ == "__main__":
    main()
