"""
metricas.py
───────────
Todas las métricas de evaluación, con sus recuentos, para cualquier modelo:

  · por etiqueta (B-T, I-T, O, y B-E/I-E si existen): TP, FP, FN, TN, soporte,
    precisión, recuperación y F1;
  · agrupado término frente a palabra (B-T/I-T frente a todo lo demás): TP, FP,
    FN, TN, precisión, recuperación, F1, exactitud y especificidad. Es el
    «F1 específico de término» de la tesis (f1_termino);
  · medias macro y ponderada sobre las etiquetas presentes (f1_global =
    ponderada, como en la tesis), y exactitud por token;
  · a nivel de span de término: coincidencia exacta (mismo inicio y fin) y
    parcial (solapamiento), con TP, FP, FN, precisión, recuperación y F1;
    además, el desglose de los spans gold en exacto / parcial / perdido y el
    número de spans predichos espurios.

Todas las claves llevan nombres explícitos (tp_bt, fp_termino, f1_span_exacto…)
para poder volcarse tal cual a un CSV.

Además, guardar_predicciones() escribe las etiquetas gold y predichas token a
token, para recalcular después cualquier métrica sin volver a entrenar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ETIQUETAS_BASE = ["B-T", "I-T", "O"]
TAG = {"B-T": "bt", "I-T": "it", "O": "o", "B-E": "be", "I-E": "ie"}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _es_secuencias(seqs) -> bool:
    return len(seqs) > 0 and not isinstance(seqs[0], str)


def _aplanar(seqs):
    if _es_secuencias(seqs):
        return [str(x) for s in seqs for x in s]
    return [str(x) for x in seqs]


def spans_termino(labels):
    """Spans de término (B-T I-T*) de una secuencia: [(inicio, fin_exclusivo)]."""
    spans, start = [], None
    for i, lab in enumerate(labels):
        if lab == "B-T":
            if start is not None:
                spans.append((start, i))
            start = i
        elif lab == "I-T":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def metricas_span(y_true_seqs, y_pred_seqs) -> dict:
    tp_ex = fp_ex = fn_ex = 0
    tp_pa_gold = tp_pa_pred = n_gold = n_pred = 0
    exacto = parcial = perdido = espurio = 0
    for yt, yp in zip(y_true_seqs, y_pred_seqs):
        g, p = spans_termino(list(yt)), spans_termino(list(yp))
        gs, ps = set(g), set(p)
        tp_ex += len(gs & ps)
        fp_ex += len(ps - gs)
        fn_ex += len(gs - ps)
        n_gold += len(g)
        n_pred += len(p)
        for (a, b) in g:
            solapa = [(c, d) for (c, d) in p if c < b and d > a]
            if not solapa:
                perdido += 1
            elif (a, b) in ps:
                exacto += 1
            else:
                parcial += 1
            if solapa:
                tp_pa_gold += 1
        for (c, d) in p:
            if any(a < d and b > c for (a, b) in g):
                tp_pa_pred += 1
            else:
                espurio += 1
    p_ex, r_ex, f_ex = _prf(tp_ex, fp_ex, fn_ex)
    p_pa = tp_pa_pred / n_pred if n_pred else 0.0
    r_pa = tp_pa_gold / n_gold if n_gold else 0.0
    f_pa = 2 * p_pa * r_pa / (p_pa + r_pa) if p_pa + r_pa else 0.0
    return {
        "n_spans_gold": n_gold, "n_spans_pred": n_pred,
        "tp_span_exacto": tp_ex, "fp_span_exacto": fp_ex, "fn_span_exacto": fn_ex,
        "p_span_exacto": p_ex, "r_span_exacto": r_ex, "f1_span_exacto": f_ex,
        "tp_span_parcial_gold": tp_pa_gold, "tp_span_parcial_pred": tp_pa_pred,
        "p_span_parcial": p_pa, "r_span_parcial": r_pa, "f1_span_parcial": f_pa,
        "spans_gold_exactos": exacto, "spans_gold_parciales": parcial,
        "spans_gold_perdidos": perdido, "spans_pred_espurios": espurio,
    }


def metricas_completas(y_true, y_pred) -> dict:
    """y_true / y_pred: listas de secuencias (o listas planas) de etiquetas."""
    seqs = _es_secuencias(y_true)
    if seqs:
        y_true = [[str(x) for x in s] for s in y_true]
        y_pred = [[str(x) for x in s] for s in y_pred]
    yt, yp = _aplanar(y_true), _aplanar(y_pred)
    n = len(yt)
    etiquetas = ETIQUETAS_BASE + [l for l in ("B-E", "I-E") if l in set(yt) | set(yp)]
    out = {"n_tokens": n}
    sum_f = sum_p = sum_r = 0.0
    w_f = w_p = w_r = 0.0
    for lab in etiquetas:
        t = TAG[lab]
        tp = sum(1 for a, b in zip(yt, yp) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(yt, yp) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(yt, yp) if a == lab and b != lab)
        tn = n - tp - fp - fn
        p, r, f = _prf(tp, fp, fn)
        sop = tp + fn
        out.update({f"tp_{t}": tp, f"fp_{t}": fp, f"fn_{t}": fn, f"tn_{t}": tn, f"soporte_{t}": sop,
                    f"p_{t}": p, f"r_{t}": r, f"f1_{t}": f})
        sum_f += f; sum_p += p; sum_r += r
        w_f += f * sop; w_p += p * sop; w_r += r * sop
    k = len(etiquetas)
    out.update({"p_macro": sum_p / k, "r_macro": sum_r / k, "f1_macro": sum_f / k,
                "p_global": w_p / n if n else 0.0, "r_global": w_r / n if n else 0.0,
                "f1_global": w_f / n if n else 0.0,
                "exactitud_token": sum(1 for a, b in zip(yt, yp) if a == b) / n if n else 0.0})

    es_t = lambda x: x in ("B-T", "I-T")  # noqa: E731
    tp = sum(1 for a, b in zip(yt, yp) if es_t(a) and es_t(b))
    fp = sum(1 for a, b in zip(yt, yp) if not es_t(a) and es_t(b))
    fn = sum(1 for a, b in zip(yt, yp) if es_t(a) and not es_t(b))
    tn = n - tp - fp - fn
    p, r, f = _prf(tp, fp, fn)
    pn, rn, fn_ = _prf(tn, fn, fp)  # clase «palabra»
    out.update({"tp_termino": tp, "fp_termino": fp, "fn_termino": fn, "tn_termino": tn,
                "p_termino": p, "r_termino": r, "f1_termino": f,
                "p_palabra": pn, "r_palabra": rn, "f1_palabra": fn_,
                "exactitud_binaria": (tp + tn) / n if n else 0.0,
                "especificidad_termino": tn / (tn + fp) if tn + fp else 0.0})
    if seqs:
        out.update(metricas_span(y_true, y_pred))
    return out


def guardar_predicciones(path: Path, sentences_or_keys, y_true, y_pred, extra: dict | None = None):
    """Escribe un CSV con una fila por token: doc, oración, posición, token, gold, pred
    (+ columnas constantes de `extra`, p. ej. lengua, nivel y semilla)."""
    filas = []
    for s, yt, yp in zip(sentences_or_keys, y_true, y_pred):
        rows = s["rows"] if isinstance(s, dict) and "rows" in s else None
        for i, (a, b) in enumerate(zip(yt, yp)):
            fila = {"doc_id": s["doc_id"] if isinstance(s, dict) else s[0],
                    "sentence_id": s["sentence_id"] if isinstance(s, dict) else s[1],
                    "i": i, "gold": a, "pred": b}
            if rows is not None:
                fila["token"] = rows[i]["token"]
                fila["entrada"] = rows[i].get("entrada", "")
            filas.append(fila)
    df = pd.DataFrame(filas)
    for k, v in (extra or {}).items():
        df[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)
