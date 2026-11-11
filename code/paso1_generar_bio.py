"""
paso1_generar_bio.py
────────────────────
Genera los ficheros BIO a nivel de token de los dos subcorpus:

  1. repara los artefactos del PDF (clean(), el mismo método que the glossary);
  2. analiza el texto con spaCy (por defecto los modelos transformer que cita
     la tesis: en_core_web_trf / es_dep_news_trf);
  3. proyecta sobre el texto todas las formas de las 261 entradas de the glossary
     (flexión, variantes, siglas, préstamos), sin solapamientos;
  4. calcula para cada token las mismas columnas de rasgos que el fichero BIO
     anterior (tesis/bio_token_ml.py), de modo que los scripts del CRF y del
     BiLSTM-CRF funcionan sin cambios.

Salidas (en $DATOS_GENERADOS, por defecto ../datos_generados):
  {lang}_bio_tokens.csv             anotación completa: B-T/I-T (términos),
                                    B-E/I-E (entidades) y O
  {lang}_bio_tokens_terms_only.csv  solo términos: las entidades pasan a O
  {lang}_resumen_anotacion.csv      apariciones por entrada (comprobación)

Columnas añadidas respecto al BIO anterior:
  entrada  slug de la entrada de the glossary del span (unidad de la reserva de tipos)
  clase    T (término) / E (entidad) / O

Uso:  python paso1_generar_bio.py en
      python paso1_generar_bio.py es
      SPACY_EN=en_core_web_sm python paso1_generar_bio.py en --docs 3   (prueba rápida)
"""
from __future__ import annotations

import math
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glossary_match import Matcher, clean, load_jobs  # noqa: E402

DATOS = Path(os.environ.get("DATOS", HERE.parent / "datos"))
OUT = Path(os.environ.get("DATOS_GENERADOS", HERE.parent / "datos_generados"))
OUT.mkdir(parents=True, exist_ok=True)

SPACY_MODELS = {
    "en": os.environ.get("SPACY_EN", "en_core_web_trf"),
    "es": os.environ.get("SPACY_ES", "es_dep_news_trf"),
}
CHUNK_CHARS = 50_000


def normalize(text: str, lang: str) -> str:
    """Igual que en el BIO anterior: minúsculas y, en español, sin tildes
    (solo para token_lower y lemma; la columna token conserva las tildes)."""
    text = text.lower()
    if lang == "es":
        text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def chunks(text: str, size: int = CHUNK_CHARS):
    """Trocea en fragmentos de ~size caracteres cortando tras un punto; devuelve (inicio, fin)."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            end = cut + 2 if cut != -1 else end
        yield start, end
        start = end


def juilland_dispersion(freqs):
    if not freqs or sum(freqs) == 0:
        return 0.0
    mean = sum(freqs) / len(freqs)
    sd = math.sqrt(sum((f - mean) ** 2 for f in freqs) / len(freqs))
    return max(0.0, 1 - sd / mean) if mean else 0.0


def label_tokens(doc, offset, spans):
    """Asigna a cada token del fragmento su etiqueta BIO y la entrada de the glossary."""
    labels = ["O"] * len(doc)
    entrada = [""] * len(doc)
    for s, e, slug, tipo in spans:
        span = doc.char_span(s - offset, e - offset, alignment_mode="expand")
        if span is None or len(span) == 0:
            continue
        if any(labels[t.i] != "O" for t in span):
            continue
        letra = "T" if tipo == "term" else "E"
        for k, t in enumerate(span):
            labels[t.i] = ("B-" if k == 0 else "I-") + letra
            entrada[t.i] = slug
    return labels, entrada


def process_language(lang: str, max_docs: int | None = None):
    t0 = time.time()
    gpu = False
    try:
        import torch
        if torch.cuda.is_available():          # solo GPU NVIDIA (Colab); en el Mac, CPU
            gpu = spacy.require_gpu()
    except ImportError:
        pass
    nlp = spacy.load(SPACY_MODELS[lang], exclude=["ner"])
    nlp.max_length = 2_000_000
    print(f"[{lang}] spaCy: {SPACY_MODELS[lang]}  ·  GPU: {gpu}", flush=True)

    jobs, tipo, case_sensitive = load_jobs(DATOS / "glossary")
    matcher = Matcher(jobs, tipo, case_sensitive, lang)
    if matcher.conflictos:
        print(f"[{lang}] aviso: {len(matcher.conflictos)} cadenas compartidas por dos entradas "
              f"(se queda la primera): {matcher.conflictos[:5]}", flush=True)

    paths = sorted((DATOS / "corpus" / lang).glob("*.txt"))[:max_docs]
    rows = []
    lemma_global, lemma_docsets = Counter(), defaultdict(set)
    lemma_sentence_sets, lemma_pos = defaultdict(set), defaultdict(Counter)
    lemma_doc_freqs = defaultdict(Counter)
    lemma_doc_positions = defaultdict(list)
    doc_lexical_variety = {}
    resumen = Counter()

    for n_doc, path in enumerate(paths, 1):
        doc_id = f"{lang.upper()}::{path.stem}"
        text = clean(path.read_text(encoding="utf-8", errors="replace"))
        spans = matcher.find(text)
        for _, _, slug, _ in spans:
            resumen[slug] += 1

        doc_rows = []
        sent_idx = 0
        token_offset = 0
        noun_tokens_doc = []
        for (cs, ce) in chunks(text):
            doc = nlp(text[cs:ce])
            chunk_spans = [(s, e, sl, ti) for s, e, sl, ti in spans if s >= cs and e <= ce]
            labels, entrada = label_tokens(doc, cs, chunk_spans)
            in_np, np_head = set(), set()
            try:
                for ch in doc.noun_chunks:
                    in_np.update(t.i for t in ch)
                    np_head.add(ch.root.i)
            except (NotImplementedError, ValueError):
                pass
            for sent in doc.sents:
                toks = [t for t in sent if not t.is_space]
                if not toks:
                    continue
                for k, tok in enumerate(toks):
                    lemma = normalize(tok.lemma_ or tok.text, lang)
                    lab = labels[tok.i]
                    doc_rows.append({
                        "language": lang.upper(), "doc_id": doc_id, "path": path.name,
                        "sentence_id": sent_idx, "token_id": token_offset + tok.i,
                        "token": tok.text, "token_lower": normalize(tok.text, lang),
                        "lemma": lemma,
                        "lemma_suffix3": lemma[-3:] if len(lemma) >= 3 else lemma,
                        "lemma_suffix4": lemma[-4:] if len(lemma) >= 4 else lemma,
                        "pos": tok.pos_, "tag": tok.tag_, "dep": tok.dep_, "shape": tok.shape_,
                        "label_bio": lab, "clase": lab[2:] if lab != "O" else "O",
                        "entrada": entrada[tok.i],
                        "is_stop": int(tok.is_stop), "is_punct": int(tok.is_punct),
                        "is_alpha": int(tok.is_alpha), "is_propn": int(tok.pos_ == "PROPN"),
                        "is_noun": int(tok.pos_ == "NOUN"), "is_verb": int(tok.pos_ == "VERB"),
                        "is_adj": int(tok.pos_ == "ADJ"),
                        "is_in_noun_phrase": int(tok.i in in_np),
                        "is_in_noun_chunk_head": int(tok.i in np_head),
                        "has_adjective_modifier": int(any(c.dep_ == "amod" or c.pos_ == "ADJ" for c in tok.children)),
                        "has_pre_modifier": int(any(c.i < tok.i and c.pos_ in {"ADJ", "NOUN", "PROPN", "NUM"}
                                                    for c in tok.children)),
                        "position_in_sentence": round(k / max(len(toks) - 1, 1), 6),
                        "has_digit": int(any(c.isdigit() for c in tok.text)),
                        "token_length": len(tok.text),
                    })
                    lemma_global[lemma] += 1
                    lemma_docsets[lemma].add(doc_id)
                    lemma_sentence_sets[lemma].add(f"{doc_id}::{sent_idx}")
                    lemma_pos[lemma][tok.pos_] += 1
                    lemma_doc_freqs[lemma][doc_id] += 1
                    lemma_doc_positions[(doc_id, lemma)].append(k / max(len(toks) - 1, 1))
                    if not tok.is_punct:
                        noun_tokens_doc.append(lemma)
                sent_idx += 1
            token_offset += len(doc)
        for r in doc_rows:
            r["doc_length_tokens"] = len(doc_rows)
        doc_lexical_variety[doc_id] = len(set(noun_tokens_doc)) / max(len(noun_tokens_doc), 1)
        rows.extend(doc_rows)
        n_term = sum(1 for r in doc_rows if r["label_bio"] == "B-T")
        n_ent = sum(1 for r in doc_rows if r["label_bio"] == "B-E")
        print(f"[{lang}] {n_doc}/{len(paths)} {path.stem[:60]}: {len(doc_rows)} tokens, "
              f"{n_term} términos, {n_ent} entidades ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    n_docs = len(paths)
    df["freq_in_text"] = df.groupby(["doc_id", "lemma"])["lemma"].transform("size").astype(float)
    avg = {k: sum(v) / len(v) for k, v in lemma_doc_positions.items()}
    df["avg_position_in_sentence"] = [avg[(d, l)] for d, l in zip(df["doc_id"], df["lemma"])]
    df["freq_global"] = df["lemma"].map(lemma_global).astype(float)
    df["document_frequency"] = df["lemma"].map(lambda x: len(lemma_docsets[x])).astype(float)
    df["contextual_diversity"] = df["lemma"].map(lambda x: len(lemma_sentence_sets[x])).astype(float)
    doc_ids = [f"{lang.upper()}::{p.stem}" for p in paths]
    disp = {l: juilland_dispersion([c.get(d, 0) for d in doc_ids]) for l, c in lemma_doc_freqs.items()}
    df["dispersion"] = df["lemma"].map(disp)
    df["lexical_variety"] = df["doc_id"].map(doc_lexical_variety).astype(float)
    for pos, col in [("NOUN", "noun"), ("VERB", "verb"), ("ADJ", "adj"), ("PROPN", "propn")]:
        probs = {l: c.get(pos, 0) / max(sum(c.values()), 1) for l, c in lemma_pos.items()}
        df[f"lemma_pos_{col}_prob"] = df["lemma"].map(probs).astype(float)
    df["tf_idf"] = (df["freq_in_text"] / df["doc_length_tokens"].clip(lower=1)) * np.log(
        (n_docs + 1) / (df["document_frequency"] + 1))
    df["label"] = np.where(df["label_bio"] == "O", "O", "T")

    full = OUT / f"{lang}_bio_tokens.csv"
    df.to_csv(full, index=False)
    terms = df.copy()
    es_ent = terms["clase"] == "E"
    terms.loc[es_ent, ["label_bio", "clase", "label"]] = ["O", "O", "O"]
    terms.loc[es_ent, "entrada"] = ""
    terms.to_csv(OUT / f"{lang}_bio_tokens_terms_only.csv", index=False)

    res = pd.DataFrame([{"entrada": s, "tipo": tipo.get(s, "?"), "apariciones": n}
                        for s, n in resumen.most_common()])
    res.to_csv(OUT / f"{lang}_resumen_anotacion.csv", index=False)
    print(f"[{lang}] FIN: {len(df)} tokens · spans término={int((df.label_bio == 'B-T').sum())} "
          f"entidad={int((df.label_bio == 'B-E').sum())} · entradas con apariciones: "
          f"T={sum(1 for s in resumen if tipo.get(s) == 'term')} E={sum(1 for s in resumen if tipo.get(s) == 'entity')} "
          f"· {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    lang = sys.argv[1]
    max_docs = int(sys.argv[sys.argv.index("--docs") + 1]) if "--docs" in sys.argv else None
    process_language(lang, max_docs)
