"""
Preprocesa ACTER EN (4 dominios, anotación IO sin entidades nombradas) al
mismo formato de tokens que el corpus TECLA, para validación externa del
CRF (20 rasgos) y el BiLSTM-CRF.

- Lee los TSV token\tIO, reconstruye oraciones (línea en blanco = separador)
- Convierte IO -> BIO (I tras O o inicio = B-T; I tras I = I-T)
- Procesa con spaCy en_core_web_lg SOBRE LOS TOKENS ORIGINALES (Doc pre-tokenizado,
  sin re-tokenizar) para extraer los mismos rasgos lingüísticos que TECLA
- Las columnas estadísticas (freq_global, document_frequency, contextual_diversity,
  tf_idf) se inicializan a 0: el pipeline sin-fuga las recalcula desde train
- lemma_pos_*_prob se calculan sobre el propio ACTER (propiedad léxica, no de término)

Salida: acter_en_bio_tokens.csv (con columna 'domain' para LODO)
"""
from pathlib import Path
import pandas as pd
import spacy
from spacy.tokens import Doc
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ACTER = HERE.parent / "acter_en"
DOMAINS = ["corp", "equi", "htfl", "wind"]
ANN_SUBPATH = "annotated/annotations/sequential_annotations/io_annotations/without_named_entities"

nlp = spacy.load("en_core_web_lg", disable=["ner"])

def read_tsv_sentences(path: Path):
    """Devuelve lista de oraciones; cada oración es lista de (token, io_label)."""
    sents, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            cur.append((parts[0], parts[1].strip()))
    if cur:
        sents.append(cur)
    return sents

def io_to_bio(labels):
    bio = []
    prev = "O"
    for l in labels:
        if l == "I":
            bio.append("B-T" if prev != "I" else "I-T")
        else:
            bio.append("O")
        prev = l
    return bio

rows = []
lemma_pos_counts = defaultdict(Counter)

for domain in DOMAINS:
    ann_dir = ACTER / domain / ANN_SUBPATH
    for tsv in sorted(ann_dir.glob("*.tsv")):
        doc_id = tsv.stem
        sents = read_tsv_sentences(tsv)
        for sent_id, sent in enumerate(sents):
            tokens = [t for t, _ in sent]
            io = [l for _, l in sent]
            bio = io_to_bio(io)
            doc = Doc(nlp.vocab, words=tokens)
            doc = nlp(doc)

            np_tokens, np_heads = set(), set()
            for chunk in doc.noun_chunks:
                np_heads.add(chunk.root.i)
                for t in chunk:
                    np_tokens.add(t.i)

            for i, tok in enumerate(doc):
                lemma_pos_counts[tok.lemma_.lower()][tok.pos_] += 1
                has_amod = any(c.dep_ == "amod" for c in tok.children)
                has_premod = any(c.i < tok.i and c.dep_ in ("amod", "compound", "nummod") for c in tok.children)
                rows.append({
                    "domain": domain,
                    "doc_id": doc_id,
                    "sentence_id": sent_id,
                    "token_id": i,
                    "token": tok.text,
                    "token_lower": tok.text.lower(),
                    "lemma": tok.lemma_,
                    "shape": tok.shape_,
                    "pos": tok.pos_,
                    "tag": tok.tag_,
                    "dep": tok.dep_,
                    "is_stop": tok.is_stop,
                    "is_punct": tok.is_punct,
                    "is_alpha": tok.is_alpha,
                    "has_digit": any(c.isdigit() for c in tok.text),
                    "is_noun": tok.pos_ == "NOUN",
                    "is_verb": tok.pos_ == "VERB",
                    "is_adj": tok.pos_ == "ADJ",
                    "is_propn": tok.pos_ == "PROPN",
                    "is_in_noun_phrase": i in np_tokens,
                    "is_in_noun_chunk_head": i in np_heads,
                    "has_adjective_modifier": has_amod,
                    "has_pre_modifier": has_premod,
                    "freq_global": 0.0,
                    "document_frequency": 0.0,
                    "contextual_diversity": 0.0,
                    "tf_idf": 0.0,
                    "label_bio": bio[i],
                    "label": "T" if bio[i] != "O" else "O",
                })
    print(f"{domain}: acumulado {len(rows):,} tokens", flush=True)

df = pd.DataFrame(rows)

# Probabilidades POS dominante por lema (propiedad léxica)
def pos_prob(lemma, pos):
    c = lemma_pos_counts.get(str(lemma).lower())
    if not c:
        return 0.0
    total = sum(c.values())
    return c.get(pos, 0) / total if total else 0.0

lem = df["lemma"]
df["lemma_pos_noun_prob"] = [pos_prob(l, "NOUN") for l in lem]
df["lemma_pos_verb_prob"] = [pos_prob(l, "VERB") for l in lem]
df["lemma_pos_adj_prob"] = [pos_prob(l, "ADJ") for l in lem]
df["lemma_pos_propn_prob"] = [pos_prob(l, "PROPN") for l in lem]

out = HERE / "acter_en_bio_tokens.csv"
df.to_csv(out, index=False)

terms = df[df.label_bio != "O"]
print(f"\nTotal: {len(df):,} tokens | {df.doc_id.nunique()} docs | 4 dominios")
print(f"Tokens término: {len(terms):,} ({len(terms)/len(df)*100:.1f}%) | tipos únicos: {terms.token_lower.nunique():,}")
print(f"✅ Guardado: {out.name}", flush=True)
