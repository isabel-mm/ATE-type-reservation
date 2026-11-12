"""
Converts ACTER EN (four domains, IO annotation without named entities) to the token
format of our corpus, for the external validation.

- Reads the token\tIO TSV files and rebuilds the sentences (a blank line separates them).
- Converts IO to BIO (I after O or at the start = B-T; I after I = I-T).
- Runs spaCy en_core_web_lg ON THE ORIGINAL TOKENS (pre-tokenised Doc, no
  re-tokenisation) to obtain the same linguistic features as for our corpus.
- The statistical columns (freq_global, document_frequency, contextual_diversity,
  tf_idf) are set to 0: the models recompute them from the training partition.
- lemma_pos_*_prob are computed on ACTER itself (a lexical property, not a term one).

Output: acter_en_bio_tokens.csv (with a 'domain' column for LODO)
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
    """Returns a list of sentences, each a list of (token, io_label)."""
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

# Part-of-speech probabilities per lemma (a lexical property)
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
print(f"\nTotal: {len(df):,} tokens | {df.doc_id.nunique()} docs | 4 domains")
print(f"Term tokens: {len(terms):,} ({len(terms)/len(df)*100:.1f}%) | unique types: {terms.token_lower.nunique():,}")
print(f"Saved: {out.name}", flush=True)
