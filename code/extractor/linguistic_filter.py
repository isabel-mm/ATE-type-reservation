"""
Linguistic Filter
------------------------

This module implements the linguistic filtering stage of the hybrid
terminology extraction pipeline. It applies POS-based grammatical
patterns to preprocessed documents in order to identify structurally
plausible term candidates (unithood).

The module does NOT compute frequencies, weights or rankings.

Pattern syntax
--------------
Pattern elements can be:
- A POS tag (uppercase, e.g. 'NOUN', 'ADJ') — matched against the token's POS.
- A literal token (lowercase, e.g. 'of', 'de') — matched against the token's
  lowercased surface form. This allows patterns like 'NOUN+ of NOUN+'.
- POS tag with '+' suffix (e.g. 'NOUN+') — one or more consecutive matches.
- POS group with '*' suffix (e.g. '(ADJ|NOUN)*') — zero or more matches.
"""

from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import yaml


# POS tags of context words for NC-value: nouns, adjectives and verbs
# (Frantzi et al. 2000).
_CONTEXT_POS: frozenset = frozenset({"NOUN", "ADJ", "VERB"})

# Universal Dependencies POS tag set
_UD_POS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
    "X", "SPACE",
}

# ── Function-word filter ───────────────────────────────────────────────
# POS tags that should NEVER appear inside a terminological candidate.
# ADP is intentionally excluded: prepositions appear legitimately as
# structural connectors in prepositional patterns (EN_NP_OF, ES_NP_DE).
_BLOCKED_POS: frozenset = frozenset({
    "DET", "PRON", "AUX", "CCONJ", "SCONJ", "INTJ", "PUNCT", "SPACE", "PART",
})

# Lemmas of function words that must not appear in a CT at any position.
# "of" / "de" are deliberately excluded: they are matched as literal
# tokens in prepositional patterns and always occupy an internal position.
_FUNCTION_LEMMAS: Dict[str, frozenset] = {
    "en": frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "it", "its", "they", "their", "we", "our", "you", "your",
        "he", "his", "she", "her", "who", "which", "what", "whose",
        "such", "all", "each", "every", "any", "both", "other", "same",
        "not", "no", "nor",
        "be", "have", "do", "will", "would", "can", "could",
        "may", "might", "shall", "should", "must",
        # inflected forms that the tagger occasionally labels as nouns
        "is", "are", "was", "were", "has", "had", "does", "did",
    }),
    "es": frozenset({
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "aquel", "aquella", "aquellos", "aquellas",
        "su", "sus", "mi", "mis", "tu", "tus",
        "nuestro", "nuestra", "nuestros", "nuestras",
        "vuestro", "vuestra",
        "se", "le", "les", "lo", "me", "te",
        "que", "cual", "cuyo", "cuya", "quien", "donde",
        "todo", "toda", "todos", "todas",
        "cada", "cualquier", "cualquiera",
        "otro", "otra", "mismo", "misma",
        "no", "ni", "ser", "estar", "haber", "tener",
        # inflected forms that the tagger sometimes takes for nouns
        # (as in "es decir", for instance)
        "es", "son", "era", "eran", "fue", "fueron", "sea", "sean",
        "ha", "han", "hay", "está", "están",
    }),
}


def _is_pos_tag(element: str) -> bool:
    """Return True if element is a known Universal Dependencies POS tag."""
    return element in _UD_POS_TAGS


def _is_clean_candidate(
    tokens: List[Dict[str, Any]],
    language: str,
    extra_stopwords: frozenset = frozenset(),
) -> bool:
    """
    Return True if the candidate span contains no functional tokens.

    Rejects spans where any token has:
    - A POS tag in _BLOCKED_POS (DET, PRON, AUX, CCONJ, SCONJ …)
    - A lemma in the language-specific _FUNCTION_LEMMAS set
    - A form or lemma in *extra_stopwords* (e.g. the stopwords of the other
      language, for abstracts written in it)

    ADP (prepositions) are NOT blocked: they appear as structural
    connectors inside prepositional patterns and never at span edges
    when extracted by those patterns.
    """
    func_lemmas = _FUNCTION_LEMMAS.get(language, frozenset())
    for tok in tokens:
        if tok.get("pos", "") in _BLOCKED_POS:
            return False
        if tok.get("lemma", "").lower() in func_lemmas:
            return False
        if extra_stopwords and (
            tok.get("text", "").lower() in extra_stopwords
            or tok.get("lemma", "").lower() in extra_stopwords
        ):
            return False
    return True


# ---------------------------------------------------------------------
# Grammar loading
# ---------------------------------------------------------------------

def load_grammar(grammar_path: Path) -> Dict[str, Any]:
    """
    Load the POS-based grammar from a YAML file.

    Parameters
    ----------
    grammar_path : Path
        Path to the grammar.yaml file.

    Returns
    -------
    dict
        Parsed grammar grouped by language.
    """
    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = yaml.safe_load(f)
    return grammar


# ---------------------------------------------------------------------
# Core linguistic filter
# ---------------------------------------------------------------------

def extract_candidates(
    documents: List[Dict[str, Any]],
    language: str,
    grammar_path: Path,
    max_span_tokens: int = 6,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract term candidates from preprocessed documents using
    POS-based grammatical patterns.

    Parameters
    ----------
    documents : list of dict
        Preprocessed documents. Each document must contain:
        - doc_id : str
        - tokens : list of dicts with at least 'text', 'lemma', 'pos'

    language : str
        Language code (e.g. 'en', 'es').

    grammar_path : Path
        Path to the POS grammar YAML file.

    extra_stopwords : iterable of str, optional
        Additional words that may not appear in a candidate (compared,
        lowercased, with the form and the lemma of each token).

    Returns
    -------
    list of dict
        List of candidate term occurrences.
    """

    grammar = load_grammar(grammar_path)

    if language not in grammar:
        raise ValueError(f"No grammar defined for language '{language}'")

    patterns = grammar[language]
    candidates = []
    extra = frozenset(w.lower() for w in (extra_stopwords or ()))

    for doc in documents:
        doc_id = doc.get("doc_id")

        # Use sentence-segmented tokens when available so that patterns
        # never match across sentence boundaries.
        sentences = doc.get("sentences")
        if sentences is None:
            sentences = [doc.get("tokens", [])]

        for pattern in patterns:
            pattern_id = pattern["id"]
            pos_pattern = pattern["pos_pattern"].split()

            for sentence_tokens in sentences:
                pos_sequence = [tok["pos"] for tok in sentence_tokens]
                token_sequence = [tok["text"] for tok in sentence_tokens]

                matches = _match_pos_pattern(pos_sequence, token_sequence, pos_pattern)

                for start, end in matches:
                    span_tokens = sentence_tokens[start:end]

                    # Skip spans that exceed the maximum allowed length
                    if end - start > max_span_tokens:
                        continue

                    # Skip spans that contain functional/stopword tokens
                    if not _is_clean_candidate(span_tokens, language, extra):
                        continue

                    term_text = " ".join(tok["text"] for tok in span_tokens)
                    term_lemma = " ".join(tok["lemma"] for tok in span_tokens)

                    # Context words (±2 tokens) for NC-value computation:
                    # nouns, adjectives and verbs, as lowercased lemmas.
                    context_words: List[str] = []
                    for pos in range(max(0, start - 2), start):
                        tok = sentence_tokens[pos]
                        if tok.get("pos", "") in _CONTEXT_POS:
                            context_words.append(tok["lemma"].lower())
                    for pos in range(end, min(len(sentence_tokens), end + 2)):
                        tok = sentence_tokens[pos]
                        if tok.get("pos", "") in _CONTEXT_POS:
                            context_words.append(tok["lemma"].lower())

                    candidates.append({
                        "term": term_text,
                        "lemma": term_lemma,
                        "tokens": span_tokens,
                        "length": len(span_tokens),
                        "pattern_id": pattern_id,
                        "doc_id": doc_id,
                        "language": language,
                        "context_words": context_words,
                    })

    return candidates


# ---------------------------------------------------------------------
# Pattern matching utilities
# ---------------------------------------------------------------------

def _elem_matches(
    pos_sequence: List[str],
    token_sequence: List[str],
    j: int,
    element: str,
) -> bool:
    """Return True if position j matches a single pattern element."""
    if _is_pos_tag(element):
        return pos_sequence[j] == element
    return token_sequence[j].lower() == element.lower()


def _match_rest(
    pos_sequence: List[str],
    token_sequence: List[str],
    pattern: List[str],
    k: int,
    j: int,
) -> List[int]:
    """
    Return all possible end positions for matching pattern[k:] starting at j.

    Uses backtracking so that greedy operators (*) do not prevent
    subsequent pattern elements from matching.
    """
    seq_len = len(pos_sequence)

    if k == len(pattern):
        return [j]

    pat = pattern[k]

    if pat.endswith("*"):
        # Zero-or-more: try all lengths from 0 up to max consecutive matches.
        base = pat[:-1].strip("()")
        allowed = base.split("|")
        results: List[int] = []
        pos = j
        while True:
            results.extend(_match_rest(pos_sequence, token_sequence, pattern, k + 1, pos))
            if pos < seq_len and pos_sequence[pos] in allowed:
                pos += 1
            else:
                break
        return results

    if pat.endswith("+"):
        # One-or-more: must match at least one, then try all extensions.
        base = pat[:-1]
        if j >= seq_len or not _elem_matches(pos_sequence, token_sequence, j, base):
            return []
        results = []
        pos = j + 1
        while True:
            results.extend(_match_rest(pos_sequence, token_sequence, pattern, k + 1, pos))
            if pos < seq_len and _elem_matches(pos_sequence, token_sequence, pos, base):
                pos += 1
            else:
                break
        return results

    # Exact single element
    if j >= seq_len or not _elem_matches(pos_sequence, token_sequence, j, pat):
        return []
    return _match_rest(pos_sequence, token_sequence, pattern, k + 1, j + 1)


def _match_pos_pattern(
    pos_sequence: List[str],
    token_sequence: List[str],
    pattern: List[str],
) -> List[tuple]:
    """
    Match a POS pattern against a POS/token sequence with backtracking.

    Supports:
    - POS tags (e.g. 'NOUN') — matched against pos_sequence.
    - Literal tokens (e.g. 'of', 'de') — matched against lowercased token_sequence.
    - One-or-more with '+' suffix (e.g. 'NOUN+').
    - Zero-or-more with '*' suffix (e.g. '(ADJ|NOUN)*') — never greedy-consumes
      tokens needed by subsequent elements.

    Returns
    -------
    list of (start, end) tuples
    """
    matches = []
    seq_len = len(pos_sequence)

    for i in range(seq_len):
        for end in _match_rest(pos_sequence, token_sequence, pattern, 0, i):
            if end > i:  # exclude zero-length matches
                matches.append((i, end))

    return matches
