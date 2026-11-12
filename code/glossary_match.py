"""
glossary_match.py
───────────────
Finds every occurrence of the glossary forms in a text, with the same method used to
count their frequencies in the glossary:

  - PDF conversion artefacts are repaired first (soft hyphens, split words, ligatures,
    typographic apostrophes, accents detached from their letter) and spaces are
    normalised;
  - matching is exact, case-insensitive and within word boundaries; in Spanish it also
    ignores accents; the strings listed in "sensibles_mayusculas" are case-sensitive;
  - each entry contributes all its forms: inflection, formal variants, synonyms,
    loanwords and, for acronyms, the expanded form;
  - the strings in "excluye" are recognised but not annotated, and neither are the
    uses ruled out by "excluye_si" / "solo_si" or the metadata labels of articles
    ("Palabras clave:", "Keywords:").

Unlike the glossary count, all entries are searched in a single pass, from the longest
string to the shortest, so that the annotation has no overlaps: the complete
terminological sequence is annotated, not the terms it contains.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

LIGATURES = {'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi',
             'ﬄ': 'ffl', 'ﬅ': 'st', 'ﬆ': 'st'}
SPACING_ACCENTS = {'´': '́', '`': '̀', '¨': '̈', '˜': '̃', 'ˆ': '̂'}


def clean(text: str) -> str:
    """Repairs the PDF conversion artefacts (as in the glossary)."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'­\s*', '', text)
    text = re.sub(r'([A-Za-záéíóúñü])-\s*\n\s*([a-záéíóúñü])', r'\1\2', text)
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace('’', "'").replace('‘', "'")
    text = re.sub('([´`¨˜ˆ])([A-Za-zı])',
                  lambda m: unicodedata.normalize('NFC', m.group(2).replace('ı', 'i') + SPACING_ACCENTS[m.group(1)]),
                  text)
    return re.sub(r'\s+', ' ', text).strip()


def fold(s: str, lang: str) -> str:
    """Removes Spanish accents while keeping the length of the text."""
    if lang != 'es':
        return s
    return ''.join(unicodedata.normalize('NFD', c)[0] for c in s)


METADATA_LABELS = {'palabras clave', 'palabra clave', 'palabras claves', 'keywords', 'keyword', 'key words'}
LIST_AFTER = re.compile(r'\s*[:.]?\s*(?:[^\s.;,·•:]+(?:\s+[^\s.;,·•:]+){0,4}\s*[;,·•]\s*){2,}')


def is_metadata_label(text: str, m) -> bool:
    s = m.group(0)
    if s.lower() not in METADATA_LABELS or not s[0].isupper():
        return False
    after = text[m.end():m.end() + 120]
    return after.lstrip().startswith(':') or s.isupper() or bool(LIST_AFTER.match(after))


def context_excluded(text: str, m, form: dict) -> bool:
    a, b = max(0, m.start() - 100), min(len(text), m.end() + 100)
    window = text[a:b]

    def covers(pats):
        return any(x.start() + a <= m.start() and x.end() + a >= m.end()
                   for p in pats for x in re.finditer(p, window))

    if form.get('solo_si') and not covers(form['solo_si']):
        return True
    return bool(form.get('excluye_si')) and covers(form['excluye_si'])


def load_jobs(glossary_dir: Path):
    """Returns {slug: {lang: [forms]}}, the type of each entry (term/entity) and the
    case-sensitive strings."""
    norm = json.loads((glossary_dir / 'normalizacion.json').read_text(encoding='utf-8'))
    entradas = json.loads((glossary_dir / 'entradas.json').read_text(encoding='utf-8'))
    tipo = {e['slug']: e['type'] for e in entradas}
    siglas = {k: v for k, v in norm.get('siglas', {}).items() if not k.startswith('_')}
    case_sensitive = set(norm.get('sensibles_mayusculas', {}).get('cadenas', []))

    jobs = {}
    for e in norm['entradas']:
        jobs[e['slug']] = {l: [dict(f) for f in fs] for l, fs in e['formas'].items()}
    for slug, langs in norm['formas_contadas'].items():
        if slug.startswith('_'):
            continue
        jobs[slug] = {l: [{'forma': s[0], 'relacion': 'lema', 'cuenta': s}] for l, s in langs.items()}
    for slug, expansions in siglas.items():
        for lang, forms in jobs.get(slug, {}).items():
            if forms and forms[0]['relacion'] == 'lema':
                forms[0]['relacion'] = 'sigla'
            if not any(f['relacion'] == 'forma_desarrollada' for f in forms):
                forms.append({'forma': expansions[0], 'relacion': 'forma_desarrollada', 'cuenta': expansions})
    sin_tipo = [s for s in jobs if s not in tipo]
    if sin_tipo:
        print(f"  warning: {len(sin_tipo)} entries of normalizacion.json without a type in entradas.json: {sin_tipo[:5]}")
    return jobs, tipo, case_sensitive


class Matcher:
    """A single pattern with every string of every entry of a language."""

    def __init__(self, jobs, tipo, case_sensitive, lang):
        self.lang = lang
        self.forms = {}          # (slug, i) -> forma
        owner = {}               # lowercased string -> (slug, i) | None
        conflictos = []
        for slug, langs in jobs.items():
            for i, f in enumerate(langs.get(lang, [])):
                self.forms[(slug, i)] = f
                for s in f.get('excluye', []):
                    owner.setdefault(fold(s, lang).lower(), None)
        for slug, langs in jobs.items():
            for i, f in enumerate(langs.get(lang, [])):
                for s in f['cuenta']:
                    k = fold(s, lang).lower()
                    if k in owner and owner[k] is not None and owner[k][0] != slug:
                        conflictos.append((s, owner[k][0], slug))
                        continue
                    owner.setdefault(k, (slug, i))
        self.owner = owner
        self.tipo = tipo
        self.conflictos = conflictos
        exact = {fold(s, lang).lower(): fold(s, lang) for s in case_sensitive}
        alts = [f'(?-i:{re.escape(exact[s])})' if s in exact else re.escape(s)
                for s in sorted(owner, key=len, reverse=True)]
        self.pattern = re.compile(r'(?<!\w)(?:' + '|'.join(alts) + r')(?!\w)', re.IGNORECASE)

    def find(self, text_clean: str):
        """Takes a text already repaired with clean(); returns [(start, end, slug, type)]."""
        folded = fold(text_clean, self.lang)
        out = []
        for m in self.pattern.finditer(folded):
            o = self.owner.get(m.group(0).lower())
            if o is None or is_metadata_label(folded, m) or context_excluded(folded, m, self.forms[o]):
                continue
            slug = o[0]
            out.append((m.start(), m.end(), slug, self.tipo.get(slug, 'term')))
        return out
