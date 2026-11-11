"""
glossary_match.py
───────────────
Localiza en un texto todas las apariciones de las formas del glosario the glossary,
con el mismo método con que el glosario cuenta sus frecuencias
(glossary/_dev/recount_variants.py):

  - antes de buscar se reparan los artefactos de la conversión desde PDF
    (guiones blandos, palabras partidas, ligaduras, apóstrofos tipográficos,
    tildes separadas de su letra) y se normalizan los espacios;
  - coincidencia exacta de la cadena, sin distinguir mayúsculas y con límites
    de palabra; en español, también sin distinguir tildes; las cadenas de
    «sensibles_mayusculas» sí distinguen mayúsculas;
  - cada entrada aporta todas sus formas: flexión, variantes formales,
    sinónimos, préstamos y, en las siglas, la forma desarrollada;
  - las cadenas de «excluye» se reconocen pero no se anotan; los usos que
    descartan «excluye_si» / «solo_si» y las etiquetas de metadatos de los
    artículos («Palabras clave:», «Keywords:») tampoco.

Diferencia con el recuento del glosario: aquí todas las entradas se buscan en
una sola pasada, de la cadena más larga a la más corta, para obtener una
anotación sin solapamientos (criterio de delimitación del cap. 3: se anota la
cadena terminológica completa, no los términos que contiene).
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
    """Repara los artefactos de la conversión desde PDF (idéntico al glosario)."""
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
    """En español elimina las tildes conservando la longitud del texto."""
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
    """Devuelve {slug: {lang: [formas]}} con la misma construcción que recount_variants.py,
    el tipo de cada entrada (term/entity) y las cadenas sensibles a mayúsculas."""
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
        print(f"  aviso: {len(sin_tipo)} entradas de normalizacion.json sin tipo en entradas.json: {sin_tipo[:5]}")
    return jobs, tipo, case_sensitive


class Matcher:
    """Patrón único con todas las cadenas de todas las entradas de una lengua."""

    def __init__(self, jobs, tipo, case_sensitive, lang):
        self.lang = lang
        self.forms = {}          # (slug, i) -> forma
        owner = {}               # cadena plegada en minúsculas -> (slug, i) | None
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
        """Recibe el texto ya reparado con clean(); devuelve [(inicio, fin, slug, tipo)]."""
        folded = fold(text_clean, self.lang)
        out = []
        for m in self.pattern.finditer(folded):
            o = self.owner.get(m.group(0).lower())
            if o is None or is_metadata_label(folded, m) or context_excluded(folded, m, self.forms[o]):
                continue
            slug = o[0]
            out.append((m.start(), m.end(), slug, self.tipo.get(slug, 'term')))
        return out
