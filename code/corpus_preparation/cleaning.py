"""
Text cleaning and linguistic normalisation utilities for the el extractor corpus module.

This module offers (all optional, activated by flag):

Structural noise
    rejoin_hyphens_flag             Reúne palabras partidas por guión al final de línea.
    structure                       Elimina números de página y líneas muy cortas.

Document metadata
    remove_section_headers_flag     Elimina líneas que son solo un título de sección
                                    (Introduction, Metodología, Conclusiones…).
    remove_keyword_lines_flag       Elimina líneas "Keywords: …" / "Palabras clave: …".
    remove_captions_flag            Elimina pies de figura y tabla (Figure 1, Tabla 3…).
    remove_urls_flag                Elimina líneas que son solo URL, DOI o email.
    remove_publication_metadata_flag Elimina copyright, licencias CC, fechas de recepción.
    remove_journal_metadata_flag    Elimina líneas de volumen/número/páginas de revista.
    remove_inline_citations_flag    Elimina marcadores de cita numérica [1], [2,3].
    remove_statistical_lines_flag   Elimina líneas mayoritariamente numéricas (tablas, stats).
    remove_bibliography_flag        Elimina la sección de referencias/bibliografía.

Lexical filtering
    remove_stopwords_flag           Elimina palabras vacías (spaCy + NLTK + lista extra).
    stoplist                        Stoplist personalizada (lista o ruta a .txt).
    lemmatize_flag                  Lematiza el texto con spaCy.

QA (no modifica el texto, solo informa)
    flag_discourse_noise(text, language)  Devuelve líneas que son solo marcadores
                                          de discurso genérico.

Design principles:
- Nothing is applied implicitly.
- Every operation is explicit, controlled by flags.
- Always fails gracefully (never breaks corpus building).
"""

import re
from pathlib import Path
import spacy
from .utils import load_stoplist

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "extractor" / "resources"

# ---------------------------------------------------------------------------
# SECTION HEADERS
# Títulos estructurales de sección que, si aparecen solos en una línea,
# son ruido estructural (no cuerpo del texto).
# ---------------------------------------------------------------------------

_SECTION_HEADERS: dict[str, frozenset] = {
    "en": frozenset({
        "abstract", "introduction", "background", "related work",
        "methodology", "methods", "method", "materials", "materials and methods",
        "results", "discussion", "conclusion", "conclusions",
        "limitations", "acknowledgements", "acknowledgments",
        "references", "bibliography", "appendix", "supplementary material",
        "keywords", "key words", "funding", "conflict of interest",
        "author contributions", "data availability",
    }),
    "es": frozenset({
        "resumen", "introducción", "antecedentes", "trabajo relacionado",
        "metodología", "métodos", "método", "materiales", "materiales y métodos",
        "resultados", "discusión", "conclusión", "conclusiones",
        "limitaciones", "agradecimientos", "agradecimiento",
        "referencias", "bibliografía", "apéndice", "material suplementario",
        "palabras clave", "financiación", "conflicto de intereses",
        "contribución de los autores", "disponibilidad de datos",
    }),
}

# Prefijos de líneas de metadatos de palabras clave
_KEYWORD_PREFIXES = re.compile(
    r"^\s*(?:keywords?|key\s*words?|palabras?\s*claves?|mots[\s\-]clés?"
    r"|paraules?\s*claus?|palavras?\s*[\-]?chave)\s*[:\-]",
    re.IGNORECASE,
)

# Patrón de número de sección al inicio de línea (ej: "1.", "2.3", "IV.")
_SECTION_NUM = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*|[IVXLCDM]+\.?\s*)")

# ---------------------------------------------------------------------------
# STOPWORD MANAGEMENT (spaCy + NLTK + user)
# ---------------------------------------------------------------------------

import nltk
from nltk.corpus import stopwords as nltk_stopwords

NLTK_LANG_MAP = {
    "es": "spanish",
    "en": "english",
}


def get_language_stopwords(language, extra_stoplist=None):
    """
    Returns a unified stopword set combining:
    - spaCy's default stopwords
    - NLTK stopwords (if available)
    - optional extra stoplist provided by user

    Always returns a Python set().
    """

    lang = language.lower()
    combined = set()

    # ------------------------------- spaCy STOPWORDS ---------------------------
    try:
        nlp = _get_spacy_model(lang)
        combined |= {w.lower() for w in nlp.Defaults.stop_words}
    except Exception:
        print(f"⚠️ Could not load spaCy stopwords for '{lang}'.")

    # ------------------------------- NLTK STOPWORDS ----------------------------
    nltk_lang = NLTK_LANG_MAP.get(lang)
    if nltk_lang:
        try:
            nltk_sw = nltk_stopwords.words(nltk_lang)
        except LookupError:
            print(f"📥 Downloading NLTK stopwords for '{nltk_lang}'...")
            nltk.download("stopwords")
            nltk_sw = nltk_stopwords.words(nltk_lang)
        except Exception:
            print(f"⚠️ Error loading NLTK stopwords for '{lang}'.")
            nltk_sw = []
        combined |= {w.lower() for w in nltk_sw}

    # ------------------------------- EXTRA STOPLIST ----------------------------
    if extra_stoplist:
        if isinstance(extra_stoplist, list):
            combined |= {w.lower() for w in extra_stoplist}

        elif isinstance(extra_stoplist, str):  # path to .txt file
            try:
                extra = load_stoplist(extra_stoplist)
                combined |= {w.lower() for w in extra}
            except Exception:
                print(f"⚠️ Could not load extra stoplist file '{extra_stoplist}'. Skipping.")

    return combined


# ---------------------------------------------------------------------------
# SPACY MODELS
# ---------------------------------------------------------------------------

SPACY_MODEL_TABLE = {
    "en": "en_core_web_lg",
    "es": "es_core_news_md",
}

_SPACY_MODELS = {}


def _get_spacy_model(language):
    """
    Robust spaCy model loader:

    1. Try to load installed model.
    2. If not installed, download it.
    3. If download fails, fallback to spacy.blank(language).

    Always returns an NLP object.
    """

    import spacy.cli

    lang = language.lower()
    model_name = SPACY_MODEL_TABLE.get(lang)

    # Unsupported language -> blank model
    if not model_name:
        if lang not in _SPACY_MODELS:
            nlp = spacy.blank(lang)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SPACY_MODELS[lang] = nlp
            print(f"✅ spaCy blank model initialised for language '{language}'")
        return _SPACY_MODELS[lang]

    # Cached model
    if model_name in _SPACY_MODELS:
        return _SPACY_MODELS[model_name]

    # Try loading installed model
    try:
        nlp = spacy.load(model_name)
        _SPACY_MODELS[model_name] = nlp
        print(f"✅ spaCy model '{model_name}' loaded successfully")
        return nlp

    except OSError:
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            _SPACY_MODELS[model_name] = nlp
            print(f"✅ spaCy model '{model_name}' downloaded and loaded successfully")
            return nlp

        # spacy.cli.download aborts with SystemExit when pip is unavailable
        # (e.g. inside the packaged desktop app)
        except (Exception, SystemExit):
            nlp = spacy.blank(lang)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SPACY_MODELS[model_name] = nlp
            print(
                f"⚠️ spaCy model '{model_name}' unavailable. "
                f"Using blank model for '{language}'."
            )
            return nlp


# ---------------------------------------------------------------------------
# STRUCTURAL CLEANING
# ---------------------------------------------------------------------------

PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")


def clean_structure(text, remove_short_lines=True, remove_page_numbers=True):
    """Remove page numbers and extremely short lines."""
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if remove_page_numbers and PAGE_NUMBER_LINE.fullmatch(stripped):
            continue

        if remove_short_lines and len(stripped.split()) < 3:
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned)


def remove_numeric_only_lines(text):
    """Remove lines containing only numbers or numeric markers."""
    pattern = r'^\s*[\(\[\{\-–—\*•●"]*\s*\d+\s*[\)\]\}\-–—\*•●"."]*\s*$'
    return "\n".join(
        line for line in text.splitlines()
        if not re.fullmatch(pattern, line)
    )


# ---------------------------------------------------------------------------
# BIBLIOGRAPHY REMOVAL
# ---------------------------------------------------------------------------
# [2026-11-07] Se quitan las listas de referencias como bloques, en cualquier
# posición del documento, en lugar de cortar desde el encabezado hasta el final:
# así se conservan los capítulos, apéndices y notas que siguen a las referencias
# de un capítulo de libro o de un volumen colectivo. Cada bloque empieza en un
# encabezado («References», «Bibliografía»…) al que siguen al menos tres
# entradas («Apellido, Nombre … año», «[12] … año», «SIGLA = Autor año», una URL)
# y termina donde dejan de aparecer entradas durante más de 1.200 caracteres, o
# en un encabezado de apéndice, notas o capítulo. También se quita el índice de
# materias final. Comprobado documento a documento sobre el corpus de la tesis
# (tesis/sin_bibliografia_2026-11/limpieza_v2/).

_BIB_SECTIONS = {
    "es": [r'\bREFERENCIAS?(?:\s+(?:BIBLIOGR[ÁA]\S{1,3}CAS?|CITADAS?))?\b', r'\bBIBLIOGRAF[ÍI]A(?:\s+(?:CITADA|B[ÁA]SICA|COMPLEMENTARIA))?\b',
           r'\bOBRAS\s+CITADAS\b', r'\bABREVIATURAS\s+Y\s+REFERENCIAS\s+BIBLIOGR[ÁA]\S{1,3}CAS\b'],
    "en": [r'\bREFERENCES(?:\s+CITED)?\b', r'\bREFERENCE\s+LIST\b', r'\bBIBLIOGRAPHY\b', r'\bWORKS\s+CITED\b', r'\bLITERATURE\s+CITED\b'],
}

YEAR = r"(?:(?:1[5-9]|20)\d\d[a-z]?|en\s+prensa|in\s+press|s\.\s?f\.|n\.\s?d\.)"
# Entrada bibliográfica: «Apellido, Nombre|I.» (o «APELLIDO, N.») y un año poco después.
L_UP = "A-ZÀ-ÞА-ЯЁ"
ENTRY = re.compile(
    r"[" + L_UP + r"][\w'’`´\-]+(?:\s[" + L_UP + r"][\w'’`´\-]+)?,\s+(?:[" + L_UP + r"][\w'’`´\-]*\.?[\s\-]?){1,4}[^\n]{0,300}?\b" + YEAR + r"\b"   # Apellido, Nombre … año
    r"|\[\d{1,3}\][^\n]{0,400}?\b" + YEAR + r"\b"                                                   # [12] … año
    r"|(?:^|(?<=\s))\d{1,3}\.\s[^\n]{0,400}?\b" + YEAR + r"\b"
    r"|\b[" + L_UP + r"]\.(?:\s?[" + L_UP + r"]\.)?\s[" + L_UP + r"]{2,}[\w\-]*[;,][^\n]{0,300}?\b" + YEAR + r"\b"                                     # 12. … año
    r"|^[^\n=]{1,60}\s=\s[^\n]{0,250}?\b" + YEAR + r"\b"                                           # SIGLA = Autor año
    r"|https?://\S+|<?www\.\S+", re.M)                                                         # URL de una fuente
GAP, MIN_ENTRIES, LOOK, BRIDGE, LOOK_MIN, SHORT_TAIL = 1200, 3, 1500, 3000, 2500, 1500
# Encabezados que cierran siempre una lista de referencias
STOP = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:Appendix|Appendices|APPENDIX|APPENDICES|Anexos?|ANEXOS?|Ap[ée]ndices?|AP[ÉE]NDICES?|"
    r"Notas|NOTAS|Notes|NOTES|Chapter|CHAPTER|Cap[íi]tulo|CAP[ÍI]TULO|CAP´ITULO|Solutions)"
    r"(?=\s*$|\s*[:.]|\s+(?:\d|[IVXLC]+\b|[A-Z]\b|[A-ZÀ-Þ][a-zà-ÿ]))", re.M)

def is_stop(text, a, b):
    """¿Hay entre a y b un encabezado de apéndice, notas o capítulo? Se ignoran las
    cabeceras de página, que se repiten a lo largo del documento."""
    for m in STOP.finditer(text, a, b):
        ls = text.rfind("\n", 0, m.start()) + 1; le = text.find("\n", m.start()); le = len(text) if le < 0 else le
        line = text[ls:le].strip()
        if line and text.count(line) >= 3 and len(line) < 120:
            continue                                   # cabecera de página
        if re.search(r"\bpp?\.|ISBN|\(" + YEAR + r"\)", line + " " + text[m.start():m.start() + 70]):
            continue                                   # «Capítulo VIII, pp. 175…» dentro de una referencia
        return True
    return False
INDEX_TITLES = {"en": r"(?:Subject\s+|Author\s+|Name\s+|General\s+)?Index|INDEX", "es": r"(?i:[ÍI]ndice\s+(?:anal[íi]tico|tem[áa]tico|de\s+materias|onom[áa]stico|alfab[ée]tico))"}

def _heading_starts(text, lang):
    sec = _BIB_SECTIONS[lang]
    title_res = [re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*|[IVXLCDM]+(?:\.\d+)*\.?\s+)?(?:" + t + r")\s*:?\s*$", re.I) for t in sec]
    fused_res = [re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*|[IVXLCDM]+(?:\.\d+)*\.?\s+)?(?i:" + t + r")\s*:?\s+(?:[A-ZÀ-Þ]|\d|\[)") for t in sec]
    out, off = [], 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s and any(r.match(s) for r in (title_res if len(s) < 60 else []) + fused_res):
            out.append(off)
        off += len(line)
    surname = (r"\s*:?\s+[A-ZÀ-Þ][\w'’`´\-]+(?:(?:\s+[A-ZÀ-Þ][\w'’`´\-]+)?,"
               r"|(?:\s+(?:y|and|&)\s+[A-ZÀ-Þ][\w'’`´\-]+|\s+et\s+al\.)?\s+\(" + YEAR + r"\);)")
    for t in sec:
        for m in re.finditer(r"(?<=[\s.\)\]])(?i:" + t + r")(?=" + surname + r")", text):
            w = m.group(0)
            if w[0].isupper() and (w.isupper() or w[1:2].islower()):
                out.append(m.start())
    return sorted(set(out))


def _line(text, pos):
    ls = text.rfind("\n", 0, pos) + 1; le = text.find("\n", pos); le = len(text) if le < 0 else le
    return text[ls:le].strip()


def _running_header(text, m):
    """La «entrada» es en realidad una cabecera de página repetida («… Liendo, P. y otros»)."""
    around = text[max(0, m.start() - 45):m.start() + 12]
    return len(around.strip()) > 30 and text.count(around) >= 3


def block_end(text, start):
    """Fin del bloque de referencias que empieza en `start`, o None si no lo es."""
    ms = [m for m in ENTRY.finditer(text, start) if not _running_header(text, m)]
    if len([m for m in ms if m.start() < start + LOOK_MIN]) < MIN_ENTRIES:
        # lista breve al final del documento («References Biber, D. 1988…»)
        if len(text) - start <= SHORT_TAIL and re.search(r"\b" + YEAR + r"\b", text[start:]):
            return len(text)
        return None
    last = ms[0]
    for k, m in enumerate(ms[1:], 1):
        gap = m.start() - last.end()
        if is_stop(text, last.end(), m.start()) or re.search(r"\b(?:Appendix|Anexo|Ap[ée]ndice)\s+(?:\d+|[A-Z])\s*[:.]", text[last.end():m.start()]):
            break
        if gap > GAP:
            # una tabla o un pie de página en medio de la lista: se salta si
            # detrás vuelven, enseguida, varias entradas
            again = [x for x in ms[k:] if x.start() < m.start() + LOOK]
            if gap > BRIDGE or len(again) < 3:
                break
        last = m
    # hasta el final de la línea de la última entrada y sus líneas de continuación breves
    end = text.find("\n", last.end()); end = len(text) if end < 0 else end
    rest = text[end:]
    dense = 1000 * len(re.findall(r"\b" + YEAR + r"\b", rest)) / max(1, len(rest))
    if ((len(rest) <= SHORT_TAIL and re.search(r"\b" + YEAR + r"\b", rest) or len(rest) <= 2 * SHORT_TAIL and dense >= 3) and not is_stop(text, end, len(text))
            and not re.search(r"\b(?:Appendix|APPENDIX|Anexo|ANEXO|Ap[ée]ndice|AP[ÉE]NDICE)\b", rest)):
        return len(text)                      # restos de la lista al final del documento
    SIGNAL = re.compile(r"\d+\s*[–\-]\s*\d+|\bpp?\.|\b(?:In|En|ed\.|eds\.|edd\.|Press|University|Universidad|Publishing|"
                        r"Benjamins|Routledge|Springer|Elsevier|ELRA|Association|Editorial|Gredos|Ariel)\b|https?://|doi", re.I)
    for _ in range(4):
        nxt = text.find("\n", end + 1)
        line = text[end + 1: nxt if nxt >= 0 else len(text)]
        prev = text[text.rfind("\n", 0, end) + 1:end].rstrip()
        if not line.strip() or len(line) > 170 or STOP.match(line) or nxt < 0:
            break
        unfinished = not re.search(r"[.)”\"»]$", prev) or prev.endswith(("\xad", "-"))
        if not (unfinished or SIGNAL.search(line)):
            break
        end = nxt
    rest = text[end:]
    dense = 1000 * len(re.findall(r"\b" + YEAR + r"\b", rest)) / max(1, len(rest))
    if ((len(rest) <= SHORT_TAIL and re.search(r"\b" + YEAR + r"\b", rest) or len(rest) <= 2 * SHORT_TAIL and dense >= 3) and not is_stop(text, end, len(text))
            and not re.search(r"\b(?:Appendix|APPENDIX|Anexo|ANEXO|Ap[ée]ndice|AP[ÉE]NDICE)\b", rest)):
        return len(text)                      # restos de la lista al final del documento
    return end

def index_start(text, lang):
    """Índice de materias final («Index», «Índice analítico»…), solo o pegado a la primera entrada."""
    n = len(text)
    rx = re.compile(r"(?:^|(?<=[\s.]))(?:" + INDEX_TITLES[lang] + r")(?=\s*$|\s+[" + L_UP + r"a-z#\d])", re.M)
    for m in rx.finditer(text):
        w = m.group(0)
        if m.start() >= 0.85 * n and w[0].isupper():
            tail = text[m.end():m.end() + 3000]
            years = len(re.findall(r"\b(?:1[5-9]|20)\d\d\b", tail))
            if len(re.findall(r",\s*\d{1,4}\b", tail)) >= 25 and years <= 8:
                return m.start()
    return None


def bibliography_spans(text, language="en"):
    """Tramos (inicio, fin) de las listas de referencias y del índice de materias final."""
    lang = language.lower()
    if lang not in _BIB_SECTIONS:
        return []
    out = []
    for s in _heading_starts(text, lang):
        if out and s < out[-1][1]:
            continue
        e = block_end(text, s)
        if e is not None:
            out.append((s, e))
    i = index_start(text, lang)
    if i is not None and any(a <= i < b for a, b in out):
        i = None                                   # «Science Citation Index» dentro de una referencia
    if i is not None:
        out = [(a, b) for a, b in out if a < i] + [(i, len(text))]
    return out



def remove_bibliography(text, language="en"):
    """Texto sin sus listas de referencias ni el índice de materias final."""
    sp = bibliography_spans(text, language)
    if not sp:
        return text
    parts, last = [], 0
    for a, b in sp:
        parts.append(text[last:a]); last = b
    parts.append(text[last:])
    return "\n".join(p.strip("\n") for p in parts if p.strip()).strip()


# ---------------------------------------------------------------------------
# STOPWORDS & LEMMATISATION
# ---------------------------------------------------------------------------

def remove_stopwords(text, stopwords):
    return " ".join(t for t in text.split() if t.lower() not in stopwords)


def apply_user_stoplist(text, stoplist):
    stopset = {w.lower() for w in stoplist}
    return " ".join(t for t in text.split() if t.lower() not in stopset)


def lemmatize_text(text, language="en"):
    nlp = _get_spacy_model(language)
    doc = nlp(text)
    return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)


# ---------------------------------------------------------------------------
# SECTION HEADER LINE REMOVAL
# Elimina líneas que son solo un título de sección (Introduction, Metodología…)
# posiblemente precedido por un número de sección.
# ---------------------------------------------------------------------------

def remove_section_header_lines(text: str, language: str = "en") -> str:
    """
    Elimina líneas que consisten únicamente en un título de sección estructural,
    opcionalmente precedido por un número (ej: "1. Introduction", "3.2 Resultados").

    Solo actúa sobre líneas cortas aisladas (< 80 caracteres) para evitar
    borrar frases legítimas que empiecen por la misma palabra.
    """
    lang = language.lower()
    headers = _SECTION_HEADERS.get(lang, frozenset())
    if not headers:
        return text

    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) >= 80:
            cleaned.append(line)
            continue

        # Quitar número de sección inicial si lo hay
        core = _SECTION_NUM.sub("", stripped).strip().rstrip(":").strip().lower()

        if core in headers:
            continue  # línea de cabecera → descartar

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# KEYWORD METADATA LINE REMOVAL
# Elimina líneas del tipo "Keywords: corpus, NLP, annotation"
# ---------------------------------------------------------------------------

def remove_keyword_lines(text: str) -> str:
    """
    Elimina líneas que son listas de palabras clave (metadatos de artículos).
    Detecta prefijos en inglés, español, francés, catalán y portugués.
    """
    return "\n".join(
        line for line in text.splitlines()
        if not _KEYWORD_PREFIXES.match(line)
    )


# ---------------------------------------------------------------------------
# DISCOURSE NOISE DETECTION (QA)
# Devuelve líneas que consisten únicamente en marcadores de discurso
# genérico, sin contenido léxico sustantivo.
# ---------------------------------------------------------------------------

def _load_discourse_markers(language: str) -> frozenset:
    """Carga marcadores de discurso genérico desde las stoplists del extractor."""
    filename = f"stop_{language}.txt"
    path = _RESOURCES_DIR / filename
    if not path.exists():
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        )


def flag_discourse_noise(text: str, language: str = "en") -> list[str]:
    """
    Devuelve una lista de líneas del texto que consisten únicamente
    en marcadores de discurso genérico (sin contenido léxico real).

    Útil para revisión QA: estas líneas son candidatas a ser ruido
    estructural (headers sueltos, pies de página, etc.).

    No modifica el texto; solo informa.
    """
    markers = _load_discourse_markers(language)
    if not markers:
        return []

    suspicious = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        # Línea corta que coincide exactamente con un marcador
        if stripped in markers and len(stripped.split()) <= 6:
            suspicious.append(line.strip())

    return suspicious


# ---------------------------------------------------------------------------
# HYPHENATION REJOINING
# Reúne palabras partidas por guión a final de línea (artefacto de PDF/OCR).
# "termi-\nnología" → "terminología"
# ---------------------------------------------------------------------------

def rejoin_hyphenated_words(text: str) -> str:
    """
    Reúne palabras divididas por guión al final de línea.
    Solo actúa cuando el guión está pegado a una letra en ambos lados,
    para no afectar a listas con guión o guiones largos estructurales.
    """
    return re.sub(r"([a-záéíóúüñA-ZÁÉÍÓÚÜÑ\w])-\n([a-záéíóúüñA-ZÁÉÍÓÚÜÑ\w])", r"\1\2", text)


# ---------------------------------------------------------------------------
# FIGURE / TABLE CAPTION REMOVAL
# Elimina líneas que son pie de figura o tabla (no son cuerpo del texto).
# ---------------------------------------------------------------------------

_CAPTION_LINE = re.compile(
    r"^\s*(?:fig(?:ure|ura)?s?\.?|tab(?:le|la)s?\.?|gr[aá]fico?s?\.?|cuadros?\.?|image?n?s?\.?|chart?s?\.?)"
    r"\s*\d+[\.\-:]?",
    re.IGNORECASE,
)


def remove_caption_lines(text: str) -> str:
    """
    Elimina líneas que empiezan con "Figure X", "Table X", "Fig. X",
    "Figura X", "Tabla X", "Gráfico X", etc.
    Solo actúa en líneas relativamente cortas (< 200 chars) para no
    borrar párrafos que citan figuras en medio de una frase.
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if _CAPTION_LINE.match(stripped) and len(stripped) < 200:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# URL / DOI / EMAIL LINE REMOVAL
# ---------------------------------------------------------------------------

_URL_LINE = re.compile(
    r"^\s*(?:https?://|www\.|ftp://|doi\s*:\s*10\.|https?://doi\.org/)\S+\s*$",
    re.IGNORECASE,
)
_EMAIL_LINE = re.compile(
    r"^\s*[\w.+\-]+@[\w\-]+\.[a-z]{2,}\s*$",
    re.IGNORECASE,
)
_DOI_LINE = re.compile(
    r"^\s*(?:doi|DOI)\s*[:\s]\s*10\.\d{4,}/\S+\s*$",
    re.IGNORECASE,
)


def remove_url_doi_email_lines(text: str) -> str:
    """
    Elimina líneas que son exclusivamente una URL, un DOI o una dirección
    de correo electrónico. No toca líneas de texto normal que contengan
    URLs en medio de una oración.
    """
    cleaned = []
    for line in text.splitlines():
        if _URL_LINE.match(line) or _EMAIL_LINE.match(line) or _DOI_LINE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# PUBLICATION METADATA LINE REMOVAL
# Elimina líneas de copyright, licencia y fechas de recepción/aceptación.
# ---------------------------------------------------------------------------

_PUBLICATION_METADATA = re.compile(
    r"^\s*(?:"
    r"©|copyright\s|\(c\)\s*\d{4}"                          # copyright
    r"|creative\s+commons|cc\s+by"                           # licencias
    r"|this\s+(?:work|article|paper)\s+is\s+licensed"
    r"|este\s+(?:trabajo|art[íi]culo)\s+est[aá]\s+(?:bajo|publicado)"
    r"|received\s*:|recibido\s*:|re[çc]u\s*:"               # fechas
    r"|accepted\s*:|aceptado\s*:|accept[eé]\s*:"
    r"|published\s*:|publicado\s*:|publi[eé]\s*:"
    r"|available\s+online|disponible\s+en\s+l[íi]nea"
    r"|issn\s*[\:\-]|e-?issn\s*[\:\-]"                     # ISSN
    r")",
    re.IGNORECASE,
)


def remove_publication_metadata_lines(text: str) -> str:
    """
    Elimina líneas de metadatos de publicación: copyright, licencias CC,
    fechas de recepción/aceptación, ISSN.
    Solo actúa en líneas cortas (< 180 chars).
    """
    cleaned = []
    for line in text.splitlines():
        if _PUBLICATION_METADATA.match(line) and len(line.strip()) < 180:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# INLINE CITATION MARKER REMOVAL
# Elimina marcadores de cita numérica [1], [1,2,3], [1-4] dentro del texto.
# No toca citas autor-año (Smith, 2023) para no romper texto legítimo.
# ---------------------------------------------------------------------------

_INLINE_CITATION = re.compile(
    r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]"
)


def remove_inline_citations(text: str) -> str:
    """
    Elimina marcadores de cita numérica entre corchetes: [1], [2,3], [1-4].
    Deja intactas las citas autor-año entre paréntesis.
    """
    return _INLINE_CITATION.sub("", text)


# ---------------------------------------------------------------------------
# JOURNAL / VOLUME METADATA LINE REMOVAL
# Elimina líneas del tipo "Linguistics 12(3): 45–67" o "Vol. 14, No. 2, 2023"
# ---------------------------------------------------------------------------

_JOURNAL_METADATA = re.compile(
    r"^\s*(?:"
    r"vol(?:ume|umen)?\.?\s*\d+"                            # Vol. 14
    r"|n[oú]?(?:mero|\.?)?\s*\d+"                          # No. 3 / Número 2
    r"|\d+\s*\(\s*\d+\s*\)\s*[:\-–]\s*\d"                 # 12(3): 45
    r"|pp?\.?\s*\d+\s*[–\-]\s*\d+"                        # pp. 45-67
    r"|p[aá]g(?:inas?)?\.?\s*\d+"                          # pág. 45
    r")",
    re.IGNORECASE,
)


def remove_journal_metadata_lines(text: str) -> str:
    """
    Elimina líneas de metadatos de revista: volumen, número, páginas.
    Solo en líneas cortas (< 120 chars) para no afectar texto normal.
    """
    cleaned = []
    for line in text.splitlines():
        if _JOURNAL_METADATA.match(line) and len(line.strip()) < 120:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# MOSTLY-NUMERIC LINE REMOVAL
# Elimina líneas estadísticas o de tabla (coordenadas, p-valores, IDs…)
# que no aportan terminología.
# ---------------------------------------------------------------------------

_NON_NUMERIC = re.compile(r"[^\d\s\.\,\-\+\(\)\[\]%=/]")


def remove_statistical_lines(text: str, threshold: float = 0.75) -> str:
    """
    Elimina líneas en las que la proporción de caracteres numéricos y
    símbolos matemáticos supera `threshold` (por defecto 75%).

    Evita borrar años o fechas sueltas: solo actúa en líneas con
    al menos 6 caracteres.
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < 6:
            cleaned.append(line)
            continue
        non_num_chars = len(_NON_NUMERIC.findall(stripped))
        ratio_numeric = 1 - (non_num_chars / len(stripped))
        if ratio_numeric >= threshold:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# MASTER CLEANING FUNCTION
# ---------------------------------------------------------------------------

def remove_running_headers(text: str, min_repeats: int = 5, max_words: int = 12) -> str:
    """
    Remove running headers and footers: short lines that repeat, apart from
    page numbers, many times in the same document (the title of a book or
    the name of its author on every page). Left in the text, they produce
    false candidates that join the end of the header with the next line.
    """
    import re as _re
    from collections import Counter as _Counter

    def _key(line: str) -> str:
        return _re.sub(r"[\d_\W]+", " ", line).strip().lower()

    lines = text.split("\n")
    counts = _Counter(
        _key(l) for l in lines
        if l.strip() and len(l.split()) <= max_words
    )
    repeated = {k for k, n in counts.items() if k and n >= min_repeats}
    if not repeated:
        return text
    return "\n".join(
        l for l in lines
        if not (len(l.split()) <= max_words and _key(l) in repeated)
    )


def clean_text(
    text,
    structure=False,
    remove_stopwords_flag=False,
    stoplist=None,
    extra_stoplist=None,
    lemmatize_flag=False,
    remove_bibliography_flag=False,
    remove_section_headers_flag=False,
    remove_keyword_lines_flag=False,
    rejoin_hyphens_flag=False,
    remove_captions_flag=False,
    remove_urls_flag=False,
    remove_publication_metadata_flag=False,
    remove_inline_citations_flag=False,
    remove_journal_metadata_flag=False,
    remove_statistical_lines_flag=False,
    remove_running_headers_flag=True,
    language="en",
):
    """
    Main public cleaning function.
    Fully modular: the user decide cada componente.

    Flags disponibles
    -----------------
    rejoin_hyphens_flag             : Reúne palabras partidas por guión al final de línea.
    remove_captions_flag            : Elimina pies de figura y tabla (Figure 1, Tabla 3…).
    remove_urls_flag                : Elimina líneas que son solo URL, DOI o email.
    remove_publication_metadata_flag: Elimina copyright, licencias CC, fechas de recepción.
    remove_inline_citations_flag    : Elimina marcadores de cita numérica [1], [2,3].
    remove_journal_metadata_flag    : Elimina líneas de volumen/número/páginas de revista.
    remove_statistical_lines_flag   : Elimina líneas mayoritariamente numéricas (tablas, stats).
    remove_running_headers_flag     : Elimina encabezados y pies de página repetidos (activado por defecto).
    remove_section_headers_flag     : Elimina líneas que son solo un título de sección.
    remove_keyword_lines_flag       : Elimina líneas "Keywords: …" / "Palabras clave: …".
    remove_bibliography_flag        : Elimina la sección de referencias/bibliografía.
    structure                       : Elimina números de página y líneas muy cortas.
    """

    cleaned = text

    # 1. Reunir guiones de fin de línea (antes de cualquier otra operación)
    if rejoin_hyphens_flag:
        cleaned = rejoin_hyphenated_words(cleaned)

    # 1b. Encabezados y pies de página repetidos
    if remove_running_headers_flag:
        cleaned = remove_running_headers(cleaned)

    # 2. Eliminar marcadores de cita inline [1], [2,3]
    if remove_inline_citations_flag:
        cleaned = remove_inline_citations(cleaned)

    # 3. Bibliografía
    if remove_bibliography_flag:
        cleaned = remove_bibliography(cleaned, language)
        cleaned = remove_numeric_only_lines(cleaned)

    # 4. Cabeceras de sección
    if remove_section_headers_flag:
        cleaned = remove_section_header_lines(cleaned, language)

    # 5. Líneas de palabras clave
    if remove_keyword_lines_flag:
        cleaned = remove_keyword_lines(cleaned)

    # 6. Pies de figura/tabla
    if remove_captions_flag:
        cleaned = remove_caption_lines(cleaned)

    # 7. URLs, DOIs, emails
    if remove_urls_flag:
        cleaned = remove_url_doi_email_lines(cleaned)

    # 8. Metadatos de publicación (copyright, fechas, ISSN)
    if remove_publication_metadata_flag:
        cleaned = remove_publication_metadata_lines(cleaned)

    # 9. Metadatos de revista (Vol., No., pp.)
    if remove_journal_metadata_flag:
        cleaned = remove_journal_metadata_lines(cleaned)

    # 10. Líneas estadísticas/numéricas
    if remove_statistical_lines_flag:
        cleaned = remove_statistical_lines(cleaned)

    # 11. Limpieza estructural
    if structure:
        cleaned = clean_structure(cleaned)

    # 12. Stopwords combinadas
    if remove_stopwords_flag:
        sw = get_language_stopwords(language, extra_stoplist)
        if sw:
            cleaned = remove_stopwords(cleaned, sw)
        else:
            print(f"⚠️ No stopwords available for language '{language}'.")

    # 13. Stoplist de usuario
    if stoplist:
        if isinstance(stoplist, str):
            try:
                user_stop = load_stoplist(stoplist)
                cleaned = apply_user_stoplist(cleaned, user_stop)
            except Exception:
                print(f"⚠️ Invalid stoplist file '{stoplist}'. Skipping.")
        elif isinstance(stoplist, list):
            cleaned = apply_user_stoplist(cleaned, stoplist)
        else:
            print(f"⚠️ Ignoring invalid stoplist value: must be list or file path.")

    # 14. Lematización
    if lemmatize_flag:
        cleaned = lemmatize_text(cleaned, language)

    return cleaned
