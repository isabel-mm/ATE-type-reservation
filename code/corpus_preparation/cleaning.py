"""
Text cleaning and linguistic normalisation utilities for the corpus.

This module offers (all optional, activated by flag):

Structural noise
    rejoin_hyphens_flag             Rejoins words split by a hyphen at the end of a line.
    structure                       Removes page numbers and very short lines.

Document metadata
    remove_section_headers_flag     Removes lines that are only a section title
                                    (Introduction, Metodología, Conclusiones…).
    remove_keyword_lines_flag       Removes "Keywords: …" / "Palabras clave: …" lines.
    remove_captions_flag            Removes figure and table captions (Figure 1, Tabla 3…).
    remove_urls_flag                Removes lines that are only a URL, DOI or email address.
    remove_publication_metadata_flag Removes copyright, CC licences, submission dates.
    remove_journal_metadata_flag    Removes journal volume/issue/page lines.
    remove_inline_citations_flag    Removes numeric citation markers [1], [2,3].
    remove_statistical_lines_flag   Removes mostly numeric lines (tables, statistics).
    remove_bibliography_flag        Removes the reference lists.

Lexical filtering
    remove_stopwords_flag           Removes stopwords (spaCy + NLTK + extra list).
    stoplist                        Custom stoplist (a list or the path to a .txt file).
    lemmatize_flag                  Lemmatises the text with spaCy.

QA (reports only, the text is not modified)
    flag_discourse_noise(text, language)  Returns lines that consist only of generic
                                          discourse markers.

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
# Section titles that, on a line of their own, are structural noise
# rather than body text.
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

# Prefixes of keyword metadata lines
_KEYWORD_PREFIXES = re.compile(
    r"^\s*(?:keywords?|key\s*words?|palabras?\s*claves?|mots[\s\-]clés?"
    r"|paraules?\s*claus?|palavras?\s*[\-]?chave)\s*[:\-]",
    re.IGNORECASE,
)

# Section number at the start of a line (e.g. "1.", "2.3", "IV.")
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
        print(f"Warning: could not load spaCy stopwords for '{lang}'.")

    # ------------------------------- NLTK STOPWORDS ----------------------------
    nltk_lang = NLTK_LANG_MAP.get(lang)
    if nltk_lang:
        try:
            nltk_sw = nltk_stopwords.words(nltk_lang)
        except LookupError:
            print(f" Downloading NLTK stopwords for '{nltk_lang}'...")
            nltk.download("stopwords")
            nltk_sw = nltk_stopwords.words(nltk_lang)
        except Exception:
            print(f"Warning: error loading NLTK stopwords for '{lang}'.")
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
                print(f"Warning: could not load extra stoplist file '{extra_stoplist}'. Skipping.")

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
            print(f"spaCy blank model initialised for language '{language}'")
        return _SPACY_MODELS[lang]

    # Cached model
    if model_name in _SPACY_MODELS:
        return _SPACY_MODELS[model_name]

    # Try loading installed model
    try:
        nlp = spacy.load(model_name)
        _SPACY_MODELS[model_name] = nlp
        print(f"spaCy model '{model_name}' loaded successfully")
        return nlp

    except OSError:
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            _SPACY_MODELS[model_name] = nlp
            print(f"spaCy model '{model_name}' downloaded and loaded successfully")
            return nlp

        # spacy.cli.download aborts with SystemExit when pip is unavailable
        # (e.g. inside the packaged desktop app)
        except (Exception, SystemExit):
            nlp = spacy.blank(lang)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SPACY_MODELS[model_name] = nlp
            print(
                f"Warning: spaCy model '{model_name}' unavailable. "
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
# Reference lists are removed as blocks, wherever they occur in the document,
# instead of cutting from the heading to the end of the text: this keeps the
# chapters, appendices and notes that follow the references of a book chapter or
# an edited volume. A block starts at a heading ("References", "Bibliografía"…)
# followed by at least three entries ("Surname, Name … year", "[12] … year",
# "ACRONYM = Author year", a URL) and ends where no entry appears for more than
# 1,200 characters, or at an appendix, notes or chapter heading. A final subject
# index is removed too. Checked document by document on our corpus.

_BIB_SECTIONS = {
    "es": [r'\bREFERENCIAS?(?:\s+(?:BIBLIOGR[ÁA]\S{1,3}CAS?|CITADAS?))?\b', r'\bBIBLIOGRAF[ÍI]A(?:\s+(?:CITADA|B[ÁA]SICA|COMPLEMENTARIA))?\b',
           r'\bOBRAS\s+CITADAS\b', r'\bABREVIATURAS\s+Y\s+REFERENCIAS\s+BIBLIOGR[ÁA]\S{1,3}CAS\b'],
    "en": [r'\bREFERENCES(?:\s+CITED)?\b', r'\bREFERENCE\s+LIST\b', r'\bBIBLIOGRAPHY\b', r'\bWORKS\s+CITED\b', r'\bLITERATURE\s+CITED\b'],
}

YEAR = r"(?:(?:1[5-9]|20)\d\d[a-z]?|en\s+prensa|in\s+press|s\.\s?f\.|n\.\s?d\.)"
# Reference entry: "Surname, Name|I." (or "SURNAME, N.") followed shortly by a year.
L_UP = "A-ZÀ-ÞА-ЯЁ"
ENTRY = re.compile(
    r"[" + L_UP + r"][\w'’`´\-]+(?:\s[" + L_UP + r"][\w'’`´\-]+)?,\s+(?:[" + L_UP + r"][\w'’`´\-]*\.?[\s\-]?){1,4}[^\n]{0,300}?\b" + YEAR + r"\b"   # Surname, Name … year
    r"|\[\d{1,3}\][^\n]{0,400}?\b" + YEAR + r"\b"                                                   # [12] … year
    r"|(?:^|(?<=\s))\d{1,3}\.\s[^\n]{0,400}?\b" + YEAR + r"\b"
    r"|\b[" + L_UP + r"]\.(?:\s?[" + L_UP + r"]\.)?\s[" + L_UP + r"]{2,}[\w\-]*[;,][^\n]{0,300}?\b" + YEAR + r"\b"                                     # 12. … year
    r"|^[^\n=]{1,60}\s=\s[^\n]{0,250}?\b" + YEAR + r"\b"                                           # ACRONYM = Author year
    r"|https?://\S+|<?www\.\S+", re.M)                                                         # URL of a source
GAP, MIN_ENTRIES, LOOK, BRIDGE, LOOK_MIN, SHORT_TAIL = 1200, 3, 1500, 3000, 2500, 1500
# Headings that always close a reference list
STOP = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:Appendix|Appendices|APPENDIX|APPENDICES|Anexos?|ANEXOS?|Ap[ée]ndices?|AP[ÉE]NDICES?|"
    r"Notas|NOTAS|Notes|NOTES|Chapter|CHAPTER|Cap[íi]tulo|CAP[ÍI]TULO|CAP´ITULO|Solutions)"
    r"(?=\s*$|\s*[:.]|\s+(?:\d|[IVXLC]+\b|[A-Z]\b|[A-ZÀ-Þ][a-zà-ÿ]))", re.M)

def is_stop(text, a, b):
    """Is there an appendix, notes or chapter heading between a and b? Running headers,
    which repeat throughout the document, are ignored."""
    for m in STOP.finditer(text, a, b):
        ls = text.rfind("\n", 0, m.start()) + 1; le = text.find("\n", m.start()); le = len(text) if le < 0 else le
        line = text[ls:le].strip()
        if line and text.count(line) >= 3 and len(line) < 120:
            continue                                   # running header
        if re.search(r"\bpp?\.|ISBN|\(" + YEAR + r"\)", line + " " + text[m.start():m.start() + 70]):
            continue                                   # "Capítulo VIII, pp. 175…" inside a reference
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
    """The "entry" is in fact a repeated running header ("… Liendo, P. y otros")."""
    around = text[max(0, m.start() - 45):m.start() + 12]
    return len(around.strip()) > 30 and text.count(around) >= 3


def block_end(text, start):
    """End of the reference block starting at `start`, or None if it is not one."""
    ms = [m for m in ENTRY.finditer(text, start) if not _running_header(text, m)]
    if len([m for m in ms if m.start() < start + LOOK_MIN]) < MIN_ENTRIES:
        # short list at the end of the document ("References Biber, D. 1988…")
        if len(text) - start <= SHORT_TAIL and re.search(r"\b" + YEAR + r"\b", text[start:]):
            return len(text)
        return None
    last = ms[0]
    for k, m in enumerate(ms[1:], 1):
        gap = m.start() - last.end()
        if is_stop(text, last.end(), m.start()) or re.search(r"\b(?:Appendix|Anexo|Ap[ée]ndice)\s+(?:\d+|[A-Z])\s*[:.]", text[last.end():m.start()]):
            break
        if gap > GAP:
            # a table or a footer in the middle of the list: skip it if several
            # entries follow right after
            again = [x for x in ms[k:] if x.start() < m.start() + LOOK]
            if gap > BRIDGE or len(again) < 3:
                break
        last = m
    # up to the end of the last entry's line and its short continuation lines
    end = text.find("\n", last.end()); end = len(text) if end < 0 else end
    rest = text[end:]
    dense = 1000 * len(re.findall(r"\b" + YEAR + r"\b", rest)) / max(1, len(rest))
    if ((len(rest) <= SHORT_TAIL and re.search(r"\b" + YEAR + r"\b", rest) or len(rest) <= 2 * SHORT_TAIL and dense >= 3) and not is_stop(text, end, len(text))
            and not re.search(r"\b(?:Appendix|APPENDIX|Anexo|ANEXO|Ap[ée]ndice|AP[ÉE]NDICE)\b", rest)):
        return len(text)                      # leftovers of the list at the end of the document
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
        return len(text)                      # leftovers of the list at the end of the document
    return end

def index_start(text, lang):
    """Final subject index ("Index", "Índice analítico"…), alone or attached to its first entry."""
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
    """Spans (start, end) of the reference lists and of the final subject index."""
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
        i = None                                   # "Science Citation Index" inside a reference
    if i is not None:
        out = [(a, b) for a, b in out if a < i] + [(i, len(text))]
    return out



def remove_bibliography(text, language="en"):
    """The text without its reference lists and final subject index."""
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
# Removes lines that are only a section title (Introduction, Metodología…),
# possibly preceded by a section number.
# ---------------------------------------------------------------------------

def remove_section_header_lines(text: str, language: str = "en") -> str:
    """
    Removes lines that consist only of a section title, optionally preceded by a
    number (e.g. "1. Introduction", "3.2 Resultados").

    Only short lines (< 80 characters) are affected, so that sentences starting with
    the same word are kept.
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

        # drop the section number, if any
        core = _SECTION_NUM.sub("", stripped).strip().rstrip(":").strip().lower()

        if core in headers:
            continue  # heading line: drop it

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# KEYWORD METADATA LINE REMOVAL
# Removes lines such as "Keywords: corpus, NLP, annotation"
# ---------------------------------------------------------------------------

def remove_keyword_lines(text: str) -> str:
    """
    Removes keyword lists (article metadata). Prefixes in English, Spanish, French,
    Catalan and Portuguese are recognised.
    """
    return "\n".join(
        line for line in text.splitlines()
        if not _KEYWORD_PREFIXES.match(line)
    )


# ---------------------------------------------------------------------------
# DISCOURSE NOISE DETECTION (QA)
# Returns lines made only of generic discourse markers, with no lexical
# content.
# ---------------------------------------------------------------------------

def _load_discourse_markers(language: str) -> frozenset:
    """Loads generic discourse markers from the extractor stoplists."""
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
    Returns the lines of the text that consist only of generic discourse markers
    (no real lexical content).

    Useful for quality checks: these lines are likely structural noise (stray
    headings, footers, etc.).

    The text is not modified.
    """
    markers = _load_discourse_markers(language)
    if not markers:
        return []

    suspicious = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        # short line identical to a marker
        if stripped in markers and len(stripped.split()) <= 6:
            suspicious.append(line.strip())

    return suspicious


# ---------------------------------------------------------------------------
# HYPHENATION REJOINING
# Rejoins words split by a hyphen at the end of a line (a PDF/OCR artefact):
# "termi-\nnología" → "terminología"
# ---------------------------------------------------------------------------

def rejoin_hyphenated_words(text: str) -> str:
    """
    Rejoins words split by a hyphen at the end of a line. Only applies when the
    hyphen touches a letter on both sides, so that dashed lists and structural
    dashes are left alone.
    """
    return re.sub(r"([a-záéíóúüñA-ZÁÉÍÓÚÜÑ\w])-\n([a-záéíóúüñA-ZÁÉÍÓÚÜÑ\w])", r"\1\2", text)


# ---------------------------------------------------------------------------
# FIGURE / TABLE CAPTION REMOVAL
# Removes figure and table captions (not body text).
# ---------------------------------------------------------------------------

_CAPTION_LINE = re.compile(
    r"^\s*(?:fig(?:ure|ura)?s?\.?|tab(?:le|la)s?\.?|gr[aá]fico?s?\.?|cuadros?\.?|image?n?s?\.?|chart?s?\.?)"
    r"\s*\d+[\.\-:]?",
    re.IGNORECASE,
)


def remove_caption_lines(text: str) -> str:
    """
    Removes lines starting with "Figure X", "Table X", "Fig. X", "Figura X",
    "Tabla X", "Gráfico X", etc. Only fairly short lines (< 200 characters) are
    affected, so that paragraphs that mention a figure are kept.
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
    Removes lines that are only a URL, a DOI or an email address. Lines of running
    text that contain a URL are kept.
    """
    cleaned = []
    for line in text.splitlines():
        if _URL_LINE.match(line) or _EMAIL_LINE.match(line) or _DOI_LINE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# PUBLICATION METADATA LINE REMOVAL
# Removes copyright, licence and submission/acceptance date lines.
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
    Removes publication metadata lines: copyright, CC licences,
    submission/acceptance dates, ISSN. Only short lines (< 180 characters).
    """
    cleaned = []
    for line in text.splitlines():
        if _PUBLICATION_METADATA.match(line) and len(line.strip()) < 180:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# INLINE CITATION MARKER REMOVAL
# Removes numeric citation markers [1], [1,2,3], [1-4] in the text.
# Author-year citations (Smith, 2023) are kept.
# ---------------------------------------------------------------------------

_INLINE_CITATION = re.compile(
    r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]"
)


def remove_inline_citations(text: str) -> str:
    """
    Removes numeric citation markers in brackets: [1], [2,3], [1-4].
    Author-year citations in parentheses are kept.
    """
    return _INLINE_CITATION.sub("", text)


# ---------------------------------------------------------------------------
# JOURNAL / VOLUME METADATA LINE REMOVAL
# Removes lines such as "Linguistics 12(3): 45–67" or "Vol. 14, No. 2, 2023"
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
    Removes journal metadata lines: volume, issue, pages. Only short lines
    (< 120 characters), so that running text is not affected.
    """
    cleaned = []
    for line in text.splitlines():
        if _JOURNAL_METADATA.match(line) and len(line.strip()) < 120:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# MOSTLY-NUMERIC LINE REMOVAL
# Removes statistical or table lines (coordinates, p-values, IDs…), which
# contain no terminology.
# ---------------------------------------------------------------------------

_NON_NUMERIC = re.compile(r"[^\d\s\.\,\-\+\(\)\[\]%=/]")


def remove_statistical_lines(text: str, threshold: float = 0.75) -> str:
    """
    Removes lines in which the share of digits and mathematical symbols exceeds
    `threshold` (75% by default). Lines shorter than 6 characters are kept, so that
    isolated years or dates survive.
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
    Fully modular: every step is switched on by its own flag.

    Flags
    -----
    rejoin_hyphens_flag             : rejoins words split by a hyphen at the end of a line.
    remove_captions_flag            : removes figure and table captions (Figure 1, Tabla 3…).
    remove_urls_flag                : removes lines that are only a URL, DOI or email address.
    remove_publication_metadata_flag: removes copyright, CC licences, submission dates.
    remove_inline_citations_flag    : removes numeric citation markers [1], [2,3].
    remove_journal_metadata_flag    : removes journal volume/issue/page lines.
    remove_statistical_lines_flag   : removes mostly numeric lines (tables, statistics).
    remove_running_headers_flag     : removes repeated running headers and footers (on by default).
    remove_section_headers_flag     : removes lines that are only a section title.
    remove_keyword_lines_flag       : removes "Keywords: …" / "Palabras clave: …" lines.
    remove_bibliography_flag        : removes the reference lists.
    structure                       : removes page numbers and very short lines.
    """

    cleaned = text

    # 1. rejoin end-of-line hyphens (before anything else)
    if rejoin_hyphens_flag:
        cleaned = rejoin_hyphenated_words(cleaned)

    # 1b. repeated running headers and footers
    if remove_running_headers_flag:
        cleaned = remove_running_headers(cleaned)

    # 2. numeric citation markers [1], [2,3]
    if remove_inline_citations_flag:
        cleaned = remove_inline_citations(cleaned)

    # 3. reference lists
    if remove_bibliography_flag:
        cleaned = remove_bibliography(cleaned, language)
        cleaned = remove_numeric_only_lines(cleaned)

    # 4. section titles
    if remove_section_headers_flag:
        cleaned = remove_section_header_lines(cleaned, language)

    # 5. keyword lines
    if remove_keyword_lines_flag:
        cleaned = remove_keyword_lines(cleaned)

    # 6. figure and table captions
    if remove_captions_flag:
        cleaned = remove_caption_lines(cleaned)

    # 7. URLs, DOIs, emails
    if remove_urls_flag:
        cleaned = remove_url_doi_email_lines(cleaned)

    # 8. publication metadata (copyright, dates, ISSN)
    if remove_publication_metadata_flag:
        cleaned = remove_publication_metadata_lines(cleaned)

    # 9. journal metadata (Vol., No., pp.)
    if remove_journal_metadata_flag:
        cleaned = remove_journal_metadata_lines(cleaned)

    # 10. statistical or numeric lines
    if remove_statistical_lines_flag:
        cleaned = remove_statistical_lines(cleaned)

    # 11. structural cleaning
    if structure:
        cleaned = clean_structure(cleaned)

    # 12. stopwords
    if remove_stopwords_flag:
        sw = get_language_stopwords(language, extra_stoplist)
        if sw:
            cleaned = remove_stopwords(cleaned, sw)
        else:
            print(f"Warning: no stopwords available for language '{language}'.")

    # 13. user stoplist
    if stoplist:
        if isinstance(stoplist, str):
            try:
                user_stop = load_stoplist(stoplist)
                cleaned = apply_user_stoplist(cleaned, user_stop)
            except Exception:
                print(f"Warning: invalid stoplist file '{stoplist}'. Skipping.")
        elif isinstance(stoplist, list):
            cleaned = apply_user_stoplist(cleaned, stoplist)
        else:
            print(f"Warning: ignoring invalid stoplist value: must be list or file path.")

    # 14. lemmatisation
    if lemmatize_flag:
        cleaned = lemmatize_text(cleaned, language)

    return cleaned
