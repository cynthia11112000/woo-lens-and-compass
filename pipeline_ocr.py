from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PDF_PATH = Path(__file__).parent / "test.pdf"

# WOO (Wet open overheid) redaction codes.
# Valid grounds are all under article 5.1.x or 5.2.x, e.g. 5.1.2e, 5.1.1, 5.2.1.
# This intentionally excludes plain decimals (2.5), money amounts (50.000), etc.
REDACTION_CODE_RE = re.compile(r"\b(5\.[12]\.[1-9][a-z]{0,2})\b")

# ---------------------------------------------------------------------------
# Multi-region search constants
# ---------------------------------------------------------------------------

# Fractions used to define search regions as (x0, y0, x1, y1) of page dimensions.
_TOP_FRAC  = 0.12   # top-strip height
_SIDE_FRAC = 0.30   # side-strip width
_BTM_FRAC  = 0.10   # bottom-strip height for doc-code search (wider than page-num strip)

# Text-layer doc-code search regions.
# Top regions use a broad \d{4} match (safe — no body text appears in the header strip).
# Bottom regions require a leading zero (0\d{3}) to avoid false positives from
# 4-digit numbers in email body text; all WOO catalogue codes are 0001-0999.
# Each entry: (name, bbox_fracs, leading_zero_required)
_DOC_CODE_TEXT_REGIONS: list[tuple[str, tuple[float, float, float, float], bool]] = [
    # (name, (x0_frac, y0_frac, x1_frac, y1_frac), leading_zero_required)
    ("top-right",  (1 - _SIDE_FRAC, 0.00,          1.00,       _TOP_FRAC), False),
    ("top-left",   (0.00,           0.00,           _SIDE_FRAC, _TOP_FRAC), False),
    ("top-full",   (0.00,           0.00,           1.00,       _TOP_FRAC), False),
    ("btm-right",  (1 - _SIDE_FRAC, 1 - _BTM_FRAC, 1.00,       1.00     ), True),
    ("btm-left",   (0.00,           1 - _BTM_FRAC,  _SIDE_FRAC, 1.00     ), True),
    ("btm-middle", (0.30,           1 - _BTM_FRAC,  0.70,       1.00     ), True),
]

# Raster-OCR doc-code search: all corners and edges.
# Raster stamps are unambiguous (they contain only the code, nothing else), so
# searching bottom regions is safe.  This covers alternative WOO formats where
# the code appears at the bottom-centre or other corners.
_DOC_CODE_RASTER_REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("top-right",  (1 - _SIDE_FRAC, 0.00,          1.00,       _TOP_FRAC)),
    ("top-left",   (0.00,           0.00,           _SIDE_FRAC, _TOP_FRAC)),
    ("btm-right",  (1 - _SIDE_FRAC, 1 - _BTM_FRAC, 1.00,       1.00)),  # before btm-middle
    ("btm-middle", (0.30,           1 - _BTM_FRAC,  0.70,       1.00)),
    ("btm-left",   (0.00,           1 - _BTM_FRAC,  _SIDE_FRAC, 1.00)),
    ("top-full",   (0.00,           0.00,           1.00,       _TOP_FRAC)),
]

# Within-document page-number stamp search: bottom-right only.
# The WOO raster stamp is always placed in the bottom-right corner.
# Searching other bottom regions risks picking up digits from email body text
# (e.g. redaction codes "5.1.2e" → OCR extracts "512" with digits-only filter).
_PAGE_NUM_H_FRAC = 0.06   # tight strip: keeps crop below "Pagina X van Y" text
_PAGE_NUM_W_FRAC = 0.12

_PAGE_NUM_REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("btm-right",  (1 - _PAGE_NUM_W_FRAC, 1 - _PAGE_NUM_H_FRAC, 1.00, 1.00)),
]


# ---------------------------------------------------------------------------
# Low-level OCR helpers
# ---------------------------------------------------------------------------

def _ocr_bottom_right(image: Image.Image) -> int | None:
    """
    OCR a pre-cropped PIL image and return the integer found, or None.
    Shared by all within-document page-number detection paths.
    """
    text = pytesseract.image_to_string(
        image,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    return int(m.group(1)) if m else None


def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Prefers Dutch, falls back to English."""
    try:
        return pytesseract.image_to_string(image, lang="nld+eng")
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng")


def _is_searchable(pdf_path: Path) -> bool:
    """Return True if the PDF has any embedded (selectable) text."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False


# ---------------------------------------------------------------------------
# Multi-region document-code detection
# ---------------------------------------------------------------------------

def _is_year(code: str) -> bool:
    """Return True if a 4-digit code looks like a calendar year (1900–2099).

    Year numbers regularly appear in document headers, footers, and email
    date fields, so they must be excluded from stamp detection to avoid false
    positives.  Genuine WOO document codes are sequential catalogue numbers
    (0001, 0002, …) and never fall in this range in practice.
    """
    return 1900 <= int(code) <= 2099


def _find_doc_code_words(page) -> str | None:
    """
    Search text-layer regions for a 4-digit document code.

    Top regions use a broad \\d{4} match (safe — no body text in the header strip).
    Bottom regions require a leading zero (0\\d{3}) to avoid false positives from
    4-digit numbers in email body text.  All WOO codes are 0001–0999.
    Regions are tried in priority order; within each region the rightmost
    matching word is returned.  Year-range numbers (1900–2099) are excluded.
    """
    w, h = page.width, page.height
    words = page.extract_words()
    for _name, (x0f, y0f, x1f, y1f), leading_zero in _DOC_CODE_TEXT_REGIONS:
        x0, y0, x1, y1 = w * x0f, h * y0f, w * x1f, h * y1f
        pattern = r"0\d{3}" if leading_zero else r"\d{4}"
        candidates = [
            (wd["x0"], wd["text"].strip())
            for wd in words
            if (x0 <= wd["x0"] and wd["x1"] <= x1
                and y0 <= wd["top"] and wd["bottom"] <= y1
                and re.fullmatch(pattern, wd["text"].strip())
                and not _is_year(wd["text"].strip()))
        ]
        if candidates:
            return max(candidates, key=lambda c: c[0])[1]
    return None


# Matches isolated leading-zero WOO codes (e.g. "0144").  Defined at module
# level to avoid recompiling on every call to _find_doc_code_raster().
_LEADING_ZERO_RE = re.compile(r"\b(0\d{3})\b")


def _find_doc_code_raster(image: Image.Image) -> str | None:
    """
    OCR all corner/edge regions of a rendered page image for a 4-digit document
    code.

    Regions in _DOC_CODE_RASTER_REGIONS are tried in priority order; the first
    unambiguous match wins.

    Two-stage matching per region:
      1. Primary  : isolated leading-zero code  ``\\b(0\\d{3})\\b``
                    All known WOO catalogue codes begin with 0 (e.g. 0001, 0144).
                    Requiring the leading zero prevents article-number fragments
                    like "5122" (from 5.1.2.2) from being treated as doc codes.
      2. Fallback : first '0NNN' substring — bottom regions only.
                    Handles noise-merged barcode strings like "7601441" -> "0144".
                    Restricted to bottom regions because top-strip OCR can pick up
                    long email-body numbers (e.g. "825202570600") from which the
                    fallback would wrongly extract a spurious code.
    """
    w, h = image.size
    for name, (x0f, y0f, x1f, y1f) in _DOC_CODE_RASTER_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        text = pytesseract.image_to_string(
            crop, config="--psm 6 -c tessedit_char_whitelist=0123456789"
        )
        m = _LEADING_ZERO_RE.search(text)
        if m and not _is_year(m.group(1)):
            return m.group(1)
        # Fallback: '0NNN' embedded in a longer barcode (at least 2 preceding digits).
        # Pattern: "\d{2,}(0\d{3})" matches barcodes like "7601441" → "0144" and
        # "542240014" → "0014" but NOT short sequences like "00001" (only 5 chars,
        # insufficient for \d{2,} + 0\d{3} = min 6).  Bottom regions only — top-strip
        # OCR may yield long email-body numbers where this could still give false hits.
        if name.startswith("btm"):
            m = re.search(r"\d{2,}(0\d{3})", text)
            if m:
                return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Multi-region within-document page-number detection
# ---------------------------------------------------------------------------

def _find_within_doc_page_raster(image: Image.Image) -> int | None:
    """
    Search the defined page-number regions of a rendered image for the WOO
    within-document page-number stamp.

    The stamp is a raster image added by WOO (not in the text layer).  Regions
    are searched in priority order; returns the first non-None result.
    """
    w, h = image.size
    for _name, (x0f, y0f, x1f, y1f) in _PAGE_NUM_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        result = _ocr_bottom_right(crop)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Code normalisation and annotation
# ---------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    """Strip any spurious second letter from an OCR-garbled redaction code.

    WOO article 5 sub-grounds carry at most one letter suffix (e.g. 5.1.2e).
    OCR occasionally appends an extra letter (e.g. '5.1.2el'); we keep only
    the first letter so it counts toward the correct ground.
    """
    return re.sub(r"([a-z])[a-z]+$", r"\1", code)


def _annotate_redactions(text: str) -> str:
    """Replace bare WOO redaction codes with [REDACTED: …] markers."""
    return REDACTION_CODE_RE.sub(
        lambda m: f"[REDACTED: {_normalize_code(m.group(1))}]", text
    )


# ---------------------------------------------------------------------------
# Document categorisation
# ---------------------------------------------------------------------------

# Each entry is (category_label, [regex_patterns]).
# Patterns are matched case-insensitively against the full document text.
# The category with the most matches wins; ties go to the first entry below.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    # NOTE: E-mail patterns are scored against the FIRST PAGE only (see _categorize_document).
    ("E-mail", [
        # Dutch headers
        r"\bvan:\s",
        r"\baan:\s",
        r"\bonderwerp:\s",
        r"\bverzonden:\s",          # "Verzonden:" = Sent (Dutch Outlook)
        r"\bdatum:\s",              # "Datum:" used in Dutch Outlook replies
        # English headers
        r"\bfrom:\s",
        r"\bto:\s",
        r"\bsubject:\s",
        r"\bsent:\s",               # "Sent:" — Outlook timestamp line
        r"\bcc:\s",
        r"\bbcc:\s",
        # Meeting / calendar invites (Outlook meeting request format)
        r"\brequired attendees:\s",
        r"\boptional attendees:\s",
        r"\bstart date/time:\s",
        r"\bend date/time:\s",
        r"\blocation:\s.*\b(?:webex|teams|skype|zoom|vergader|room)\b",
        # Forwarding / threading markers (both languages)
        r"\boriginal message\b",    # "-----Original Message-----"
        r"\bforwarded message\b",   # "---- Forwarded Message ----"
        r"\bdoorgestuurd bericht\b",
        r"@[\w.-]+\.\w{2,}",       # any email address
    ]),
    ("Nota", [
        r"\bnota\s+aan\b",           # "Nota aan de Minister" — highly distinctive opener
        r"\bnotitie\b",
        r"\bnota\b",
        r"\bter informatie\b",
        r"\bter besluitvorming\b",
        r"\bter advisering\b",
        r"\bdienstonderdeel\b",
        r"\bministerraad\b",
    ]),
    ("Report", [
        r"\brapport\b",
        r"\brapportage\b",
        r"\bonderzoeksrapport\b",
        r"\bsamenvatting\b",
        r"\bconclusie[s]?\b",
        r"\baanbeveling(en)?\b",
        r"\bbevinding(en)?\b",
    ]),
    ("Timeline", [
        r"\btijdlijn\b",
        r"\bchronologie\b",
        r"\bchronologisch\b",
        r"\boverzicht\b.*\bdatum\b",
        r"\bperiode\b",
        r"\b\d{4}\s*[-–]\s*\d{4}\b",  # year ranges like "2020–2021"
    ]),
    ("Vergadernotulen", [
        r"\bnotulen\b",
        r"\bvergadering\b",
        r"\bagendapunt\b",
        r"\bbijeenkomst\b",
        r"\boverleg\b",
        r"\bactions?\b",             # action items in English-language minutes
    ]),
    ("Brief", [
        r"\bgeachte\b",
        r"\bhoogachtend\b",
        r"\bmet vriendelijke groet\b",
        r"\bkenmerk\b",
        r"\bbetreft:\s",
        r"\buw brief\b",
        r"\buw kenmerk\b",
    ]),
    ("Presentatie", [
        r"\bpresentatie\b",
        r"\bslide[s]?\b",
        r"\bpowerpoint\b",
        r"\bdiapositive[s]?\b",
    ]),
    ("Advies", [
        r"\badvies\b",
        r"\badviseur\b",
        r"\bwij\s+adviseren\b",
        r"\bop\s+grond\s+van\b",
        r"\bjuridisch\s+advies\b",
        r"\bbeleidsadvies\b",
    ]),
    ("Convenant", [
        r"\bconvenant\b",
        r"\bsamenwerkingsovereenkomst\b",
        r"\bovereenkomst\b",
        r"\bpartijen\s+komen\s+overeen\b",
        r"\bondergetekenden\b",
    ]),
    ("Protocol", [
        r"\bprotocol\b",
        r"\bwerkprotocol\b",
        r"\bprocedure\b",
        r"\bwerkwijze\b",
        r"\bhandleiding\b",
    ]),
]

# Matches common Dutch and ISO date formats for timeline density detection.
_DATE_RE = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"           # dd-mm-yy(yy)
    r"|\b\d{4}[-/]\d{2}[-/]\d{2}\b"                 # yyyy-mm-dd
    r"|\b\d{1,2}\s+(?:jan|feb|mrt|maa|apr|mei|jun|jul|aug|sep|okt|nov|dec)\w*\s*\d{4}\b",
    re.IGNORECASE,
)


def _categorize_document(text: str, first_page: str = "") -> str:
    """
    Classify a WOO document into a category based on keyword scoring.

    *first_page* should be the text of the first page of the document.
    If omitted, the first 1000 characters of *text* are used as a proxy.
    """
    t = text.lower()
    fp = (first_page or text[:1000]).lower()

    scores: dict[str, int] = {}
    for label, patterns in _CATEGORY_RULES:
        if label == "E-mail":
            scores[label] = sum(1 for p in patterns if re.search(p, fp))
        else:
            scores[label] = sum(1 for p in patterns if re.search(p, t))

    if re.search(r"^\s*(nota|notitie)\b", fp):
        scores["Nota"] = scores.get("Nota", 0) + 3
    if re.search(r"^\s*tijdlijn\b", fp):
        scores["Timeline"] = scores.get("Timeline", 0) + 3

    if len(_DATE_RE.findall(t)) >= 5:
        scores["Timeline"] = scores.get("Timeline", 0) + 2

    best_label, best_score = "Other", 0
    for label, score in scores.items():
        if score > best_score:
            best_label, best_score = label, score
    return best_label


# ---------------------------------------------------------------------------
# Auto-split helpers (used when no document stamps are detected)
# ---------------------------------------------------------------------------

def _is_new_doc_boundary(p: dict) -> bool:
    """
    Return True when high-confidence signals indicate a new document starts
    on this page.  Used in both forward-fill and auto-split modes.

    Signals (in priority order):
      1. WOO within-document page stamp == 1
      2. "Pagina 1 van N" footer in the text layer
    """
    if p["within_doc_page"] == 1:
        return True
    if re.search(r"\bpagina\s+1\s+van\b", p["text"], re.IGNORECASE):
        return True
    return False


def _is_fresh_email_start(text: str) -> bool:
    """
    Return True if this page starts a fresh email document.

    Rules (applied in order):
    1. First-line veto: if the first non-empty line is neither an email header
       field nor a document stamp/code, it's a continuation page → False.
    2. Count ≥ 2 distinct email header fields in the first 20 lines.
    """
    _FIELD_RE = re.compile(
        r"^[ \t]*(?:van|from|aan|to|onderwerp|subject|datum|date|sent|verzonden|betreft|afzender|cc)\s*:",
        re.IGNORECASE,
    )
    # OCR misreads D→O, r→comma, and digits 5/S or S/5.
    # Pattern: D/O · ocn · (r|,)? · digits-or-S
    _STAMP_RE = re.compile(r"^[ \t]*(?:\d{4}|[do]ocn[r,]?\s*[\ds]+)", re.IGNORECASE)
    lines = text.splitlines()
    first_line = next((ln for ln in lines if ln.strip()), None)
    if first_line is not None and not _FIELD_RE.match(first_line) and not _STAMP_RE.match(first_line):
        return False
    count = sum(1 for line in lines[:20] if _FIELD_RE.match(line))
    return count >= 2


def _is_fresh_nota_start(text: str) -> bool:
    """
    Return True if this page starts a fresh Nota/Brief document.

    Dutch government formal documents (nota's, brieven) typically open with a
    metadata block containing fields like Contactpersoon, Datum, and
    Onze referentie / Ons kenmerk.  Two or more of these in the first 15 lines
    is a strong boundary signal — these fields are rare in email body text.

    Same first-line veto as _is_fresh_email_start: if the first non-empty line
    is not one of these fields, it's a continuation page.
    """
    _NOTA_FIELD_RE = re.compile(
        r"^[ \t]*(?:contactpersoon|datum|onze\s+referentie|ons\s+kenmerk"
        r"|uw\s+referentie|uw\s+kenmerk|behandeld\s+door|doorkiesnummer"
        r"|telefoon|bijlage[n]?|onderwerp|betreft)\s*:",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    first_ne = next((ln for ln in lines if ln.strip()), None)
    if first_ne is not None and not _NOTA_FIELD_RE.match(first_ne):
        return False
    count = sum(1 for line in lines[:15] if _NOTA_FIELD_RE.match(line))
    return count >= 2


def _auto_split_boundaries(page_data: list[dict]) -> list[bool]:
    """
    Determine document boundaries when no raster stamps were found anywhere.

    Returns a list of booleans (one per page); True means a new document starts
    at that page.

    Priority:
      1. within_doc_page == 1  OR  'Pagina 1 van' in text  (high confidence)
      2. Fresh email header block                           (medium confidence)
      3. Fresh nota/brief header block                      (medium confidence)
    """
    n = len(page_data)
    is_new = [False] * n
    is_new[0] = True   # first page always starts a document

    for i in range(1, n):
        p = page_data[i]
        text = p["text"]
        if _is_new_doc_boundary(p) or _is_fresh_email_start(text) or _is_fresh_nota_start(text):
            is_new[i] = True

    return is_new


# ---------------------------------------------------------------------------
# Semantic boundary detection (sentence-transformer similarity)
# ---------------------------------------------------------------------------

# Module-level cache so the model loads once per process, not once per call.
_SENTENCE_MODEL = None


def _get_sentence_model():
    """Lazy-load and cache the multilingual sentence-transformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "semantic_split requires the sentence-transformers package. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        print("[data_import] Loading sentence-transformer model (first run downloads ~120 MB)…")
        _SENTENCE_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("[data_import] Model ready.")
    return _SENTENCE_MODEL


# Pattern used to strip redaction markers before embedding — they are structured
# noise that would distort topical similarity scores.
_REDACTION_MARKER_RE = re.compile(r"\[REDACTED:[^\]]+\]")


def _normalize_text_for_embedding(text: str, max_chars: int = 1000) -> str:
    """Prepare a page's text for embedding.

    Strips redaction markers, collapses whitespace, and truncates to max_chars
    (roughly 200 tokens — well within the 512-token model limit).  Returns an
    empty string when fewer than 20 non-whitespace characters remain.
    """
    t = _REDACTION_MARKER_RE.sub(" ", text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t[:max_chars]
    return t if len(t.replace(" ", "")) >= 20 else ""


def _embed_pages(texts: list[str], model) -> list[np.ndarray | None]:
    """Embed page texts in one batch; pages with empty text get None."""
    non_empty = [(i, t) for i, t in enumerate(texts) if t]
    result: list[np.ndarray | None] = [None] * len(texts)
    if not non_empty:
        return result
    indices, batched = zip(*non_empty)
    vectors = model.encode(
        list(batched),
        normalize_embeddings=True,   # unit vectors → cosine = dot product
        show_progress_bar=False,
        batch_size=32,
    )
    for idx, vec in zip(indices, vectors):
        result[idx] = vec
    return result


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] between two L2-normalised 1-D arrays."""
    return float(np.dot(a, b))   # dot product of unit vectors = cosine


def _semantic_boundaries(
    embeddings: list[np.ndarray | None],
) -> list[float | None]:
    """Return per-page cosine similarity to the previous page (None at index 0
    and wherever either neighbour has no embedding)."""
    scores: list[float | None] = [None]
    for i in range(1, len(embeddings)):
        prev, curr = embeddings[i - 1], embeddings[i]
        if prev is not None and curr is not None:
            scores.append(_cosine_sim(prev, curr))
        else:
            scores.append(None)
    return scores


def _auto_split_boundaries_semantic(
    page_data: list[dict],
    embeddings: list[np.ndarray | None],
    threshold: float,
) -> list[bool]:
    """Fuse semantic similarity with existing heuristic signals.

    Signal priority (highest to lowest):
      1. HIGH   within_doc_page==1  OR  'Pagina 1 van' in text  →  always split
      2. MEDIUM cosine_sim < threshold (embedding available)      →  split
      3. MEDIUM fresh email header on this page                   →  split
      4. DEFAULT                                                   →  continue
    """
    scores = _semantic_boundaries(embeddings)
    n = len(page_data)
    is_new = [False] * n
    is_new[0] = True

    for i in range(1, n):
        p = page_data[i]
        score = scores[i]

        text = p["text"]
        if _is_new_doc_boundary(p):                        # signal 1
            is_new[i] = True
        elif score is not None and score < threshold:      # signal 2
            is_new[i] = True
        elif _is_fresh_email_start(text):                  # signal 3
            is_new[i] = True
        elif _is_fresh_nota_start(text):                   # signal 4
            is_new[i] = True

    return is_new


# ---------------------------------------------------------------------------
# Document-code assignment helpers
# ---------------------------------------------------------------------------

def _log_page(num: int, code: str, wpn: int | None,
              codes: Counter, inherited: bool) -> None:
    """Print a single per-page assignment line to stdout."""
    status = f"doc={code}" + (" (inherited)" if inherited else "")
    if wpn is not None:
        status += f"  page-in-doc={wpn}"
    if codes:
        status += f"  redactions={dict(codes)}"
    print(f"  Page {num}: {status}")


def _build_docs_forward_fill(page_data: list[dict]) -> dict:
    """
    Assign document codes using detected stamps with forward-fill fallback.

    Pages whose code is not detectable inherit the previous page's code.
    A page where _is_new_doc_boundary() fires but no code is found gets an
    auto-generated 'unknown_N' code.
    """
    docs: dict[str, dict] = defaultdict(
        lambda: {"pages": [], "text_parts": [], "code_counts": Counter(),
                 "page_nums_in_doc": []}
    )
    last_code: str | None = None
    unknown_count = 0

    for p in page_data:
        i   = p["page_num"]
        det = p["detected_code"]
        wpn = p["within_doc_page"]

        inherited = False
        if det:
            doc_code  = det
            last_code = det
        elif _is_new_doc_boundary(p) and last_code:
            # New document starts here, but no code was readable
            unknown_count += 1
            doc_code  = f"unknown_{unknown_count}"
            last_code = doc_code
        elif last_code:
            doc_code  = last_code
            inherited = True
        else:
            doc_code = "unknown"

        docs[doc_code]["pages"].append(i)
        docs[doc_code]["text_parts"].append(p["text"])
        docs[doc_code]["code_counts"].update(p["code_counts"])
        docs[doc_code]["page_nums_in_doc"].append(wpn)
        _log_page(i, doc_code, wpn, p["code_counts"], inherited)

    return docs


def _build_docs_auto_split(
    page_data: list[dict],
    boundaries: list[bool] | None = None,
) -> dict:
    """
    Assign auto-generated document codes (auto_001, auto_002, …) based on
    automatically detected document boundaries.

    If *boundaries* is None, the heuristic _auto_split_boundaries() is used.
    Pass a pre-computed list to use semantic or other boundaries instead.
    """
    if boundaries is None:
        boundaries = _auto_split_boundaries(page_data)
    docs: dict[str, dict] = defaultdict(
        lambda: {"pages": [], "text_parts": [], "code_counts": Counter(),
                 "page_nums_in_doc": []}
    )
    doc_counter  = 0
    current_code = "auto_001"

    for p, is_new in zip(page_data, boundaries):
        if is_new:
            doc_counter += 1
            current_code = f"auto_{doc_counter:03d}"

        docs[current_code]["pages"].append(p["page_num"])
        docs[current_code]["text_parts"].append(p["text"])
        docs[current_code]["code_counts"].update(p["code_counts"])
        docs[current_code]["page_nums_in_doc"].append(p["within_doc_page"])
        _log_page(p["page_num"], current_code, p["within_doc_page"],
                  p["code_counts"], False)

    return docs


def _finalize_docs(docs_raw: dict, method: str) -> dict[str, dict]:
    """Convert raw per-document accumulator into the final output dict."""
    result = {}
    for code in sorted(docs_raw):
        data     = docs_raw[code]
        combined = "\n\n".join(data["text_parts"])
        first_pg = data["text_parts"][0] if data["text_parts"] else ""
        result[code] = {
            "doc_code":        code,
            "pages":           data["pages"],
            "page_nums_in_doc": data["page_nums_in_doc"],
            "text":            combined,
            "annotated_text":  _annotate_redactions(combined),
            "redaction_codes": dict(data["code_counts"]),
            "category":        _categorize_document(combined, first_pg),
            "method":          method,
        }
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_pdf(
    pdf_path: Path = PDF_PATH,
    ocr_supplement: bool = False,
    semantic_split: bool = False,
    semantic_threshold: float = 0.35,
    page_range: tuple[int, int] | None = None,
) -> dict[str, dict]:
    """
    Extract text from *pdf_path* and group pages by their 4-digit document code.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to process.
    ocr_supplement : bool, default False
        When True, every page of a searchable PDF is also rendered at 300 DPI
        and OCR'd.  The redaction-code count per page is taken as the max of
        the text-layer count and the OCR count, capturing image-only stamps
        that are invisible to the text layer.  Adds ~3 s per page.
    semantic_split : bool, default False
        When True and no stamps are found, uses sentence-transformer cosine
        similarity to detect document boundaries by topic shift.  Falls back
        to the heuristic auto-split when this is False.  Requires the
        sentence-transformers package.
    semantic_threshold : float, default 0.35
        Cosine similarity below which an adjacent-page transition is treated
        as a document boundary.  Only used when semantic_split=True.
        Tune downward (e.g. 0.20) to merge more aggressively; upward (e.g.
        0.50) to split more finely.
    Returns
    -------
    dict[str, dict]
        Keyed by document code (4-digit stamp, 'unknown_N', or 'auto_NNN'):

        {
            "0001": {
                "doc_code":         str,
                "pages":            list[int],        # 1-based PDF page numbers
                "page_nums_in_doc": list[int | None], # within-doc page stamps
                "text":             str,               # concatenated raw text
                "annotated_text":   str,               # text with [REDACTED: …] markers
                "redaction_codes":  dict[str, int],    # code → total count
                "category":         str,               # one of 7 categories
                "method":           "direct" | "ocr",
            },
            ...
        }

    Document detection strategy (trust hierarchy)
    ----------------------------------------------
    1. Stamped indicators found  →  forward-fill with 'unknown_N' for readable
       boundaries that lack a code.
    2. No stamps found anywhere  →
         semantic_split=True  →  sentence-transformer similarity + heuristics
         semantic_split=False →  heuristic auto-split only (email headers,
                                 'Pagina 1 van', within-doc page stamp == 1)

    Region search order
    -------------------
    Doc-code detection searches text-layer regions (_DOC_CODE_TEXT_REGIONS)
    first, then raster regions (_DOC_CODE_RASTER_REGIONS).  Page-number
    detection uses _PAGE_NUM_REGIONS.  All lists are tried in priority order,
    so the pipeline handles PDFs where the stamp is not in the primary WOO
    location.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    searchable = _is_searchable(pdf_path)
    method     = "direct" if searchable else "ocr"

    images = None
    if not searchable:
        print("[data_import] Image-based PDF — running OCR (DPI=300, may take a moment).")
        images = convert_from_path(pdf_path, dpi=300)
    else:
        print("[data_import] Searchable PDF — using direct text extraction.")

    if ocr_supplement and searchable:
        print("[data_import] OCR supplement enabled — rendering all pages at 300 DPI.")

    # ── Stage 1: per-page data collection ────────────────────────────────────
    page_data: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if page_range is not None:
            pr_start = max(1, page_range[0])
            pr_end   = min(total_pages, page_range[1])
            print(f"[data_import] Page range: {pr_start}–{pr_end} of {total_pages} total pages.")
            pages_iter = [(idx, pdf.pages[idx - 1]) for idx in range(pr_start, pr_end + 1)]
        else:
            pages_iter = list(enumerate(pdf.pages, start=1))

        for i, page in pages_iter:

            if searchable:
                # One 200 DPI render serves both page-number and doc-code detection.
                rendered = page.to_image(resolution=200).original

                within_doc_page = _find_within_doc_page_raster(rendered)

                detected_code = _find_doc_code_words(page)
                # At a stamp-boundary (page 1 of doc) the text layer can misread
                # the code from email body text — raster OCR corrects this.
                # Also try raster when the text layer found nothing at all.
                if within_doc_page == 1 or detected_code is None:
                    raster = _find_doc_code_raster(rendered)
                    if raster:
                        detected_code = raster

                text    = page.extract_text() or ""
                was_ocr = False
                if len(text.strip()) < 20:
                    # Rasterized page embedded in an otherwise searchable PDF
                    was_ocr = True
                    try:
                        text = _ocr_image(page.to_image(resolution=300).original)
                    except Exception:
                        pass

            else:
                # Fully image-based PDF — everything comes from pre-rendered images
                rendered        = images[i - 1] if images and i - 1 < len(images) else None
                within_doc_page = _find_within_doc_page_raster(rendered) if rendered else None
                detected_code   = _find_doc_code_raster(rendered) if rendered else None
                text            = _ocr_image(rendered) if rendered else ""
                was_ocr         = True

            # Redaction-code counting (with optional OCR supplement)
            page_codes: Counter = Counter(
                _normalize_code(c) for c in REDACTION_CODE_RE.findall(text)
            )
            if ocr_supplement and searchable and not was_ocr:
                try:
                    hi_res    = page.to_image(resolution=300).original
                    ocr_text  = _ocr_image(hi_res)
                    ocr_codes = Counter(
                        _normalize_code(c) for c in REDACTION_CODE_RE.findall(ocr_text)
                    )
                    all_keys   = set(page_codes) | set(ocr_codes)
                    page_codes = Counter(
                        {k: max(page_codes[k], ocr_codes[k]) for k in all_keys}
                    )
                except Exception:
                    pass

            page_data.append({
                "page_num":        i,
                "detected_code":   detected_code,
                "within_doc_page": within_doc_page,
                "text":            text,
                "code_counts":     page_codes,
            })

    # ── Stage 2: assign document codes ───────────────────────────────────────
    has_stamps = any(p["detected_code"] for p in page_data)
    if has_stamps:
        docs_raw = _build_docs_forward_fill(page_data)
    elif semantic_split:
        print("[data_import] No stamps found — using semantic document splitting.")
        model = _get_sentence_model()
        norm_texts = [_normalize_text_for_embedding(p["text"]) for p in page_data]
        embeddings = _embed_pages(norm_texts, model)
        scores = _semantic_boundaries(embeddings)
        candidates = [
            (i + 1, f"{s:.3f}")
            for i, s in enumerate(scores[1:], start=1)
            if s is not None and s < semantic_threshold
        ]
        print(f"[data_import] Semantic: {len(candidates)} boundary candidate(s)"
              f" at threshold={semantic_threshold}")
        for pg, sc in candidates:
            print(f"  Page {pg}: similarity={sc}")
        boundaries = _auto_split_boundaries_semantic(page_data, embeddings, semantic_threshold)
        docs_raw = _build_docs_auto_split(page_data, boundaries)
    else:
        print("[data_import] No stamps found — using heuristic document splitting.")
        docs_raw = _build_docs_auto_split(page_data)

    result = _finalize_docs(docs_raw, method)
    n_pages = sum(len(d["pages"]) for d in result.values())
    print(f"[data_import] Done. {n_pages} page(s) → {len(result)} document(s).")
    return result
