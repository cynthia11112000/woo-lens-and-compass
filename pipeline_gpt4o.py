"""
pipeline_gpt4o.py — Full-page GPT-4o VLM pipeline for WOO documents.

Uses OpenAI GPT-4o vision to:
  1. Extract page text with layout and structure preservation
  2. Detect document boundaries (new document starts)
  3. Classify document type (Email, Nota, Brief, etc.)
  4. Identify document codes and within-doc page numbers

Drop-in alternative to pipeline_ocr.py with the same output schema.
Use main_gpt4o.py as the entry point.

Usage:
    from pipeline_gpt4o import load_pdf_vlm
    docs = load_pdf_vlm(pdf_path, api_key="sk-...", max_pages=10)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

OPENAI_MODEL    = "gpt-4o"
_BOUNDARY_MODEL = "gpt-4o-mini"   # text-only boundary pass — no vision needed, much cheaper
_RENDER_DPI     = 200
_MAX_IMG_PX     = 1568      # GPT-4o "high" detail works best ≤ 2048px; 1568 is optimal tile size
_MAX_RETRIES    = 3
_MAX_WORKERS    = 10        # concurrent GPT-4o API calls

# WOO redaction code pattern — same as pipeline_ocr.py
_REDACTION_RE = re.compile(r"5\.[12]\.[1-9][a-z]{0,2}", re.IGNORECASE)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are analyzing pages from Dutch government WOO (Wet Open Overheid) disclosure dossiers. "
    "These documents were released under Dutch freedom-of-information law and may contain any type "
    "of internal government communication.\n"
    "Key context you must apply:\n"
    "- IMPORTANT: not all WOO dossiers use inventory stamps. Your primary task is always to "
    "identify document boundaries from the content itself (headers, page counters, document type "
    "changes). Stamps, when present, are a useful label — not the primary split signal.\n"
    "- Some dossiers add a 4-digit inventory code (0001–0999) stamped in a corner or at the "
    "bottom-centre of each page. Others use a 7-digit barcode stamp (format 760XXXD where XXX is "
    "the 3-digit document number). Many dossiers have no stamps at all.\n"
    "- Sensitive content is redacted with solid black rectangles. The applicable WOO legal ground "
    "is printed in small text next to each black box, in the format 5.1.X or 5.2.X "
    "(e.g. '5.1.2e', '5.1.1', '5.2.1'). These are articles of the Wet Open Overheid.\n"
    "- The dossier typically starts with an Inventarislijst: a table listing all documents with "
    "their codes, titles, page counts, and the WOO decision (openbaar/deels openbaar/niet openbaar).\n"
    "- Document types in order of frequency: internal emails, forwarded email threads, memos "
    "(nota's), formal letters (brieven), reports, meeting minutes, chat exports, and legal decisions.\n"
    "Respond ONLY with a valid JSON object matching the schema in the user message. "
    "Do not include markdown code fences or any text outside the JSON object."
)

_PAGE_PROMPT = """\
Analyze this Dutch government document page and return a JSON object with exactly these fields:

{
  "is_new_document": <true if this is clearly the first page of a new document>,
  "doc_code": "<4-digit stamp code like 0143, or null if not visible>",
  "within_doc_page": <integer — page number within this document, e.g. 1, 2, 3 — or null>,
  "category": "<Email | Chat | Nota | Report | Brief | Timeline | Vergadernotulen | Inventarislijst | Other>",
  "doc_subtype": "<email | chat_sms | nota | brief | factuur | besluit | kamerbrief | vergaderverslag | persbericht | rapport | presentatie | advies | convenant | tijdlijn | protocol | other>",
  "has_redactions": <true if black boxes or censored areas are visible>,
  "doc_date": "<primary date as YYYY-MM-DDTHH:MM if a time is visible, else YYYY-MM-DD, or null — see METADATA RULES>",
  "doc_sender": "<author or sender as a plain name string, or null>",
  "email_start": <true if a new email header block starts on this page, else false>,
  "email_from": "<sender from Van:/From: field, or null>",
  "email_to": "<recipient(s) from Aan:/To: field, or null>",
  "email_cc": "<CC field, or null>",
  "email_subject": "<subject from Onderwerp:/Subject: field, or null>",
  "email_date": "<sent date as YYYY-MM-DDTHH:MM if a time is visible, else YYYY-MM-DD, or null>",
  "chat_name": "<name of the chat group or contact if this is a Chat page, else null>",
  "chat_messages": <array of message objects if this is a Chat page, else []>,
  "text": "<all readable text from the page — see TEXT EXTRACTION RULES>"
}

TEXT EXTRACTION RULES:
- Extract ALL text in reading order, preserving paragraph and section structure.
- For email documents, preserve header fields each on their own line:
    Van: ...   Aan: ...   CC: ...   Onderwerp: ...   Datum: ...   Verzonden: ...
- For tables (e.g. Inventarislijst), reproduce rows as tab- or pipe-separated text so the
  structure is recoverable. Include all column values including codes and page counts.
- For page headers/footers (logo text, document reference numbers, "VERTROUWELIJK" stamps):
  include them at the top or bottom of the text field respectively.
- For redacted email addresses where only the domain remains (black box before @domain.nl):
    write: [REDACTED]@domain.nl
- For redacted sections (solid black rectangles / censored areas):
    If a WOO article code (e.g. "5.1.2e") is printed next to the box:  write [REDACTED: 5.1.2e]
    If no article code is visible:                                       write [REDACTED]
- For media in chat exports ([Foto], [Video], [Bestand], [Sticker], [Audio]):
    include the placeholder exactly as printed, e.g. "[Foto]" or "[Video weggelaten]".
- If the page is blank or contains only a stamp/page number with no readable content:
    set text to "" (empty string).
- Use blank lines to separate paragraphs and sections.

DOCUMENT BOUNDARY RULES:
- is_new_document = true ONLY when ONE OR MORE of these conditions hold at the very top of the page:
    * Footer or stamp reads "Pagina 1 van N" or "1/N"
    * A complete new email header block (Van: + Aan: + Onderwerp:) appears at line 1–3
    * A new memo/letter heading with a fresh date and reference number appears at line 1–3
    * The page shows a new Inventarislijst table header
    * A forwarded-message block ("-----Doorgestuurd bericht-----" / "-----Forwarded Message-----")
      appears at the very top, clearly starting a new forwarded document
- is_new_document = false in ALL of the following cases:
    * within_doc_page is 2 or higher
    * The page starts with a salutation like "Geachte" or "Beste" but has NO header block above it
      (this is the continuation of a letter or email body, not a new document)
    * The page starts mid-sentence, mid-paragraph, or mid-list (clear continuation)
    * Quoted/forwarded email headers appear only deep in the body, not at the very top
    * No stamp is visible and the preceding page had a stamp (continuation is the default)
    * The page is blank or contains only a page separator / cover sheet
- doc_code: check ALL four corners AND the bottom-centre for a stamp (format 0001–0999).
    * 4-digit code:  use as-is → "0143"
    * 7-digit barcode (e.g. 7601430): digits 4–6 with leading zero → "0143"
    * Year-like numbers (1900–2099) are NOT document codes — ignore them
- within_doc_page: return X if a stamp or footer explicitly reads "Pagina X van N" or "pag. X".
  Return null if no explicit page-within-document counter is visible.

CATEGORY RULES (pick the best match; when ambiguous prefer the more specific type):
- Email:          Van/Aan/Onderwerp header block; From/To/Subject block; Outlook meeting invite
                  (Required Attendees, Start Date/Time); reply body with quoted headers below
- Chat:           Screenshot or print of a messaging app (WhatsApp, Signal, Teams, SMS, iMessage)
                  showing speech bubbles or a forensic chat export (sms.db, Native Messages)
- Nota:           Internal memo with "NOTA", "Memorandum", or "Briefing" heading; "Aan:", "Van:", "Betreft:"
- Brief:          Formal letter with address block, date, and salutation ("Geachte", "Beste", "Dear")
                  NOTE: a Brief addressed to a minister or parliament member that also has email-like
                  header fields should still be classified as Brief, not Email.
- Report:         Multi-page document with chapter headings, "Rapport", "Onderzoek", "Analyse"
- Timeline:       Chronological list of events or dates, "Tijdlijn", "Chronologisch overzicht"
- Vergadernotulen: Meeting minutes or agenda; "Verslag", "Notulen", "Agenda", "Aanwezig:"
- Inventarislijst: Table inventorying WOO dossier documents — columns: code, title, pages, decision
- Other:          Does not fit any of the above

DOC_SUBTYPE RULES — return exactly one of:
  email | chat_sms | nota | brief | factuur | besluit | kamerbrief | vergaderverslag | persbericht | rapport | presentatie | advies | convenant | tijdlijn | protocol | other
- email:          Van:/Aan:/Onderwerp: or From:/To:/Subject: header fields
- chat_sms:       Chat bubbles, forensic metadata ("FG ProMax", "sms.db", "Status: Sent/Read")
- nota:           "NOTA" heading with Aan:/Van:/Betreft: block
- brief:          Address block + "Geachte"/"Beste"/"Dear" salutation
- factuur:        "Factuur", "BTW", "Bedrag", invoice/order numbers, payment totals
- besluit:        "Besluit", "ingevolge artikel", legal operative clauses, "overwegende dat"
- kamerbrief:     "Aan de Voorzitter van de Tweede Kamer" or "Aan de Voorzitter van de Eerste Kamer"
- vergaderverslag: "Verslag", "Notulen", "Aanwezig:", "Agenda", action-item lists
- persbericht:    Press-release heading, embargo line, "Persbericht", public statement format
- rapport:        Chapter headings/numbered sections spanning multiple pages, "Rapport", "Onderzoek"
- presentatie:    Slide deck layout, "Presentatie", "Slide", bullet-heavy pages, PowerPoint style
- advies:         "Advies", legal or policy advisory opinion, "Op grond van", "wij adviseren"
- convenant:      "Convenant", "Overeenkomst", "Samenwerkingsovereenkomst", signature blocks between multiple parties
- tijdlijn:       Chronological event list, "Tijdlijn", "Chronologisch overzicht", table of dated events
- protocol:       "Protocol", "Procedure", step-by-step working instructions or standards document
- other:          Use only when none of the signals above apply

CHAT PAGE RULES (only when category = "Chat"):
- chat_name: the group or contact name shown at the top of the screen (e.g. "Bezoek praktische zaken").
  If the name is redacted, write "[REDACTED]".
- Extract every visible message into chat_messages (array, top-to-bottom order):
  {
    "sender_position": "left" | "right",   // left = incoming, right = outgoing (device owner)
    "sender_label": "<name or redaction code above the bubble; '[REDACTED]' if blacked out>",
    "timestamp": "<HH:MM visible on the message, or null>",
    "is_system_message": <true for date separators, join/leave events, missed-call notices>,
    "content": "<full message text; use [REDACTED] for blacked-out words; media as [Foto]/[Video]/[Audio]/[Bestand]>"
  }
- System messages (date separators like "Vandaag", "Gisteren", "15 oktober 2024"; join/leave
  notifications; "Berichten en oproepen zijn end-to-end versleuteld") should be included with
  is_system_message = true, sender_position = "left", sender_label = null.
- If a sender name is consistently replaced by the same redaction code (e.g. "5.1.2e"), use
  that code as sender_label across all that person's messages so they can be tracked.
- text field: plain concatenation of all non-system message contents (for search/sorting).

METADATA RULES (apply to ALL document types, including emails):
- doc_date: the primary date this document was written, sent, or issued.
    * Sources: letterhead date, "Datum:", "Date:", "Verzonden:", "Sent:", "Start Date/Time:"
    * Dutch date formats to parse:
        "3 januari 2024"          → "2024-01-03"
        "15 okt. 2024"            → "2024-10-15"   (jan feb mrt/maa apr mei jun jul aug sep okt nov dec)
        "ma 06-03-2024 14:32"     → "2024-03-06T14:32"  (strip day abbreviation)
        "Verzonden: di 14:32"     combined with a visible date → use the date + "T14:32"
        "06/03/2024 14:32:05"     → "2024-03-06T14:32"  (strip seconds)
    * If a time is visible: return "YYYY-MM-DDTHH:MM" (strip seconds if present).
    * For email threads: use the date/time of the most recent (topmost) email on this page.
    * Return null ONLY if no date is visible anywhere on the page.
- doc_sender: name only — no email addresses, no angle brackets.
    * Email: Van:/From: name
    * "Verzonden namens [Name]" or "On behalf of [Name]": use the delegated name
    * Nota/Brief: name or team from signature or "Van:" header line
    * Report: organisation or author from title page
    * Return null if not determinable.

EMAIL HEADER RULES (only when category = "Email"):
- email_start = true when this page begins a NEW email. Signals:
    * Fresh Van:/From: + Aan:/To: + Onderwerp: block at lines 1–5
    * Outlook meeting invite: From: + Required Attendees: + Subject: + Start Date/Time:
    * Reply body starting with salutation/text before any quoted headers on this page
      (e.g. page begins with "Hoi," then shows "-----Oorspronkelijk bericht-----" below)
    * First page of a forwarded email marked "FW:" or "Fwd:" in the subject
- email_start = false if the page is a continuation of an email body or quoted thread.
- email_from: name only; strip email addresses.
    * "Verzonden namens [Name]" → use the delegated name
    * Partially redacted (e.g. "[REDACTED]@minbuza.nl"): write "[REDACTED]"
- email_to, email_cc: names only, comma-separated for multiple recipients. Null if absent.
- email_subject: include FW:/Re:/Fwd: prefixes as-is. "[REDACTED]" if fully redacted.
- email_date: parse Datum:/Verzonden:/Sent:/Date:/Start Date/Time: → YYYY-MM-DD or YYYY-MM-DDTHH:MM.
    Strip day abbreviations and seconds. Null if not visible or not parseable.
- If multiple emails start on the same page: report only the FIRST one's header fields.
- Non-email pages: email_start = false, all email_* fields = null.
"""


# ── Category normalisation ────────────────────────────────────────────────────
# Maps GPT-4o category labels → canonical labels used by visualisation.py
_GPT4O_TO_CATEGORY: dict[str, str] = {
    # GPT-4o raw labels
    "email":           "E-mail",
    "Email":           "E-mail",
    "chat":            "Chat",
    "Chat":            "Chat",
    "nota":            "Nota",
    "report":          "Report",
    "brief":           "Brief",
    "timeline":        "Timeline",
    "vergadernotulen":  "Vergadernotulen",
    "inventarislijst":  "Inventarislijst",
    "Inventarislijst":  "Inventarislijst",
    "other":            "Other",
    # Canonical pass-through (already normalised, e.g. loaded from cache)
    "E-mail":           "E-mail",
    "Nota":             "Nota",
    "Report":           "Report",
    "Brief":            "Brief",
    "Timeline":         "Timeline",
    "Vergadernotulen":  "Vergadernotulen",
    "Other":            "Other",
}

_VALID_DOC_SUBTYPES: set[str] = {
    "email",
    "chat_sms",
    "nota",
    "brief",
    "factuur",
    "besluit",
    "kamerbrief",
    "vergaderverslag",
    "persbericht",
    "rapport",
    "presentatie",
    "advies",
    "convenant",
    "tijdlijn",
    "protocol",
    "other",
}

_CATEGORY_TO_SUBTYPE: dict[str, str] = {
    "E-mail": "email",
    "Chat": "chat_sms",
    "Nota": "nota",
    "Brief": "brief",
    "Report": "rapport",
    "Vergadernotulen": "vergaderverslag",
    "Timeline": "tijdlijn",
    "Inventarislijst": "other",
    "Other": "other",
}


def _normalise_category(raw: str) -> str:
    return _GPT4O_TO_CATEGORY.get(raw.strip(), "Other")


def _normalise_doc_subtype(raw: str | None, category: str) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in _VALID_DOC_SUBTYPES:
        return value
    return _CATEGORY_TO_SUBTYPE.get(category, "other")


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image(img: Image.Image) -> str:
    """Resize to optimal GPT-4o size and base64-encode as JPEG."""
    w, h = img.size
    if max(w, h) > _MAX_IMG_PX:
        scale = _MAX_IMG_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── JSON repair ───────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> dict:
    """
    Close a JSON string that was truncated mid-value (hit max_tokens).
    Walks the string to close any open string literal, then closes open
    arrays/objects in reverse order so json.loads can parse the result.
    """
    s = raw.rstrip()

    # Determine whether we ended inside a string literal
    in_string = False
    escaped   = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        s += '"'   # close the open string

    # Count unclosed brackets/braces (ignoring string contents)
    stack     = []
    in_string = False
    escaped   = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    s += "".join(reversed(stack))
    return json.loads(s)


# ── GPT-4o call ───────────────────────────────────────────────────────────────

def _call_gpt4o(image_b64: str, client, page_index: int, pdf_page_num: int) -> dict:
    """Call GPT-4o vision with retry logic. Returns parsed JSON dict."""
    page_hint = f"[PDF page {pdf_page_num}]\n\n"
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=8192,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": page_hint + _PAGE_PROMPT},
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content or "{}"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                result = _repair_truncated_json(raw)
                print(f"  [gpt4o] p{page_index+1}: JSON truncated — repaired, text may be partial")
                return result

        except json.JSONDecodeError as e:
            print(f"  [gpt4o] p{page_index+1}: JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            print(f"  [gpt4o] p{page_index+1}: API error (attempt {attempt+1}): {err[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {}


# ── Text helpers ──────────────────────────────────────────────────────────────

def _count_redaction_codes(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for m in _REDACTION_RE.finditer(text):
        counts[m.group().lower()] += 1
    return dict(counts)


def _annotate_text(text: str) -> str:
    # Negative lookbehind: skip codes already inside [REDACTED: ...] to prevent
    # GPT-4o's own [REDACTED: 5.1.2e] from becoming [REDACTED: [REDACTED: 5.1.2e]].
    # "[REDACTED: " is exactly 11 chars — a fixed-length lookbehind.
    return re.sub(
        r"(?<!\[REDACTED: )5\.[12]\.[1-9][a-z]{0,2}",
        lambda m: f"[REDACTED: {m.group()}]",
        text,
        flags=re.IGNORECASE,
    )


def _normalise_doc_code(raw) -> str | None:
    """Validate and zero-pad a doc code string. Returns '0143' style or None."""
    if not raw:
        return None
    s = re.sub(r"\D", "", str(raw))   # digits only
    if len(s) == 7:
        # 7-digit barcode: extract digits 4-6 and prefix with 0
        s = "0" + s[3:6]
    elif len(s) >= 4:
        s = s[-4:].zfill(4)
    else:
        return None
    # Must be WOO-style 0NNN (leading zero, not a year)
    if re.fullmatch(r"0\d{3}", s):
        return s
    return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_cache(page_data: list[dict], cache_path: Path) -> None:
    """Save per-page text/metadata to JSON. Images are excluded (re-rendered on load)."""
    payload = [
        {
            "page_num":        p["page_num"],
            "text":            p["text"],
            "doc_code":        p["doc_code"],
            "is_new_document": p["is_new_document"],
            "within_doc_page": p["within_doc_page"],
            "category":        p["category"],
            "doc_subtype":     p.get("doc_subtype") or "other",
            "chat_name":       p.get("chat_name"),
            "chat_messages":   p.get("chat_messages") or [],
            "email_start":     p.get("email_start", False),
            "email_from":      p.get("email_from"),
            "email_to":        p.get("email_to"),
            "email_cc":        p.get("email_cc"),
            "email_subject":   p.get("email_subject"),
            "email_date":      p.get("email_date"),
            "doc_date":        p.get("doc_date"),
            "doc_sender":      p.get("doc_sender"),
        }
        for p in page_data
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"dpi": _RENDER_DPI, "pages": payload}, f, ensure_ascii=False, indent=2)
    print(f"[gpt4o] Cache saved → {cache_path}")


def _load_cache_pages(cache_path: Path) -> tuple[list[dict], int, list[dict]]:
    """Load page metadata from JSON cache. Returns (pages_without_images, dpi, boundary_docs)."""
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    dpi      = data.get("dpi", _RENDER_DPI)
    pages    = data["pages"]
    boundary = data.get("boundary_documents") or []
    # Re-normalise category in case cache was written before the fix
    for p in pages:
        p["category"] = _normalise_category(str(p.get("category") or "Other"))
        p["doc_subtype"] = _normalise_doc_subtype(p.get("doc_subtype"), p["category"])
    return pages, dpi, boundary


def _update_cache_boundaries(documents: list[dict], cache_path: Path) -> None:
    """Append pass-2 boundary decisions to an existing cache file."""
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        data["boundary_documents"] = documents
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[gpt4o] Boundary decisions saved to cache → {cache_path}")
    except Exception as e:
        print(f"[gpt4o] Warning: could not update cache boundaries: {e}")


def _update_cache_emails(emails_by_doc: dict, cache_path: Path) -> None:
    """Store pass-3 full-document email extraction results in the cache file."""
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        data["emails_by_doc"] = emails_by_doc
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[gpt4o] Email extraction saved to cache → {cache_path}")
    except Exception as e:
        print(f"[gpt4o] Warning: could not update cache emails: {e}")


def docs_from_cache(cache_path: Path, pdf_path: Path) -> dict[str, dict]:
    """
    Rebuild the docs dict from a saved JSON cache without calling the API.
    Re-renders page images from the PDF at the original DPI.

    Args:
        cache_path: Path to the JSON cache file written by load_pdf_vlm().
        pdf_path:   Original PDF (needed to re-render thumbnails).

    Returns:
        Same dict[str, dict] as load_pdf_vlm().
    """
    print(f"[gpt4o] Loading cache from {cache_path} ...")
    pages_meta, dpi, boundary_docs = _load_cache_pages(cache_path)
    total = len(pages_meta)

    print(f"[gpt4o] Re-rendering {total} page images from PDF at {dpi} DPI...")
    from pdf2image import convert_from_path
    all_images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=dpi)

    # Attach images to page metadata (cache stores 1-based page_num)
    page_data: list[dict] = []
    for p in pages_meta:
        idx = p["page_num"] - 1
        img = all_images[idx] if idx < len(all_images) else all_images[-1]
        page_data.append({**p, "image": img})

    return _finalise_pipeline(page_data, boundary_docs=boundary_docs or None)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_pdf_vlm(
    pdf_path:   Path,
    api_key:    str | None = None,
    max_pages:  int | None = None,
    cache_path: Path | None = None,
    page_range: tuple[int, int] | None = None,
) -> dict[str, dict]:
    """
    GPT-4o full-page VLM pipeline. Drop-in replacement for load_pdf().

    Args:
        pdf_path:   Path to the PDF file.
        api_key:    OpenAI API key (falls back to OPENAI_API_KEY env var).
        max_pages:  Only process the first N pages (test mode).
        cache_path: If given, save extracted page data to this JSON file after the run.

    Returns:
        dict[str, dict] with the same schema as pipeline_ocr.load_pdf().
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenAI API key provided. "
            "Set the OPENAI_API_KEY environment variable or pass api_key=..."
        )

    client = OpenAI(api_key=api_key)

    # ── Convert PDF to images ────────────────────────────────────────────────
    print(f"[gpt4o] Converting PDF to images at {_RENDER_DPI} DPI...")
    from pdf2image import convert_from_path
    images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=_RENDER_DPI)

    all_images = images
    if page_range is not None:
        pr_start = max(1, page_range[0])
        pr_end   = min(len(all_images), page_range[1])
        images = all_images[pr_start - 1:pr_end]
        page_offset = pr_start - 1  # offset to get real PDF page numbers
        print(f"[gpt4o] Page range: {pr_start}–{pr_end} of {len(all_images)} total pages.")
    elif max_pages is not None:
        images = all_images[:max_pages]
        page_offset = 0
        print(f"[gpt4o] Test mode — processing first {len(images)} pages.")
    else:
        page_offset = 0

    total = len(images)
    print(f"[gpt4o] Analysing {total} pages with {OPENAI_MODEL} (detail=high, workers={_MAX_WORKERS})...")

    # ── Stage 1: per-page GPT-4o analysis (parallel) ────────────────────────
    def _process_page(args: tuple[int, Image.Image]) -> tuple[int, dict]:
        i, img = args
        image_b64 = _encode_image(img)
        return i, _call_gpt4o(image_b64, client, i, pdf_page_num=i + 1 + page_offset)

    raw_results: dict[int, dict] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_process_page, (i, img)): i for i, img in enumerate(images)}
        for future in as_completed(futures):
            i, result = future.result()
            completed += 1
            print(f"[gpt4o] Page {completed}/{total} done (pdf page {i + 1 + page_offset})")
            raw_results[i] = result

    page_data: list[dict] = []
    for i, img in enumerate(images):
        result = raw_results.get(i, {})

        text            = str(result.get("text") or "").strip()
        is_new_doc      = bool(result.get("is_new_document", False))
        doc_code        = _normalise_doc_code(result.get("doc_code"))
        within_doc_page = result.get("within_doc_page")
        category        = _normalise_category(str(result.get("category") or "Other"))
        doc_subtype     = _normalise_doc_subtype(result.get("doc_subtype"), category)

        if within_doc_page is not None:
            try:
                within_doc_page = int(within_doc_page)
            except (ValueError, TypeError):
                within_doc_page = None

        chat_name     = result.get("chat_name") or None
        chat_messages = result.get("chat_messages") or []
        if not isinstance(chat_messages, list):
            chat_messages = []

        email_start   = bool(result.get("email_start", False)) if category == "E-mail" else False
        email_from    = result.get("email_from")    or None
        email_to      = result.get("email_to")      or None
        email_cc      = result.get("email_cc")      or None
        email_subject = result.get("email_subject") or None
        email_date    = result.get("email_date")    or None
        doc_date      = result.get("doc_date")      or None
        doc_sender    = result.get("doc_sender")    or None

        page_data.append({
            "page_num":        i + 1 + page_offset,
            "image":           img,
            "text":            text,
            "doc_code":        doc_code,
            "is_new_document": is_new_doc,
            "within_doc_page": within_doc_page,
            "category":        category,
            "doc_subtype":     doc_subtype,
            "doc_date":        doc_date,
            "doc_sender":      doc_sender,
            "chat_name":       chat_name,
            "chat_messages":   chat_messages,
            "email_start":     email_start,
            "email_from":      email_from,
            "email_to":        email_to,
            "email_cc":        email_cc,
            "email_subject":   email_subject,
            "email_date":      email_date,
        })

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path:
        save_cache(page_data, cache_path)

    return _finalise_pipeline(page_data, client=client, cache_path=cache_path)


def _build_emails_from_pages(doc_code: str, email_pages: list[dict]) -> list[dict]:
    """Assemble individual email dicts from per-page GPT-4o email metadata."""
    emails: list[dict] = []
    current: dict | None = None
    email_idx = 0

    def _split_email_datetime(value: str | None) -> tuple[str | None, str | None]:
        if not value:
            return None, None
        if "T" in value:
            date_part, time_part = value.split("T", 1)
            return date_part, time_part[:5]
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})", value)
        if m:
            return m.group(1), m.group(2)
        return value, None

    for p in email_pages:
        if p.get("email_start") or current is None:
            if current is not None:
                emails.append(current)
            email_idx += 1
            date_part, time_part = _split_email_datetime(p.get("email_date"))
            current = {
                "id":      f"{doc_code}.{email_idx}",
                "subject": p.get("email_subject"),
                "sender":  p.get("email_from"),
                "to":      p.get("email_to"),
                "cc":      p.get("email_cc"),
                "date":    date_part,
                "time":    time_part,
                "text":    p["text"],
            }
        else:
            current["text"] += "\n\n" + p["text"]

    if current:
        emails.append(current)

    return emails


# ── Pass 2: context-aware boundary detection ──────────────────────────────────

_BOUNDARY_SYSTEM = (
    "You are analyzing a Dutch government WOO (Wet Open Overheid) disclosure dossier. "
    "You receive a compact per-page summary and must draw document boundaries and, for "
    "email documents, individual email boundaries. "
    "Not all dossiers use inventory stamps — rely on document content and structure first; "
    "use stamp codes only to confirm or label boundaries you have already identified. "
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_BOUNDARY_PROMPT = """\
Below is a {n}-page WOO dossier summary. Each line:
  p<num>: stamp=<4-digit code|null> wpn=<within-doc page|null> new=<Y|N> cont=<Y|N> cat=<category> eml=<Y|N> | "<first 300 chars of text>"
  (new=Y means pass-1 suggested a new doc starts here; cont=Y means the page text begins mid-sentence — a near-certain continuation; eml=Y means email header fields were detected)

{summary}

Return ONLY this JSON (all page numbers are the p<num> indices above, 1-based):
{{
  "documents": [
    {{
      "doc_code": "<stamp code if consistently present for this document, else null>",
      "start_page": <integer>,
      "email_starts": [<page numbers where individual emails start — include the first page of email docs>]
    }}
  ]
}}

DOCUMENT BOUNDARY RULES — apply signals in priority order:

1. DEFINITIVE (never override):
   - cont=Y means a sentence continued from the previous page. This page CANNOT start a new document.
   - Every page must belong to exactly one document; list documents in ascending start_page order with no gaps.

2. STRONGEST content signals (each alone is sufficient to start a new document):
   - wpn=1 AND new=Y together confirm a new document start.
   - The text preview begins with "Pagina 1 van" or "1/" in a footer/stamp context.
   - The text preview shows a complete, fresh document header at the very top: a full
     Van/Aan/Onderwerp email block, a new memo heading with date + reference, or a new
     letter/rapport title page.

3. MODERATE content signals (combine two or more to split):
   - new=Y alone (pass-1 detected a new header, but could be a quoted/forwarded header in a body)
   - wpn=1 alone (within-doc page resets to 1)
   - Category changes significantly between consecutive pages (e.g. Email → Nota)
   - Text preview clearly shows a new letterhead, new ministry logo, or new document reference

4. STAMP CODES — use only to confirm or label, never as the sole split signal:
   - Same stamp on consecutive pages: confirms they are the same document (do NOT split).
   - Stamp changes between pages: supports a split already suggested by content signals;
     do NOT split on a stamp change alone if content signals show continuation.
   - No stamps at all: rely entirely on signals 1–3. This is normal for many dossiers.
   - A stamp=null page following a stamped page with no content split signals = continuation.

5. DEFAULT — when in doubt, keep pages together. Spurious splits create empty documents;
   missed splits only merge two documents. Merging is the safer error.

EMAIL BOUNDARY RULES:
- email_starts: list page numbers where a genuinely NEW email begins — a fresh
  Van:/From: + Aan:/To: + Onderwerp:/Subject: header block at the top of the page.
- eml=Y fires on QUOTED and FORWARDED headers too — do NOT treat eml=Y alone as a new email start.
- A continuation page (quoted reply chain, body text, page 2+ of a long email) is NOT a new email
  start even if eml=Y.
- One email can span multiple pages; do not create multiple email_starts for the same email.
- Always include the first page of an email-type document in email_starts.
- Leave email_starts empty ([]) for non-email documents.
"""


# ── Full-document email extraction (pass 3) ───────────────────────────────────

_EMAIL_EXTRACT_SYSTEM = (
    "You are a structured data extractor for Dutch government WOO (Wet Open Overheid) email documents. "
    "Extract every individual email from the provided document text with accurate metadata. "
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_EMAIL_EXTRACT_PROMPT = """\
Below is the full text of WOO document {code} — an email document (possibly a thread).

Extract every individual email and return:
{{
  "emails": [
    {{
      "subject":     "<Onderwerp/Subject/Betreft line, or null>",
      "sender":      "<From/Van/Afzender — name only, no email address; '[REDACTED]' if redacted>",
      "to":          "<To/Aan field, or null>",
      "cc":          "<CC field, or null>",
            "date":        "<date as YYYY-MM-DD — parse any format; null if not found>",
            "time":        "<time as HH:MM if visible in the source, else null>",
      "attachments": ["<filename or description>"],
      "body":        "<email body text only — no header lines>"
    }}
  ]
}}

Extraction rules:
- List emails oldest-first (in a reply chain the quoted original is oldest).
- A new email starts at a Van:/From:/Afzender: + Aan:/To: + Onderwerp:/Subject:/Betreft: header block,
  OR at a separator line like "-----Oorspronkelijk bericht-----" / "-----Original Message-----".
- A reply body appearing BEFORE the first quoted header is the newest email — list it LAST.
- date: parse ALL Dutch and European formats to YYYY-MM-DD:
    "3 januari 2024"       → "2024-01-03"
    "maandag 6 mei 2024"   → "2024-05-06"
    "06-03-2024"           → "2024-03-06"
    "DD/MM/YYYY"           → "YYYY-MM-DD"
  Return null only if truly no date is present.
- time: if a visible sent time exists, return it as HH:MM. Otherwise return null.
- attachments: look for "Bijlage:", "Bijlagen:", "Attachment:", or filenames (.pdf, .docx, .xlsx, etc.).
  Return [] if none found.
- body: message body only — strip all Van/Aan/CC/Onderwerp/Datum header lines from the top.
  Preserve [REDACTED: 5.1.2e] markers exactly as found.
- sender: name only. Strip angle brackets and email addresses entirely.
  If redacted, use "[REDACTED]".
- If this is a single email (no thread), return an array with one item.

Document text:
{text}
"""

_EMAIL_EXTRACT_MAX_CHARS = 24_000   # ~6 000 tokens — keeps cost low for gpt-4o-mini


def _extract_emails_full_doc(code: str, text: str, client) -> list[dict]:
    """
    Full-document email extraction via GPT-4o-mini.

    Sends the complete email document text in a single call so GPT can see
    the whole thread, correctly identify split points, parse Dutch dates,
    and extract attachment lists — without page-boundary fragmentation.

    Falls back to an empty list on failure; the caller should then fall back
    to _build_emails_from_pages().
    """
    truncated = text[:_EMAIL_EXTRACT_MAX_CHARS]
    prompt    = _EMAIL_EXTRACT_PROMPT.format(code=code, text=truncated)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_BOUNDARY_MODEL,   # gpt-4o-mini — text only, no vision needed
                max_tokens=4096,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _EMAIL_EXTRACT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            data = json.loads(response.choices[0].message.content or "{}")
            raw  = data.get("emails") or []
            result: list[dict] = []
            for i, em in enumerate(raw, 1):
                result.append({
                    "id":          f"{code}.{i}",
                    "subject":     em.get("subject")     or None,
                    "sender":      em.get("sender")      or None,
                    "to":          em.get("to")          or None,
                    "cc":          em.get("cc")          or None,
                    "date":        em.get("date")        or None,
                    "time":        em.get("time")        or None,
                    "attachments": em.get("attachments") or [],
                    "text":        em.get("body")        or "",
                })
            print(f"  [gpt4o-email] {code}: {len(result)} email(s) via full-doc extraction")
            return result
        except json.JSONDecodeError as e:
            print(f"  [gpt4o-email] {code}: JSON error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"  [gpt4o-email] {code}: API error (attempt {attempt + 1}): {str(e)[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return []


_STAMP_NOISE_RE = re.compile(r"^\s*(?:\d{4,}|[A-Z]{2,}\d+|\[PDF page \d+\])\s*", re.IGNORECASE)


def _continuation_flag(text: str) -> str:
    """Return 'Y' if the page text starts mid-sentence (strong continuation signal)."""
    # Strip leading stamp/barcode noise to get to the actual first word
    clean = _STAMP_NOISE_RE.sub("", text).lstrip()
    if not clean:
        return "N"
    # Starts with a lowercase letter → almost certainly continues from previous page
    return "Y" if clean[0].islower() else "N"


def _build_page_summary(page_data: list[dict]) -> str:
    """Build the compact per-page text table sent to pass 2."""
    lines = []
    for i, p in enumerate(page_data, 1):
        stamp   = p["doc_code"] or "null"
        wpn     = str(p["within_doc_page"]) if p["within_doc_page"] is not None else "null"
        new     = "Y" if p.get("is_new_document") else "N"
        cat     = p.get("category") or "Other"
        eml     = "Y" if p.get("email_start") else "N"
        text    = p.get("text") or ""
        cont    = _continuation_flag(text)
        extra   = ""
        if p.get("email_start"):
            if p.get("email_from"):
                extra += f' from="{str(p["email_from"])[:30]}"'
            if p.get("email_subject"):
                extra += f' subj="{str(p["email_subject"])[:40]}"'
        preview = " ".join(text.split())[:500]
        lines.append(
            f'p{i:03d}: stamp={stamp} wpn={wpn} new={new} cont={cont} cat={cat} eml={eml}{extra} | "{preview}"'
        )
    return "\n".join(lines)


def _call_boundary_pass(summary: str, n_pages: int, client) -> list[dict]:
    """Single GPT-4o-mini text-only call that returns document + email boundaries."""
    prompt = _BOUNDARY_PROMPT.format(n=n_pages, summary=summary)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_BOUNDARY_MODEL,
                max_tokens=4096,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _BOUNDARY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return data.get("documents") or []
        except json.JSONDecodeError as e:
            print(f"  [gpt4o-pass2] JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"  [gpt4o-pass2] API error (attempt {attempt + 1}): {str(e)[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return []


def _build_docs_from_boundaries(
    page_data: list[dict],
    documents: list[dict],
) -> dict[str, dict]:
    """Assign pages to documents based on pass-2 boundary decisions."""
    if not documents:
        docs_raw: dict[str, dict] = {}
        for p in page_data:
            _append_page(docs_raw, "auto_001", p)
        return docs_raw

    n           = len(page_data)
    docs_sorted = sorted(documents, key=lambda d: int(d.get("start_page", 1)))

    idx_to_code: dict[int, str] = {}
    email_start_idxs: set[int]  = set()

    for i, doc in enumerate(docs_sorted):
        start    = int(doc.get("start_page", 1))
        end      = (int(docs_sorted[i + 1].get("start_page", n + 1)) - 1
                    if i + 1 < len(docs_sorted) else n)
        raw_code = doc.get("doc_code")
        code     = (_normalise_doc_code(raw_code) if raw_code else None) or f"auto_{i + 1:03d}"
        for idx in range(start, end + 1):
            idx_to_code[idx] = code
        for ep in (doc.get("email_starts") or []):
            email_start_idxs.add(int(ep))

    docs_raw: dict[str, dict] = {}
    for seq_idx, p in enumerate(page_data, 1):
        code  = idx_to_code.get(seq_idx, f"auto_{seq_idx:03d}")
        p_mod = dict(p)
        # Override pass-1 email_start with pass-2 decisions (more context-aware)
        p_mod["email_start"] = seq_idx in email_start_idxs
        _append_page(docs_raw, code, p_mod)

    return docs_raw


def _finalise_pipeline(
    page_data: list[dict],
    client=None,
    cache_path: Path | None = None,
    boundary_docs: list[dict] | None = None,
) -> dict[str, dict]:
    """Stage 2 + 3: assign doc codes and build the final docs dict."""
    if boundary_docs is not None:
        # Cached boundary decisions — skip pass 2 entirely
        docs_raw = _build_docs_from_boundaries(page_data, boundary_docs)
        method   = "gpt4o-2pass"
    elif client is not None:
        print(f"[gpt4o] Pass 2: identifying boundaries across {len(page_data)} pages...")
        summary   = _build_page_summary(page_data)
        documents = _call_boundary_pass(summary, len(page_data), client)
        if documents:
            docs_raw = _build_docs_from_boundaries(page_data, documents)
            method   = "gpt4o-2pass"
            if cache_path:
                _update_cache_boundaries(documents, cache_path)
        else:
            print("[gpt4o] Pass 2 failed — falling back to stamp-based logic.")
            has_stamps = any(p["doc_code"] for p in page_data)
            docs_raw   = (_build_docs_forward_fill(page_data) if has_stamps
                          else _build_docs_boundary(page_data))
            method     = "gpt4o-stamp" if has_stamps else "gpt4o-boundary"
    else:
        # No client, no cached boundaries — original stamp/boundary logic
        has_stamps = any(p["doc_code"] for p in page_data)
        if has_stamps:
            docs_raw = _build_docs_forward_fill(page_data)
            method   = "gpt4o-stamp"
        else:
            print("[gpt4o] No doc-code stamps found — using VLM boundary detection.")
            docs_raw = _build_docs_boundary(page_data)
            method   = "gpt4o-boundary"

    docs: dict[str, dict] = {}
    for code, d in docs_raw.items():
        full_text       = "\n\n".join(d["texts"])
        annotated       = _annotate_text(full_text)
        redaction_codes = _count_redaction_codes(full_text)

        # First non-"Other" page category wins: email headers only appear on the
        # first page, so majority vote would wrongly demote multi-page emails.
        category = next((c for c in d["categories"] if c != "Other"), "Other")
        doc_subtype = next((s for s in d.get("doc_subtypes", []) if s and s != "other"), None)
        if not doc_subtype:
            doc_subtype = _CATEGORY_TO_SUBTYPE.get(category, "other")

        # Chat: use most common chat name across pages
        chat_names    = d.get("chat_names") or []
        chat_name     = Counter(chat_names).most_common(1)[0][0] if chat_names else None
        chat_messages = d.get("chat_messages") or []

        # Build structured emails for E-mail documents.
        # Prefer the full-document GPT-4o-mini pass (sees the whole thread at once,
        # handles Dutch dates, attachments, reply chains correctly).
        # Fall back to per-page assembly if extraction fails or client is unavailable.
        if category == "E-mail":
            if client is not None:
                structured_emails = _extract_emails_full_doc(code, annotated, client)
                if not structured_emails:
                    structured_emails = _build_emails_from_pages(code, d.get("email_pages", []))
            else:
                structured_emails = _build_emails_from_pages(code, d.get("email_pages", []))
        else:
            structured_emails = []

        # First non-null doc_date / doc_sender across all pages of this document
        doc_date   = next((v for v in d.get("doc_dates", [])   if v), None)
        doc_sender = next((v for v in d.get("doc_senders", []) if v), None)

        docs[code] = {
            "doc_code":         code,
            "pages":            d["pages"],
            "pdf_pages":        d.get("pdf_pages") or [],
            "page_nums_in_doc": d["page_nums_in_doc"],
            "text":             full_text,
            "annotated_text":   annotated,
            "redaction_codes":  redaction_codes,
            "category":         category,
            "doc_subtype":      doc_subtype,
            "method":           method,
            "doc_date":         doc_date,
            "doc_sender":       doc_sender,
            "chat_name":        chat_name,
            "chat_messages":    chat_messages,
            "emails":           structured_emails,
        }

    if cache_path:
        emails_by_doc = {
            code: doc["emails"]
            for code, doc in docs.items()
            if doc.get("emails")
        }
        if emails_by_doc:
            _update_cache_emails(emails_by_doc, cache_path)

    total = len(page_data)
    print(f"[gpt4o] Done — {len(docs)} documents from {total} pages.")
    return docs


# ── Assignment helpers ────────────────────────────────────────────────────────

def _build_docs_forward_fill(page_data: list[dict]) -> dict[str, dict]:
    """Stamp-based assignment with forward-fill. Same logic as pipeline_ocr."""
    docs_raw: dict[str, dict] = {}
    last_code     = None
    unknown_count = 0

    for p in page_data:
        detected = p["doc_code"]
        wpn      = p["within_doc_page"]
        # Never start a new doc if VLM explicitly says this is page 2+
        is_new   = (p["is_new_document"] or wpn == 1) and (wpn is None or wpn == 1)

        if detected:
            current_code = detected
            last_code    = detected
        elif is_new and last_code is not None:
            # Boundary but no new code — create unknown slot
            unknown_count += 1
            current_code  = f"unknown_{unknown_count}"
            last_code     = current_code
        else:
            current_code = last_code or "unknown_1"
            if not last_code:
                unknown_count += 1
                current_code  = f"unknown_{unknown_count}"
                last_code     = current_code

        _append_page(docs_raw, current_code, p)

    return docs_raw


def _build_docs_boundary(page_data: list[dict]) -> dict[str, dict]:
    """Boundary-only assignment (no stamps). Uses VLM is_new_document signal."""
    docs_raw: dict[str, dict] = {}
    doc_index = 0

    for p in page_data:
        is_new = p["is_new_document"] or p["within_doc_page"] == 1
        if is_new or doc_index == 0:
            doc_index += 1
        code = f"auto_{doc_index:03d}"
        _append_page(docs_raw, code, p)

    return docs_raw


def _append_page(docs_raw: dict, code: str, p: dict) -> None:
    if code not in docs_raw:
        docs_raw[code] = {
            "doc_code":         code,
            "pages":            [],
            "pdf_pages":        [],
            "page_nums_in_doc": [],
            "texts":            [],
            "categories":       [],
            "doc_subtypes":     [],
            "doc_dates":        [],   # per-page doc_date values; first non-null wins
            "doc_senders":      [],   # per-page doc_sender values; first non-null wins
            "chat_names":       [],
            "chat_messages":    [],
            "email_pages":      [],
        }
    docs_raw[code]["pages"].append(p["image"])
    docs_raw[code]["pdf_pages"].append(p["page_num"])
    docs_raw[code]["page_nums_in_doc"].append(p["within_doc_page"])
    docs_raw[code]["texts"].append(p["text"])
    docs_raw[code]["categories"].append(p["category"])
    docs_raw[code]["doc_subtypes"].append(_normalise_doc_subtype(p.get("doc_subtype"), p.get("category", "Other")))
    if p.get("doc_date"):
        docs_raw[code]["doc_dates"].append(p["doc_date"])
    if p.get("doc_sender"):
        docs_raw[code]["doc_senders"].append(p["doc_sender"])
    if p.get("chat_name"):
        docs_raw[code]["chat_names"].append(p["chat_name"])
    docs_raw[code]["chat_messages"].extend(p.get("chat_messages") or [])
    docs_raw[code]["email_pages"].append({
        "email_start":   p.get("email_start", False),
        "email_from":    p.get("email_from"),
        "email_to":      p.get("email_to"),
        "email_cc":      p.get("email_cc"),
        "email_subject": p.get("email_subject"),
        "email_date":    p.get("email_date"),
        "text":          p["text"],
    })
