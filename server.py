"""
WOOLens + WOO Organizer unified server.

Serves the frontend and proxies requests to Dutch government WOO databases.
Also exposes /api/analyse — a server-side pipeline endpoint (SSE) that runs
either the OCR pipeline (pdfplumber + tesseract) or the GPT-4o vision pipeline
on a PDF and returns structured emails + timeline JSON.

Run:
    python server.py
    # or: uvicorn server:app --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

app = FastAPI(title="WOOLens + WOO Organizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_HOSTS = {
    "pid.wooverheid.nl",
    "open.overheid.nl",
    "woozm.nl",
    "rijksoverheid.nl",
    "www.rijksoverheid.nl",
}
INDEX_HTML    = Path(__file__).parent / "index.html"
HEADERS       = {"User-Agent": "Mozilla/5.0 (compatible; WOOLens/1.0; +https://woozm.nl)"}

# Valid pipeline identifiers.
_PIPELINE_OCR   = "ocr"
_PIPELINE_GPT4O = "gpt4o"

# Prevent concurrent pipeline runs — stdout capture is process-wide.
_pipeline_lock = threading.Lock()

# Last uploaded PDF kept for in-app page preview.
_SESSION_PDF: Path = Path(tempfile.gettempdir()) / "woo_session.pdf"


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> Response:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return Response(
        content=INDEX_HTML.read_text(encoding="utf-8"),
        media_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/favicon.svg")
async def favicon():
    favicon_path = Path(__file__).parent / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="favicon.svg not found")
    return FileResponse(favicon_path, media_type="image/svg+xml")


# ── /proxy?url=XXX ───────────────────────────────────────────────────────────

@app.get("/proxy")
async def proxy(url: str = Query(..., description="Full URL to proxy")):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail=f"Host not allowed: {parsed.hostname}")
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Only https URLs are allowed")

    async def _stream():
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            async with client.stream("GET", url, headers=HEADERS) as r:
                async for chunk in r.aiter_bytes(chunk_size=65536):
                    yield chunk

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        head = await client.head(url, headers=HEADERS)
        content_type = head.headers.get("content-type", "application/octet-stream")

    return StreamingResponse(_stream(), media_type=content_type)


# ── /api/session_pdf & /api/session_pdf_range ────────────────────────────────

@app.get("/api/session_pdf")
async def session_pdf():
    """Serve the most recently analysed PDF for in-app page preview."""
    if not _SESSION_PDF.exists():
        raise HTTPException(status_code=404, detail="No session PDF available")
    return FileResponse(
        _SESSION_PDF,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/session_pdf_range")
async def session_pdf_range(
    start: int = Query(..., description="First page (1-indexed, inclusive)"),
    end:   int = Query(..., description="Last page (1-indexed, inclusive)"),
):
    """Extract a page range from the session PDF and return it as a new PDF."""
    if not _SESSION_PDF.exists():
        raise HTTPException(status_code=404, detail="No session PDF available")
    if start > end:
        end = start  # be lenient: single-page doc or bad metadata
    import io
    from pypdf import PdfReader, PdfWriter
    try:
        reader = PdfReader(str(_SESSION_PDF), strict=False)
        writer = PdfWriter()
        for page_num in range(start - 1, min(end, len(reader.pages))):
            writer.add_page(reader.pages[page_num])
        buf = io.BytesIO()
        writer.write(buf)
        content = buf.getvalue()
        if not content:
            raise ValueError("Empty PDF output")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )
    except Exception:
        # Fall back to serving the full session PDF so the user still sees the document
        return FileResponse(
            _SESSION_PDF,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )


# ── /infobox?pid=XXX ─────────────────────────────────────────────────────────

@app.get("/infobox")
async def infobox(pid: str = Query(...)):
    target = f"https://pid.wooverheid.nl/?pid={pid}&infobox=true"
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        r = await client.get(target, headers=HEADERS)
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail="Upstream error")
    return Response(content=r.content, media_type="application/json")


# ── /search?q=XXX&page=N ─────────────────────────────────────────────────────

@app.get("/search")
async def search(q: str = Query(...), page: int = Query(1)):
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        r = await client.get(
            "https://woozm.nl/search",
            params={"q": q, "page": page, "json": "true"},
            headers=HEADERS,
        )
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail="Upstream error")
    return Response(content=r.content, media_type="application/json")


# ── /text?pid=XXX ────────────────────────────────────────────────────────────

@app.get("/text")
async def text(pid: str = Query(...)):
    target = f"https://pid.wooverheid.nl/?pid={pid}&text=true"
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(target, headers=HEADERS)
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail="Upstream error")
    return Response(
        content=json.dumps({"text": r.text}),
        media_type="application/json",
    )


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _write_temp_pdf(pdf_bytes: bytes) -> Path:
    """Write bytes to a named temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.close()
    return Path(tmp.name)


# HTTP / server error strings that can appear when a gov portal returns an
# error page instead of document text (HTTP 200 with error body, or OCR of
# an error-page screenshot).  Emails whose *entire* text matches are dropped.
_HTTP_ERROR_RE = re.compile(
    r"^(?:internal\s+server\s+error|bad\s+gateway|service\s+unavailable|"
    r"not\s+found|forbidden|unauthorized|gateway\s+timeout|"
    r"\d{3}\s+(?:error|bad\s+gateway|service\s+unavailable))\s*$",
    re.IGNORECASE,
)

# Matches email header field lines — used to skip them when scanning for
# the first meaningful body line in non-email documents.
_HEADER_LINE_RE = re.compile(
    r"^(?:van|aan|from|to|cc|bcc|datum|onderwerp|subject|verzonden|sent)\s*:",
    re.IGNORECASE,
)


def _get_first_email_subject(doc: dict) -> Optional[str]:
    """Return the subject of the first email in a structured email doc, or None."""
    # GPT-4o pipeline: structured emails already extracted — no re-parsing needed.
    structured = doc.get("emails") or []
    if structured:
        return structured[0].get("subject") or None
    # OCR pipeline fallback: parse from raw text.
    try:
        from email_splitter import split_emails
        emails = split_emails(doc.get("annotated_text") or doc.get("text", ""),
                              doc.get("doc_code", ""))
        return (emails[0].get("subject") or None) if emails else None
    except Exception:
        return None


def _extract_title(doc: dict) -> str:
    """Best-effort title: email subject, or first non-empty text line."""
    if doc.get("category") == "E-mail":
        subject = _get_first_email_subject(doc)
        if subject:
            return subject[:120]
    for line in doc.get("text", "").split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            return stripped[:120]
    return doc.get("doc_code", "Document")


def _extract_sender(doc: dict) -> str:
    """Best-effort sender from the first Van:/From: line."""
    m = re.search(r"(?i)(?:van|from)\s*:\s*(.+)", doc.get("text", "")[:1000])
    return m.group(1).strip()[:80] if m else ""


def _generate_description(doc: dict) -> str:
    """Short human-readable description for document cards."""
    category        = doc.get("category", "Other")
    text            = doc.get("text", "")
    redaction_codes = doc.get("redaction_codes", {})

    if category == "E-mail":
        subject = _get_first_email_subject(doc)
        return subject[:100] if subject else "E-mail"

    for line in text.split("\n")[:20]:
        stripped = line.strip()
        if (len(stripped) > 8
                and not _HEADER_LINE_RE.match(stripped)
                and not re.fullmatch(r"[\d\s.\-\/]+", stripped)):
            return stripped[:100]

    if redaction_codes:
        top_codes = ", ".join(list(redaction_codes.keys())[:3])
        return f"Geredigeerd document ({top_codes})"

    return category


_DOC_SUBTYPE_LABELS = {
    "email": "E-mail",
    "chat_sms": "Chat",
    "nota": "Nota",
    "brief": "Brief",
    "factuur": "Factuur",
    "besluit": "Besluit",
    "kamerbrief": "Kamerbrief",
    "vergaderverslag": "Vergaderverslag",
    "persbericht": "Persbericht",
    "rapport": "Rapport",
    "presentatie": "Presentatie",
    "advies": "Advies",
    "convenant": "Convenant",
    "tijdlijn": "Tijdlijn",
    "protocol": "Protocol",
    "other": "Overig",
}

# Maps OCR-pipeline category strings to subtypes for docs where doc_subtype
# was not set by the pipeline (OCR pipeline only produces category, not subtype).
_CATEGORY_TO_SUBTYPE_FALLBACK = {
    "Presentatie": "presentatie",
    "Advies": "advies",
    "Convenant": "convenant",
    "Timeline": "tijdlijn",
    "Protocol": "protocol",
    "Report": "rapport",
    "Vergadernotulen": "vergaderverslag",
    "Brief": "brief",
    "Nota": "nota",
}


def _display_type_label(category: str, doc_subtype: str) -> str:
    """Return user-facing type label for timeline/cards."""
    cat = (category or "").strip()
    if _is_chat_category(cat):
        return "Chat"
    if (cat or "").lower().startswith("e-mail") or (cat or "").lower() == "email":
        return "E-mail"
    key = (doc_subtype or "other").strip().lower()
    return _DOC_SUBTYPE_LABELS.get(key, cat or "Overig")


def _nonchat_summary_fallback(doc: dict) -> str:
    """Fallback one-sentence summary for non-email/non-chat documents."""
    text = re.sub(r"\s+", " ", (doc.get("annotated_text") or doc.get("text") or "").strip())
    if not text:
        label = _display_type_label(doc.get("category", "Overig"), doc.get("doc_subtype") or "other")
        return f"Dit {label.lower()}-document bevat inhoud uit het Woo-dossier."
    short = text[:220].rstrip(" ,.;:")
    if not short.endswith("."):
        short += "."
    return short


def _generate_nonchat_ai_summary(doc: dict, api_key: Optional[str], cache: dict[str, str]) -> str:
    """Generate a concise AI sentence for non-email/non-chat documents."""
    text = (doc.get("annotated_text") or doc.get("text") or "").strip()
    snippet = re.sub(r"\s+", " ", text)[:4000]
    if not snippet:
        return _nonchat_summary_fallback(doc)

    digest_input = f"{doc.get('doc_subtype') or 'other'}::{snippet}"
    digest = hashlib.sha1(digest_input.encode("utf-8", errors="ignore")).hexdigest()
    if digest in cache:
        return cache[digest]

    if not api_key:
        summary = _nonchat_summary_fallback(doc)
        cache[digest] = summary
        return summary

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        subtype_label = _display_type_label(doc.get("category", "Overig"), doc.get("doc_subtype") or "other")
        prompt = (
            "Je krijgt tekst uit een Nederlands Woo-document. "
            "Schrijf precies 1 korte zin (max 24 woorden) die samenvat wat zichtbaar is in dit document. "
            "Noem geen geredigeerde personen. Geef alleen de zin, zonder labels.\n\n"
            f"Documenttype: {subtype_label}\n"
            f"Tekst:\n{snippet}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = re.sub(r"\s+", " ", (resp.choices[0].message.content or "").strip())
        if not out:
            out = _nonchat_summary_fallback(doc)
        if not out.endswith("."):
            out += "."
        cache[digest] = out
        return out
    except Exception as exc:
        print(f"[server] Non-chat summary generation failed: {exc}")
        summary = _nonchat_summary_fallback(doc)
        cache[digest] = summary
        return summary


def _is_chat_category(category: str) -> bool:
    t = (category or "").strip().lower()
    return "chat" in t or "whatsapp" in t or "teams" in t or "sms" in t or "berichtenverkeer" in t


def _format_chat_title_date(date_str: str) -> str:
    """Render chat title as DD-MM-YYYY (timeline card title only date)."""
    raw = (date_str or "").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r"(\d{2})[\/-](\d{2})[\/-](\d{4})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return raw or "Datum onbekend"


def _chat_summary_fallback(doc: dict) -> str:
    """Fallback one-sentence chat summary without extra API calls."""
    msgs = doc.get("chat_messages") or []
    parts: list[str] = []
    if isinstance(msgs, list):
        for msg in msgs[:4]:
            body = (msg.get("content") or "").strip()
            if body:
                clean = re.sub(r"\s+", " ", body)
                parts.append(clean)
            if len(parts) >= 2:
                break
    if not parts:
        text = re.sub(r"\s+", " ", (doc.get("text") or "").strip())
        if text:
            parts.append(text[:180])
    merged = " ".join(parts).strip()
    if not merged:
        return "Chatgesprek over praktische afstemming en informatie-uitwisseling."
    merged = merged[:220].rstrip(" ,.;:")
    if not merged.endswith("."):
        merged += "."
    return merged


def _generate_chat_ai_summary(doc: dict, api_key: Optional[str], cache: dict[str, str]) -> str:
    """Generate a one-sentence AI summary for one chat day/document."""
    msgs = doc.get("chat_messages") or []
    if not isinstance(msgs, list):
        msgs = []

    lines: list[str] = []
    for msg in msgs[:20]:
        ts = (msg.get("timestamp") or "").strip()
        sender = (msg.get("sender_label") or "").strip()
        body = re.sub(r"\s+", " ", (msg.get("content") or "").strip())
        if not body:
            continue
        prefix = " ".join(v for v in [ts, sender] if v)
        lines.append(f"{prefix}: {body}" if prefix else body)

    if not lines:
        return _chat_summary_fallback(doc)

    joined = "\n".join(lines)
    digest = hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()
    if digest in cache:
        return cache[digest]

    if not api_key:
        summary = _chat_summary_fallback(doc)
        cache[digest] = summary
        return summary

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Vat deze Nederlandse Woo-chatberichten samen in precies 1 zin (max 24 woorden). "
            "Noem geen namen van geredigeerde personen. Geef alleen de zin, zonder labels.\n\n"
            f"Chatberichten:\n{joined}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            text = _chat_summary_fallback(doc)
        if not text.endswith("."):
            text += "."
        cache[digest] = text
        return text
    except Exception as exc:
        print(f"[server] Chat summary generation failed: {exc}")
        summary = _chat_summary_fallback(doc)
        cache[digest] = summary
        return summary


def _split_email_datetime(email: dict) -> tuple[str, str]:
    """Return normalized (date, time) from explicit fields or an ISO datetime."""
    date_val = (email.get("date") or "").strip()
    time_val = (email.get("time") or "").strip()
    if time_val:
        return date_val, time_val
    if "T" in date_val:
        date_part, time_part = date_val.split("T", 1)
        return date_part, time_part[:5]
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})", date_val)
    if m:
        return m.group(1), m.group(2)
    return date_val, ""

_CHAT_LINE_PATTERNS = [
    re.compile(r"^\[(?P<stamp>[^\]]+)\]\s*(?P<sender>[^:]{1,120})\s*:\s*(?P<body>.+)$"),
    re.compile(r"^(?P<sender>.+?)\s*\((?P<stamp>[^)]+)\)\s*:\s*(?P<body>.+)$"),
]
_CHAT_REDACT_RE = re.compile(r"\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]|\b5\.[12]\.\d[a-z]{0,2}\b", re.IGNORECASE)
_CHAT_SYSTEM_RE = re.compile(r"\b(dit\s+bericht\s+is\s+verwijderd|this\s+message\s+was\s+deleted)\b", re.IGNORECASE)
# Matches a body that is entirely redaction markers, WOO codes, and whitespace —
# no actual readable text remains.
_CHAT_ALL_REDACTED_RE = re.compile(
    r"^[\s,;.|]*(?:\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]|\b5\.[12]\.\d[a-z]{0,2}\b)[\s,;.|]*(?:(?:\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]|\b5\.[12]\.\d[a-z]{0,2}\b)[\s,;.|]*)*$",
    re.IGNORECASE,
)

def _normalise_chat_body(body: str) -> str:
    """Collapse bodies that are nothing but redaction codes into a single placeholder."""
    text = (body or "").strip()
    if text and _CHAT_ALL_REDACTED_RE.match(text):
        return "[Bericht weggelakt]"
    return text


def _chat_message_flags(body: str) -> tuple[bool, bool]:
    text = (body or "").strip()
    return bool(_CHAT_REDACT_RE.search(text)), bool(_CHAT_SYSTEM_RE.search(text))

def _chat_thread_id(code: str, participants: list[str], fallback_name: str = "") -> str:
    parts = [p.strip() for p in participants if p and p.strip()]
    key = "|".join(parts) or (fallback_name.strip() or code or "chat")
    safe = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:60] or "chat"
    return f"{code}-chat-{safe}"

def _build_chat_conversation(doc: dict, code: str, doc_date_str: str = "") -> Optional[dict]:
    category = (doc.get("category") or "").strip().lower()
    raw_messages = doc.get("chat_messages") or []
    chat_name = (doc.get("chat_name") or "").strip()
    if not doc_date_str:
        raw = doc.get("date")
        doc_date_str = raw if isinstance(raw, str) else (raw.strftime("%Y-%m-%d") if raw else "")

    messages = []
    if isinstance(raw_messages, list) and raw_messages:
        for msg in raw_messages:
            body = _normalise_chat_body(msg.get("content") or "")
            if not body:
                continue
            sender = (msg.get("sender_label") or ("Eigenaar" if msg.get("sender_position") == "right" else "Onbekend")).strip() or "Onbekend"
            stamp = (msg.get("timestamp") or "").strip()
            iso_stamp = f"{doc_date_str}T{stamp}" if doc_date_str and stamp and re.fullmatch(r"\d{1,2}:\d{2}", stamp) else (stamp or doc_date_str)
            is_redacted, is_system = _chat_message_flags(body)
            messages.append({
                "sender": sender,
                "timestamp": iso_stamp or stamp or doc_date_str,
                "body": body,
                "isRedacted": is_redacted,
                "isSystemMessage": is_system,
            })

    if not messages:
        text = doc.get("annotated_text") or doc.get("text") or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            matched = None
            for pattern in _CHAT_LINE_PATTERNS:
                matched = pattern.match(line)
                if matched:
                    break
            if not matched:
                continue
            sender = matched.group("sender").strip()
            stamp = matched.group("stamp").strip()
            body = _normalise_chat_body(matched.group("body"))
            if not sender or not body:
                continue
            is_redacted, is_system = _chat_message_flags(body)
            messages.append({
                "sender": sender,
                "timestamp": stamp,
                "body": body,
                "isRedacted": is_redacted,
                "isSystemMessage": is_system,
            })

    if not messages and category != "chat":
        return None
    if not messages:
        return {
            "threadId": _chat_thread_id(code, [], chat_name),
            "deelnemers": [chat_name] if chat_name else [],
            "laatsteBericht": "",
            "berichten": [],
            "chatName": chat_name or code,
            "sourceDocId": code,
        }

    participants = []
    seen = set()
    for msg in messages:
        sender = (msg.get("sender") or "").strip()
        if sender and sender.lower() not in seen:
            seen.add(sender.lower())
            participants.append(sender)

    latest_text = next((m.get("body") or "" for m in reversed(messages) if (m.get("body") or "").strip()), "")
    return {
        "threadId": _chat_thread_id(code, participants, chat_name),
        "deelnemers": participants,
        "laatsteBericht": latest_text[:200],
        "berichten": messages,
        "chatName": chat_name or (", ".join(participants[:3]) if participants else code),
        "sourceDocId": code,
    }


def _pick_emails(doc: dict, code: str) -> list[dict]:
    """
    Choose the best email split for one E-mail document.

    Runs both the GPT-4o structured list and the text-based splitter, then
    keeps whichever finds more individual emails.  When the text splitter
    wins, any GPT-4o metadata (subject/sender/date) that the text splitter
    missed is copied in for the emails where indices align.
    """
    from email_splitter import split_emails

    gpt_emails = doc.get("emails") or []  # List[dict]

    try:
        text_emails = split_emails(doc.get("annotated_text") or doc["text"], code)
    except Exception as exc:
        print(f"[server] Warning: text email split failed for {code}: {exc}")
        text_emails = []

    if len(text_emails) <= len(gpt_emails):
        return gpt_emails or text_emails

    # Text splitter found more splits — enrich with GPT-4o metadata where available.
    for i, em in enumerate(text_emails):
        if i >= len(gpt_emails):
            break
        g = gpt_emails[i]
        if not em.get("subject") and g.get("subject"):
            em["subject"] = g["subject"]
        if not em.get("sender") and g.get("sender"):
            em["sender"] = g["sender"]
        if not em.get("date") and g.get("date"):
            em["date"] = g["date"]
        if not em.get("time") and g.get("time"):
            em["time"] = g["time"]
    return text_emails


def _pipeline_to_json(docs: dict, api_key: Optional[str] = None) -> dict:
    """
    Convert pipeline output (dict[str, dict]) to frontend-ready JSON.

    PIL Image objects in 'pages' are excluded; only page counts are forwarded.
    """
    from text_sorting import sort_documents

    sorted_docs = sort_documents(docs)

    emails        = []  # List[dict]
    chats         = []  # List[dict]
    others        = []  # List[dict]
    timeline      = []  # List[dict]
    all_redaction = {}  # Dict[str, int]  # Accumulate redaction codes
    total_pages   = 0
    chat_summary_cache: dict[str, str] = {}
    doc_summary_cache: dict[str, str] = {}

    for code, doc in sorted_docs.items():
        dt       = doc.get("date")
        date_str = dt.strftime("%Y-%m-%d") if dt else ""
        category = doc.get("category", "Other")

        # pdf_pages is set by GPT-4o pipeline; OCR stores PIL Images in 'pages'.
        raw_pages = doc.get("pages", [])
        pdf_pages = doc.get("pdf_pages") or (
            raw_pages if raw_pages and isinstance(raw_pages[0], int) else []
        )
        total_pages += len(raw_pages)

        doc_subtype = doc.get("doc_subtype") or _CATEGORY_TO_SUBTYPE_FALLBACK.get(category, "other")
        is_chat_doc = _is_chat_category(category)
        is_email_doc = (category or "").strip().lower() in {"e-mail", "email"}
        title = _extract_title(doc)
        description = _generate_description(doc)
        if is_chat_doc:
            title = _format_chat_title_date(date_str)
            description = _generate_chat_ai_summary(doc, api_key, chat_summary_cache)
        elif not is_email_doc:
            description = _generate_nonchat_ai_summary(doc, api_key, doc_summary_cache)

        display_type = _display_type_label(category, doc_subtype)

        timeline.append({
            "id":             code,
            "type":           display_type,
            "raw_type":       category,
            "doc_subtype":    doc_subtype,
            "date":           date_str,
            "title":          title,
            "description":    description,
            "sender":         doc.get("doc_sender") or _extract_sender(doc),
            "preview":        (doc.get("annotated_text") or doc.get("text", ""))[:800],
            "pdf_page_start": pdf_pages[0]  if pdf_pages else None,
            "pdf_page_end":   pdf_pages[-1] if pdf_pages else None,
        })

        if category == "E-mail":
            try:
                for em in _pick_emails(doc, code):
                    email_date, email_time = _split_email_datetime(em)
                    emails.append({
                        "id":          em.get("id", f"{code}.?"),
                        "subject":     em.get("subject")     or "",
                        "sender":      em.get("sender")      or "",
                        "to":          em.get("to")          or "",
                        "cc":          em.get("cc")          or "",
                        "date":        email_date,
                        "time":        email_time,
                        "attachments": em.get("attachments") or [],
                        "text":        em.get("text")        or "",
                    })
            except Exception as exc:
                print(f"[server] Warning: email processing failed for {code}: {exc}")

        chat_conv = _build_chat_conversation(doc, code, date_str)
        if chat_conv:
            if is_chat_doc:
                chat_conv["chatDate"] = date_str
                chat_conv["aiSummary"] = description
            chats.append(chat_conv)

        doc_redaction = dict(doc.get("redaction_codes", {}))
        for rc, count in doc_redaction.items():
            all_redaction[rc] = all_redaction.get(rc, 0) + count

        if not is_email_doc and not is_chat_doc:
            others.append({
                "id":             code,
                "type":           display_type,
                "doc_subtype":    doc_subtype,
                "date":           date_str,
                "title":          title,
                "description":    description,
                "sender":         doc.get("doc_sender") or _extract_sender(doc),
                "preview":        (doc.get("annotated_text") or doc.get("text", ""))[:800],
                "pdf_page_start": pdf_pages[0]  if pdf_pages else None,
                "pdf_page_end":   pdf_pages[-1] if pdf_pages else None,
                "redaction_codes": doc_redaction,
            })

    # Drop emails whose entire text is a generic HTTP/server error string.
    # These arise when a gov portal returns an error page with HTTP 200,
    # or when the OCR extracts text from an error-page screenshot in the PDF.
    emails = [e for e in emails if not _HTTP_ERROR_RE.match((e.get("text") or "").strip())]

    return {
        "emails":   emails,
        "chats":    chats,
        "others":   others,
        "timeline": timeline,
        "stats": {
            "docs":           len(docs),
            "pages":          total_pages,
            "redactionCodes": all_redaction,
        },
    }


# ── SSE pipeline runner ───────────────────────────────────────────────────────

async def _run_pipeline_sse(
    pipeline: str,
    pdf_path: Path,
    api_key: Optional[str],
    page_range: Optional[Tuple[int, int]] = None,
) -> AsyncIterator[str]:
    """
    Async generator — runs the chosen pipeline in a background thread,
    forwards captured stdout lines as SSE progress events, then emits
    a final 'done' event with the serialised result.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Optional[dict]] = asyncio.Queue()

    class _StdoutCapture:
        """Thread-safe stdout replacement that forwards lines to the asyncio queue."""
        def __init__(self) -> None:
            self._buf = ""

        def write(self, text: str) -> None:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    asyncio.run_coroutine_threadsafe(
                        q.put({"type": "log", "msg": line}), loop
                    )

        def flush(self) -> None:
            pass

    def _run_in_thread() -> None:
        if not _pipeline_lock.acquire(blocking=False):
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "msg": "Er loopt al een pipeline. Wacht tot deze klaar is of herstart de server."}), loop
            )
            asyncio.run_coroutine_threadsafe(q.put(None), loop)
            return
        old_stdout = sys.stdout
        capture = _StdoutCapture()
        try:
            sys.stdout = capture
            # Deferred imports: pipeline modules are heavy (pdf2image, tesseract,
            # openai) and only needed when a run actually starts.
            if pipeline == _PIPELINE_OCR:
                from pipeline_ocr import load_pdf
                result = load_pdf(pdf_path, page_range=page_range)
            else:
                from pipeline_gpt4o import load_pdf_vlm
                result = load_pdf_vlm(pdf_path, api_key=api_key, page_range=page_range)
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "result", "data": result}), loop
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "msg": str(exc)}), loop
            )
        finally:
            sys.stdout = old_stdout
            _pipeline_lock.release()
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    yield _sse({"type": "progress", "msg": "Pipeline gestart…", "pct": 3})

    result_data = None  # Optional[dict]
    while True:
        item = await q.get()
        if item is None:
            break
        if item["type"] == "log":
            # Thread emits "log"; frontend expects "progress" — translate here.
            yield _sse({"type": "progress", "msg": item["msg"], "pct": None})
        elif item["type"] == "result":
            result_data = item["data"]
        elif item["type"] == "error":
            yield _sse({"type": "error", "msg": item["msg"]})
            return

    if result_data is not None:
        yield _sse({
            "type": "progress",
            "msg":  "Documenten sorteren en e-mails extraheren…",
            "pct":  93,
        })
        try:
            output = _pipeline_to_json(result_data, api_key=api_key)
            output["pdf_url"] = "/api/session_pdf"
            yield _sse({"type": "done", **output})
        except Exception as exc:
            yield _sse({"type": "error", "msg": f"Resultaten verwerken mislukt: {exc}"})


async def _pipeline_with_cleanup(
    pipeline: str,
    pdf_path: Path,
    api_key: Optional[str],
    page_range: Optional[Tuple[int, int]] = None,
) -> AsyncIterator[str]:
    """Wraps _run_pipeline_sse to delete the temp PDF when done.
    Saves a copy to _SESSION_PDF first so the frontend can display PDF pages."""
    shutil.copy2(pdf_path, _SESSION_PDF)
    try:
        async for chunk in _run_pipeline_sse(pipeline, pdf_path, api_key, page_range=page_range):
            yield chunk
    finally:
        pdf_path.unlink(missing_ok=True)


# ── /api/analyse ─────────────────────────────────────────────────────────────

@app.post("/api/analyse")
async def analyse(
    pipeline:   str       = Form(..., description="'ocr' or 'gpt4o'"),
    url:        Optional[str] = Form(None, description="PDF URL (for online dossiers)"),
    api_key:    Optional[str] = Form(None, description="OpenAI API key (gpt4o only)"),
    file:       Optional[UploadFile] = File(None, description="Uploaded PDF file"),
    page_start: Optional[int] = Form(None, description="First page to process (1-indexed, inclusive)"),
    page_end:   Optional[int] = Form(None, description="Last page to process (1-indexed, inclusive)"),
):
    """
    Run the OCR or GPT-4o pipeline on a PDF and stream progress + results via SSE.

    Accepts either:
      - multipart field 'file' (uploaded PDF), or
      - form field 'url'       (PDF downloaded via proxy)

    Returns text/event-stream with JSON events:
      {"type": "progress", "msg": "...", "pct": 0-100 | null}
      {"type": "done",     "emails": [...], "timeline": [...], "stats": {...}}
      {"type": "error",    "msg": "..."}
    """
    if pipeline not in (_PIPELINE_OCR, _PIPELINE_GPT4O):
        raise HTTPException(
            status_code=400,
            detail=f"pipeline must be '{_PIPELINE_OCR}' or '{_PIPELINE_GPT4O}'",
        )

    # ── Obtain PDF bytes ──────────────────────────────────────────────────────
    if file is not None:
        pdf_bytes = await file.read()
    elif url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                r = await client.get(url, headers=HEADERS)
            if not r.is_success:
                raise HTTPException(
                    status_code=502,
                    detail=f"Kon PDF niet ophalen: HTTP {r.status_code}",
                )
            pdf_bytes = r.content
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"PDF ophalen mislukt: {exc}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Geef 'file' of 'url' op",
        )

    pdf_path = _write_temp_pdf(pdf_bytes)

    page_range = (page_start, page_end) if page_start and page_end else None

    return StreamingResponse(
        _pipeline_with_cleanup(pipeline, pdf_path, api_key or None, page_range=page_range),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /api/inventarislijst ─────────────────────────────────────────────────────

@app.post("/api/inventarislijst")
async def analyse_inventarislijst(
    api_key:    str            = Form(...,  description="OpenAI API key for GPT-4o-mini"),
    file:       Optional[UploadFile] = File(None, description="Separate Inventarislijst PDF"),
    pdf_url:    Optional[str]     = Form(None, description="Remote PDF URL to download"),
    page_start: Optional[int]     = Form(None, description="First page in session PDF (1-indexed)"),
    page_end:   Optional[int]     = Form(None, description="Last page in session PDF (1-indexed)"),
):
    """
    Extract the document inventory table from a WOO Inventarislijst.

    Accepts:
      - multipart 'file' — a dedicated Inventarislijst PDF upload
      - 'pdf_url' — a remote PDF URL (downloaded server-side via proxy)
      - no file + page_start/page_end — a page range within the current session PDF
      - no file + no range — the entire session PDF

    Returns JSON: {"items": [...], "total": N}
    Each item: {code, title, date, pages, decision, grounds}
    """
    if file is not None:
        pdf_path   = _write_temp_pdf(await file.read())
        is_tmp     = True
        page_range = None
    elif pdf_url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                r = await client.get(pdf_url, headers=HEADERS)
            if not r.is_success:
                raise HTTPException(
                    status_code=502,
                    detail=f"Kon inventarislijst PDF niet ophalen: HTTP {r.status_code}",
                )
            pdf_path = _write_temp_pdf(r.content)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"PDF ophalen mislukt: {exc}")
        is_tmp     = True
        page_range = None
    elif _SESSION_PDF.exists():
        pdf_path   = _SESSION_PDF
        is_tmp     = False
        page_range = (page_start, page_end) if page_start and page_end else None
    else:
        raise HTTPException(
            status_code=400,
            detail="Geen PDF beschikbaar. Upload eerst een dossier of geef een apart bestand op.",
        )

    try:
        from pipeline_inventarislijst import extract_inventarislijst
        items = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: extract_inventarislijst(pdf_path, api_key, page_range=page_range),
        )
        return {"items": items, "total": len(items)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if is_tmp:
            pdf_path.unlink(missing_ok=True)


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
