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
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, Union

import os
from pathlib import Path as _Path

# Load .env if present (before anything reads os.environ)
_env_file = _Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

app = FastAPI(title="WOOLens + WOO Organizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Compass router ────────────────────────────────────────────────────────────
try:
    from routers.compass import router as _compass_router
    app.include_router(_compass_router)
except Exception as _compass_err:  # noqa: BLE001
    import warnings
    warnings.warn(f"WOOLens Compass router could not be loaded: {_compass_err}")

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
_session_language: str = "nl"   # persisted per server session; updated by /api/analyse


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


@app.get("/api/config")
async def api_config() -> dict:
    """Expose minimal runtime config to the frontend (key presence indicator only)."""
    return {"openai_api_key": "set" if os.environ.get("OPENAI_API_KEY") else ""}


from fastapi import Request
from fastapi.responses import JSONResponse

@app.post("/api/openai/chat")
async def openai_chat_proxy(request: Request) -> JSONResponse:
    """Proxy OpenAI chat completions through the server so the API key stays backend-only.
    Accepts an optional 'api_key' field in the JSON body to override the server env var."""
    body_bytes = await request.body()
    try:
        body_json = json.loads(body_bytes)
    except Exception:
        body_json = {}

    # Allow the frontend to supply a key (e.g. user-entered) — strip it before forwarding.
    _front_key = body_json.pop("api_key", None)
    if _front_key and not _front_key.startswith("sk-"):
        _front_key = None
    api_key = _front_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Geen OpenAI API-sleutel geconfigureerd. Vul een sleutel in."}},
        )
    forward_body = json.dumps(body_json).encode()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            content=forward_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/favicon.svg")
async def favicon():
    favicon_path = Path(__file__).parent / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="favicon.svg not found")
    return FileResponse(favicon_path, media_type="image/svg+xml")


# ── /pdfs/{filename} ─────────────────────────────────────────────────────────

@app.get("/pdfs/{filename:path}")
async def serve_pdf(filename: str):
    pdf_path = Path(__file__).parent / "pdfs" / filename
    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf")


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


_LANG_SUFFIX = {
    "en":  "Respond entirely in English.",
    "pap": (
        "Respond entirely in Papiamentu (the creole language spoken on Curaçao and Bonaire). "
        "Use natural, clear Papiamentu. Keep legal terms in Dutch where no Papiamentu equivalent exists, "
        "but explain them in Papiamentu."
    ),
}


def _generate_nonchat_ai_summary(
    doc: dict, api_key: Optional[str], cache: dict[str, str], language: str = "nl"
) -> str:
    """Generate a concise AI sentence for non-email/non-chat documents."""
    text = (doc.get("annotated_text") or doc.get("text") or "").strip()
    snippet = re.sub(r"\s+", " ", text)[:4000]
    if not snippet:
        return _nonchat_summary_fallback(doc)

    digest_input = f"{language}::{doc.get('doc_subtype') or 'other'}::{snippet}"
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
        lang_instr = _LANG_SUFFIX.get(language, "")
        lang_suffix = f"\n\n{lang_instr}" if lang_instr else ""
        prompt = (
            "Je krijgt tekst uit een Nederlands Woo-document. "
            "Schrijf precies 1 korte zin (max 24 woorden) die samenvat wat zichtbaar is in dit document. "
            "Noem geen geredigeerde personen. Geef alleen de zin, zonder labels.\n\n"
            f"Documenttype: {subtype_label}\n"
            f"Tekst:\n{snippet}"
            f"{lang_suffix}"
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


def _generate_chat_ai_summary(
    doc: dict, api_key: Optional[str], cache: dict[str, str], language: str = "nl"
) -> str:
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
    digest = hashlib.sha1(f"{language}::{joined}".encode("utf-8", errors="ignore")).hexdigest()
    if digest in cache:
        return cache[digest]

    if not api_key:
        cache[digest] = ""
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        lang_instr = _LANG_SUFFIX.get(language, "")
        lang_suffix = f"\n\n{lang_instr}" if lang_instr else ""
        prompt = (
            "Vat deze Nederlandse Woo-chatberichten samen in precies 1 zin (max 24 woorden). "
            "Noem geen namen van geredigeerde personen. Geef alleen de zin, zonder labels.\n\n"
            f"Chatberichten:\n{joined}"
            f"{lang_suffix}"
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
        cache[digest] = ""
        return ""


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


def _pipeline_to_json(docs: dict, api_key: Optional[str] = None, language: str = "nl") -> dict:
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
            description = _generate_chat_ai_summary(doc, api_key, chat_summary_cache, language)
        elif not is_email_doc:
            description = _generate_nonchat_ai_summary(doc, api_key, doc_summary_cache, language)

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
    language: str = "nl",
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
            output = _pipeline_to_json(result_data, api_key=api_key, language=language)
            output["pdf_url"] = "/api/session_pdf"
            yield _sse({"type": "done", **output})
        except Exception as exc:
            yield _sse({"type": "error", "msg": f"Resultaten verwerken mislukt: {exc}"})


async def _pipeline_with_cleanup(
    pipeline: str,
    pdf_path: Path,
    api_key: Optional[str],
    page_range: Optional[Tuple[int, int]] = None,
    language: str = "nl",
) -> AsyncIterator[str]:
    """Wraps _run_pipeline_sse to delete the temp PDF when done.
    Saves a copy to _SESSION_PDF first so the frontend can display PDF pages."""
    shutil.copy2(pdf_path, _SESSION_PDF)
    try:
        async for chunk in _run_pipeline_sse(pipeline, pdf_path, api_key, page_range=page_range, language=language):
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
    language:   Optional[str] = Form(None, description="Output language: nl (default), en, or pap"),
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

    global _session_language
    _lang = language if language in ("nl", "en", "pap") else "nl"
    _session_language = _lang

    return StreamingResponse(
        _pipeline_with_cleanup(pipeline, pdf_path, api_key or None, page_range=page_range, language=_lang),
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


# ── /api/context?pid=XXX — contextual panel: news + rijksoverheid + woo dossiers ─

_context_cache: dict[str, tuple[float, dict]] = {}   # cache_key -> (timestamp, result)
_CONTEXT_CACHE_TTL = 600                              # 10 minutes

_CONTEXT_EMPTY: dict = {"search_terms": [], "sections": {"nieuws": [], "rijksoverheid": [], "woo_dossier": []}}


@app.post("/api/context")
async def api_context(body: dict = Body(...)) -> dict:
    """Return contextual panel data for a dossier: news, Rijksoverheid releases, Woogle related dossiers.

    Accepts a JSON body with:
      pid, title, description, body_name, top_subjects, organisations, date_range, besluit_subject

    1. Fetches dossier metadata from pid.wooverheid.nl (when pid is given).
    2. Asks GPT-4o to extract 3-5 searchable entities from all available context.
    3. Fetches NewsAPI, Rijksoverheid API, and Woogle in parallel.
    4. Falls back to Google News RSS if NewsAPI returns 0 results.
    5. Caches results per pid/title for 1 hour.
    """
    pid             = (body.get("pid") or "").strip()
    title           = (body.get("title") or "").strip()
    description     = (body.get("description") or "").strip()
    body_name       = (body.get("body_name") or "").strip()
    top_subjects    = [str(s).strip() for s in (body.get("top_subjects") or []) if s][:8]
    organisations   = [str(o).strip() for o in (body.get("organisations") or []) if o][:6]
    date_range      = body.get("date_range") or []
    besluit_subject = (body.get("besluit_subject") or "").strip()

    # Compute news search date bounds: 30 days before dossier start, 60 days after end
    _news_from = ""
    _news_to   = ""
    if len(date_range) >= 2 and date_range[0]:
        from datetime import date as _d, timedelta as _td
        try:
            _news_from = (_d.fromisoformat(str(date_range[0])[:10]) - _td(days=30)).isoformat()
            _news_to   = (_d.fromisoformat(str(date_range[1] or date_range[0])[:10]) + _td(days=60)).isoformat()
        except Exception:
            pass

    cache_key = (pid or title[:80]).strip()
    if not cache_key:
        return _CONTEXT_EMPTY

    now = time.time()

    # ── Step 1: get dossier metadata from pid.wooverheid.nl ──────────────────
    pid_title    = ""
    pid_desc     = ""
    pid_meta_raw: dict = {}   # full infobox response, used in GPT context
    if pid:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(
                    f"https://pid.wooverheid.nl/?pid={pid}&infobox=true",
                    headers=HEADERS,
                )
            if r.is_success:
                meta          = r.json()
                pid_meta_raw  = meta  # keep everything for GPT context
                pid_title     = (
                    meta.get("dc_title") or
                    meta.get("title") or
                    meta.get("titel") or ""
                ).strip()
                pid_desc      = (
                    meta.get("dc_description") or
                    meta.get("dcterms_description") or
                    meta.get("dcterms_abstract") or
                    meta.get("description") or
                    meta.get("beschrijving") or
                    meta.get("summary") or ""
                ).strip()
                print(f"[context] PID metadata: title={pid_title!r}, desc={pid_desc[:80]!r}, fields={list(meta.keys())}")
        except Exception as e:
            print(f"[context] Metadata fetch failed: {e}")

    # Detect Ministerie van Buitenlandse Zaken — triggers international context enrichment
    _is_buza = any(
        kw in (body_name + " " + pid_title + " " + pid).lower()
        for kw in ("buitenlandse", "buza", "mnre1013")
    )
    if _is_buza:
        print(f"[context] BuZa dossier detected — enabling international context extraction")

    # Best title: prefer PID-fetched > frontend title
    def _clean_title(raw: str) -> str:
        raw = re.sub(r'\.\w{2,5}$', '', raw.strip())
        raw = re.sub(r'\s*\(\d+\)\s*$', '', raw)
        raw = re.sub(r'[\s_-]+\d+$', '', raw)
        return raw.strip()

    best_title = _clean_title(pid_title or title)
    best_desc  = pid_desc or description

    has_any_context = bool(best_title or body_name or top_subjects or organisations or besluit_subject)
    if not has_any_context:
        return _CONTEXT_EMPTY

    # Serve from cache only if result has actual, high-quality search terms
    if cache_key in _context_cache:
        ts, cached = _context_cache[cache_key]
        cached_terms = cached.get("search_terms") or []
        _QUICK_BLOCK = {"woo organizer", "woolens", "snelwijzer", "woo verzoek",
                        "openbaarmaking", "inventarislijst", "bestuursorgaan"}
        # Also reject sentence-fragment terms (contain verb/adj words or too many words)
        _SENTENCE_FRAG = {"is", "zijn", "heeft", "was", "werd", "verantwoordelijk",
                          "betreft", "gaat", "doet", "stelt", "geeft", "maakt",
                          "voor", "van", "het", "een", "aan", "door", "met"}
        def _term_is_clean(t):
            tl = t.lower()
            if any(b in tl for b in _QUICK_BLOCK):
                return False
            words = tl.split()
            if len(words) > 5:
                return False
            if any(w in _SENTENCE_FRAG for w in words[1:]):
                return False
            return True
        cache_is_clean = cached_terms and len(cached_terms) >= 3 and all(_term_is_clean(t) for t in cached_terms)
        if now - ts < _CONTEXT_CACHE_TTL and cache_is_clean:
            return cached
        # Invalidate bad or empty cache entry
        _context_cache.pop(cache_key, None)

    # ── Step 2: extract entities with GPT-4o from ALL available context ──────
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    search_terms: list[str] = []
    dossier_summary: dict = {}
    resolution_identifiers: list[str] = []

    # Post-processing helpers — reject sentence fragments, filename IDs, and app/process noise
    _SENTENCE_WORDS = {
        "is", "zijn", "heeft", "hebben", "was", "waren", "werd", "werden",
        "verantwoordelijk", "betreft", "gaat", "doet", "stelt", "geeft",
        "vraagt", "meldt", "besluit", "zegt", "blijkt", "komt", "staat",
    }
    # Generic FOI-process terms and app names that are never useful as news search queries
    _BLOCKLIST = {
        "woo organizer", "woolens", "woo lens", "snelwijzer", "woo-analyse",
        "woo verzoek", "woo-verzoek", "woo dossier", "woo-dossier",
        "openbaarmaking", "inventarislijst", "besluitbrief", "bestuursorgaan",
        "wet open overheid", "open overheid", "wob verzoek", "wob-verzoek",
    }

    def _clean_term(raw: str) -> str:
        """Trim to max 5 words, strip trailing Dutch stop/connector words."""
        _TRAIL = {"de","het","een","en","in","van","bij","op","met","over","voor",
                  "na","uit","tot","aan","door","te","om","als","of","dat","dit",
                  "die","deze","is","zijn","heeft","was","werd","naar","der","den"}
        words = raw.strip().split()[:5]
        while words and words[-1].lower() in _TRAIL:
            words.pop()
        return " ".join(words)

    # Internal/operational Dutch lead words — these signal an email subject or work
    # instruction, not a news-searchable topic (e.g. "Graag zsm akkoord: VNVR resolutie")
    _INTERNAL_LEADS = {
        "graag", "nota", "voorbereiding", "bijlage", "reactie", "fwd", "fw",
        "re", "betreft", "betreffende", "uitwerking", "tekst", "concept",
        "verzoek", "aanvraag", "verwerking", "update", "info", "vraag",
    }

    def _is_bad_term(t: str) -> bool:
        tl = t.lower().strip()
        if tl in _BLOCKLIST:
            return True
        # Partial blocklist match (e.g. "WOO Organizer analyse")
        if any(b in tl for b in _BLOCKLIST):
            return True
        words = tl.split()
        if len(words) > 5:
            return True
        if re.match(r'^[\w][\w-]*-\d{4}[-.\d]*$', t):   # filename/ID pattern
            return True
        # First word is an internal/operational lead — not a news topic
        if words and words[0].rstrip(':') in _INTERNAL_LEADS:
            return True
        for w in words[1:]:
            if w in _SENTENCE_WORDS:
                return True
        return False

    # Strip app/tool title from context if it slipped through
    _APP_NAMES = re.compile(
        r'\b(woo\s*organizer|woolens|woo\s*lens|snelwijzer)\b', re.IGNORECASE
    )
    safe_title = best_title if best_title and not _APP_NAMES.search(best_title) else ""

    if openai_key:
        # Build rich context for GPT — content-first ordering
        ctx_lines = []
        # Email subjects first — these are the actual document topics
        if top_subjects:
            ctx_lines.append("EMAIL SUBJECTS FROM DOCUMENTS (primary content signal):")
            for s in top_subjects[:8]:
                ctx_lines.append(f"  - {s[:120]}")
        if organisations:
            ctx_lines.append(f"ORGANISATIONS MENTIONED: {', '.join(organisations)}")
        if besluit_subject:
            ctx_lines.append(f"DECISION SUBJECT: {besluit_subject}")
        # Full PID website metadata — highest-quality official source
        if pid_meta_raw:
            ctx_lines.append("PID WEBSITE METADATA (official government dossier page):")
            _pid_field_map = [
                # Dublin Core variants
                ("dc_title",             "Title"),
                ("dc_description",       "Description"),
                ("dcterms_description",  "Description"),
                ("dcterms_abstract",     "Abstract"),
                ("dc_subject",           "Subject/Keywords"),
                ("dcterms_subject",      "Subject"),
                ("dc_creator",           "Creator"),
                ("dc_publisher",         "Publisher"),
                ("dcterms_publisher",    "Publisher"),
                ("dc_date",              "Date"),
                ("dcterms_date",         "Date"),
                ("dcterms_type",         "Document type"),
                ("dc_type",              "Type"),
                ("dcterms_spatial",      "Location"),
                ("dc_coverage",          "Coverage"),
                ("dcterms_relation",     "Related"),
                # Common alternative key names (API may use these instead)
                ("title",                "Title"),
                ("titel",                "Title"),
                ("description",          "Description"),
                ("beschrijving",         "Description"),
                ("summary",              "Abstract"),
                ("subject",              "Subject/Keywords"),
                ("onderwerp",            "Subject/Keywords"),
                ("keywords",             "Subject/Keywords"),
                ("identifier",           "Identifier"),
                ("publisher",            "Publisher"),
                ("aanbieder",            "Publisher"),
                ("creator",              "Creator"),
                ("date",                 "Date"),
                ("besluitdatum",         "Decision date"),
                ("publicatiedatum",      "Publication date"),
                ("type",                 "Type"),
                ("coverage",             "Coverage"),
            ]
            seen_vals: set = set()
            for field, label in _pid_field_map:
                val = pid_meta_raw.get(field)
                if not val:
                    continue
                # Handle lists (some fields return arrays)
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val if v)
                val = str(val).strip()[:300]
                if val and val not in seen_vals:
                    seen_vals.add(val)
                    ctx_lines.append(f"  {label}: {val}")
        # Always include best_desc and safe_title — critical even if pid_meta_raw fields don't match
        if best_desc and best_desc not in (seen_vals if pid_meta_raw else set()):
            ctx_lines.append(f"DESCRIPTION: {best_desc[:400]}")
        if safe_title:
            ctx_lines.append(f"DOSSIER TITLE: {safe_title}")
        if body_name:
            ctx_lines.append(f"MINISTRY/BODY: {body_name}")
        if date_range and len(date_range) >= 2 and date_range[0]:
            ctx_lines.append(f"DOCUMENT PERIOD: {date_range[0]} to {date_range[1]}")
        # BuZa international context: explicitly instruct GPT to extract named entities from title
        if _is_buza:
            _intl_ref = pid_title or safe_title
            if _intl_ref:
                ctx_lines.append("")
                ctx_lines.append("=== INTERNATIONALE CONTEXT (Ministerie van Buitenlandse Zaken) ===")
                ctx_lines.append(f"VOLLEDIGE DOSSIERTITEL: {_intl_ref}")
                ctx_lines.append("INSTRUCTIE: Extraheer uit bovenstaande titel ALLE:")
                ctx_lines.append("  - Landen (bijv. Israël, Palestijnse Gebieden, Nederland)")
                ctx_lines.append("  - Internationale organisaties (bijv. VN, AVVN, Veiligheidsraad, NATO)")
                ctx_lines.append("  - Gewapende conflicten en operaties bij naam")
                ctx_lines.append("  - Diplomatieke partijen en verdragen")
                ctx_lines.append("Deze specifieke namen MOETEN letterlijk terugkomen in de zoekqueries.")
                ctx_lines.append("VERBODEN: Generaliseer NOOIT naar 'humanitaire bescherming' of 'internationale betrekkingen'")
                ctx_lines.append("         als de titel concrete landen/conflicten noemt.")
        rich_context = "\n".join(ctx_lines)
        print(f"[context] Sending to GPT:\n{rich_context[:800]}")

        _SYSTEM_PROMPT = (
            "You are the Snelwijzer search expert. Your goal is to generate high-precision "
            "search queries for a News API based on the uploaded WOO documents.\n\n"
            "### STEP 1 — RESOLUTION & IDENTIFIER SCAN (do this first)\n"
            "Scan ALL provided text for formal identifiers:\n"
            "- UN/AVVN resolutions: A/RES/*, A/ES-*/L.*, S/RES/*, A/HRC/*, ES-10/*, etc.\n"
            "- EU/NATO/treaty references: e.g. Verdrag van Lissabon, Sanctieverordening (EU) 833/2014\n"
            "- Named operations, programmes, missions (e.g. Operatie Protective Edge, EUNAVFOR)\n"
            "- Named agreements or accords (Oslo, Abraham, JCPOA)\n"
            "List EVERY identifier found in 'resolution_identifiers'. Empty list [] if none found.\n\n"
            "### STEP 2 — GENERATE SEARCH QUERIES\n"
            "Generate 5-6 high-quality, noun-only search queries.\n\n"
            "### PRIORITY ORDER\n"
            "1. RESOLUTION IDENTIFIERS from Step 1 — each gets its own dedicated search query "
            "(e.g. 'Resolutie A/ES-10/L.27 Nederland', 'VN-stemming A/ES-10/L.25 Gaza')\n"
            "2. PID WEBSITE METADATA / DOSSIER TITLE — extract ALL named parties, countries, "
            "organisations, conflicts, and operations mentioned. The title and description are "
            "the single most authoritative signal. If the title says 'conflict tussen Israël en "
            "Palestijnse Gebieden', EVERY search query must reflect this specific conflict. "
            "Never generalise ('humanitaire bescherming') when the title names specific parties.\n"
            "3. Email subjects from the documents — reveal real topics discussed\n"
            "4. Specific names: politician names, locations, organisations\n"
            "5. The decision subject and description\n\n"
            "### RULES\n"
            "1. NO STOP WORDS: Remove (de, het, een, van, in, voor, met, op, aan, door, te).\n"
            "2. NOUN-ONLY: Queries must consist ONLY of nouns and proper entities.\n"
            "3. SPECIFICITY: Resolution numbers > names > descriptions.\n"
            "4. NO META-TERMS: NEVER include 'WOO', 'Organizer', 'WOOLens', 'Snelwijzer', "
            "'openbaarmaking', 'inventarislijst', 'besluit', 'bestuursorgaan', 'open overheid', "
            "'woo verzoek', or any FOI process words.\n"
            "5. NO FILE CODES: Never use internal document IDs (e.g. 'nl.mnre1013.2i.2024.65').\n"
            "6. BOOLEAN LOGIC: Use AND/OR where useful to combine entities.\n"
            "7. Max 6 words per query.\n\n"
            "BAD: 'WOO Organizer', 'Woo verzoek BuZa', 'openbaarmaking documenten'\n"
            "GOOD: 'Resolutie A/ES-10/L.27 Nederland stem', 'VN-stemming Gaza humanitair wapenstilstand', "
            "'sancties Rusland exportcontrole BuZa', 'Kaag Gaza humanitaire corridor'\n\n"
            "### OUTPUT FORMAT (JSON ONLY)\n"
            "{\n"
            "  \"resolution_identifiers\": [\"A/ES-10/L.27\", \"A/ES-10/L.25\"],\n"
            "  \"dossier_summary\": {\n"
            "    \"topic\": \"Concise 1-sentence description of the political/social subject\",\n"
            "    \"relevant_timeframe\": \"Relevant date range for the events\"\n"
            "  },\n"
            "  \"search_queries\": [\n"
            "    \"Resolutie A/ES-10/L.27 Nederland\",\n"
            "    \"VN-stemming A/ES-10/L.25 Gaza wapenstilstand\",\n"
            "    \"Nederland VN-stemming Gaza humanitair\",\n"
            "    \"Kaag Gaza humanitaire corridor\",\n"
            "    \"sancties Rusland exportcontrole\"\n"
            "  ]\n"
            "}"
        )

        try:
            from openai import OpenAI as _OAI
            _oa = _OAI(api_key=openai_key)
            resp = _oa.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Generate search queries for this Woo dossier:\n\n"
                            f"{rich_context}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            dossier_summary = parsed.get("dossier_summary") or {}
            resolution_identifiers = [str(r).strip() for r in (parsed.get("resolution_identifiers") or []) if r]
            raw_queries = parsed.get("search_queries") or []
            cleaned = [_clean_term(str(t)) for t in raw_queries if t]
            search_terms = [t for t in cleaned if t and not _is_bad_term(t)][:6]
            print(f"[context] Resolutions found: {resolution_identifiers}")
            print(f"[context] GPT entities: {search_terms}")
            print(f"[context] Dossier summary: {dossier_summary}")
        except Exception as e:
            print(f"[context] GPT extraction failed: {e}")

    # Fallback: extract clean terms from available metadata — always through _clean_term + _is_bad_term
    if not search_terms:
        _looks_like_id = bool(re.match(r'^[\w][\w-]*-\d{4}[-.\d]*$', safe_title or ""))
        _candidates = []
        if safe_title and not _looks_like_id:
            _candidates.append(safe_title)
        if besluit_subject:
            _candidates.append(besluit_subject)
        if body_name:
            _candidates.append(body_name)
        for _cand in _candidates:
            _t = _clean_term(_cand)
            if _t and not _is_bad_term(_t):
                search_terms = [_t]
                print(f"[context] Using fallback: {search_terms!r} (from {_cand!r})")
                break
        if not search_terms:
            print(f"[context] No usable fallback, title was: {best_title!r}")
    if not search_terms:
        return _CONTEXT_EMPTY

    # Pad up to 6 terms so the frontend always has multiple chip options to choose from.
    # Pull additional candidates from metadata: subjects → organisations → body → title words.
    if len(search_terms) < 6:
        _pad_sources: list[str] = []
        for s in top_subjects:
            _pad_sources.append(s)
        for o in organisations:
            _pad_sources.append(o)
        if besluit_subject:
            _pad_sources.append(besluit_subject)
        if body_name:
            _pad_sources.append(body_name)
        if safe_title:
            # Also try 2-word windows from the title as individual topic chips
            _tw = [w for w in safe_title.split() if len(w) > 3]
            for _i in range(0, len(_tw) - 1, 2):
                _pad_sources.append(" ".join(_tw[_i:_i + 2]))
        _seen_lower = {t.lower() for t in search_terms}
        for _cand in _pad_sources:
            if len(search_terms) >= 6:
                break
            _t = _clean_term(_cand)
            if _t and not _is_bad_term(_t) and _t.lower() not in _seen_lower:
                search_terms.append(_t)
                _seen_lower.add(_t.lower())
                print(f"[context] Padded term: {_t!r}")

    # ── Step 3: fetch all sources in parallel ─────────────────────────────────
    primary_term = search_terms[0]
    # AND-join top 3 entities for precise intersection queries
    and_query = " AND ".join(search_terms[:3])

    async def _fetch_newsapi() -> list[dict]:
        newsapi_key = os.environ.get("NEWSAPI_KEY", "")
        if not newsapi_key:
            # Re-read .env in case key was added after server start
            _env_path = Path(__file__).parent / ".env"
            if _env_path.exists():
                for _ln in _env_path.read_text().splitlines():
                    _ln = _ln.strip()
                    if _ln.startswith("NEWSAPI_KEY="):
                        newsapi_key = _ln.split("=", 1)[1].strip()
                        if newsapi_key:
                            os.environ["NEWSAPI_KEY"] = newsapi_key
                        break
        if not newsapi_key:
            return []
        try:
            _na_params: dict = {
                "q":        and_query,
                "language": "nl",
                "sortBy":   "relevancy",
                "pageSize": 6,
                "apiKey":   newsapi_key,
            }
            if _news_from:
                _na_params["from"] = _news_from
            if _news_to:
                _na_params["to"] = _news_to
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(
                    "https://newsapi.org/v2/everything",
                    params=_na_params,
                )
            if r.is_success:
                return [
                    {
                        "title":       (a.get("title") or "").strip(),
                        "source":      ((a.get("source") or {}).get("name") or "").strip(),
                        "date":        (a.get("publishedAt") or "")[:10],
                        "url":         (a.get("url") or "").strip(),
                        "description": (a.get("description") or "")[:120].strip(),
                        "source_type": "nieuws",
                    }
                    for a in (r.json().get("articles") or [])[:6]
                    if a.get("title") and a.get("url")
                ]
        except Exception:
            pass
        return []

    async def _fetch_rijksoverheid() -> list[dict]:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(
                    "https://opendata.rijksoverheid.nl/v1/sources/rijksoverheid/infotypes/pressrelease",
                    params={"output": "json", "rows": 3, "query": primary_term},
                    headers=HEADERS,
                )
            if r.is_success:
                data = r.json()
                items = data if isinstance(data, list) else (
                    data.get("docs") or
                    data.get("response", {}).get("docs") or []
                )
                results = []
                for item in items[:3]:
                    t = (item.get("title") or "").strip()
                    u = (item.get("url") or item.get("link") or "").strip()
                    d = (item.get("lastmodified") or item.get("date") or "")[:10]
                    desc = re.sub(r"<[^>]+>", "", (item.get("introduction") or item.get("description") or ""))[:120].strip()
                    if t:
                        results.append({
                            "title":       t,
                            "source":      "Rijksoverheid",
                            "date":        d,
                            "url":         u,
                            "description": desc,
                            "source_type": "rijksoverheid",
                        })
                return results
        except Exception:
            pass
        return []

    async def _fetch_woogle() -> list[dict]:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(
                    "https://woogle.wooverheid.nl/api/search",
                    params={"q": and_query, "type": "2i", "rows": 3},
                    headers=HEADERS,
                )
            if r.is_success:
                data = r.json()
                items = data.get("response", {}).get("docs") or data.get("docs") or []
                results = []
                for item in items[:3]:
                    t    = (item.get("dc_title") or item.get("title") or "").strip()
                    pid_val = (item.get("pid") or item.get("id") or "").strip()
                    u    = f"https://open.overheid.nl/documenten/{pid_val}" if pid_val else ""
                    d    = (item.get("dc_date") or item.get("date") or "")[:10]
                    desc = (item.get("dc_description") or item.get("description") or "")[:120].strip()
                    body = (item.get("dc_publisher_name") or item.get("publisher") or "Woogle").strip()
                    if t:
                        results.append({
                            "title":       t,
                            "source":      body,
                            "date":        d,
                            "url":         u,
                            "description": desc,
                            "source_type": "woo_dossier",
                        })
                return results
        except Exception:
            pass
        return []

    async def _fetch_google_rss() -> list[dict]:
        try:
            query = " ".join(search_terms[:3])
            if _news_from:
                query += f" after:{_news_from}"
            if _news_to:
                query += f" before:{_news_to}"
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={urllib.parse.quote(query)}&hl=nl&gl=NL&ceid=NL:nl"
            )
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(rss_url, headers=HEADERS)
            if r.is_success:
                root    = ET.fromstring(r.text)
                channel = root.find("channel")
                results = []
                if channel is not None:
                    for item in channel.findall("item")[:6]:
                        t_el   = item.findtext("title")       or ""
                        l_el   = item.findtext("link")        or ""
                        p_el   = item.findtext("pubDate")     or ""
                        d_el   = item.findtext("description") or ""
                        src_el = item.find("source")
                        source = (src_el.text or "").strip() if src_el is not None else ""
                        desc   = re.sub(r"<[^>]+>", "", d_el)[:120].strip()
                        if t_el and l_el:
                            results.append({
                                "title":       t_el.strip(),
                                "source":      source,
                                "date":        p_el[:16],
                                "url":         l_el.strip(),
                                "description": desc,
                                "source_type": "nieuws",
                            })
                return results
        except Exception:
            pass
        return []

    # Run three main fetches in parallel
    news_items, rijks_items, woogle_items = await asyncio.gather(
        _fetch_newsapi(),
        _fetch_rijksoverheid(),
        _fetch_woogle(),
    )

    # Google RSS fallback when NewsAPI returned nothing
    if not news_items:
        news_items = await _fetch_google_rss()

    print(f"[context] Results — nieuws:{len(news_items)} rijks:{len(rijks_items)} woogle:{len(woogle_items)}")

    result = {
        "search_terms":          search_terms,
        "resolution_identifiers": resolution_identifiers,
        "pid_title":       pid_title,
        "pid_desc":        pid_desc,
        "dossier_summary": dossier_summary,
        "date_range":      date_range,
        "sections": {
            "nieuws":        news_items,
            "rijksoverheid": rijks_items,
            "woo_dossier":   woogle_items,
        },
    }
    # Only cache if at least one section has results, so failed attempts can retry
    if news_items or rijks_items or woogle_items:
        _context_cache[cache_key] = (now, result)
    return result


@app.get("/api/news")
async def api_news(q: str = "", from_date: str = "", to_date: str = "") -> dict:
    """Fetch news for a single search term. Used by clickable term chips in Snelwijzer."""
    q = q.strip()[:200]
    if not q:
        return {"articles": []}

    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    articles: list[dict] = []

    if newsapi_key:
        try:
            _params: dict = {"q": q, "language": "nl", "sortBy": "relevancy", "pageSize": 6, "apiKey": newsapi_key}
            if from_date:
                _params["from"] = from_date[:10]
            if to_date:
                _params["to"] = to_date[:10]
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get("https://newsapi.org/v2/everything", params=_params)
            if r.is_success:
                articles = [
                    {
                        "title":  (a.get("title") or "").strip(),
                        "source": ((a.get("source") or {}).get("name") or "").strip(),
                        "date":   (a.get("publishedAt") or "")[:10],
                        "url":    (a.get("url") or "").strip(),
                    }
                    for a in (r.json().get("articles") or [])
                    if a.get("title") and a.get("url")
                ]
        except Exception:
            pass

    # Google RSS fallback
    if not articles:
        try:
            rss_query = q
            if from_date:
                rss_query += f" after:{from_date[:10]}"
            if to_date:
                rss_query += f" before:{to_date[:10]}"
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={urllib.parse.quote(rss_query)}&hl=nl&gl=NL&ceid=NL:nl"
            )
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(rss_url, headers=HEADERS)
            if r.is_success:
                root    = ET.fromstring(r.text)
                channel = root.find("channel")
                if channel is not None:
                    for item in channel.findall("item")[:6]:
                        t_el   = item.findtext("title")   or ""
                        l_el   = item.findtext("link")    or ""
                        p_el   = item.findtext("pubDate") or ""
                        src_el = item.find("source")
                        source = (src_el.text or "").strip() if src_el is not None else ""
                        if t_el and l_el:
                            articles.append({
                                "title":  t_el.strip(),
                                "source": source,
                                "date":   p_el[:16],
                                "url":    l_el.strip(),
                            })
        except Exception:
            pass

    return {"articles": articles[:6]}


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
