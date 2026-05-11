"""
compass/pdf_parser.py

Extract structured data from Besluit (decision letter) and Inventarislijst
(inventory list) PDFs using pdfplumber.
"""
from __future__ import annotations

import io
import re
from typing import Optional

import pdfplumber

# ── Dutch month names ──────────────────────────────────────────────────────────

_MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "augustus": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
}

_DATE_RE = re.compile(
    r"(\d{1,2})\s+(januari|februari|maart|april|mei|juni|juli|augustus|"
    r"september|oktober|november|december)\s+(\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_REFUSAL_RE = re.compile(r"5\.1\.\d+[a-z]?", re.IGNORECASE)
_DAYS_RE = re.compile(r"(\d+)\s*(?:werkdagen|kalenderdagen|dagen)", re.IGNORECASE)


def _find_dates(text: str) -> list[str]:
    """Return all dates found as ISO strings (YYYY-MM-DD)."""
    results = []
    for m in _DATE_RE.finditer(text):
        day = int(m.group(1))
        month = _MONTHS.get(m.group(2).lower(), 0)
        year = int(m.group(3))
        if month:
            results.append(f"{year:04d}-{month:02d}-{day:02d}")
    for m in _ISO_DATE_RE.finditer(text):
        results.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return results


def _days_between(d1: str, d2: str) -> int:
    """Calculate calendar days between two ISO date strings."""
    from datetime import date
    try:
        a = date.fromisoformat(d1)
        b = date.fromisoformat(d2)
        return abs((b - a).days)
    except Exception:
        return 0


# ── Besluit parser ─────────────────────────────────────────────────────────────

def parse_besluit_text(text: str) -> dict:
    """
    Parse raw besluit text and return a structured dict.
    All fields are best-effort — missing info stays as empty string / False / 0.
    """
    result = {
        "request_date": "",
        "receipt_date": "",
        "decision_date": "",
        "processing_days": 0,
        "deadline_extended": False,
        "outcome": "",
        "body_name": "",
        "subject": "",
        "requested_doc_types": [],
        "refusal_grounds_cited": [],
        "appeal_deadline_weeks": 0,
        "appeal_body": "",
        "contact_person": "",
        "default_notice": False,
    }

    lower = text.lower()

    # Dates — heuristic: first = request, second = receipt/decision
    dates = _find_dates(text)
    unique_dates = list(dict.fromkeys(dates))  # preserve order, deduplicate
    if len(unique_dates) >= 1:
        result["decision_date"] = unique_dates[-1]
    if len(unique_dates) >= 2:
        result["request_date"] = unique_dates[0]
    if len(unique_dates) >= 3:
        result["receipt_date"] = unique_dates[1]

    # Processing days
    if result["request_date"] and result["decision_date"]:
        result["processing_days"] = _days_between(
            result["request_date"], result["decision_date"]
        )

    # Deadline extension
    if any(kw in lower for kw in ("verdaging", "verdaagd", "verlenging", "verlengd")):
        result["deadline_extended"] = True

    # Outcome
    if re.search(r"volledig\s+ingewilligd", lower):
        result["outcome"] = "volledig ingewilligd"
    elif re.search(r"gedeeltelijk\s+ingewilligd|deels\s+ingewilligd", lower):
        result["outcome"] = "deels ingewilligd"
    elif re.search(r"\bgeweigerd\b", lower):
        result["outcome"] = "geweigerd"
    elif re.search(r"openbaar\s+gemaakt", lower):
        result["outcome"] = "deels ingewilligd"

    # Refusal grounds
    grounds = list(dict.fromkeys(_REFUSAL_RE.findall(text)))
    result["refusal_grounds_cited"] = grounds

    # Appeal info
    appeal_weeks_m = re.search(r"(\d+)\s*wek(?:en)?", lower)
    if appeal_weeks_m and "bezwaar" in lower:
        result["appeal_deadline_weeks"] = int(appeal_weeks_m.group(1))
    elif "zes weken" in lower:
        result["appeal_deadline_weeks"] = 6

    if "rechtbank" in lower:
        m = re.search(r"rechtbank\s+([a-zäöü\-]+)", lower)
        result["appeal_body"] = "Rechtbank " + (m.group(1).title() if m else "")
    elif "bezwaar" in lower:
        result["appeal_body"] = "Bij het bestuursorgaan (bezwaarschrift)"

    # Body name — look for "ministerie van", "rijksdienst", "gemeente" etc.
    body_m = re.search(
        r"(ministerie\s+van\s+[\w\s]+?(?=\n|,|\.)|"
        r"rijksdienst\s+[\w\s]+?(?=\n|,|\.)|"
        r"gemeente\s+[\w]+)",
        text,
        re.IGNORECASE,
    )
    if body_m:
        result["body_name"] = body_m.group(0).strip()[:80]

    # Subject — look for "betreft:", "onderwerp:", "verzoek om"
    subj_m = re.search(r"(?:betreft|onderwerp)\s*:?\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if subj_m:
        result["subject"] = subj_m.group(1).strip()[:200]
    else:
        req_m = re.search(r"verzoek\s+(?:om|tot)\s+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if req_m:
            result["subject"] = req_m.group(1).strip()[:200]

    # Contact person — email address or "contactpersoon"
    contact_m = re.search(r"[\w.\-+]+@[\w.\-]+\.\w{2,}", text)
    if contact_m:
        result["contact_person"] = contact_m.group(0)

    # Ingebrekestelling (formal notice for being late)
    if "ingebrekestelling" in lower or "dwangsom" in lower:
        result["default_notice"] = True

    return result


def parse_besluit_from_upload(file_bytes: bytes) -> dict:
    """Extract and parse a besluit PDF from raw bytes."""
    raw_text = _extract_pdf_text(file_bytes)
    parsed = parse_besluit_text(raw_text)
    parsed["_raw_text"] = raw_text
    return parsed


# ── Inventarislijst parser ─────────────────────────────────────────────────────

_GROUND_SPLIT_RE = re.compile(r"[,;\s]+")

_STATUS_MAP = {
    "openbaar": "Openbaar",
    "deels openbaar": "Deels Openbaar",
    "niet openbaar": "Niet Openbaar",
    "reeds openbaar": "Openbaar",
    "buiten verzoek": "Buiten Verzoek",
    "buiten": "Buiten Verzoek",
}


def _normalise_status(raw: str) -> str:
    raw_lower = raw.strip().lower()
    for key, val in _STATUS_MAP.items():
        if key in raw_lower:
            return val
    return raw.strip() or "Onbekend"


def _parse_grounds(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    parts = _GROUND_SPLIT_RE.split(raw.strip())
    return [p for p in parts if re.match(r"5\.[12]\.\d+[a-z]?", p, re.IGNORECASE)]


def parse_inventory_rows(rows: list[list]) -> list[dict]:
    """
    Parse raw table rows from pdfplumber into structured dicts.

    Expected columns (flexible index matching):
    Nr | Unieke ID | Datum | Documentnaam | Beoordeling | Weigeringsgronden
    """
    parsed = []
    for row in rows:
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue
        cells = [str(c or "").strip() for c in row]
        # Skip header rows
        if any(kw in cells[0].lower() for kw in ("nr", "nummer", "id", "unieke")):
            continue
        # Expect at least 4 columns
        if len(cells) < 4:
            continue

        # Try to find columns by content heuristics
        # Col 0: Nr (numeric or empty)
        # Col 1: Unieke ID (alphanumeric code)
        # Col 2: Datum (date-like)
        # Col 3: Documentnaam (longest text)
        # Col 4: Beoordeling (status)
        # Col 5: Weigeringsgronden (5.1.x codes or empty)

        if len(cells) >= 6:
            nr = cells[0]
            doc_id = cells[1]
            date = cells[2]
            name = cells[3]
            status_raw = cells[4]
            grounds_raw = cells[5]
        elif len(cells) == 5:
            nr = cells[0]
            doc_id = cells[1]
            date = cells[2]
            name = cells[3]
            status_raw = cells[4]
            grounds_raw = ""
        else:
            nr = cells[0]
            doc_id = ""
            date = ""
            name = cells[-1]
            status_raw = ""
            grounds_raw = ""

        status = _normalise_status(status_raw)
        grounds = _parse_grounds(grounds_raw)

        parsed.append({
            "nr": nr,
            "id": doc_id,
            "date": date,
            "name": name,
            "status": status,
            "refusal_grounds": grounds,
        })
    return parsed


def parse_inventory_from_upload(file_bytes: bytes) -> list[dict]:
    """Extract inventory table rows from a PDF and return parsed row dicts."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        all_rows: list[list] = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                all_rows.extend(table)
    return parse_inventory_rows(all_rows)


# ── Generic PDF text extractor ─────────────────────────────────────────────────

def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF using pdfplumber."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        parts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)
