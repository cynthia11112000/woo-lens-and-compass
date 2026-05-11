"""
email_splitter.py

Splits every E-mail document returned by load_pdf() into individual emails.

Both Dutch Outlook and English Outlook formats are supported:

  Dutch  : Van:  Afzender:  Aan:  Onderwerp:  Betreft:  Verzonden:  Datum:  CC:  BCC:
  English: From:  To:  Subject:  Sent:  Received:  CC:  BCC:

A new email is recognised by two complementary signals:

  1. A header field type that already appeared in the current email reappears
     (e.g. a second "Van:" means the first email has ended and a new one begins).
     Dutch and English names are normalised before comparison, so "Van:" and
     "From:" count as the same field type.

  2. A header field line appears after EXIT_BODY_THRESHOLD or more consecutive
     non-header lines — i.e. we clearly re-entered a header block from body text.

Outlook "-----Original Message-----" / "-----Oorspronkelijk bericht-----" separators
are also treated as split points.

Output IDs follow the pattern <doc_code>.<n>, e.g. "0003.1", "0003.2".
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Dutch month names for date normalisation
# ---------------------------------------------------------------------------

_DUTCH_MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "augustus": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
    # abbreviated
    "jan": 1, "feb": 2, "mrt": 3, "maa": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12,
}


def _normalize_date(raw: str | None) -> str | None:
    """Convert common Dutch/European date strings to YYYY-MM-DD or YYYY-MM-DDTHH:MM.

    Handles:
      - Already ISO: 2024-01-03
      - European:    DD-MM-YYYY  or  DD/MM/YYYY
      - Dutch long:  3 januari 2024  (optionally preceded by weekday name)
    - With time:   maandag 6 mei 2024 14:32  →  2024-05-06T14:32

    Returns the original string unchanged when parsing is not possible
    (the frontend's normalizeDateStr() will attempt a second parse).
    """
    if not raw:
        return None
    s = raw.strip()

    # Already ISO (optionally with time: 2024-01-03T14:32 or 2024-01-03 14:32)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{1,2}:\d{2}))?", s)
    if m:
        base = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return f"{base}T{m.group(4)}" if m.group(4) else base

    # European: DD-MM-YYYY or DD/MM/YYYY, optionally with time
    m = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})(?:\s+(\d{1,2}:\d{2}))?", s)
    if m:
        d, mo, y, t = m.group(1), m.group(2), m.group(3), m.group(4)
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
            base = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
            return f"{base}T{t}" if t else base

    # Dutch long: "3 januari 2024" or "maandag 3 januari 2024 14:32"
    m = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})(?:\s+(\d{1,2}:\d{2}))?", s)
    if m:
        d, mon_str, y, t = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        mon = _DUTCH_MONTHS.get(mon_str) or _DUTCH_MONTHS.get(mon_str[:3])
        if mon:
            base = f"{y}-{str(mon).zfill(2)}-{d.zfill(2)}"
            return f"{base}T{t}" if t else base

    return raw  # Return raw — frontend will try again


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

def _extract_time(normalized_date: str | None) -> str | None:
    """Extract time from a normalized date string.
    
    If the date is in format YYYY-MM-DDTHH:MM, returns HH:MM.
    Otherwise returns None.
    """
    if not normalized_date or "T" not in normalized_date:
        return None
    try:
        parts = normalized_date.split("T")
        if len(parts) == 2:
            return parts[1]
    except Exception:
        pass
    return None


# Matches a header field line and captures the field name.
_FIELD_TYPE_RE = re.compile(
    r"^[ \t]*(?P<field>from|van|afzender|to|aan|subject|onderwerp|betreft|sent|verzonden|datum|date|received|cc|bcc)\s*:",
    re.IGNORECASE,
)

# Map Dutch field names → English so "Van:" and "From:" are treated as the same.
_FIELD_NORMALIZE = {
    "van":      "from",
    "afzender": "from",
    "aan":      "to",
    "onderwerp": "subject",
    "betreft":  "subject",
    "verzonden": "sent",
    "datum":    "sent",
}

# Matches Outlook "---- Original / Forwarded / Oorspronkelijk bericht ----" separators.
_SEPARATOR_RE = re.compile(
    r"[-_=]{4,}.*?(?:original message|forwarded message|doorgestuurd bericht|oorspronkelijk bericht).*",
    re.IGNORECASE,
)

# Field extractors for metadata display.
_SUBJECT_RE = re.compile(r"^[ \t]*(?:subject|onderwerp|betreft)\s*:\s*(.+)",  re.IGNORECASE)
_FROM_RE    = re.compile(r"^[ \t]*(?:from|van|afzender)\s*:\s*(.+)",          re.IGNORECASE)
_DATE_RE    = re.compile(r"^[ \t]*(?:sent|verzonden|datum|date)\s*:\s*(.+)",   re.IGNORECASE)
_TO_RE      = re.compile(r"^[ \t]*(?:to|aan)\s*:\s*(.+)",                     re.IGNORECASE)
_CC_RE      = re.compile(r"^[ \t]*cc\s*:\s*(.+)",                             re.IGNORECASE)

# Matches explicit attachment-list header lines.
_ATTACH_RE  = re.compile(
    r"^[ \t]*(?:bijlage[n]?|attachment[s]?)\s*:\s*(.+)",
    re.IGNORECASE,
)

# Recognises standalone filenames with common extensions.
_FILENAME_RE = re.compile(
    r"\b[\w\-. ]+\.(?:pdf|docx?|xlsx?|pptx?|msg|eml|zip|txt|csv|rtf|odt|ods)\b",
    re.IGNORECASE,
)

# Redacted email-address patterns: "< @domain>" or "< ,@domain>" or "<@domain>"
# where the local part has been blacked out.  Normalise to "<[REDACTED]@domain>".
_REDACTED_ADDR_RE = re.compile(r"<\s*,?\s*@([\w.-]+\.\w{2,})\s*>", re.IGNORECASE)


def _normalize_redacted_addrs(text: str) -> str:
    """Replace redacted email address fragments with a legible placeholder."""
    return _REDACTED_ADDR_RE.sub(r"<[REDACTED]@\1>", text)


def _normalize_field(name: str) -> str:
    """Normalise a header field name to its English canonical form."""
    return _FIELD_NORMALIZE.get(name.lower(), name.lower())


# ---------------------------------------------------------------------------
# Attachment extraction
# ---------------------------------------------------------------------------

def _extract_attachments(lines: list[str], limit: int = 30) -> list[str]:
    """Return a deduplicated list of attachment names/filenames from the header block.

    Two patterns are recognised:
      1. Explicit label:  "Bijlage: report.pdf"  or  "Attachments: file1.docx; file2.xlsx"
      2. Bare filename:   a line in the header area whose content is only a filename.
    """
    found: list[str] = []
    seen: set[str]   = set()

    def _add(name: str) -> None:
        name = name.strip().strip(";,")
        if name and name not in seen:
            seen.add(name)
            found.append(name)

    for line in lines[:limit]:
        m = _ATTACH_RE.match(line)
        if m:
            # Extract filenames from the value part of the Bijlage(n)/Attachment(s) line.
            fnames = _FILENAME_RE.findall(m.group(1))
            if fnames:
                for f in fnames:
                    _add(f)
            else:
                # Value may be a plain name without an extension (e.g. "Bijlage: verslag")
                for part in re.split(r"[;,]", m.group(1)):
                    _add(part)
        else:
            # A line that is *only* a filename (no other text) — common after "Bijlagen: 3"
            stripped = line.strip()
            if _FILENAME_RE.fullmatch(stripped):
                _add(stripped)

    return found


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

# After this many consecutive non-header lines we consider the header block over.
_EXIT_BODY_THRESHOLD = 6
# A header preceded by a gap of > _GAP_LINES AND > _BODY_LINES body lines also
# starts a new email (catches cases where the body is long but has no repeated field).
_GAP_LINES  = 4
_BODY_LINES = 2


def _find_split_points(lines: list[str]) -> list[int]:
    """
    Return line indices where new emails begin.

    For each line in document order:
    - Outlook separator  → always start a new email.
    - Header field line  → start a new email if any of:
        a) we are not currently in a header block (came from body text), OR
        b) the same field type was already seen in this email's headers, OR
        c) large line gap from last header AND several body lines seen.
      Otherwise add the field to the current email's header set.
    - Any other line     → increment body-line counter; exit header-block mode
                           once _EXIT_BODY_THRESHOLD body lines accumulate.
    """
    split_points: list[int] = []
    in_header_block = False
    seen_fields: set[str] = set()
    last_header_idx = -999
    body_lines = 0

    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            split_points.append(i)
            in_header_block = True
            seen_fields = set()
            last_header_idx = i
            body_lines = 0
            continue

        m = _FIELD_TYPE_RE.match(line)
        if not m:
            if in_header_block:
                body_lines += 1
                if body_lines >= _EXIT_BODY_THRESHOLD:
                    in_header_block = False
            continue

        field = _normalize_field(m.group("field"))
        gap   = i - last_header_idx

        new_email = (
            not in_header_block                              # re-entered from body text
            or field in seen_fields                          # repeated field → new email
            or (gap > _GAP_LINES and body_lines > _BODY_LINES)  # gap + body evidence
        )

        if new_email:
            split_points.append(i)
            seen_fields = {field}
        else:
            seen_fields.add(field)

        in_header_block = True
        last_header_idx = i
        body_lines = 0

    return split_points


def _extract_field(lines: list[str], pattern: re.Pattern, limit: int = 15) -> str | None:
    """Return the first match of *pattern* in the first *limit* lines."""
    for line in lines[:limit]:
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_emails(text: str, doc_code: str) -> list[dict]:
    """
    Split the concatenated text of an e-mail document into individual emails.

    Returns a list of dicts, one per email:
        {
            "id":          str,          # e.g. "0003.2"
            "subject":     str | None,
            "sender":      str | None,
            "to":          str | None,
            "cc":          str | None,
            "date":        str | None,   # normalised to YYYY-MM-DD where possible
            "time":        str | None,   # HH:MM if a time is visible, else None
            "attachments": list[str],    # filenames / attachment names
            "text":        str,
            "warning":     str | None,   # non-None when splitting was ambiguous
        }
    """
    text = _normalize_redacted_addrs(text)
    lines = text.splitlines()
    split_points = _find_split_points(lines)

    if not split_points:
        return [{
            "id": f"{doc_code}.1",
            "subject": None,
            "sender": None,
            "to": None,
            "cc": None,
            "date": None,
            "time": None,
            "attachments": [],
            "text": text.strip(),
            "warning": "No header-field cluster found — kept as single block",
        }]

    emails = []
    email_idx = 0

    if split_points[0] > 0:
        preamble = lines[:split_points[0]]
        preamble_text = "\n".join(preamble).strip()
        if preamble_text:
            email_idx += 1
            emails.append({
                "id": f"{doc_code}.{email_idx}",
                "subject": None,
                "sender": None,
                "to": None,
                "cc": None,
                "date": None,
                "time": None,
                "attachments": [],
                "text": preamble_text,
                "warning": "Reply body before quoted headers — no header fields found",
            })

    for j, start in enumerate(split_points):
        end = split_points[j + 1] if j + 1 < len(split_points) else len(lines)
        block_lines = lines[start:end]
        block_text = "\n".join(block_lines).strip()

        if not block_text:
            continue

        raw_date = _extract_field(block_lines, _DATE_RE)
        normalized_date = _normalize_date(raw_date)
        date_only = normalized_date.split("T", 1)[0] if normalized_date else None
        time_val = _extract_time(normalized_date)

        email_idx += 1
        emails.append({
            "id": f"{doc_code}.{email_idx}",
            "subject": _extract_field(block_lines, _SUBJECT_RE),
            "sender": _extract_field(block_lines, _FROM_RE),
            "to": _extract_field(block_lines, _TO_RE),
            "cc": _extract_field(block_lines, _CC_RE),
            "date": date_only,
            "time": time_val,
            "attachments": _extract_attachments(block_lines),
            "text": block_text,
            "warning": None,
        })

    return emails
