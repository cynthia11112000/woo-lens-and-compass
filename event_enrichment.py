"""
event_enrichment.py — Post-processing step for the GPT-4o WOO pipeline.

Takes the extracted emails produced by pipeline_gpt4o and sends them through
a second GPT-4o call that groups, classifies, and enriches them into structured
timeline events with headlines, summaries, tags, signals and redaction info.

Only used by main_gpt4o.py — the OCR pipeline does not run this step.
"""

import json
import time
import re
from datetime import datetime, date
from pathlib import Path

# ── System prompt ─────────────────────────────────────────────────────────────

_ENRICH_SYSTEM = """\
You are a deterministic data transformation assistant.

Your task is to process FOIA (Woo) JSON data into structured timeline events for further use in an application UI (powered by GPT-4o).

Focus on:
* consistency
* structured output
* minimal ambiguity

Do NOT be creative. Follow rules strictly.

---

# INPUT

JSON with:
* "emails": raw messages (email text OR OCR-extracted chat text)
* messages may include:
  * emails
  * WhatsApp screenshots (OCR text)
  * Microsoft Teams chats (OCR text)

---

# OUTPUT

Return ONLY valid JSON. No explanations.

---

# STEP 1 — MESSAGE TYPE CLASSIFICATION

For each message:
Classify as:

## "email" if:
* has subject field OR
* contains "From:", "Sent:", "To:"

## "chat" if:
* short lines
* conversational structure
* multiple speakers
* timestamps inside text
* no email headers

If uncertain: default to "email"

---

# STEP 2 — EMAIL THREAD GROUPING

Normalize subject:
* lowercase
* remove prefixes repeatedly: "re:", "fw:", "fwd:"
* trim whitespace

Group emails by normalized subject. Each group = 1 thread.

---

# STEP 3 — CHAT CONVERSATION GROUPING

Group messages into conversations if:
* timestamps are close
* same context
* conversational structure

Each conversation = 1 event. Be tolerant to OCR errors.

---

# STEP 4 — DATETIME EXTRACTION

Extract datetime per message. Format: "YYYY-MM-DDTHH:MM:SS"
If only date: set time = "00:00:00"
Event datetime: earliest message datetime in group.
Also include:
* "date": YYYY-MM-DD
* "time": HH:MM

---

# STEP 5 — EVENT CREATION

Each:
* email thread      → event_type = "email_thread"
* chat conversation → event_type = "chat_conversation"

---

# STEP 6 — CONTENT CLEANING

Remove:
* email headers
* signatures
* disclaimers
* system noise

Keep: meaningful message content only.

---

# STEP 7 — REDACTION HANDLING

Detect patterns: "[REDACTED: X]"
Extract legal code (e.g. "5.1.2.a").

Output:
"redactions": [
  {
    "law": "Woo art. 5.1 lid 2 sub a"
  }
]

In summaries: include "Persoonsgegevens verwijderd (Woo art. ...)"
Do not infer missing data.

---

# STEP 8 — EVENT ENRICHMENT

Generate:

## headline
* max 8 words
* no jargon
* no raw subject reuse

## ui_summary
* max 12 words
* very short

## detail_summary
* 1–2 sentences

## actors
* organizations only if clear
* else empty array

## tags (max 4)
Allowed:
* "intern overleg"
* "voorbereiding"
* "besluitvorming"
* "document gedeeld"
* "communicatie"
* "bezoek"
* "vertrouwelijk"
* "chat"

Rule: if chat → must include "chat"

---

# STEP 9 — UI FIELDS (STRICT)

## importance
* "high"   → decisions / key actions
* "medium" → coordination / preparation
* "low"    → minor communication

## phase
Choose one:
* "voorbereiding"
* "afstemming"
* "uitvoering"
* "nazorg"

## signals (max 3)
Choose from:
* "tijdsdruk"
* "interne afstemming"
* "extern contact"
* "vertrouwelijk overleg"

## confidence
* float 0–1
* lower if OCR unclear

---

# STEP 10 — OUTPUT FORMAT

{
  "events": [
    {
      "id":             "event_1",
      "datetime":       "YYYY-MM-DDTHH:MM:SS",
      "date":           "YYYY-MM-DD",
      "time":           "HH:MM",
      "event_type":     "...",
      "headline":       "...",
      "ui_summary":     "...",
      "detail_summary": "...",
      "actors":         [],
      "tags":           [],
      "importance":     "...",
      "phase":          "...",
      "signals":        [],
      "confidence":     0.0,
      "redactions":     [],
      "source_emails":  []
    }
  ]
}

---

# HARD RULES

* Output must be valid JSON
* No additional text
* No hallucination
* No missing required fields
* Be consistent across similar inputs

---

Process the input JSON.\
"""

# ── Constants ─────────────────────────────────────────────────────────────────

_BATCH_SIZE   = 25   # emails per GPT-4o call — keep context manageable
_MAX_RETRIES  = 3
_RETRY_DELAY  = 2    # seconds between retries (doubles each attempt)
_MODEL        = "gpt-4o"

_CHAT_DAY_PROMPT = """Je krijgt WhatsApp berichten van één dag uit een Nederlands Woo-dossier.
Genereer alleen een JSON object met deze velden:
{
    'title': 'één zin die de kern van deze dag beschrijft, maximaal 12 woorden',
    'theme': 'één woord thema-label zoals Desinformatie, Coördinatie, Monitoring',
    'summary': '2-3 zinnen samenvatting van wat er die dag werd besproken',
    'key_topics': ['maximaal 3 concrete onderwerpen, geen stopwoorden, niet: redacted https www']
}
Noem geen namen van geredacteerde personen. Return alleen JSON, geen andere tekst."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_chat_doc(doc: dict) -> bool:
    """Detect whether a document is a chat/SMS style document."""
    category = (doc.get("category") or "").strip().lower()
    subtype = (doc.get("doc_subtype") or "").strip().lower()
    return category == "chat" or subtype == "chat_sms"


def _extract_doc_messages(doc: dict, code: str) -> list[dict]:
    """Extract structured message items from one document with splitter fallback."""
    from email_splitter import split_emails

    structured: list[dict] = doc.get("emails") or []
    if structured:
        return structured

    try:
        return split_emails(doc.get("annotated_text") or doc.get("text", ""), code)
    except Exception as exc:
        print(f"[enrich] Warning: message extraction failed for {code}: {exc}")
        return []


def _extract_non_chat_emails_from_docs(docs: dict) -> list[dict]:
    """
    Flatten all emails from the sorted docs dict.

    Tries structured GPT-4o email metadata first; falls back to text splitter.
    """
    emails: list[dict] = []
    for code, doc in docs.items():
        if doc.get("category") != "E-mail":
            continue
        if _is_chat_doc(doc):
            # Keep chat documents out of the email-thread enrichment flow.
            continue
        emails.extend(_extract_doc_messages(doc, code))

    return emails


def _parse_date_value(value) -> datetime | None:
    """Parse multiple date formats into datetime (time defaults to 00:00)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    normalized = raw.replace("/", "-").replace(".", "-")
    fmts = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass

    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", normalized)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", normalized)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d-%m-%Y")
        except ValueError:
            pass

    return None


def _parse_time_value(value: str | None) -> str | None:
    """Normalize HH:MM time string."""
    if not value:
        return None
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b", str(value))
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"


def _extract_chat_messages_from_docs(docs: dict) -> dict[str, list[dict]]:
    """Group chat messages by day key DD-MM-YYYY across all chat documents."""
    grouped: dict[str, list[dict]] = {}

    for code, doc in docs.items():
        if not _is_chat_doc(doc):
            continue

        doc_dt = _parse_date_value(doc.get("date"))
        messages = _extract_doc_messages(doc, code)

        for idx, msg in enumerate(messages, start=1):
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            msg_dt = _parse_date_value(msg.get("date"))
            if msg_dt is None:
                msg_dt = _parse_date_value(text)
            if msg_dt is None:
                msg_dt = doc_dt
            if msg_dt is None:
                continue

            time_hm = _parse_time_value(msg.get("time")) or _parse_time_value(text) or "00:00"
            day_key = msg_dt.strftime("%d-%m-%Y")

            grouped.setdefault(day_key, []).append({
                "id": msg.get("id") or f"{code}.{idx}",
                "doc_code": code,
                "date": msg_dt.strftime("%Y-%m-%d"),
                "time": time_hm,
                "sender": msg.get("sender") or "",
                "text": text,
            })

    for day_key, items in grouped.items():
        grouped[day_key] = sorted(items, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("id") or ""))

    return grouped


def _clean_theme(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return "Chat"
    one_word = raw.split()[0]
    one_word = re.sub(r"[^0-9A-Za-zÀ-ÿ_-]", "", one_word)
    return one_word or "Chat"


def _clean_key_topics(values) -> list[str]:
    if not isinstance(values, list):
        return []
    banned = {"redacted", "https", "http", "www"}
    out: list[str] = []
    for item in values:
        topic = str(item or "").strip()
        if not topic:
            continue
        normalized = topic.lower()
        if normalized in banned:
            continue
        if any(b in normalized.split() for b in banned):
            continue
        out.append(topic)
        if len(out) >= 3:
            break
    return out


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def _call_chat_day_enrich(client, day_key: str, messages: list[dict]) -> dict:
    """One GPT call per chat day-group (DD-MM-YYYY)."""
    payload = {
        "date": day_key,
        "messages": [
            {
                "time": m.get("time") or "00:00",
                "sender": m.get("sender") or "",
                "text": m.get("text") or "",
            }
            for m in messages
        ],
    }

    user_input = _CHAT_DAY_PROMPT + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": user_input}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)

            title = _truncate_words(str(data.get("title") or "WhatsApp-gesprek"), 12)
            theme = _clean_theme(data.get("theme"))
            summary = str(data.get("summary") or "").strip()
            key_topics = _clean_key_topics(data.get("key_topics"))

            return {
                "title": title or "WhatsApp-gesprek",
                "theme": theme,
                "summary": summary,
                "key_topics": key_topics,
            }
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                print(f"[enrich] Chat day {day_key} failed after {_MAX_RETRIES} attempts: {exc}")
                break
            wait = _RETRY_DELAY * (2 ** attempt)
            print(f"[enrich] Chat day {day_key} attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)

    return {
        "title": "WhatsApp-gesprek",
        "theme": "Chat",
        "summary": "",
        "key_topics": [],
    }


def _build_chat_day_events(client, grouped: dict[str, list[dict]]) -> list[dict]:
    """Create one timeline event per day-group of chat messages."""
    if not grouped:
        return []

    events: list[dict] = []
    day_keys = sorted(grouped.keys(), key=lambda d: datetime.strptime(d, "%d-%m-%Y"))

    for idx, day_key in enumerate(day_keys, start=1):
        day_messages = grouped[day_key]
        if not day_messages:
            continue

        enrich = _call_chat_day_enrich(client, day_key, day_messages)
        first = day_messages[0]
        date_iso = first.get("date") or datetime.strptime(day_key, "%d-%m-%Y").strftime("%Y-%m-%d")
        first_time = first.get("time") or "00:00"

        summary = enrich.get("summary") or ""
        ui_summary = _truncate_words(summary, 12) if summary else "Dagoverzicht chatberichten"
        theme = enrich.get("theme") or "Chat"

        event = {
            "id": f"chat_day_{idx}",
            "datetime": f"{date_iso}T{first_time}:00",
            "date": date_iso,
            "time": first_time,
            "event_type": "chat_conversation",
            "headline": enrich.get("title") or "WhatsApp-gesprek",
            "title": enrich.get("title") or "WhatsApp-gesprek",
            "theme": theme,
            "badge_label": theme,
            "ui_summary": ui_summary,
            "detail_summary": summary,
            "summary": summary,
            "key_topics": enrich.get("key_topics") or [],
            "actors": [],
            "tags": ["chat", theme.lower()],
            "importance": "medium",
            "phase": "afstemming",
            "signals": ["interne afstemming"],
            "confidence": 0.8,
            "redactions": [],
            "source_emails": [m.get("id") for m in day_messages if m.get("id")],
            "chat_messages": day_messages,
        }
        events.append(event)

    return events


def _call_enrich(client, batch: list[dict], batch_num: int, total_batches: int) -> list[dict]:
    """Call GPT-4o with one batch of emails. Returns list of events."""
    input_payload = json.dumps({"emails": batch}, ensure_ascii=False)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _ENRICH_SYSTEM},
                    {"role": "user",   "content": input_payload},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            return data.get("events") or []

        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                print(f"[enrich] Batch {batch_num}/{total_batches} failed after {_MAX_RETRIES} attempts: {exc}")
                return []
            wait = _RETRY_DELAY * (2 ** attempt)
            print(f"[enrich] Batch {batch_num}/{total_batches} attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)

    return []


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_events(docs: dict, api_key: str) -> list[dict]:
    """
    Enrich all emails in *docs* into structured timeline events.

    Args:
        docs:    Sorted docs dict as returned by sort_documents().
        api_key: OpenAI API key.

    Returns:
        Flat list of event dicts matching the output schema in the system prompt.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    emails = _extract_non_chat_emails_from_docs(docs)
    chat_messages_by_day = _extract_chat_messages_from_docs(docs)

    if not emails and not chat_messages_by_day:
        print("[enrich] No enrichable email/chat messages found in docs — skipping enrichment.")
        return []

    print(f"[enrich] {len(emails)} non-chat emails to enrich across {_MODEL}.")
    print(f"[enrich] {len(chat_messages_by_day)} chat day-group(s) to enrich across {_MODEL}.")

    all_events: list[dict] = []

    if emails:
        # Split non-chat emails into batches and keep current enrichment behavior.
        batches = [emails[i: i + _BATCH_SIZE] for i in range(0, len(emails), _BATCH_SIZE)]
        total = len(batches)
        event_offset = 0

        for idx, batch in enumerate(batches, start=1):
            print(f"[enrich] Batch {idx}/{total} ({len(batch)} emails)…")
            events = _call_enrich(client, batch, idx, total)

            # Re-number event IDs so they stay globally unique across batches
            for ev in events:
                raw_id = ev.get("id", "event_1")
                try:
                    local_num = int(raw_id.split("_")[-1])
                except (ValueError, AttributeError):
                    local_num = 1
                ev["id"] = f"event_{event_offset + local_num}"

            all_events.extend(events)
            event_offset += len(events)

    if chat_messages_by_day:
        print("[enrich] Building chat day events (one GPT call per day)…")
        chat_events = _build_chat_day_events(client, chat_messages_by_day)
        all_events.extend(chat_events)

    return all_events


def save_events(events: list[dict], path: Path) -> None:
    """Write enriched events to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, ensure_ascii=False, indent=2)
    print(f"[enrich] {len(events)} events saved → {path}")


def load_events(path: Path) -> list[dict]:
    """Load previously saved events from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])
