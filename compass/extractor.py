"""
compass/extractor.py

Combine data from all three Woo dossier sources (besluit, inventarislijst,
disclosed documents) and triangulate across them to surface insights.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Known Dutch ministry/agency names (for NER scan) ─────────────────────────

_KNOWN_ORGS = [
    "Ministerie van Algemene Zaken",
    "Ministerie van Binnenlandse Zaken",
    "Ministerie van Buitenlandse Zaken",
    "Ministerie van Defensie",
    "Ministerie van Economische Zaken",
    "Ministerie van Financiën",
    "Ministerie van Infrastructuur",
    "Ministerie van Justitie",
    "Ministerie van Landbouw",
    "Ministerie van Onderwijs",
    "Ministerie van Sociale Zaken",
    "Ministerie van Volksgezondheid",
    "Ministerie van Volkshuisvesting",
    "Rijkswaterstaat",
    "RIVM",
    "UWV",
    "Belastingdienst",
    "AIVD",
    "MIVD",
    "Politie",
    "OM",
    "Openbaar Ministerie",
    "CBS",
    "Centraal Bureau voor de Statistiek",
    "DNB",
    "De Nederlandsche Bank",
    "ACM",
    "Autoriteit Consument en Markt",
    "AP",
    "Autoriteit Persoonsgegevens",
]


# ── Triangulation result ───────────────────────────────────────────────────────

@dataclass
class TriangulationInsights:
    outcome_vs_disclosure: str = ""
    dominant_ground_flag: str = ""
    informal_channel_flag: str = ""
    draft_iteration_flag: str = ""
    processing_time_flag: str = ""
    cross_source_summary: str = ""


# ── Main extraction function ───────────────────────────────────────────────────

def extract_all(
    dossier_json: dict,
    besluit: Optional[dict] = None,
    inventory_rows: Optional[list[dict]] = None,
) -> dict:
    """
    Build a unified extraction dict from all available sources.
    besluit and inventory_rows are optional — degrades gracefully.
    """
    extraction: dict = {
        "besluit_available": besluit is not None,
        "inventory_available": inventory_rows is not None,
        "documents_available": bool(dossier_json),
        # besluit fields
        "request_date": "",
        "receipt_date": "",
        "decision_date": "",
        "processing_days": 0,
        "deadline_extended": False,
        "outcome": "",
        "body_name": "",
        "subject": "",
        "refusal_grounds_besluit": [],
        "besluit_raw_text": "",
        "appeal_deadline_weeks": 0,
        "appeal_body": "",
        "contact_person": "",
        "default_notice": False,
        # inventory aggregates
        "total_inventory_docs": 0,
        "fully_public": 0,
        "partially_public": 0,
        "not_public": 0,
        "disclosure_percentage": 0.0,
        "refusal_ground_counts": {},
        "has_whatsapp": False,
        "has_signal": False,
        "has_chat_export": False,
        "has_images": False,
        "buiten_verzoek_count": 0,
        "numbering_gaps": [],
        "draft_clusters": [],
        "unusual_flags": [],
        # dossier doc stats
        "total_emails": 0,
        "total_chats": 0,
        "total_other_docs": 0,
        "total_pages": 0,
        "has_redacted_chat_messages": False,
        "unique_senders": [],
        "top_subjects": [],
        "organisations_found": [],
        "date_range": ["", ""],
        "refusal_codes": {},
        # extended dossier fields
        "named_persons": [],
        "document_authors": [],
        "document_recipients": [],
        "channel_counts": {},
        "redacted_passage_count": 0,
        "doc_types_breakdown": {},
        "key_dates": [],
    }

    # ── Part 1: Besluit ──────────────────────────────────────────────────────
    if besluit:
        for key in (
            "request_date", "receipt_date", "decision_date", "processing_days",
            "deadline_extended", "outcome", "body_name", "subject",
            "appeal_deadline_weeks", "appeal_body", "contact_person", "default_notice",
        ):
            extraction[key] = besluit.get(key, extraction[key])
        extraction["refusal_grounds_besluit"] = besluit.get("refusal_grounds_cited", [])
        extraction["besluit_raw_text"] = (besluit.get("_raw_text") or "")[:2500]

    # ── Part 2: Inventarislijst ──────────────────────────────────────────────
    if inventory_rows:
        _aggregate_inventory(extraction, inventory_rows)

    # ── Part 3: Dossier JSON ─────────────────────────────────────────────────
    _aggregate_dossier(extraction, dossier_json)

    # Merge refusal codes from inventory + dossier stats
    merged_codes: dict[str, int] = {}
    for code, cnt in extraction["refusal_ground_counts"].items():
        merged_codes[code] = merged_codes.get(code, 0) + cnt
    for code, cnt in extraction["refusal_codes"].items():
        merged_codes[code] = merged_codes.get(code, 0) + cnt
    extraction["refusal_codes"] = merged_codes

    return extraction


def _aggregate_inventory(extraction: dict, rows: list[dict]) -> None:
    """Compute inventory aggregates and unusual flags."""
    total = len(rows)
    fully = sum(1 for r in rows if r["status"] == "Openbaar")
    partial = sum(1 for r in rows if r["status"] == "Deels Openbaar")
    not_pub = sum(1 for r in rows if r["status"] == "Niet Openbaar")
    buiten = sum(1 for r in rows if r["status"] == "Buiten Verzoek")

    extraction["total_inventory_docs"] = total
    extraction["fully_public"] = fully
    extraction["partially_public"] = partial
    extraction["not_public"] = not_pub
    extraction["buiten_verzoek_count"] = buiten
    extraction["disclosure_percentage"] = round(fully / total * 100, 1) if total else 0.0

    # Refusal ground counts
    ground_counts: dict[str, int] = {}
    for row in rows:
        for g in row.get("refusal_grounds", []):
            ground_counts[g] = ground_counts.get(g, 0) + 1
    extraction["refusal_ground_counts"] = ground_counts

    # Detect doc types from names
    names = [r["name"].lower() for r in rows]
    extraction["has_whatsapp"] = any("whatsapp" in n for n in names)
    extraction["has_signal"] = any("signal" in n for n in names)
    extraction["has_chat_export"] = any(
        kw in n for n in names for kw in ("chat", "teams", "telegram", "sms")
    )
    extraction["has_images"] = any(
        n.endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")) for n in names
    )

    # Numbering gaps
    nrs = []
    for r in rows:
        try:
            nrs.append(int(r["nr"]))
        except (ValueError, TypeError):
            pass
    if nrs:
        mn, mx = min(nrs), max(nrs)
        gaps = [i for i in range(mn, mx + 1) if i not in nrs]
        extraction["numbering_gaps"] = gaps

    # Draft clusters — documents with same base name (ignoring version suffix)
    from collections import Counter
    name_counts = Counter(n[:40] for n in names if n)
    extraction["draft_clusters"] = [n for n, c in name_counts.items() if c > 1]

    # Date range from inventory
    dates = sorted(r["date"] for r in rows if r.get("date") and len(r["date"]) >= 8)
    if dates:
        extraction["date_range"] = [dates[0], dates[-1]]

    # Unusual flags
    flags: list[str] = []
    if extraction["disclosure_percentage"] < 10 and total > 0:
        flags.append(
            f"Zeer lage openbaarheid — slechts {extraction['disclosure_percentage']}% "
            f"volledig openbaar gemaakt"
        )
    if extraction["has_whatsapp"] or extraction["has_signal"]:
        channels = []
        if extraction["has_whatsapp"]:
            channels.append("WhatsApp")
        if extraction["has_signal"]:
            channels.append("Signal")
        flags.append(
            f"{'/'.join(channels)}-berichten aanwezig — informele kanalen gebruikt"
        )
    if buiten > 0:
        flags.append(
            f"{buiten} document(en) valt buiten het verzoek volgens de overheid"
        )
    if extraction["numbering_gaps"]:
        flags.append(
            f"Nummering heeft gaten ({len(extraction['numbering_gaps'])} "
            f"ontbrekend) — mogelijk ontbrekende documenten"
        )
    if extraction["draft_clusters"]:
        flags.append(
            f"Meerdere versies van hetzelfde document gevonden "
            f"— interne deliberatie zichtbaar"
        )
    if total > 0 and ground_counts:
        top_ground = max(ground_counts, key=ground_counts.get)
        top_pct = round(ground_counts[top_ground] / total * 100)
        if top_pct > 70:
            flags.append(
                f"Weigeringsgrond {top_ground} gebruikt bij {top_pct}% van "
                f"documenten — dit is ongebruikelijk hoog"
            )
    extraction["unusual_flags"] = flags


def _aggregate_dossier(extraction: dict, dossier_json: dict) -> None:
    """Extract stats from the existing WOOLens dossier JSON."""
    emails = dossier_json.get("emails", [])
    chats = dossier_json.get("chats", [])
    others = dossier_json.get("others", [])
    timeline = dossier_json.get("timeline", [])
    stats = dossier_json.get("stats", {})

    extraction["total_emails"] = len(emails)
    extraction["total_chats"] = len(chats)
    extraction["total_other_docs"] = len(others)
    extraction["total_pages"] = stats.get("pages", 0)
    extraction["refusal_codes"] = dict(stats.get("redactionCodes", {}))

    # Check redacted chat messages
    for chat in chats:
        for msg in chat.get("berichten", []):
            if msg.get("isRedacted"):
                extraction["has_redacted_chat_messages"] = True
                break

    # Unique senders
    senders: set[str] = set()
    for e in emails:
        s = (e.get("sender") or "").strip()
        if s and s.lower() not in ("[gelakt]", "[redacted]", ""):
            senders.add(s)
    for chat in chats:
        for d in chat.get("deelnemers", []):
            if d and d.strip():
                senders.add(d.strip())
    for o in others:
        s = (o.get("sender") or "").strip()
        if s:
            senders.add(s)
    extraction["unique_senders"] = sorted(senders)[:20]

    # Top subject keywords from emails
    raw_subjects = [
        re.sub(r"^(?:re|fw|fwd|ant|tr)\s*:\s*", "", e.get("subject") or "", flags=re.IGNORECASE).strip()
        for e in emails
        if e.get("subject")
    ]
    # Deduplicate preserving order
    seen: set[str] = set()
    top5: list[str] = []
    for s in raw_subjects:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            top5.append(s)
        if len(top5) >= 5:
            break
    extraction["top_subjects"] = top5

    # Scan all text for known organisation names
    all_text = " ".join([
        e.get("text", "") for e in emails
    ] + [
        o.get("preview", "") for o in others
    ] + [
        (chat.get("aiSummary") or "") for chat in chats
    ])
    orgs_found = [org for org in _KNOWN_ORGS if org.lower() in all_text.lower()]
    extraction["organisations_found"] = orgs_found

    # Date range across all document types
    dates = sorted(
        item.get("date", "")
        for item in timeline
        if item.get("date") and len(item["date"]) >= 8
    )
    if dates and not extraction.get("date_range", ["", ""])[0]:
        extraction["date_range"] = [dates[0], dates[-1]]

    # Named persons from email Van/Aan/CC fields
    # NOTE: this is intentionally conservative; we prefer showing nothing
    # over showing junk like "redacted" or half-redacted addresses.
    _gelakt = {"[gelakt]", "[redacted]", "gelakt", "redacted", ""}

    def _clean_person_candidate(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return ""

        # Common wrappers: "Name <mail@x>", "mail@x (Name)", quotes
        m = re.search(r"<([^>]+@[^>]+)>", s)
        if m:
            # keep the visible name part (before <...>)
            s = s.split("<", 1)[0].strip().strip('"\'')
        s = re.sub(r"\([^)]*@[^)]*\)", "", s).strip()
        s = s.strip('"\'')

        # Drop any obvious redaction markers / Woo codes in the candidate
        if re.search(r"\b(redacted|gelakt|weggelakt|afgeschermd)\b", s, flags=re.I):
            return ""
        if re.search(r"\b5\.[12]\.\d[a-z]{0,2}\b", s, flags=re.I):
            return ""
        if re.search(r"\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]", s, flags=re.I):
            return ""

        # Emails (or fragments) should never show up as "personen"
        if "@" in s:
            return ""

        # Too short / too many symbols
        if len(s) < 3 or len(s) > 80:
            return ""
        if re.fullmatch(r"[\W_\d]+", s):
            return ""

        # Reject strings that are mostly punctuation/uppercase noise
        if sum(ch.isalpha() for ch in s) < 3:
            return ""

        # Require it to look like a name-ish label:
        # - either at least 2 words with letters
        # - or a single word with Titlecase and enough letters
        words = [w for w in re.split(r"\s+", s) if w]
        if len(words) >= 2:
            # drop if any word is extremely long (OCR garbage)
            if any(len(w) > 30 for w in words):
                return ""
            return s

        if len(words) == 1:
            w = words[0]
            if len(w) >= 4 and (w[0].isupper() and any(c.islower() for c in w[1:])):
                return s
            return ""

        return ""

    persons: dict[str, int] = {}
    for e in emails:
        for field in ("sender", "to", "cc"):
            for part in re.split(r"[,;]", e.get(field) or ""):
                p = _clean_person_candidate(part)
                if not p:
                    continue
                key = p.lower()
                if key in _gelakt:
                    continue
                persons[p] = persons.get(p, 0) + 1

    # Sort by frequency, return top 20
    extraction["named_persons"] = [p for p, _ in sorted(persons.items(), key=lambda x: -x[1])][:20]

    # Document authors from others
    authors: list[str] = []
    seen_auth: set[str] = set()
    for o in others:
        s = (o.get("sender") or "").strip()
        if s and s.lower() not in _gelakt and s.lower() not in seen_auth:
            seen_auth.add(s.lower())
            authors.append(s)
    extraction["document_authors"] = authors[:15]

    # Document recipients — unique "to" values across emails
    recipients: set[str] = set()
    for e in emails:
        for part in re.split(r"[,;]", e.get("to") or ""):
            p = part.strip()
            if p and p.lower() not in _gelakt and len(p) > 2:
                recipients.add(p)
    extraction["document_recipients"] = sorted(recipients)[:15]

    # Communication channel breakdown
    channel_counts: dict[str, int] = {}
    channel_counts["E-mail"] = len(emails)
    if chats:
        channel_counts["Chat/WhatsApp"] = len(chats)
    for o in others:
        t = (o.get("type") or o.get("doc_subtype") or "Overig").strip()
        channel_counts[t] = channel_counts.get(t, 0) + 1
    extraction["channel_counts"] = {k: v for k, v in channel_counts.items() if v > 0}

    # Redacted passage count
    _redact_re = re.compile(r"\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]", re.IGNORECASE)
    redacted_count = 0
    for e in emails:
        redacted_count += len(_redact_re.findall(e.get("text", "")))
    for o in others:
        redacted_count += len(_redact_re.findall(o.get("preview", "")))
    for chat in chats:
        for msg in chat.get("berichten", []):
            if msg.get("isRedacted"):
                redacted_count += 1
    extraction["redacted_passage_count"] = redacted_count

    # Doc type breakdown from others
    doc_types: dict[str, int] = {}
    for o in others:
        t = (o.get("type") or "Overig").strip()
        doc_types[t] = doc_types.get(t, 0) + 1
    extraction["doc_types_breakdown"] = doc_types

    # Key timeline dates (first 8 with titles)
    key_dates: list[dict] = []
    for item in sorted(timeline, key=lambda x: x.get("date", "")):
        d = item.get("date", "")
        if d and len(d) >= 8:
            key_dates.append({"date": d, "title": (item.get("title") or "")[:70]})
        if len(key_dates) >= 8:
            break
    extraction["key_dates"] = key_dates


# ── Triangulation ──────────────────────────────────────────────────────────────

def triangulate(extraction: dict) -> TriangulationInsights:
    """
    Compare all three sources and produce human-readable insight strings.
    """
    t = TriangulationInsights()

    # outcome_vs_disclosure
    outcome = extraction.get("outcome", "")
    disc_pct = extraction.get("disclosure_percentage", 0.0)
    total_inv = extraction.get("total_inventory_docs", 0)
    if outcome and total_inv > 0:
        if disc_pct <= 10 and "ingewilligd" in outcome:
            t.outcome_vs_disclosure = (
                f"Het besluit zegt '{outcome}', maar slechts {disc_pct}% van de "
                f"documenten is volledig openbaar gemaakt. Dit betekent dat vrijwel "
                f"alle gevonden documenten (deels) geheim zijn gehouden."
            )
        elif disc_pct >= 70 and "geweigerd" in outcome:
            t.outcome_vs_disclosure = (
                f"Hoewel het besluit als '{outcome}' is aangemerkt, is {disc_pct}% "
                f"van de documenten toch openbaar gemaakt."
            )
        elif disc_pct < 30 and "ingewilligd" in outcome:
            t.outcome_vs_disclosure = (
                f"Het besluit zegt '{outcome}', maar slechts {disc_pct}% is volledig "
                f"openbaar. Het merendeel van de documenten bleef (deels) geheim."
            )
        else:
            t.outcome_vs_disclosure = (
                f"Het besluit luidt '{outcome}'. Van de {total_inv} gevonden "
                f"documenten is {disc_pct}% volledig openbaar gemaakt."
            )
    elif outcome:
        t.outcome_vs_disclosure = f"De beslissing is: '{outcome}'."
    elif total_inv > 0:
        t.outcome_vs_disclosure = (
            f"Van {total_inv} gevonden documenten is {disc_pct}% volledig openbaar gemaakt."
        )

    # dominant_ground_flag
    codes = extraction.get("refusal_codes", {}) or extraction.get("refusal_ground_counts", {})
    total_docs = extraction.get("total_inventory_docs", 0) or sum(codes.values())
    if codes and total_docs > 0:
        top = max(codes, key=codes.get)
        top_pct = round(codes[top] / total_docs * 100)
        if top_pct > 70:
            t.dominant_ground_flag = (
                f"Weigeringsgrond {top} is gebruikt bij {top_pct}% van de documenten. "
                f"Dit is ongebruikelijk hoog en suggereert dat de overheid het gehele "
                f"onderwerp als bijzonder gevoelig beschouwt."
            )
        elif codes:
            top3 = sorted(codes.items(), key=lambda x: -x[1])[:3]
            parts = ", ".join(f"{c} ({n}×)" for c, n in top3)
            t.dominant_ground_flag = (
                f"Meest gebruikte weigeringsgronden: {parts}."
            )

    # informal_channel_flag
    has_wa = extraction.get("has_whatsapp", False)
    has_sig = extraction.get("has_signal", False)
    channels = []
    if has_wa:
        channels.append("WhatsApp")
    if has_sig:
        channels.append("Signal")
    if channels:
        t.informal_channel_flag = (
            f"Dit dossier bevat {'/'.join(channels)}-berichten. Dit betekent dat "
            f"overheidsfunctionarissen informele communicatiekanalen hebben gebruikt "
            f"voor dit onderwerp."
        )

    # draft_iteration_flag
    clusters = extraction.get("draft_clusters", [])
    if clusters:
        t.draft_iteration_flag = (
            f"Er zijn {len(clusters)} groep(en) documenten met dezelfde naam gevonden, "
            f"wat duidt op meerdere versies of concepten. Interne deliberatie is "
            f"zichtbaar in dit dossier."
        )

    # processing_time_flag
    proc_days = extraction.get("processing_days", 0)
    extended = extraction.get("deadline_extended", False)
    default_notice = extraction.get("default_notice", False)
    if default_notice:
        t.processing_time_flag = (
            f"De overheid heeft een ingebrekestelling ontvangen omdat zij niet op tijd "
            f"heeft beslist. De beslissing duurde {proc_days} dagen."
        )
    elif proc_days > 42:
        t.processing_time_flag = (
            f"De overheid deed er {proc_days} dagen over om te beslissen — dit is "
            f"langer dan de wettelijke termijn van 28 dagen."
        )
    elif proc_days > 28:
        t.processing_time_flag = (
            f"De beslistermijn van 28 dagen werd overschreden: de overheid deed er "
            f"{proc_days} dagen over."
        )
    elif extended:
        t.processing_time_flag = (
            "De wettelijke beslistermijn werd verlengd (verdaagd). Dit is toegestaan "
            "maar moet worden gemotiveerd."
        )

    # cross_source_summary
    parts = []
    if extraction.get("body_name"):
        parts.append(f"Dit dossier is afkomstig van {extraction['body_name']}.")
    if extraction.get("subject"):
        parts.append(f"Het verzoek ging over: {extraction['subject'][:120]}.")
    total_e = extraction.get("total_emails", 0)
    total_c = extraction.get("total_chats", 0)
    total_o = extraction.get("total_other_docs", 0)
    total_pg = extraction.get("total_pages", 0)
    counts = []
    if total_e:
        counts.append(f"{total_e} e-mail(s)")
    if total_c:
        counts.append(f"{total_c} chatgesprek(ken)")
    if total_o:
        counts.append(f"{total_o} ander(e) document(en)")
    if counts:
        parts.append(f"De openbaar gemaakte documenten omvatten {', '.join(counts)}"
                     + (f" ({total_pg} pagina's)" if total_pg else "") + ".")
    if t.outcome_vs_disclosure:
        parts.append(t.outcome_vs_disclosure)
    t.cross_source_summary = " ".join(parts[:3])

    return t
