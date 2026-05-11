"""
routers/compass.py

FastAPI router for the WOOLens Compass — contextual explanation endpoint.
Accepts multipart/form-data with optional PDF uploads.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from compass.extractor import extract_all, triangulate
from compass.generator import generate_compass, generate_journalism_review
from compass.pdf_parser import parse_besluit_from_upload, parse_inventory_from_upload

router = APIRouter(prefix="/compass", tags=["compass"])

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WOOLens/1.0; +https://woozm.nl)"}

_BOILERPLATE = re.compile(
    r"reactie op uw woo.verzoek|woo@\S+|verzoek om informatie|"
    r"besluit op woo.verzoek|uw verzoek om|kenmerk\s*:|ref\s*:",
    re.IGNORECASE,
)
_STRIP_PREFIX = re.compile(
    r"^(documenten bij|bijlage(n)? bij|re:|fwd?:)\s+",
    re.IGNORECASE,
)
# Strips WOO-specific lead-in phrases and Dutch prepositions to expose the topic keywords
_NEWS_STRIP = re.compile(
    r"^(besluit op woo-?verzoek|woo-?verzoek|reactie op|openbaarmaking|"
    r"openbaar making|inventarislijst|totstandkoming|verzoek om|"
    r"inzake|betreffende|aangaande|over de|over het|over)\s+",
    re.IGNORECASE,
)
# Dutch stop words to skip when extracting keywords
_NL_STOP = {
    "de","het","een","en","in","van","bij","op","met","over","voor","na",
    "uit","tot","aan","door","te","zijn","der","den","naar","om","als","of",
    "dat","dit","die","deze","maar","ook","nog","niet","meer","dan","zo",
}


def _news_candidates(dossier_title: str | None, top_subjects: list[str], search_query: str) -> list[str]:
    """Return candidate queries for news search, from most specific to broadest."""
    candidates: list[str] = []

    if dossier_title:
        # Strip WOO lead-ins iteratively
        clean = dossier_title.strip()
        prev = None
        while prev != clean:
            prev = clean
            clean = _NEWS_STRIP.sub("", clean).strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean and len(clean) > 8:
            candidates.append(clean[:60])
            # Also try a keyword-only version: take capitalized words (proper nouns / acronyms)
            kws = [w for w in clean.split() if w[0].isupper() and w.lower() not in _NL_STOP][:5]
            if len(kws) >= 2:
                candidates.append(" ".join(kws))

    # Non-boilerplate email subjects (abbreviated)
    for s in top_subjects:
        s = s.strip()
        if s and not _BOILERPLATE.search(s) and len(s) > 8:
            candidates.append(s[:50])

    # General search query as last resort
    if search_query:
        candidates.append(search_query[:50])

    # Deduplicate preserving order
    seen: set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]


def _build_search_query(
    subject: str | None,
    top_subjects: list[str],
    body_name: str | None,
    dossier_title: str | None,
) -> str:
    """Return the best possible search query for related dossiers / news."""
    # Priority 1: dossier title from frontend, cleaned up
    if dossier_title:
        clean = dossier_title.strip()
        # Strip WOO lead-ins and Dutch prepositions iteratively (same as _news_candidates)
        prev = None
        while prev != clean:
            prev = clean
            clean = _NEWS_STRIP.sub("", clean).strip()
            clean = _STRIP_PREFIX.sub("", clean).strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean and not _BOILERPLATE.search(clean) and len(clean) > 8:
            return clean[:80]

    # Priority 2: non-boilerplate email subjects
    for s in top_subjects:
        if s and not _BOILERPLATE.search(s) and len(s) > 5:
            return s[:80]

    # Priority 3: besluit subject, stripped of boilerplate
    if subject and not _BOILERPLATE.search(subject):
        return subject[:80]

    # Fallback: body name (ministry)
    return (body_name or "")[:80]


async def _fetch_news_articles(
    dossier_title: str | None,
    top_subjects: list[str],
    search_query: str,
    date_range: list | None = None,
    limit: int = 5,
) -> list[dict]:
    """Fetch Dutch news articles from Google News RSS, trying multiple candidate queries."""
    import xml.etree.ElementTree as ET
    from datetime import date as _date, timedelta as _td

    # Build date filter for Google News (30 days before start → 60 days after end)
    _date_filter = ""
    if date_range and len(date_range) >= 1 and date_range[0]:
        try:
            _start = _date.fromisoformat(str(date_range[0])[:10]) - _td(days=30)
            _end   = _date.fromisoformat(str((date_range[1] if len(date_range) >= 2 else None) or date_range[0])[:10]) + _td(days=60)
            _date_filter = f" after:{_start.isoformat()} before:{_end.isoformat()}"
        except Exception:
            pass

    async def _rss_fetch(query: str) -> list[dict]:
        try:
            filtered_query = (query[:80] + _date_filter).strip()
            url = (
                f"https://news.google.com/rss/search"
                f"?q={urllib.parse.quote(filtered_query)}&hl=nl&gl=NL&ceid=NL:nl"
            )
            async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
                r = await client.get(url, headers=_HEADERS)
            if not r.is_success:
                return []
            root = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None:
                return []
            articles = []
            for item in channel.findall("item")[:limit]:
                title  = (item.findtext("title")   or "").strip()
                link   = (item.findtext("link")    or "").strip()
                pub    = (item.findtext("pubDate") or "").strip()
                src_el = item.find("source")
                source = src_el.text.strip() if src_el is not None and src_el.text else ""
                if title and link:
                    articles.append({"title": title, "url": link, "source": source, "date": pub})
            return articles
        except Exception:
            return []

    for candidate in _news_candidates(dossier_title, top_subjects, search_query):
        articles = await _rss_fetch(candidate)
        if articles:
            return articles
    return []


async def _fetch_related_dossiers(query: str, limit: int = 4) -> list[dict]:
    """Search WooZM for related dossiers using subject keywords."""
    if not query.strip():
        return []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
            r = await client.get(
                "https://woozm.nl/search",
                params={"q": query[:80], "page": 1, "json": "true"},
                headers=_HEADERS,
            )
        if r.is_success:
            data = r.json()
            results = data.get("results", [])[:limit]
            return [
                {
                    "pid":   res.get("pid", ""),
                    "title": (res.get("title") or "")[:100],
                    "date":  res.get("date", ""),
                    "body":  (res.get("body_name") or res.get("body") or "")[:60],
                }
                for res in results
                if res.get("title")
            ]
    except Exception:
        pass
    return []


@router.post("/generate")
async def generate_compass_endpoint(
    dossier_json: str = Form(..., description="WOOLens dossier JSON as a string"),
    api_key: Optional[str] = Form(None, description="OpenAI API key (overrides OPENAI_API_KEY env var)"),
    body_name: Optional[str] = Form(None, description="Bestuursorgaan name (optional override)"),
    decision_outcome: Optional[str] = Form(None, description="Decision outcome (optional override)"),
    dossier_title: Optional[str] = Form(None, description="Dossier title from the frontend for context search"),
    pid: Optional[str] = Form(None, description="PID URL or identifier — used to fetch official title from pid.wooverheid.nl"),
    language: Optional[str] = Form(None, description="Output language: nl (default), en, or pap (Papiamentu)"),
    besluit_pdf: Optional[UploadFile] = File(None, description="Besluit PDF file"),
    inventory_pdf: Optional[UploadFile] = File(None, description="Inventarislijst PDF file"),
    besluit_pdf_url: Optional[str] = Form(None, description="URL of besluit PDF (used when no file is uploaded)"),
    inventory_pdf_url: Optional[str] = Form(None, description="URL of inventory PDF (used when no file is uploaded)"),
) -> dict:
    """
    Generate the Compass explanation for a Woo dossier.

    Accepts:
    - dossier_json  — JSON string with existing WOOLens analysis (required)
    - besluit_pdf   — PDF of the decision letter (optional)
    - inventory_pdf — PDF of the inventory list (optional)
    - body_name     — bestuursorgaan name override (optional)
    - decision_outcome — decision outcome override (optional)

    Returns:
    {
      "compass":            { body_explanation, inventory_explanation,
                              redaction_explanation, decision_explanation,
                              context_explanation },
      "triangulation":      { outcome_vs_disclosure, dominant_ground_flag,
                              informal_channel_flag, draft_iteration_flag,
                              processing_time_flag, cross_source_summary },
      "extraction_summary": { besluit_available, inventory_available,
                              documents_available, total_emails, total_chats,
                              total_other_docs, total_pages, total_inventory_docs,
                              disclosure_percentage, processing_days, refusal_codes,
                              has_informal_channels, unusual_flags,
                              organisations_found, date_range }
    }
    """
    # Parse dossier JSON
    try:
        dossier = json.loads(dossier_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"dossier_json is geen geldige JSON: {exc}")

    # Fetch official title from pid.wooverheid.nl — overrides the local filename
    if pid:
        _pid_id = pid.strip()
        if _pid_id.startswith("http"):
            _qs = urllib.parse.parse_qs(urllib.parse.urlparse(_pid_id).query)
            _pid_id = (_qs.get("pid") or [_pid_id])[0]
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=6) as _c:
                _r = await _c.get(
                    f"https://pid.wooverheid.nl/?pid={_pid_id}&infobox=true",
                    headers=_HEADERS,
                )
            if _r.is_success:
                _meta = _r.json()
                _pid_title = (
                    _meta.get("dc_title") or _meta.get("title") or _meta.get("titel") or ""
                ).strip()
                if _pid_title:
                    dossier_title = _pid_title
                    print(f"[compass] PID title fetched: {_pid_title!r}")
        except Exception as _e:
            print(f"[compass] PID fetch failed: {_e}")

    # Fetch besluit PDF bytes — from upload or URL
    besluit: Optional[dict] = None
    _besluit_bytes: Optional[bytes] = None
    if besluit_pdf is not None:
        _besluit_bytes = await besluit_pdf.read()
    elif besluit_pdf_url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as _c:
                _r = await _c.get(besluit_pdf_url, headers=_HEADERS)
            if _r.is_success:
                _besluit_bytes = _r.content
        except Exception as _e:
            print(f"[compass] besluit PDF URL fetch failed: {_e}")
    if _besluit_bytes:
        try:
            besluit = parse_besluit_from_upload(_besluit_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Besluit PDF kon niet worden verwerkt: {exc}"
            )

    # Apply manual overrides on top of parsed besluit
    if besluit is None and (body_name or decision_outcome):
        besluit = {}
    if besluit is not None:
        if body_name:
            besluit["body_name"] = body_name
        if decision_outcome:
            besluit["outcome"] = decision_outcome

    # Fetch inventory PDF bytes — from upload or URL
    inventory_rows: Optional[list[dict]] = None
    _inv_bytes: Optional[bytes] = None
    if inventory_pdf is not None:
        _inv_bytes = await inventory_pdf.read()
    elif inventory_pdf_url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as _c:
                _r = await _c.get(inventory_pdf_url, headers=_HEADERS)
            if _r.is_success:
                _inv_bytes = _r.content
        except Exception as _e:
            print(f"[compass] inventory PDF URL fetch failed: {_e}")
    if _inv_bytes:
        try:
            inventory_rows = parse_inventory_from_upload(_inv_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Inventarislijst PDF kon niet worden verwerkt: {exc}"
            )

    # Extract from all sources
    try:
        extraction = extract_all(dossier, besluit=besluit, inventory_rows=inventory_rows)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Data-extractie mislukt: {exc}"
        )

    # Triangulate across sources
    try:
        tri = triangulate(extraction)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Triangulatie mislukt: {exc}"
        )

    # Generate compass + journalism review with GPT-4o (run in parallel)
    try:
        _real_api_key = api_key if (api_key and api_key.startswith("sk-")) else None
        effective_key = _real_api_key or os.environ.get("OPENAI_API_KEY") or None
        _lang = language if language in ("nl", "en", "pap") else "nl"
        compass, journalism = await asyncio.gather(
            asyncio.to_thread(generate_compass, extraction, tri, effective_key, dossier_title, _lang),
            asyncio.to_thread(generate_journalism_review, dossier, effective_key, dossier_title, _lang),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"GPT-4o aanroep mislukt. Controleer of OPENAI_API_KEY is ingesteld. Fout: {exc}",
        )

    # Fetch contextualisation — related dossiers from WooZM (non-blocking best-effort)
    search_query = _build_search_query(
        subject=extraction.get("subject"),
        top_subjects=extraction.get("top_subjects", []),
        body_name=extraction.get("body_name"),
        dossier_title=dossier_title,
    )
    related_dossiers, news_articles = await asyncio.gather(
        _fetch_related_dossiers(search_query),
        _fetch_news_articles(
            dossier_title=dossier_title,
            top_subjects=extraction.get("top_subjects", []),
            search_query=search_query,
            date_range=extraction.get("date_range") or [],
        ),
    )
    policy_url = (
        f"https://www.rijksoverheid.nl/zoeken?q={urllib.parse.quote(search_query[:60])}"
        if search_query else ""
    )
    nos_url = (
        f"https://nos.nl/zoeken?q={urllib.parse.quote(search_query[:60])}"
        if search_query else ""
    )

    # Full extraction summary for the frontend
    extraction_summary = {
        # availability flags
        "besluit_available":      extraction["besluit_available"],
        "inventory_available":    extraction["inventory_available"],
        "documents_available":    extraction["documents_available"],
        # besluit fields
        "request_date":           extraction["request_date"],
        "receipt_date":           extraction["receipt_date"],
        "decision_date":          extraction["decision_date"],
        "processing_days":        extraction["processing_days"],
        "deadline_extended":      extraction["deadline_extended"],
        "default_notice":         extraction["default_notice"],
        "outcome":                extraction["outcome"],
        "body_name":              extraction["body_name"],
        "subject":                extraction["subject"],
        "refusal_grounds_besluit":extraction["refusal_grounds_besluit"],
        "appeal_deadline_weeks":  extraction["appeal_deadline_weeks"],
        "appeal_body":            extraction["appeal_body"],
        "contact_person":         extraction["contact_person"],
        # inventory fields
        "total_inventory_docs":   extraction["total_inventory_docs"],
        "fully_public":           extraction["fully_public"],
        "partially_public":       extraction["partially_public"],
        "not_public":             extraction["not_public"],
        "disclosure_percentage":  extraction["disclosure_percentage"],
        "refusal_ground_counts":  extraction["refusal_ground_counts"],
        "has_whatsapp":           extraction["has_whatsapp"],
        "has_signal":             extraction["has_signal"],
        "has_chat_export":        extraction["has_chat_export"],
        "has_images":             extraction["has_images"],
        "buiten_verzoek_count":   extraction["buiten_verzoek_count"],
        "numbering_gaps":         extraction["numbering_gaps"],
        "draft_clusters":         extraction["draft_clusters"],
        "unusual_flags":          extraction["unusual_flags"],
        # dossier fields
        "total_emails":           extraction["total_emails"],
        "total_chats":            extraction["total_chats"],
        "total_other_docs":       extraction["total_other_docs"],
        "total_pages":            extraction["total_pages"],
        "refusal_codes":          extraction["refusal_codes"],
        "has_informal_channels":  extraction["has_whatsapp"] or extraction["has_signal"],
        "has_redacted_chat_messages": extraction["has_redacted_chat_messages"],
        "unique_senders":         extraction["unique_senders"],
        "named_persons":          extraction["named_persons"],
        "document_authors":       extraction["document_authors"],
        "document_recipients":    extraction["document_recipients"],
        "channel_counts":         extraction["channel_counts"],
        "redacted_passage_count": extraction["redacted_passage_count"],
        "doc_types_breakdown":    extraction["doc_types_breakdown"],
        "top_subjects":           extraction["top_subjects"],
        "organisations_found":    extraction["organisations_found"],
        "date_range":             extraction["date_range"],
        "key_dates":              extraction["key_dates"],
    }

    return {
        "compass":            compass,
        "triangulation":      {
            "outcome_vs_disclosure": tri.outcome_vs_disclosure,
            "dominant_ground_flag":  tri.dominant_ground_flag,
            "informal_channel_flag": tri.informal_channel_flag,
            "draft_iteration_flag":  tri.draft_iteration_flag,
            "processing_time_flag":  tri.processing_time_flag,
            "cross_source_summary":  tri.cross_source_summary,
        },
        "extraction_summary": extraction_summary,
        "context": {
            "related_dossiers": related_dossiers,
            "news_articles":    news_articles,
            "policy_url":       policy_url,
            "nos_url":          nos_url,
        },
        "journalism": journalism,
    }
