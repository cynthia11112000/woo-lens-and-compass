"""
compass/generator.py

Generate the five Compass explanation sections using GPT-4o.
OPENAI_API_KEY is read from the environment.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from compass.extractor import TriangulationInsights

_client: Optional[OpenAI] = None


def _get_client(api_key: Optional[str] = None) -> OpenAI:
    global _client
    if api_key:
        return OpenAI(api_key=api_key)
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from environment
    return _client


_LANG_CFG: dict[str, dict[str, str]] = {
    "nl": {
        "sys_compass": (
            "Je bent een expert in Nederlandse overheidstransparantie. "
            "Je legt complexe Woo-dossiers uit aan gewone burgers zonder juridische kennis. "
            "Schrijf altijd in begrijpelijk Nederlands zonder jargon."
        ),
        "task": (
            "Schrijf vijf korte, begrijpelijke uitleg-paragrafen voor gewone burgers zonder juridische kennis.\n"
            "Gebruik eenvoudig Nederlands. Vermijd jargon. Elke paragraaf is 2-3 zinnen."
        ),
        "output_note": "Vul elk veld in met echte, specifieke inhoud op basis van de data hierboven.\nSchrijf uitsluitend in begrijpelijk Nederlands. Geen jargon. Geen herhalingen.",
        "sys_journalism": (
            "Je bent een onderzoeksjournalist gespecialiseerd in Nederlandse overheidsstukken. "
            "Je analyseert Woo-documenten kritisch en schrijft bevindingen op in journalistiek Nederlands. "
            "Je parafraseer altijd — je citeert nooit letterlijk."
        ),
        "journalism_intro": (
            "Je analyseert e-mails en chatberichten uit een Nederlands Woo-dossier.\n"
            "Jouw taak is een journalistieke review te schrijven die de inhoud onthult"
            "— niet alleen beschrijft wat er staat, maar wat het betekent."
        ),
        "journalism_rules": (
            "REGELS:\n"
            "- Parafraseer altijd, citeer nooit letterlijk (auteursrecht)\n"
            "- Gebruik 'Persoon A', 'Persoon B' etc. voor zwartgelakte namen\n"
            "- Wees concreet: vermijd vage termen als 'zorgwekkend' of 'interessant' zonder onderbouwing\n"
            "- Schrijf journalistiek Nederlands, alsof je een onderzoeksjournalist bent\n"
            "- Maximaal 3 key_findings"
        ),
    },
    "en": {
        "sys_compass": (
            "You are an expert in Dutch government transparency. "
            "You explain complex Woo (Government Information Act) dossiers to ordinary citizens without legal knowledge. "
            "Always write in clear, plain English without jargon."
        ),
        "task": (
            "Write five short, easy-to-understand explanation paragraphs for ordinary citizens without legal knowledge.\n"
            "Use plain English. Avoid jargon. Each paragraph is 2-3 sentences."
        ),
        "output_note": "Fill each field with real, specific content based on the data above.\nWrite exclusively in clear, plain English. No jargon. No repetition.",
        "sys_journalism": (
            "You are an investigative journalist specialising in Dutch government documents. "
            "You critically analyse Woo documents and write findings in journalistic English. "
            "Always paraphrase — never quote literally."
        ),
        "journalism_intro": (
            "You are analysing emails and chat messages from a Dutch Woo (Government Information Act) dossier.\n"
            "Your task is to write a journalistic review that reveals the content "
            "— not just describes what is there, but what it means."
        ),
        "journalism_rules": (
            "RULES:\n"
            "- Always paraphrase, never quote literally (copyright)\n"
            "- Use 'Person A', 'Person B' etc. for redacted names\n"
            "- Be concrete: avoid vague terms like 'concerning' or 'interesting' without substantiation\n"
            "- Write journalistic English, as if you are an investigative journalist\n"
            "- Maximum 3 key_findings"
        ),
    },
    "pap": {
        "sys_compass": (
            "You are an expert in Dutch government transparency. "
            "You explain complex Woo (Government Information Act) dossiers to ordinary citizens without legal knowledge. "
            "CRITICAL: Write ALL output exclusively in Papiamentu "
            "(the creole language of Curaçao, Bonaire, and Aruba). Use clear, simple Papiamentu without legal jargon."
        ),
        "task": (
            "Write five short, easy-to-understand explanation paragraphs for ordinary citizens without legal knowledge.\n"
            "CRITICAL: Write ALL output exclusively in Papiamentu (the creole language of Curaçao, Bonaire, and Aruba). "
            "Use simple Papiamentu. Avoid legal jargon. Each paragraph is 2-3 sentences."
        ),
        "output_note": (
            "Fill each field with real, specific content based on the data above.\n"
            "CRITICAL: Every single word of the output must be in Papiamentu. No Dutch, no English."
        ),
        "sys_journalism": (
            "You are an investigative journalist specialising in Dutch government documents. "
            "You critically analyse Woo documents. "
            "CRITICAL: Write ALL output exclusively in Papiamentu "
            "(the creole language of Curaçao, Bonaire, and Aruba). Always paraphrase — never quote literally."
        ),
        "journalism_intro": (
            "You are analysing emails and chat messages from a Dutch Woo (Government Information Act) dossier.\n"
            "Your task is to write a journalistic review that reveals the content — not just describes what is there, but what it means.\n"
            "CRITICAL: Write ALL output exclusively in Papiamentu."
        ),
        "journalism_rules": (
            "RULES:\n"
            "- Always paraphrase, never quote literally (copyright)\n"
            "- Use 'Persona A', 'Persona B' etc. for redacted names\n"
            "- Be concrete: avoid vague terms without substantiation\n"
            "- Write in journalistic Papiamentu\n"
            "- Maximum 3 key_findings\n"
            "- CRITICAL: Every single word of output must be in Papiamentu"
        ),
    },
}

REFUSAL_EXPLANATIONS: dict[str, str] = {
    "5.1.1":  "veiligheid van de staat - informatie die de nationale veiligheid kan schaden",
    "5.1.2a": "eenheid van de Kroon - ministeriele adviezen en kabinetsuniformiteit",
    "5.1.2b": "economische of financiele belangen van de staat of openbare lichamen",
    "5.1.2c": "opsporing en vervolging van strafbare feiten",
    "5.1.2d": "inspectie, controle en toezicht door bestuursorganen",
    "5.1.2e": "internationale betrekkingen - diplomatieke gevoeligheid",
    "5.1.2f": "milieu-informatie die nadelig kan zijn voor de beschermde belangen",
    "5.1.2g": "eerbiediging van de persoonlijke levenssfeer - privacy",
    "5.1.2h": "onevenredig benadelen van betrokken personen of derden",
    "5.1.2i": "persoonsgegevens - bescherming van privedgegevens",
    "5.2.1":  "intern beraad - concept-documenten en notulen die niet bedoeld zijn voor buiten",
    "5.2.2":  "bedrijfs- en fabricagegegevens die vertrouwelijk zijn meegedeeld",
    "5.2.3":  "persoonlijke beleidsopvattingen in ambtelijke stukken",
    "5.2.4":  "gevaar voor het goed functioneren van de overheid",
    "5.2.5":  "beveiligingsinformatie die kwaad in de hand kan werken",
}


def _explain_grounds(codes: dict[str, int]) -> str:
    """Build a human-readable list of the top refusal grounds with plain explanations."""
    if not codes:
        return "Geen specifieke weigeringsgronden geregistreerd."
    top = sorted(codes.items(), key=lambda x: -x[1])[:5]
    lines = []
    for code, count in top:
        explanation = REFUSAL_EXPLANATIONS.get(code, "reden onbekend")
        lines.append(f"• {code} ({count}×): {explanation}")
    return "\n".join(lines)


def build_prompt(extraction: dict, triangulation: TriangulationInsights, dossier_title: Optional[str] = None, language: str = "nl") -> str:
    """Build the GPT-4o prompt from all extracted and triangulated data."""
    lang = _LANG_CFG.get(language) or _LANG_CFG["nl"]

    codes_str = _explain_grounds(extraction.get("refusal_codes", {}))
    flags_str = "\n".join(f"- {f}" for f in extraction.get("unusual_flags", [])) or "Geen bijzondere vlaggen."

    prompt_parts = [
        "You have structured data from a Dutch Woo dossier (Government Information Act).",
        lang["task"],
        "",
        "=== AVAILABLE DATA ===",
        "",
    ]

    if dossier_title:
        prompt_parts += [
            "--- DOSSIERTITEL ---",
            f"Titel: {dossier_title}",
            "BELANGRIJK: De context_explanation MOET alle specifieke partijen, landen, organisaties en conflicten",
            "uit deze titel benoemen. Als de titel 'Israël', 'Palestina', 'Gaza', 'AVVN', 'VN', of andere",
            "specifieke namen bevat, moeten die letterlijk terugkomen in de context_explanation.",
            "Generaliseer NOOIT als de titel concrete namen bevat.",
            "",
        ]

    # BuZa detection: Ministerie van Buitenlandse Zaken or international keywords in title
    _body = (extraction.get("body_name") or "").lower()
    _ttl  = (dossier_title or "").lower()
    _is_buza = (
        "buitenlandse" in _body or "buza" in _body or
        any(kw in _ttl for kw in ("israël", "israel", "palestin", "gaza", "avvn", "vnvr",
                                   "veiligheidsraad", "verenigde naties", "nato", "navo"))
    )
    if _is_buza:
        prompt_parts += [
            "--- INTERNATIONALE CONTEXT (Ministerie van Buitenlandse Zaken) ---",
            "Dit dossier heeft een internationale dimensie. De context_explanation MOET:",
            "  1. Alle landen uit de dossiertitel bij naam noemen (bijv. Israël, Palestijnse Gebieden, Nederland)",
            "  2. De specifieke internationale organisaties noemen (bijv. AVVN, VN-Veiligheidsraad, NATO)",
            "  3. Het concrete conflict of de diplomatieke kwestie beschrijven — NIET generaliseren",
            "  4. NOOIT vage uitdrukkingen gebruiken als 'bescherming van burgers', 'internationale betrekkingen'",
            "     of 'diplomatieke gevoeligheid' als de titel specifieke partijen en conflicten noemt.",
            "Voorbeeld van FOUT: 'Dit dossier gaat over de bescherming van burgers in gewapende conflicten.'",
            "Voorbeeld van GOED: 'Dit dossier gaat over de Nederlandse positiebepaling bij AVVN-resoluties",
            "                     over het conflict tussen Israël en de Palestijnse Gebieden.'",
            "",
        ]

    if extraction.get("besluit_available"):
        besluit_text_snippet = (extraction.get("besluit_raw_text") or "").strip()
        prompt_parts += [
            "--- BESLUIT ---",
            f"Bestuursorgaan: {extraction.get('body_name') or 'onbekend'}",
            f"Onderwerp verzoek: {extraction.get('subject') or 'onbekend'}",
            f"Beslissing: {extraction.get('outcome') or 'onbekend'}",
            f"Aanvraagdatum: {extraction.get('request_date') or 'onbekend'}",
            f"Beslisdatum: {extraction.get('decision_date') or 'onbekend'}",
            f"Behandelduur: {extraction.get('processing_days') or 0} dagen",
            f"Termijn verlengd: {'ja' if extraction.get('deadline_extended') else 'nee'}",
            f"Ingebrekestelling: {'ja' if extraction.get('default_notice') else 'nee'}",
            f"Bezwaar mogelijk bij: {extraction.get('appeal_body') or 'onbekend'}",
            f"Bezwaartermijn: {extraction.get('appeal_deadline_weeks') or 6} weken",
            f"Weigeringsgronden (besluit): {', '.join(extraction.get('refusal_grounds_besluit') or []) or 'geen'}",
        ]
        if besluit_text_snippet:
            prompt_parts += [
                "Volledige tekst besluitbrief (gebruik dit voor de context_explanation):",
                besluit_text_snippet,
            ]
        prompt_parts.append("")

    if extraction.get("inventory_available"):
        prompt_parts += [
            "--- INVENTARISLIJST ---",
            f"Totaal documenten: {extraction.get('total_inventory_docs', 0)}",
            f"Volledig openbaar: {extraction.get('fully_public', 0)}",
            f"Deels openbaar: {extraction.get('partially_public', 0)}",
            f"Niet openbaar: {extraction.get('not_public', 0)}",
            f"Openbaarheidpercentage: {extraction.get('disclosure_percentage', 0.0)}%",
            f"WhatsApp/Signal aanwezig: {'ja' if (extraction.get('has_whatsapp') or extraction.get('has_signal')) else 'nee'}",
            f"Bijzondere vlaggen:\n{flags_str}",
            "",
        ]

    if extraction.get("documents_available"):
        prompt_parts += [
            "--- OPENBAAR GEMAAKTE DOCUMENTEN ---",
            f"E-mails: {extraction.get('total_emails', 0)}",
            f"Chatgesprekken: {extraction.get('total_chats', 0)}",
            f"Overige documenten: {extraction.get('total_other_docs', 0)}",
            f"Totaal pagina's: {extraction.get('total_pages', 0)}",
            f"Gevonden organisaties: {', '.join(extraction.get('organisations_found') or []) or 'geen'}",
            f"Onderwerpen e-mails: {'; '.join(extraction.get('top_subjects') or []) or 'geen'}",
            "",
        ]

    prompt_parts += [
        "--- WEIGERINGSGRONDEN (UITLEG) ---",
        codes_str,
        "",
        "--- TRIANGULATIE INZICHTEN ---",
        f"Besluit vs. openbaarheid: {triangulation.outcome_vs_disclosure or 'n.v.t.'}",
        f"Dominante grond: {triangulation.dominant_ground_flag or 'n.v.t.'}",
        f"Informele kanalen: {triangulation.informal_channel_flag or 'n.v.t.'}",
        f"Conceptversies: {triangulation.draft_iteration_flag or 'n.v.t.'}",
        f"Behandelduur: {triangulation.processing_time_flag or 'n.v.t.'}",
        "",
        "=== REQUESTED OUTPUT (JSON) ===",
        "",
        "Generate only the following JSON object:",
        json.dumps({
            "body_explanation": "Which government body this is and why it holds these documents (2-3 sentences)",
            "inventory_explanation": "How many documents were found, what was made public and what this means (2-3 sentences)",
            "redaction_explanation": "Why information was kept secret, in plain language with the specific grounds explained (2-3 sentences)",
            "decision_explanation": "What the decision means for the applicant and how they can appeal (2-3 sentences)",
            "context_explanation": (
                f"Paraphrase the dossier title '{dossier_title[:120]}' as the first sentence, "
                "followed by 1-2 sentences about why someone would want to request this dossier. "
                "Name all countries, parties and organisations from the title explicitly."
            ) if dossier_title else (
                "What this dossier is about and why someone would want to request this information (2-3 sentences)"
            ),
        }, indent=2, ensure_ascii=False),
        "",
        lang["output_note"],
    ]

    return "\n".join(prompt_parts)


def generate_compass(extraction: dict, triangulation: TriangulationInsights, api_key: Optional[str] = None, dossier_title: Optional[str] = None, language: str = "nl") -> dict:
    """Call GPT-4o and return the five compass explanation sections."""
    lang = _LANG_CFG.get(language) or _LANG_CFG["nl"]
    prompt = build_prompt(extraction, triangulation, dossier_title, language=language)

    response = _get_client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": lang["sys_compass"]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1200,
        timeout=60,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    result = json.loads(raw)

    # Ensure all five keys are present even if the model omitted one
    defaults = {
        "body_explanation": "—",
        "inventory_explanation": "—",
        "redaction_explanation": "—",
        "decision_explanation": "—",
        "context_explanation": "—",
    }
    for key, fallback in defaults.items():
        if not result.get(key):
            result[key] = fallback

    return result


def _build_journalism_prompt(dossier: dict, dossier_title: Optional[str] = None, language: str = "nl") -> str:
    """Build a journalistic analysis prompt from raw dossier emails and chats."""
    lang = _LANG_CFG.get(language) or _LANG_CFG["nl"]
    lines = [
        lang["journalism_intro"],
        "",
        lang["journalism_rules"],
        "",
        "Generate only this JSON object:",
        json.dumps({
            "key_findings": [
                {
                    "finding": "Korte, concrete bevinding (1-2 zinnen)",
                    "source_type": "email | chat | document",
                    "date": "YYYY-MM-DD of leeg",
                    "significance": "Waarom dit ertoe doet (1 zin)"
                }
            ],
            "communication_pattern": "Beschrijf het communicatiepatroon: wie praat met wie, hoe formeel/informeel, via welke kanalen, en wat dit suggereert over de besluitvorming (2-3 zinnen)",
            "what_is_missing": "Wat ontbreekt opvallend in dit dossier? Denk aan zwartgemaakte passages, ontbrekende bijlagen, afwezige afzenders, of gaten in de tijdlijn (2-3 zinnen)",
            "journalist_tip": "Concreet vervolgonderzoek: welke vraag zou je nu stellen, welk document zou je opvragen, of welke persoon zou je benaderen? (1-2 zinnen)"
        }, indent=2, ensure_ascii=False),
        "",
        "=== DOCUMENTEN UIT HET DOSSIER ===",
        "",
    ]

    if dossier_title:
        lines.insert(2, f"Dossiertitel: {dossier_title}")
        lines.insert(3, "")

    emails = dossier.get("emails", [])[:15]
    chats = dossier.get("chats", [])[:5]
    others = dossier.get("documents", dossier.get("other_docs", []))[:10]

    for i, e in enumerate(emails, 1):
        subject = (e.get("subject") or "").strip()[:80]
        sender  = (e.get("sender") or e.get("from") or "").strip()[:60]
        to      = (e.get("to") or "").strip()[:60]
        date    = (e.get("date") or "").strip()[:10]
        text    = (e.get("text") or "").strip()[:500]
        lines.append(f"--- E-mail {i} ---")
        if date:    lines.append(f"Datum: {date}")
        if subject: lines.append(f"Onderwerp: {subject}")
        if sender:  lines.append(f"Van: {sender}")
        if to:      lines.append(f"Aan: {to}")
        if text:    lines.append(f"Inhoud:\n{text}")
        lines.append("")

    for i, c in enumerate(chats, 1):
        summary = (c.get("aiSummary") or "").strip()[:300]
        participants = ", ".join((c.get("deelnemers") or [])[:6])
        berichten = c.get("berichten", [])[:5]
        lines.append(f"--- Chat {i} ---")
        if participants: lines.append(f"Deelnemers: {participants}")
        if summary:      lines.append(f"Samenvatting: {summary}")
        for b in berichten:
            sender = (b.get("sender") or b.get("van") or "").strip()[:40]
            text   = (b.get("text") or b.get("tekst") or "").strip()[:200]
            date   = (b.get("date") or b.get("datum") or "").strip()[:10]
            if text:
                lines.append(f"  [{date}] {sender}: {text}")
        lines.append("")

    for i, d in enumerate(others, 1):
        title   = (d.get("title") or d.get("name") or "").strip()[:80]
        text    = (d.get("text") or d.get("content") or "").strip()[:600]
        doctype = (d.get("type") or "document").strip()
        lines.append(f"--- Document {i} ({doctype}) ---")
        if title: lines.append(f"Titel: {title}")
        if text:  lines.append(f"Inhoud:\n{text}")
        lines.append("")

    return "\n".join(lines)


def generate_journalism_review(dossier: dict, api_key: Optional[str] = None, dossier_title: Optional[str] = None, language: str = "nl") -> dict:
    """Call GPT-4o to produce a journalistic analysis of dossier content."""
    # Skip if there's nothing to analyse
    has_content = (
        dossier.get("emails") or
        dossier.get("chats") or
        dossier.get("documents") or
        dossier.get("other_docs")
    )
    if not has_content:
        return {}

    lang = _LANG_CFG.get(language) or _LANG_CFG["nl"]
    prompt = _build_journalism_prompt(dossier, dossier_title, language=language)

    response = _get_client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": lang["sys_journalism"]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
        timeout=60,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    result = json.loads(raw)

    # Ensure expected keys are present
    defaults: dict = {
        "key_findings": [],
        "communication_pattern": "",
        "what_is_missing": "",
        "journalist_tip": "",
    }
    for key, fallback in defaults.items():
        if key not in result:
            result[key] = fallback

    return result
