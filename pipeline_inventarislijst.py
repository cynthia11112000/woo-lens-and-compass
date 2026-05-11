"""
pipeline_inventarislijst.py — Extract WOO document inventory from an Inventarislijst PDF.

Uses GPT-4o-mini vision to parse the inventory table rows.
Can be given a separate PDF or a page range from the main dossier PDF.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

from PIL import Image

_MODEL      = "gpt-4o-mini"
_RENDER_DPI = 150        # table text is large; lower DPI keeps cost down
_MAX_RETRIES = 3

_SYSTEM = (
    "You are extracting rows from a Dutch government WOO (Wet Open Overheid) inventarislijst — "
    "a table listing all documents in a disclosure dossier with their codes, titles, page counts, "
    "dates, and the WOO transparency decision for each. "
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_PROMPT = """\
This page is from a WOO inventarislijst (document inventory table).
Extract every document row visible on this page and return:

{
  "items": [
    {
      "code":     "<4-digit doc code e.g. '0001', or null>",
      "title":    "<document title or description>",
      "date":     "<date as YYYY-MM-DD, or null>",
      "pages":    <integer page count, or null>,
      "decision": "<openbaar | gedeeltelijk openbaar | niet openbaar | null>",
      "grounds":  ["<5.1.2e>", ...]
    }
  ]
}

Rules:
- code: 4-digit document number. Normalise to zero-padded string (e.g. "0001", "0023").
- decision: normalise to exactly one of the three Dutch values above, or null.
- grounds: WOO article codes (5.1.x or 5.2.x) listed for this document's redactions. Empty [] if none.
- If this page has no table rows (e.g. only a header or footer), return {"items": []}.
"""


def _encode_page(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def extract_inventarislijst(
    pdf_path:   Path,
    api_key:    str,
    page_range: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Extract the document inventory table from a WOO Inventarislijst PDF.

    Args:
        pdf_path:   Path to the PDF (dedicated file or main dossier PDF).
        api_key:    OpenAI API key (used for GPT-4o-mini).
        page_range: (start, end) 1-indexed inclusive page range, or None for all pages.

    Returns:
        List of dicts with keys: code, title, date, pages, decision, grounds.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    from pdf2image import convert_from_path

    client = OpenAI(api_key=api_key)

    print(f"[inventarislijst] Converting PDF to images at {_RENDER_DPI} DPI...")
    all_images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=_RENDER_DPI)

    if page_range:
        start  = max(1, page_range[0])
        end    = min(len(all_images), page_range[1])
        images = all_images[start - 1 : end]
    else:
        images = all_images

    print(f"[inventarislijst] Analysing {len(images)} page(s) with {_MODEL}...")

    all_items: list[dict] = []
    for i, img in enumerate(images):
        b64 = _encode_page(img)
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=_MODEL,
                    max_tokens=4096,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}",
                                        "detail": "high",
                                    },
                                },
                                {"type": "text", "text": _PROMPT},
                            ],
                        },
                    ],
                )
                data  = json.loads(response.choices[0].message.content or "{}")
                items = data.get("items") or []
                print(f"[inventarislijst] Pagina {i + 1}: {len(items)} rij(en) gevonden")
                all_items.extend(items)
                break
            except json.JSONDecodeError as e:
                print(f"  [inventarislijst] p{i+1}: JSON parse error (attempt {attempt+1}): {e}")
            except Exception as e:
                print(f"  [inventarislijst] p{i+1}: API error (attempt {attempt+1}): {str(e)[:100]}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

    # Deduplicate by code — same document can appear on multiple table pages
    seen:   set[str]   = set()
    result: list[dict] = []
    for item in all_items:
        code = item.get("code")
        if code and code not in seen:
            seen.add(code)
            result.append(item)
        elif not code:
            result.append(item)

    print(f"[inventarislijst] Klaar — {len(result)} document(en) gevonden.")
    return result
