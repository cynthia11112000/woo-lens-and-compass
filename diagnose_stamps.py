"""
diagnose_stamps.py

For each page of the palestina PDFs, show exactly which region produced
the detected code (or None) — so we can see whether false positives come
from the top-strip text layer or from raster OCR in a bottom region.

Usage:
    python diagnose_stamps.py palestina_combined.pdf [max_pages]
"""
import re
import sys
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

_TOP_FRAC  = 0.12
_SIDE_FRAC = 0.30
_BTM_FRAC  = 0.10
_PAGE_NUM_H_FRAC = 0.06
_PAGE_NUM_W_FRAC = 0.12

_DOC_CODE_TEXT_REGIONS = [
    ("top-right",  (1 - _SIDE_FRAC, 0.00, 1.00,       _TOP_FRAC)),
    ("top-left",   (0.00,           0.00, _SIDE_FRAC,  _TOP_FRAC)),
    ("top-full",   (0.00,           0.00, 1.00,        _TOP_FRAC)),
]

_DOC_CODE_RASTER_REGIONS = [
    ("top-right",  (1 - _SIDE_FRAC, 0.00,          1.00,       _TOP_FRAC)),
    ("top-left",   (0.00,           0.00,           _SIDE_FRAC, _TOP_FRAC)),
    ("btm-right",  (1 - _SIDE_FRAC, 1 - _BTM_FRAC, 1.00,       1.00)),  # before btm-middle
    ("btm-middle", (0.30,           1 - _BTM_FRAC,  0.70,       1.00)),
    ("btm-left",   (0.00,           1 - _BTM_FRAC,  _SIDE_FRAC, 1.00)),
    ("top-full",   (0.00,           0.00,           1.00,       _TOP_FRAC)),
]

_PAGE_NUM_REGIONS = [
    ("btm-right",  (1 - _PAGE_NUM_W_FRAC, 1 - _PAGE_NUM_H_FRAC, 1.00, 1.00)),
]

def _is_year(code: str) -> bool:
    return 1900 <= int(code) <= 2099


def diagnose_text_layer(page):
    """Return list of (region_name, all_4digit_words, picked_code)."""
    w, h = page.width, page.height
    words = page.extract_words()
    results = []
    for name, (x0f, y0f, x1f, y1f) in _DOC_CODE_TEXT_REGIONS:
        x0, y0, x1, y1 = w * x0f, h * y0f, w * x1f, h * y1f
        found = [
            wd["text"].strip()
            for wd in words
            if (x0 <= wd["x0"] and wd["x1"] <= x1
                and y0 <= wd["top"] and wd["bottom"] <= y1
                and re.fullmatch(r"\d{4}", wd["text"].strip()))
        ]
        non_year = [c for c in found if not _is_year(c)]
        picked = max(non_year, key=lambda c: next(
            wd["x0"] for wd in words if wd["text"].strip() == c
        )) if non_year else None
        results.append((name, found, picked))
    return results


_LEADING_ZERO_RE = re.compile(r"\b(0\d{3})\b")

def diagnose_raster(image: Image.Image):
    """Return list of (region_name, ocr_raw_text, picked_code)."""
    w, h = image.size
    results = []
    for name, (x0f, y0f, x1f, y1f) in _DOC_CODE_RASTER_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        text = pytesseract.image_to_string(
            crop, config="--psm 6 -c tessedit_char_whitelist=0123456789"
        ).strip()
        picked = None
        m = _LEADING_ZERO_RE.search(text)
        if m and not _is_year(m.group(1)):
            picked = m.group(1)
        # Fallback: '0NNN' embedded in barcode (at least 2 preceding digits).
        # Rejects short sequences like "00001" (5 chars, insufficient).
        elif name.startswith("btm"):
            m2 = re.search(r"\d{2,}(0\d{3})", text)
            if m2:
                picked = m2.group(1)
        results.append((name, text, picked))
    return results


def diagnose_page_num(image: Image.Image):
    """Return raw OCR text and parsed page number from the bottom-right stamp."""
    w, h = image.size
    for name, (x0f, y0f, x1f, y1f) in _PAGE_NUM_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        text = pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()
        m = re.search(r"\b(\d{1,3})\b", text)
        num = int(m.group(1)) if m else None
        return name, text, num
    return "btm-right", "", None


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("palestina_combined.pdf")
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 999999

    print(f"Diagnosing: {pdf_path}  (first {max_pages} pages)")
    print("=" * 90)

    # Convert to images for raster analysis
    print("Rendering pages at 200 DPI …")
    images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=min(max_pages, 60))
    print(f"  → {len(images)} page image(s) ready\n")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            img = images[i - 1] if i - 1 < len(images) else None

            # Text-layer detection
            txt_results = diagnose_text_layer(page)
            txt_code = next((r[2] for r in txt_results if r[2]), None)
            txt_region = next((r[0] for r in txt_results if r[2]), None)

            # Raster detection
            if img:
                rst_results = diagnose_raster(img)
                rst_code = next((r[2] for r in rst_results if r[2]), None)
                rst_region = next((r[0] for r in rst_results if r[2]), None)
                pn_name, pn_raw, pn_num = diagnose_page_num(img)
            else:
                rst_code = rst_region = None
                pn_num = None

            # Pipeline logic: raster only runs when within_doc_page==1 or txt_code is None
            raster_triggered = (pn_num == 1) or (txt_code is None)
            pipeline_code = txt_code
            if raster_triggered and rst_code:
                pipeline_code = rst_code

            # Only print interesting rows (spurious codes or missing codes)
            interesting = pipeline_code and not (pipeline_code.startswith("0") and len(pipeline_code) == 4 and not _is_year(pipeline_code) and 1 <= int(pipeline_code) <= 500)
            # Actually print all for now; filter to first region that has a candidate

            flag = ""
            if pipeline_code and _is_year(pipeline_code):
                flag = " ← YEAR"
            elif pipeline_code and not pipeline_code.startswith("0"):
                flag = " ← SUSPECT"

            # Show text-layer candidates in top regions
            txt_detail = "; ".join(
                f"{r[0]}: {r[1] or '—'} → {r[2] or '—'}"
                for r in txt_results
                if r[1]  # only regions with any 4-digit words
            )

            # Show raster results for all regions
            if img:
                rst_detail = "; ".join(
                    f"{r[0]}: {repr(r[1][:20]) if r[1] else '—'} → {r[2] or '—'}"
                    for r in rst_results
                    if r[1]
                )
            else:
                rst_detail = "(no image)"

            print(f"Page {i:>3} │ txt={txt_code or '—':>6} [{txt_region or '—'}]"
                  f" │ raster={rst_code or '—':>6} [{rst_region or '—'}]"
                  f" │ pagenum={pn_num or '—'}"
                  f" │ FINAL={pipeline_code or '—'}{flag}")
            if txt_detail:
                print(f"         │  TXT detail: {txt_detail}")
            if img and rst_detail and (flag or txt_code is None):
                print(f"         │  RST detail: {rst_detail}")

    print("\nDone.")


if __name__ == "__main__":
    main()
