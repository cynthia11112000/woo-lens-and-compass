"""
main.py — standard OCR pipeline.

Steps:
  1. pipeline_ocr       — extract text and metadata from the PDF
  2. text_sorting       — detect each document's date and sort chronologically
                          (email_splitter is used internally for e-mail documents)
  3. visualisation      — render an interactive HTML timeline

Usage:
    python main.py                              # test.pdf → woo_timeline.html
    python main.py --pdf my_dossier.pdf --out output.html
"""
import argparse
from pathlib import Path

from pipeline_ocr import load_pdf
from text_sorting import sort_documents
from visualisation import build_html

_DEFAULT_PDF = Path("test.pdf")
_DEFAULT_OUT = Path("woo_timeline.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WOO document pipeline — PDF → interactive HTML timeline"
    )
    parser.add_argument(
        "--pdf", default=str(_DEFAULT_PDF), metavar="FILE",
        help="PDF file to process (default: test.pdf)",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT), metavar="FILE",
        help="Output HTML file (default: woo_timeline.html)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("Step 1/3 — Loading and parsing PDF (with 300 DPI OCR supplement)...")
    docs = load_pdf(pdf_path, ocr_supplement=True)

    print("Step 2/3 — Extracting dates and sorting chronologically...")
    docs = sort_documents(docs)

    print("Step 3/3 — Building interactive HTML timeline...")
    build_html(docs, out_path, pdf_path=pdf_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
