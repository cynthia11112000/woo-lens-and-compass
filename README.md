# WOO Organizer

Converts Dutch WOO (*Wet Open Overheid*) disclosure PDFs into an interactive HTML timeline. Documents are split, dated, categorised, and rendered with inline email cards, page thumbnails, recipient tags, attachment markers, and redaction-code summaries.

---

## Pipelines

Two independent pipelines share the same date-extraction, sorting, and visualisation layer. They differ only in how they extract text and detect document boundaries.

| Pipeline | Entry point | Text extraction | Requires |
|---|---|---|---|
| OCR | `main.py` | pdfplumber + Tesseract | Tesseract, Poppler |
| GPT-4o VLM | `main_gpt4o.py` | GPT-4o vision API | OpenAI API key |

Both produce the same output schema and are directly benchmarkable against each other.

---

## Setup

**System dependencies (macOS)**
```bash
brew install tesseract poppler
```

**Python environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**API key** (GPT-4o pipeline only)
```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

### OCR pipeline
```bash
python main.py                                      # test.pdf → woo_timeline.html
python main.py --pdf my_dossier.pdf --out out.html
```

### GPT-4o pipeline
```bash
# Full run — processes every page, saves a cache automatically
python main_gpt4o.py --pdf my_dossier.pdf

# Test run — first 15 pages only (saves to a separate cache)
python main_gpt4o.py --pdf my_dossier.pdf --vlm-pages 15

# Regenerate HTML from an existing cache (no API cost)
python main_gpt4o.py --pdf my_dossier.pdf --from-cache my_dossier_gpt4o_cache.json

# Explicit API key
python main_gpt4o.py --pdf my_dossier.pdf --api-key sk-...
```

---

## Pipeline walkthrough

Both pipelines share the same date-extraction and sorting layer. The GPT-4o pipeline has an optional fourth step (event enrichment). Steps are described below in order.

---

### Step 1 — Text extraction and document splitting

This is the only step that differs between the two pipelines.

---

#### OCR pipeline (`pipeline_ocr.py`)

**1a. Searchability check**

`_is_searchable()` opens the PDF with pdfplumber and checks whether any page contains selectable text. The result decides whether the pipeline uses direct text extraction (fast) or full raster OCR (slower but handles image-based scans).

**1b. Stage 1 — Per-page data collection**

For every page in the PDF the pipeline collects four things:

**Doc code** — the 4-digit WOO catalogue stamp (e.g. `0144`) that identifies which document a page belongs to.

- *Text layer:* `_find_doc_code_words()` scans the top strip of the page using pdfplumber's word list. Only top regions are searched (`_DOC_CODE_TEXT_REGIONS`: top-right → top-left → top-full) because body text lower on the page can contain 4-digit numbers (article references, years) that would produce false positives. The rightmost matching word in the first matching region wins. Year numbers (1900–2099) are excluded.
- *Raster OCR:* `_find_doc_code_raster()` renders the page at 200 DPI and OCRs six corner/edge regions in priority order (`_DOC_CODE_RASTER_REGIONS`: top-right → top-left → bottom-right → bottom-middle → bottom-left → top-full). Uses a digits-only Tesseract whitelist (`--psm 6`). Two-stage matching per region:
  1. Primary: `\b(0\d{3})\b` — requires a leading zero, rejecting article-number fragments like `5122`.
  2. Fallback (bottom regions only): `\d{2,}(0\d{3})` — extracts embedded codes from merged barcode strings like `7601430` → `0143`.
- Raster is always tried when `within_doc_page == 1` or the text layer found nothing, since boundary pages are most likely to have a misread or ambiguous stamp.

**Within-doc page number** — the per-document page counter stamped on each page (e.g. page 1 of 3). `_find_within_doc_page_raster()` OCRs a tight strip at the bottom-right corner (`_PAGE_NUM_REGIONS`) with a digits-only whitelist and returns the integer found.

**Page text** — `page.extract_text()` for searchable PDFs; `_ocr_image()` (Tesseract, Dutch+English) for image-based pages or any page with fewer than 20 text characters.

**Redaction codes** — WOO legal grounds for redaction (regex `5\.[12]\.[1-9][a-z]{0,2}`, e.g. `5.1.2e`, `5.2.1`) are counted per page from the extracted text. With `ocr_supplement=True`, an additional 300 DPI OCR pass is run and counts are merged as `max(text_layer, ocr)` to capture image-only redaction stamps that the text layer misses.

**1c. Stage 2 — Document code assignment**

After all pages are collected, the pipeline decides which pages belong to the same document:

- **Stamps detected** → `_build_docs_forward_fill()`: each page inherits the most recently seen stamp code. If a page has no code but `_is_new_doc_boundary()` fires (i.e. `within_doc_page == 1` or `"Pagina 1 van"` appears in the text), an auto-generated `unknown_N` code is created for it.
- **No stamps anywhere** → auto-split mode:
  - *Default:* `_build_docs_auto_split()` uses heuristics — `within_doc_page == 1`, `"Pagina 1 van"` in text, or a fresh email header block (≥2 distinct header fields in the first 20 lines, with a first-line veto for continuation pages) appearing after non-email content. Documents receive `auto_001`, `auto_002`, … codes.
  - *With `semantic_split=True`:* `_embed_pages()` encodes each page with a multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`). `_semantic_boundaries()` computes cosine similarity between adjacent pages; a drop below `semantic_threshold` (default 0.35) is treated as a document boundary, fused with the heuristic signals.

**1d. Finalisation — `_finalize_docs()`**

For each document: page texts are joined, redaction codes across all pages are summed, `_annotate_redactions()` wraps each bare code in a `[REDACTED: …]` marker, and `_categorize_document()` classifies the document by keyword scoring against 7 category rule sets (Email patterns scored on first page only to avoid false positives from quoted email headers in body text).

---

#### GPT-4o pipeline (`pipeline_gpt4o.py`)

**1a. PDF → images**

`convert_from_path()` renders every page to a PIL image at 200 DPI. In test mode (`--vlm-pages N`) only the first N pages are processed and cached under a separate filename.

**1b. Stage 1 — Per-page GPT-4o analysis**

Each page image is encoded as a base64 JPEG (resized to ≤1568 px, the optimal tile size for GPT-4o `detail=high`) and sent to the OpenAI chat completions API with a structured JSON prompt. The model is asked to return:

| Field | What it captures |
|---|---|
| `text` | Full page text in reading order; email headers preserved on their own lines; redacted sections written as `[REDACTED]` or `<[REDACTED]@domain.nl>` |
| `is_new_document` | Whether this is the first page of a new document |
| `doc_code` | 4-digit stamp visible anywhere on the page, or null |
| `within_doc_page` | Page number within the document (`"Pagina X van N"`), or null |
| `category` | One of: Email, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, Other |
| `doc_subtype` | Fine-grained subtype within the category (e.g. `kamerbrief`, `besluit`, `persbericht`, `chat_sms`) |
| `chat_name` | Chat group or contact name (Chat pages only) |
| `chat_messages` | Array of `{sender_position, sender_label, timestamp, content}` objects (Chat pages only) |

The prompt explicitly instructs the model that `is_new_document` must be `false` when `within_doc_page ≥ 2`, preventing it from contradicting itself on continuation pages.

`_normalise_doc_code()` validates and normalises the returned code: 7-digit barcodes are decoded (`7601430` → `0143`), non-WOO codes (no leading zero) are rejected. Results are streamed to a JSON cache file after the run.

**1c. Stage 2 — Document code assignment**

Same logic as the OCR pipeline:
- Stamps found → `_build_docs_forward_fill()` with forward-fill. `is_new_document` is only trusted as a boundary signal when `within_doc_page` is 1 or absent — if the model reports page 2 or higher, the flag is ignored regardless of `is_new_document`.
- No stamps → `_build_docs_boundary()` uses only the `is_new_document` signal from the model.

**1d. Finalisation**

Text is joined per document, redaction codes are counted and annotated, and category is determined by majority vote across the per-page `category` labels returned by the model.

**Cache and rebuild**

`save_cache()` writes page metadata (text, codes, boundary flags, categories) to a JSON file. `docs_from_cache()` reloads it and re-renders page images from the original PDF, enabling full HTML regeneration without any API calls.

---

### Step 2 — Date extraction and sorting (`text_sorting.py`)

For each document a date is extracted using a strategy that depends on the category:

| Category | Strategy |
|---|---|
| E-mail | `_date_from_email()` — splits the document into individual emails via `email_splitter` and uses the `Verzonden`/`Sent` date of the first email with a parseable date |
| Nota, Brief, Report, Vergadernotulen | `_date_from_datum_field()` — searches the first 1500 characters for a `Datum:` / `Datum |` / `Datum <space>` field; falls back to `_date_from_text_scan()` if not found or not parseable |
| Timeline, Other | `_date_from_text_scan()` — scans the first 1000 characters for any `DD-MM-YYYY` or `DD monthname YYYY` pattern |

`_parse_date()` handles multiple formats: `DD-MM-YYYY`, `DD monthname YYYY`, English long date with weekday (`Wednesday, March 25, 2025`), Dutch weekday + date (`maandag 17 maart 2025`), US format with weekday (`Mon 3/17/2025`), and glued formats (`17maart2025`). Both Dutch and English month names are recognised.

Documents are sorted chronologically. Those without a parseable date are appended at the end, sorted by doc code.

---

### Step 2b — Email splitting (`email_splitter.py`)

Used internally by `text_sorting` for date extraction, and again by `visualisation` for rendering individual email cards.

`split_emails()` splits the concatenated text of an email document into individual emails. It detects boundaries by two complementary signals:

1. A header field type that already appeared in the current email reappears (e.g. a second `Van:` line) — a new email has started.
2. A header field appears after 6 or more consecutive non-header lines — the parser re-entered a header block from body text.

Outlook `"---- Original Message ----"` and `"---- Forwarded Message ----"` separators are hard split points.

Both Dutch (`Van`, `Aan`, `Onderwerp`, `Verzonden`, `CC`) and English (`From`, `To`, `Subject`, `Sent`, `CC`, `BCC`) fields are normalised before comparison, so `Van:` and `From:` are treated as the same field type. Redacted addresses like `< @minbuza.nl>` are normalised to `<[REDACTED]@minbuza.nl>` before splitting.

Each email gets an ID like `0003.2` and carries `subject`, `sender`, `to`, `cc`, `date` (YYYY-MM-DD), and `time` (HH:MM, when present) fields extracted from its header lines.

---

### Step 3 — Event enrichment (`event_enrichment.py`, GPT-4o pipeline only)

`enrich_events()` post-processes the extracted emails and chat messages into structured narrative timeline events using GPT-4o.

Emails are batched (25 per call) and grouped by normalised subject. Chat messages are grouped by day. Each batch is sent to GPT-4o with a structured prompt that returns:

| Field | Description |
|---|---|
| `headline` | One-sentence event summary |
| `ui_summary` | Short label for timeline display |
| `detail_summary` | Longer narrative description |
| `actors` | Named parties involved |
| `tags` | Thematic tags |
| `importance` | 1–5 score |
| `phase` | Timeline phase label |
| `confidence` | Model confidence in the enrichment |
| `redactions` | Redaction codes present |

Results are saved to a JSON file alongside the cache and can be reloaded without additional API calls.

---

### Step 4 — HTML timeline (`visualisation.py`)

`build_html()` produces a single self-contained HTML file with no external dependencies.

**Timeline layout**

Documents are laid out chronologically along a horizontal scrollable axis. Month labels are inserted when the year-month changes. Category filter chips at the top allow toggling visibility by document type.

**Email documents** are exploded into individual cards at build time — one card per email, positioned at the email's own sent date rather than the document's date. Each card shows:
- Van / Aan / CC as ministry pill tags. Ministry domains are resolved to short names (e.g. `Buitenlandse Zaken`). Multiple recipients from the same ministry collapse into one tag with a count badge (`×3`).
- Subject line
- Redaction code summary (counts per WOO ground)
- Collapsible body with gradient fade; click anywhere on the card to expand

**Non-email documents** show the first page as a thumbnail. Clicking the card opens a side panel where all pages of that document can be scrolled through.

**Recipient resolution** — `_classify_domain()` maps known `@ministry.nl` domains to their short ministry names. Unknown external domains appear as `@domain (Extern)`.

**Redaction display** — bare `5.1.x` / `5.2.x` codes in email bodies are rendered as styled `(REDACTED)` spans. Per-email and per-document summaries list total counts per code.

**Boilerplate stripping** — `_strip_email_boilerplate()` removes confidentiality notices and saves-paper banners from email bodies before rendering.

---

## File overview

| File | Role |
|---|---|
| `main.py` | Entry point — OCR pipeline |
| `main_gpt4o.py` | Entry point — GPT-4o pipeline |
| `pipeline_ocr.py` | PDF loading, stamp detection, OCR, document splitting |
| `pipeline_gpt4o.py` | GPT-4o full-page text extraction, boundary detection, caching |
| `event_enrichment.py` | Post-processes emails/chats into structured timeline events via GPT-4o |
| `text_sorting.py` | Date extraction and chronological sorting |
| `email_splitter.py` | Splits email document text into individual emails |
| `visualisation.py` | Interactive HTML timeline generation (standalone CLI) |
| `server.py` | FastAPI server — WOOLens proxy + `/api/analyse` SSE pipeline endpoint |
| `diagnose_stamps.py` | Diagnostic tool — inspect stamp detection per page and region |

---

## Output schema

Both pipelines return `dict[str, dict]` keyed by document code. Each document contains:

| Field | Type | Description |
|---|---|---|
| `doc_code` | `str` | 4-digit stamp code, `unknown_N`, or `auto_NNN` |
| `pages` | `list[Image]` or `list[int]` | PIL page images (GPT-4o) or 1-based page numbers (OCR) |
| `page_nums_in_doc` | `list[int\|None]` | Within-document page stamps |
| `text` | `str` | Full extracted text |
| `annotated_text` | `str` | Text with `[REDACTED: 5.1.2e]` markers |
| `redaction_codes` | `dict[str, int]` | WOO redaction codes and their counts |
| `category` | `str` | One of: E-mail, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, Other |
| `doc_subtype` | `str` | Fine-grained subtype (e.g. `kamerbrief`, `besluit`, `chat_sms`); `other` if not determined |
| `method` | `str` | Extraction method used (`direct`, `ocr`, `gpt4o-stamp`, `gpt4o-boundary`) |

The GPT-4o pipeline additionally sets `chat_name` and `chat_messages` on Chat documents.

---

## Benchmarking

Both pipelines produce the same output schema. Run both on the same PDF and open the HTML files side by side:

```bash
python main.py --pdf dossier.pdf --out timeline_ocr.html
python main_gpt4o.py --pdf dossier.pdf --out timeline_gpt4o.html
```

The GPT-4o cache preserves raw per-page extraction so the HTML can be regenerated without additional API costs:

```bash
python main_gpt4o.py --pdf dossier.pdf --from-cache dossier_gpt4o_cache.json --out timeline_gpt4o.html
```
