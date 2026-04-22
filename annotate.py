"""
annotate.py — Browser-based annotation tool for WOO document segmentation evaluation.

Usage:
    python annotate.py --pdf dossier.pdf --from-cache cache.json
    python annotate.py --pdf dossier.pdf --ocr
    python annotate.py --pdf dossier.pdf --api-key sk-...
    python annotate.py --pdf dossier.pdf
    python annotate.py --pdf dossier.pdf --from-cache cache.json --annotations existing.json
    python annotate.py --pdf dossier.pdf --port 5050
"""

import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    try:
        import flask  # noqa: F401
    except ImportError:
        missing.append("flask  →  pip install flask")
    try:
        import fitz  # noqa: F401
    except ImportError:
        missing.append("pymupdf  →  pip install pymupdf")
    if missing:
        print("[annotate] ERROR: Missing required packages:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

_check_deps()

from flask import Flask, jsonify, request, Response  # noqa: E402
import fitz  # noqa: E402

# Optional fuzzy matching for email-level metrics
try:
    from rapidfuzz import fuzz as _rfuzz
    def _fuzzy_match(a: str, b: str, threshold: int = 70) -> bool:
        return _rfuzz.partial_ratio(a.lower(), b.lower()) >= threshold
except ImportError:
    def _fuzzy_match(a: str, b: str, threshold: int = 70) -> bool:  # type: ignore[misc]
        return a.strip().lower() == b.strip().lower()

# ── Global state ──────────────────────────────────────────────────────────────

G: dict = {
    "pdf_path": None,
    "fitz_doc": None,
    "page_count": 0,
    "pdf_name": "",
    "pipeline_predictions": [],
    "pipeline_mode": "none",
    "predictions_b": [],
    "pipeline_mode_b": "none",
    "annotations_path": None,
    # persisted:
    "boundaries": [0],
    "annotations": {},
}

app = Flask(__name__)

# ── HTML frontend ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WOO Annotate</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0f1117;
    --pane-bg:   #0d1117;
    --left-bg:   #0a0d16;
    --border:    #1e2740;
    --text:      #d1d5db;
    --text-dim:  #6b7280;
    --blue:      #3b82f6;
    --yellow:    #facc15;
    --green:     #34d399;
    --red:       #f87171;
    --purple:    #a78bfa;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; font-size: 13px; }

  #app { display: flex; height: 100vh; overflow: hidden; }

  /* ── Panes ── */
  #pane-pdf  { flex: 0 0 60%; display: flex; flex-direction: column; background: var(--left-bg); border-right: 1px solid var(--border); overflow: hidden; }
  #pane-docs { flex: 0 0 20%; display: flex; flex-direction: column; background: var(--pane-bg); border-right: 1px solid var(--border); overflow: hidden; }
  #pane-form { flex: 0 0 20%; display: flex; flex-direction: column; background: var(--pane-bg); overflow: hidden; }

  /* ── Toolbar ── */
  #toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--pane-bg); border-bottom: 1px solid var(--border); flex-shrink: 0; }
  #toolbar h1 { font-size: 14px; font-weight: 600; color: var(--blue); margin-right: auto; }
  #toolbar .info { color: var(--text-dim); font-size: 11px; }
  #toolbar .badge-mode { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: #1e2740; color: var(--blue); font-family: monospace; }
  #save-flash { font-size: 11px; color: var(--green); opacity: 0; transition: opacity 0.3s; }
  #save-flash.visible { opacity: 1; }
  .btn { padding: 4px 10px; border-radius: 5px; border: 1px solid var(--border); background: #1e2740; color: var(--text); cursor: pointer; font-size: 12px; }
  .btn:hover { background: #2d3b5a; }

  /* ── PDF viewer ── */
  #pdf-scroll { flex: 1; overflow-y: auto; padding: 8px 10px; }

  .pg-wrap { position: relative; margin-bottom: 4px; }

  .bnd-zone {
    height: 14px; position: relative; cursor: pointer;
    display: flex; align-items: center;
  }
  .bnd-zone:hover { background: rgba(250, 204, 21, 0.08); }
  .bnd-zone[data-page="0"] { cursor: default; pointer-events: none; }

  .bnd-gt-line  { position: absolute; left: 0; right: 0; top: 50%; height: 3px; background: var(--yellow); transform: translateY(-50%); }
  .bnd-pred-line { position: absolute; left: 0; right: 0; top: 50%; height: 2px; border-top: 2px dashed var(--blue); transform: translateY(-50%); }

  .bnd-gt-badge {
    position: absolute; left: 6px; top: 50%; transform: translateY(-50%);
    background: var(--yellow); color: #000; font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px; white-space: nowrap; z-index: 2;
  }
  .bnd-pred-badge {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
    background: var(--blue); color: #fff; font-size: 10px; font-weight: 600;
    padding: 1px 5px; border-radius: 3px; white-space: nowrap; z-index: 2; font-family: monospace;
  }

  .pg-img-wrap {
    position: relative; background: #111827; cursor: pointer;
    border: 2px solid transparent; transition: border-color 0.15s;
    min-height: 100px;
  }
  .pg-img-wrap.selected   { border-color: var(--yellow); }
  .pg-img-wrap.in-selected { border-color: rgba(250,204,21,0.25); background: rgba(250,204,21,0.04); }

  .pg-img-wrap img { display: block; width: 100%; height: auto; }
  .pg-loading { padding: 24px; text-align: center; color: var(--text-dim); font-size: 12px; }
  .pg-num { position: absolute; bottom: 4px; right: 6px; background: rgba(0,0,0,0.65); color: #fff; font-size: 10px; padding: 1px 5px; border-radius: 3px; }

  /* ── Doc list ── */
  #pane-docs-header { flex-shrink: 0; border-bottom: 1px solid var(--border); display: flex; }
  .doc-tab { flex: 1; padding: 8px 6px; font-size: 11px; font-weight: 600; text-align: center; cursor: pointer; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid transparent; transition: color 0.15s, border-color 0.15s; }
  .doc-tab:hover { color: var(--text); }
  .doc-tab.active { color: var(--blue); border-bottom-color: var(--blue); }
  #docs-scroll { flex: 1; overflow-y: auto; }

  .doc-row {
    padding: 7px 10px; border-bottom: 1px solid var(--border);
    cursor: pointer; display: flex; flex-direction: column; gap: 2px;
  }
  .doc-row:hover { background: rgba(59,130,246,0.08); }
  .doc-row.active { background: rgba(59,130,246,0.15); }

  .doc-row-top { display: flex; align-items: center; gap: 5px; }
  .doc-idx { font-family: monospace; font-size: 10px; color: var(--text-dim); background: #1e2740; padding: 1px 4px; border-radius: 3px; }
  .doc-pages { font-size: 11px; color: var(--text-dim); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .status-green  { background: var(--green); }
  .status-yellow { background: var(--yellow); }
  .status-red    { background: var(--red); }
  .status-grey   { background: #4b5563; }

  .doc-type-label { font-size: 11px; color: var(--blue); }
  .doc-date-label { font-size: 10px; color: var(--text-dim); }

  /* ── Right pane ── */
  #form-scroll { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 12px; }

  .panel-block { background: #0f1117; border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .panel-title { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-dim); margin-bottom: 8px; }

  .pred-code    { font-family: monospace; color: var(--blue); font-size: 14px; font-weight: 700; }
  .pred-type    { font-size: 12px; margin-top: 2px; }
  .pred-date    { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .pred-method  { display: inline-block; font-size: 9px; padding: 2px 5px; border-radius: 3px; background: #1e2740; color: var(--text-dim); margin-top: 4px; font-family: monospace; }
  .pred-range   { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
  .pred-none    { color: var(--text-dim); font-size: 12px; font-style: italic; }

  .form-row { display: flex; flex-direction: column; gap: 3px; }
  .form-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .form-val   { font-size: 12px; color: var(--text-dim); padding: 4px 0; }

  select, input[type="text"], textarea {
    width: 100%; background: #0a0d16; border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; padding: 5px 7px; font-size: 12px;
    font-family: inherit;
  }
  select:focus, input[type="text"]:focus, textarea:focus {
    outline: none; border-color: var(--blue);
  }
  textarea { resize: vertical; min-height: 60px; }

  /* ── Metrics panel ── */
  #metrics-panel { padding: 10px; border-top: 1px solid var(--border); flex-shrink: 0; background: var(--pane-bg); }
  .metrics-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-dim); margin-bottom: 6px; font-weight: 600; }
  .metric-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; font-size: 11px; }
  .metric-key { color: var(--text-dim); }
  .metric-val { font-family: monospace; color: var(--text); }
  .metric-f1  { font-family: monospace; color: var(--green); font-size: 13px; font-weight: 700; }
  .metric-sep { border-top: 1px solid var(--border); margin: 5px 0; }
  /* Two-pipeline comparison metrics */
  .metric-row-2 { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; font-size: 11px; gap: 4px; }
  .metric-key-2 { color: var(--text-dim); flex: 1; }
  .metric-col   { font-family: monospace; color: var(--text); min-width: 52px; text-align: right; }
  .metric-col.better { color: var(--green); font-weight: 600; }
  .metric-col.worse  { color: var(--red); }
  .metric-2hdr  { display: flex; justify-content: flex-end; gap: 4px; margin-bottom: 4px; }
  .metric-2hdr span { font-size: 9px; min-width: 52px; text-align: right; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); font-weight: 600; }

  /* ── Email detail rows ── */
  .email-entry { background: #131925; border: 1px solid var(--border); border-radius: 5px; padding: 6px; margin-bottom: 4px; }
  .email-entry-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .email-entry-idx { font-size: 10px; font-weight: 600; color: var(--purple); }
  .email-entry-lbl { font-size: 9px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 3px; margin-bottom: 1px; }
  .btn-del-email { padding: 1px 5px; font-size: 11px; background: transparent; border: 1px solid var(--red); color: var(--red); border-radius: 3px; cursor: pointer; }
  .btn-del-email:hover { background: rgba(248,113,113,0.12); }
  #row-email-details .pred-emails { margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border); }
  .pred-email-row { display: flex; align-items: baseline; gap: 5px; font-size: 10px; color: var(--text-dim); padding: 3px 0; border-bottom: 1px dashed #1e2740; }
  .pred-email-row:last-child { border-bottom: none; }
  .pred-email-subj { color: var(--text); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .btn-copy-email { flex-shrink: 0; background: transparent; border: 1px solid var(--blue); color: var(--blue); border-radius: 3px; padding: 0 5px; font-size: 10px; cursor: pointer; line-height: 1.6; }
  .btn-copy-email:hover { background: rgba(59,130,246,0.15); }
  .pred-emails-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }

  /* ── All pipeline emails panel ── */
  #all-emails-list { max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
  .all-email-row { display: flex; align-items: flex-start; gap: 6px; padding: 5px 4px; border-bottom: 1px dashed #1e2740; }
  .all-email-row:last-child { border-bottom: none; }
  .all-email-info { flex: 1; min-width: 0; }
  .all-email-subj { font-size: 11px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .all-email-meta { font-size: 10px; color: var(--text-dim); margin-top: 1px; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3b5a; border-radius: 3px; }
</style>
</head>
<body>
<div id="app">

  <!-- Left pane -->
  <div id="pane-pdf">
    <div id="toolbar">
      <h1>WOO Annotate</h1>
      <span class="info" id="tb-info"></span>
      <span class="badge-mode" id="tb-mode"></span>
      <span id="save-flash">Saved ✓</span>
      <button class="btn" id="btn-export">Export JSON</button>
    </div>
    <div id="pdf-scroll"></div>
  </div>

  <!-- Middle pane -->
  <div id="pane-docs">
    <div id="pane-docs-header">
      <div class="doc-tab active" id="tab-gt"         onclick="switchTab('gt')">GT</div>
      <div class="doc-tab"        id="tab-pipeline"  onclick="switchTab('pipeline')">Pipe A</div>
      <div class="doc-tab"        id="tab-pipeline-b" onclick="switchTab('pipeline-b')" style="display:none">Pipe B</div>
    </div>
    <div id="docs-scroll"></div>
  </div>

  <!-- Right pane -->
  <div id="pane-form">
    <div id="form-scroll">
      <div class="panel-block" id="pred-block">
        <div class="panel-title">Pipeline prediction</div>
        <div id="pred-content"><span class="pred-none">No document selected</span></div>
      </div>
      <div class="panel-block">
        <div class="panel-title">Ground truth</div>
        <div style="display:flex;flex-direction:column;gap:8px;" id="gt-form">
          <div class="form-row">
            <div class="form-label">Page range</div>
            <div class="form-val" id="gt-range">—</div>
          </div>
          <div class="form-row">
            <div class="form-label">Type</div>
            <select id="gt-type">
              <option value="">— select —</option>
              <option>E-mail</option>
              <option>Chat</option>
              <option>Nota</option>
              <option>Brief</option>
              <option>Report</option>
              <option>Timeline</option>
              <option>Vergadernotulen</option>
              <option>Other</option>
              <option>Onbekend</option>
            </select>
          </div>
          <div class="form-row" id="row-num-emails" style="display:none">
            <div class="form-label"># emails</div>
            <span id="gt-num-emails-display" style="font-size:12px;color:var(--text-dim);padding:2px 0;">0</span>
          </div>
          <div class="form-row" id="row-other-specify" style="display:none">
            <div class="form-label">Specify</div>
            <input type="text" id="gt-other-specify" placeholder="e.g. Bijlage, Factuur…">
          </div>
          <div class="form-row" id="row-email-details" style="display:none">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <div class="form-label">Email details</div>
              <button class="btn" style="font-size:11px;padding:2px 8px;" onclick="addEmailEntry()">+ Add email</button>
            </div>
            <div id="email-rows-container"></div>
          </div>
          <div class="form-row">
            <div class="form-label">Date</div>
            <input type="text" id="gt-date" placeholder="17 maart 2021">
          </div>
          <div class="form-row">
            <div class="form-label">Notes</div>
            <textarea id="gt-notes" placeholder="date only in header scan / type ambiguous / heavy redaction"></textarea>
          </div>
        </div>
      </div>
      <div class="panel-block" id="all-emails-block">
        <div class="panel-title">All pipeline emails</div>
        <div id="all-emails-list"><span class="pred-none">Loading…</span></div>
      </div>
    </div>
    <div id="metrics-panel">
      <div class="metrics-title">Metrics</div>
      <div id="metrics-content"><span style="color:var(--text-dim);font-size:11px;">Loading…</span></div>
    </div>
  </div>
</div>

<script>
'use strict';

const DOC_TYPES = ['E-mail','Chat','Nota','Brief','Report','Timeline','Vergadernotulen','Other','Onbekend'];

const S = {
  pageCount: 0,
  pdfName: '',
  pipelineMode: 'none',
  pipelineModeB: 'none',
  predictions: [],
  predictions_b: [],
  boundaries: [0],
  annotations: {},
  selectedStart: 0,
  activeTab: 'gt',
  currentPage: 0,
  metrics: {},
  metrics_b: null,
};

// ── Save debounce ─────────────────────────────────────────────────────────────
let saveTimer = null;
function scheduleSave(immediate) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(doSave, immediate ? 0 : 400);
}

async function doSave() {
  saveTimer = null;
  const body = { boundaries: S.boundaries, annotations: S.annotations };
  try {
    const r = await fetch('/api/state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.metrics) {
      S.metrics   = data.metrics;
      S.metrics_b = data.metrics_b || null;
      renderMetrics();
      renderDocList();
    }
    flashSaved();
  } catch(e) {
    console.error('Save failed', e);
  }
}

function flashSaved() {
  const el = document.getElementById('save-flash');
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 1500);
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiGet(url) {
  const r = await fetch(url);
  return r.json();
}

// ── GT doc list computation ───────────────────────────────────────────────────
function getGtDocs() {
  const bounds = [...S.boundaries].sort((a,b) => a-b);
  return bounds.map((start, i) => {
    const end = (i + 1 < bounds.length) ? bounds[i+1] - 1 : S.pageCount - 1;
    const ann  = S.annotations[String(start)] || {};
    return { start, end, type_gt: ann.type_gt || '', date_gt: ann.date_gt || '', notes: ann.notes || '', num_emails_gt: ann.num_emails_gt ?? null, emails_gt: ann.emails_gt || [] };
  });
}

function predictionForPage(page, preds) {
  // Find the pipeline prediction whose page range contains this page
  preds = preds || S.predictions;
  for (const pred of preds) {
    if (pred.start_page <= page && page <= pred.end_page) return pred;
  }
  return null;
}

function predictionForDoc(doc) {
  return predictionForPage(doc.start);
}

function docStatusColor(doc) {
  const ann  = S.annotations[String(doc.start)] || {};
  const pred = predictionForDoc(doc);
  const hasType = !!ann.type_gt;
  const hasDate = !!ann.date_gt;
  if (!hasType && !hasDate) return 'grey';
  if (!pred) return 'grey';
  const typeMatch = hasType && ann.type_gt === pred.type;
  const dateMatch = hasDate && ann.date_gt === pred.date_raw;
  if (typeMatch && dateMatch) return 'green';
  if (typeMatch || dateMatch) return 'yellow';
  return 'red';
}

// ── Render: PDF viewer ────────────────────────────────────────────────────────
const loadedPages = new Set();
const imgEls      = {};

function buildPdfViewer() {
  const container = document.getElementById('pdf-scroll');
  container.innerHTML = '';

  const predPageSet = new Set(S.predictions.map(p => p.start_page));
  const predByPage  = {};
  for (const p of S.predictions) predByPage[p.start_page] = p;

  for (let n = 0; n < S.pageCount; n++) {
    const wrap = document.createElement('div');
    wrap.className = 'pg-wrap';
    wrap.dataset.page = n;

    // Boundary zone
    const bz = document.createElement('div');
    bz.className = 'bnd-zone';
    bz.dataset.page = n;
    if (n > 0) {
      bz.title = `Click to toggle GT boundary at page ${n+1}`;
      bz.addEventListener('click', () => toggleBoundary(n));
    }
    wrap.appendChild(bz);

    // Image wrap
    const iw = document.createElement('div');
    iw.className = 'pg-img-wrap';
    iw.dataset.page = n;
    if (n > 0) iw.addEventListener('click', () => toggleBoundary(n));

    const loader = document.createElement('div');
    loader.className = 'pg-loading';
    loader.textContent = `Page ${n+1}`;
    iw.appendChild(loader);

    const img = document.createElement('img');
    img.dataset.src = `/api/page/${n}`;
    img.style.display = 'none';
    img.alt = `Page ${n+1}`;
    img.addEventListener('load', () => {
      loader.remove();
      img.style.display = '';
    });
    iw.appendChild(img);
    imgEls[n] = img;

    const pgNum = document.createElement('div');
    pgNum.className = 'pg-num';
    pgNum.textContent = n + 1;
    iw.appendChild(pgNum);

    wrap.appendChild(iw);
    container.appendChild(wrap);
  }

  setupIntersectionObserver();
  updateBoundaryMarkers();
  updatePageHighlights();
}

function setupIntersectionObserver() {
  const pdfScroll = document.getElementById('pdf-scroll');

  // Visibility observer for lazy loading
  const lazyObs = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const iw = entry.target;
      const n  = parseInt(iw.dataset.page, 10);
      if (!loadedPages.has(n)) {
        loadedPages.add(n);
        const img = imgEls[n];
        if (img && img.dataset.src) {
          fetch(img.dataset.src)
            .then(r => r.json())
            .then(d => { img.src = d.image; })
            .catch(() => {});
        }
      }
    }
  }, { root: pdfScroll, rootMargin: '300px' });

  // Current page observer
  const pageObs = new IntersectionObserver((entries) => {
    let best = null, bestRatio = -1;
    for (const entry of entries) {
      if (entry.intersectionRatio > bestRatio) {
        best      = entry.target;
        bestRatio = entry.intersectionRatio;
      }
    }
    if (best) {
      S.currentPage = parseInt(best.dataset.page, 10);
      renderPredPanel(S.currentPage);
    }
  }, { root: pdfScroll, threshold: [0.1, 0.5, 1.0] });

  document.querySelectorAll('.pg-img-wrap').forEach(el => {
    lazyObs.observe(el);
    pageObs.observe(el);
  });
}

function updateBoundaryMarkers() {
  const boundSet   = new Set(S.boundaries);
  const predByPage = {};
  for (const p of S.predictions) {
    if (p.start_page > 0) predByPage[p.start_page] = p;
  }

  document.querySelectorAll('.bnd-zone').forEach(bz => {
    const n  = parseInt(bz.dataset.page, 10);
    bz.innerHTML = '';
    if (n === 0) return;

    const isGt   = boundSet.has(n);
    const isPred = !!predByPage[n];

    if (isGt) {
      const line  = document.createElement('div');
      line.className = 'bnd-gt-line';
      bz.appendChild(line);

      const badge = document.createElement('div');
      badge.className = 'bnd-gt-badge';
      const gtDocs = getGtDocs();
      const docIdx = gtDocs.findIndex(d => d.start === n);
      badge.textContent = `Doc ${docIdx + 1} ▶ p${n + 1}`;
      bz.appendChild(badge);
    }

    if (isPred) {
      const pred  = predByPage[n];
      const line  = document.createElement('div');
      line.className = 'bnd-pred-line';
      bz.appendChild(line);

      const badge = document.createElement('div');
      badge.className = 'bnd-pred-badge';
      badge.textContent = pred.doc_code;
      bz.appendChild(badge);
    }
  });
}

function updatePageHighlights() {
  const gtDocs  = getGtDocs();
  const selDoc  = gtDocs.find(d => d.start === S.selectedStart);
  const selStart = selDoc ? selDoc.start : -1;
  const selEnd   = selDoc ? selDoc.end   : -1;

  document.querySelectorAll('.pg-img-wrap').forEach(iw => {
    const n = parseInt(iw.dataset.page, 10);
    iw.classList.remove('selected', 'in-selected');
    if (n === selStart) iw.classList.add('selected');
    else if (n > selStart && n <= selEnd) iw.classList.add('in-selected');
  });
}

// ── Render: doc list ──────────────────────────────────────────────────────────
function switchTab(tab) {
  S.activeTab = tab;
  document.getElementById('tab-gt').classList.toggle('active', tab === 'gt');
  document.getElementById('tab-pipeline').classList.toggle('active', tab === 'pipeline');
  document.getElementById('tab-pipeline-b').classList.toggle('active', tab === 'pipeline-b');
  renderDocList();
}

function renderDocList() {
  if (S.activeTab === 'pipeline')   { renderPipelineList(S.predictions);   return; }
  if (S.activeTab === 'pipeline-b') { renderPipelineList(S.predictions_b); return; }

  const scroll  = document.getElementById('docs-scroll');
  const gtDocs  = getGtDocs();
  scroll.innerHTML = '';

  gtDocs.forEach((doc, i) => {
    const row = document.createElement('div');
    row.className = 'doc-row' + (doc.start === S.selectedStart ? ' active' : '');
    row.dataset.start = doc.start;

    const color = docStatusColor(doc);

    const top = document.createElement('div');
    top.className = 'doc-row-top';

    const idx = document.createElement('span');
    idx.className = 'doc-idx';
    idx.textContent = `#${i + 1}`;
    top.appendChild(idx);

    const pages = document.createElement('span');
    pages.className = 'doc-pages';
    pages.textContent = `p${doc.start + 1}–${doc.end + 1}`;
    top.appendChild(pages);

    const dot = document.createElement('span');
    dot.className = `status-dot status-${color}`;
    top.appendChild(dot);

    row.appendChild(top);

    if (doc.type_gt) {
      const tl = document.createElement('div');
      tl.className = 'doc-type-label';
      tl.textContent = doc.type_gt;
      row.appendChild(tl);
    }
    if (doc.date_gt) {
      const dl = document.createElement('div');
      dl.className = 'doc-date-label';
      dl.textContent = doc.date_gt;
      row.appendChild(dl);
    }

    row.addEventListener('click', () => selectDoc(doc.start));
    scroll.appendChild(row);
  });
}

function renderPipelineList(preds) {
  const scroll = document.getElementById('docs-scroll');
  scroll.innerHTML = '';

  if (!preds || preds.length === 0) {
    scroll.innerHTML = '<div style="padding:12px;color:var(--text-dim);font-size:12px;">No pipeline predictions loaded.</div>';
    return;
  }

  preds.forEach((pred, i) => {
    const row = document.createElement('div');
    row.className = 'doc-row';

    const top = document.createElement('div');
    top.className = 'doc-row-top';

    const idx = document.createElement('span');
    idx.className = 'doc-idx';
    idx.textContent = `#${i + 1}`;
    top.appendChild(idx);

    const code = document.createElement('span');
    code.style.cssText = 'font-family:monospace;font-size:10px;color:var(--blue)';
    code.textContent = pred.doc_code || '—';
    top.appendChild(code);

    const pages = document.createElement('span');
    pages.className = 'doc-pages';
    pages.textContent = `p${pred.start_page + 1}–${pred.end_page + 1}`;
    top.appendChild(pages);

    const methodBadge = document.createElement('span');
    methodBadge.style.cssText = 'font-size:9px;padding:1px 4px;border-radius:3px;background:#1e2740;color:var(--text-dim);font-family:monospace;';
    methodBadge.textContent = pred.method === 'gpt4o-stamp' ? 'stamp' : 'boundary';
    top.appendChild(methodBadge);

    row.appendChild(top);

    if (pred.type) {
      const tl = document.createElement('div');
      tl.className = 'doc-type-label';
      tl.textContent = pred.type + (pred.num_emails_pipeline != null ? ` · ${pred.num_emails_pipeline} emails` : '');
      row.appendChild(tl);
    }
    if (pred.date_raw) {
      const dl = document.createElement('div');
      dl.className = 'doc-date-label';
      dl.textContent = pred.date_raw;
      row.appendChild(dl);
    }

    row.addEventListener('click', () => {
      scrollToPage(pred.start_page);
    });

    scroll.appendChild(row);
  });
}

// ── Render: pipeline prediction panel (updates on scroll) ────────────────────
function renderPredPanel(page) {
  const pred      = predictionForPage(page);
  const predBlock = document.getElementById('pred-content');
  if (!predBlock) return;

  if (pred) {
    let predHtml = `
      <div class="pred-code">${esc(pred.doc_code)}</div>
      <div class="pred-type">${esc(pred.type || '—')}</div>
      <div class="pred-date">${esc(pred.date_raw || '—')}</div>
      <span class="pred-method">${esc(pred.method || '')}</span>
      <div class="pred-range">p${pred.start_page + 1}–${pred.end_page + 1}</div>
    `;
    if (pred.emails_pipeline && pred.emails_pipeline.length > 0) {
      predHtml += `<div class="pred-emails">` +
        `<div class="pred-emails-header">` +
          `<span class="panel-title">Pipeline emails (${pred.emails_pipeline.length})</span>` +
          `<button class="btn" style="font-size:10px;padding:2px 7px;" onclick="copyAllPipelineEmails()">Copy all → GT</button>` +
        `</div>`;
      for (let _i = 0; _i < pred.emails_pipeline.length; _i++) {
        const em = pred.emails_pipeline[_i];
        predHtml += `<div class="pred-email-row">` +
          `<button class="btn-copy-email" onclick="copyOnePipelineEmail(${_i})" title="Copy to GT">↓</button>` +
          `<span class="pred-email-subj">${esc(em.subject || '—')}</span>`;
        if (em.date) predHtml += `<span>${esc(em.date)}</span>`;
        predHtml += `</div>`;
      }
      predHtml += `</div>`;
    }
    predBlock.innerHTML = predHtml;
  } else {
    predBlock.innerHTML = '<span class="pred-none">No prediction at this position</span>';
  }
}

// ── Render: right pane ────────────────────────────────────────────────────────
function renderForm() {
  const gtDocs = getGtDocs();
  const doc    = gtDocs.find(d => d.start === S.selectedStart);

  if (!doc) {
    renderPredPanel(S.selectedStart);
    document.getElementById('gt-range').textContent = '—';
    document.getElementById('gt-type').value  = '';
    document.getElementById('gt-num-emails').value = '';
    document.getElementById('gt-date').value  = '';
    document.getElementById('gt-notes').value = '';
    document.getElementById('row-num-emails').style.display = 'none';
    document.getElementById('row-email-details').style.display = 'none';
    document.getElementById('email-rows-container').innerHTML = '';
    return;
  }

  // Pipeline prediction panel: use selectedStart (set synchronously before renderForm runs)
  renderPredPanel(S.selectedStart);

  const pred = predictionForPage(S.selectedStart);

  // GT form
  const type    = doc.type_gt || '';
  const isEmail = type === 'E-mail';
  const isOther = type === 'Other';
  document.getElementById('gt-range').textContent          = `p${doc.start + 1}–${doc.end + 1}`;
  document.getElementById('gt-type').value                 = type;
  document.getElementById('gt-other-specify').value        = doc.other_type || '';
  document.getElementById('gt-date').value                 = doc.date_gt  || '';
  document.getElementById('gt-notes').value                = doc.notes    || '';
  document.getElementById('row-num-emails').style.display    = isEmail ? '' : 'none';
  document.getElementById('row-email-details').style.display = isEmail ? '' : 'none';
  document.getElementById('row-other-specify').style.display = isOther ? '' : 'none';

  // Rebuild email detail rows
  const container = document.getElementById('email-rows-container');
  container.innerHTML = '';
  for (const em of (doc.emails_gt || [])) {
    _appendEmailEntry(container, em.subject || '', em.sender || '', em.date || '');
  }

  // Update email count display (with pipeline hint)
  const emailCount = (doc.emails_gt || []).length;
  const disp = document.getElementById('gt-num-emails-display');
  if (disp) {
    const hint = (isEmail && pred && pred.num_emails_pipeline != null)
      ? ` — pipeline: ${pred.num_emails_pipeline}` : '';
    disp.textContent = `${emailCount}${hint}`;
  }
}

// ── Render: metrics ───────────────────────────────────────────────────────────
function _fmtN(v, digits) { digits = digits ?? 3; return typeof v === 'number' ? v.toFixed(digits) : '—'; }

function _row2(label, va, vb, higher_is_better) {
  higher_is_better = higher_is_better !== false;
  const na = typeof va === 'number', nb = typeof vb === 'number';
  let classA = '', classB = '';
  if (na && nb && va !== vb) {
    const aWins = higher_is_better ? va > vb : va < vb;
    classA = aWins ? 'better' : 'worse';
    classB = aWins ? 'worse'  : 'better';
  }
  return `<div class="metric-row-2"><span class="metric-key-2">${label}</span>` +
         `<span class="metric-col ${classA}">${_fmtN(va)}</span>` +
         `<span class="metric-col ${classB}">${_fmtN(vb)}</span></div>`;
}

function renderMetrics() {
  const m  = S.metrics;
  const mb = S.metrics_b;
  const el = document.getElementById('metrics-content');
  if (!m || Object.keys(m).length === 0) {
    el.innerHTML = '<span style="color:var(--text-dim);font-size:11px;">—</span>';
    return;
  }

  const dual = mb && Object.keys(mb).length > 0;

  if (!dual) {
    // Single-pipeline layout
    el.innerHTML = `
      <div class="metric-row"><span class="metric-key">Boundary F1</span><span class="metric-f1">${_fmtN(m.boundary_f1)}</span></div>
      <div class="metric-row"><span class="metric-key">P / R</span><span class="metric-val">${_fmtN(m.boundary_precision)} / ${_fmtN(m.boundary_recall)}</span></div>
      <div class="metric-row"><span class="metric-key">TP / FP / FN</span><span class="metric-val">${m.boundary_tp ?? '—'} / ${m.boundary_fp ?? '—'} / ${m.boundary_fn ?? '—'}</span></div>
      <div class="metric-sep"></div>
      <div class="metric-row"><span class="metric-key">Type acc.</span><span class="metric-val">${_fmtN(m.type_accuracy)} (${m.type_correct ?? 0}/${m.type_annotated ?? 0})</span></div>
      <div class="metric-row"><span class="metric-key">Email count acc.</span><span class="metric-val">${m.email_count_accuracy != null ? _fmtN(m.email_count_accuracy) : '—'} (${m.email_count_correct ?? 0}/${m.email_count_annotated ?? 0})</span></div>
      ${m.email_subject_f1 != null ? `
      <div class="metric-row"><span class="metric-key">Email subj. F1</span><span class="metric-val" style="color:var(--green)">${_fmtN(m.email_subject_f1)}</span></div>
      <div class="metric-row"><span class="metric-key">Email date acc.</span><span class="metric-val">${m.email_date_accuracy != null ? _fmtN(m.email_date_accuracy) : '—'} (${m.email_date_correct ?? 0}/${m.email_date_annotated ?? 0})</span></div>
      <div class="metric-row"><span class="metric-key">Email sender acc.</span><span class="metric-val">${m.email_sender_accuracy != null ? _fmtN(m.email_sender_accuracy) : '—'} (${m.email_sender_correct ?? 0}/${m.email_sender_annotated ?? 0})</span></div>` : ''}
      <div class="metric-row"><span class="metric-key">Date rate</span><span class="metric-val">${_fmtN(m.date_extraction_rate)} (${m.date_count ?? 0})</span></div>
      <div class="metric-sep"></div>
      <div class="metric-row"><span class="metric-key">GT docs</span><span class="metric-val">${m.gt_doc_count ?? '—'}</span></div>
      <div class="metric-row"><span class="metric-key">Pipeline docs</span><span class="metric-val">${m.pipeline_doc_count ?? '—'}</span></div>
    `;
    return;
  }

  // Dual-pipeline side-by-side comparison
  const modeA = esc(S.pipelineMode  || 'A');
  const modeB = esc(S.pipelineModeB || 'B');
  let html =
    `<div class="metric-2hdr"><span>${modeA}</span><span>${modeB}</span></div>` +
    _row2('Bound. F1',   m.boundary_f1,          mb.boundary_f1) +
    _row2('Precision',   m.boundary_precision,    mb.boundary_precision) +
    _row2('Recall',      m.boundary_recall,       mb.boundary_recall) +
    `<div class="metric-sep"></div>` +
    _row2('Type acc.',   m.type_accuracy,         mb.type_accuracy) +
    _row2('Email cnt.',  m.email_count_accuracy,  mb.email_count_accuracy);

  if (m.email_subject_f1 != null || (mb && mb.email_subject_f1 != null)) {
    html +=
      _row2('Email subj.F1', m.email_subject_f1,    mb.email_subject_f1) +
      _row2('Email date',    m.email_date_accuracy,  mb.email_date_accuracy) +
      _row2('Email sender',  m.email_sender_accuracy, mb.email_sender_accuracy);
  }
  html +=
    `<div class="metric-sep"></div>` +
    _row2('Pipe docs', m.pipeline_doc_count, mb.pipeline_doc_count, false) +
    `<div class="metric-row"><span class="metric-key">GT docs</span><span class="metric-val">${m.gt_doc_count ?? '—'}</span></div>`;
  el.innerHTML = html;
}

// ── Actions ───────────────────────────────────────────────────────────────────
function toggleBoundary(n) {
  if (n === 0) return;
  const idx = S.boundaries.indexOf(n);
  if (idx >= 0) {
    S.boundaries.splice(idx, 1);
    // If selected doc started here, move selection to previous
    if (S.selectedStart === n) {
      const gtDocs = getGtDocs();
      const prev   = gtDocs[Math.max(0, gtDocs.findIndex(d => d.start >= n) - 1)];
      S.selectedStart = prev ? prev.start : 0;
    }
  } else {
    S.boundaries.push(n);
    S.boundaries.sort((a,b) => a - b);
    S.selectedStart = n;
  }

  updateBoundaryMarkers();
  updatePageHighlights();
  renderDocList();
  renderForm();
  scheduleSave(false);
}

function selectDoc(start) {
  saveFormToState();
  S.selectedStart = start;
  updatePageHighlights();
  renderDocList();
  renderForm();
  scrollToPage(start);
}

function scrollToPage(n) {
  const el = document.querySelector(`.pg-wrap[data-page="${n}"]`);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getAnnotation(start) {
  if (!S.annotations[String(start)]) {
    S.annotations[String(start)] = { type_gt: '', date_gt: '', notes: '', other_type: '' };
  }
  return S.annotations[String(start)];
}

function saveFormToState() {
  const ann    = getAnnotation(S.selectedStart);
  ann.type_gt    = document.getElementById('gt-type').value;
  ann.other_type = document.getElementById('gt-other-specify').value.trim();
  ann.date_gt    = document.getElementById('gt-date').value.trim();
  ann.notes      = document.getElementById('gt-notes').value;

  // Collect per-email detail rows
  const emails = [];
  document.querySelectorAll('#email-rows-container .email-entry').forEach(entry => {
    emails.push({
      subject: entry.querySelector('.email-subj')?.value?.trim() || '',
      sender:  entry.querySelector('.email-sndr')?.value?.trim() || '',
      date:    entry.querySelector('.email-edate')?.value?.trim() || '',
    });
  });
  ann.emails_gt     = emails;
  ann.num_emails_gt = emails.length > 0 ? emails.length : null;
  // Update the live count display
  const disp = document.getElementById('gt-num-emails-display');
  if (disp) disp.textContent = emails.length;
}

function nextUnannotated() {
  const gtDocs = getGtDocs();
  const curIdx = gtDocs.findIndex(d => d.start === S.selectedStart);
  const start  = (curIdx < 0) ? 0 : curIdx + 1;
  for (let i = 0; i < gtDocs.length; i++) {
    const j   = (start + i) % gtDocs.length;
    const doc = gtDocs[j];
    const ann = S.annotations[String(doc.start)] || {};
    if (!ann.type_gt || !ann.date_gt) {
      selectDoc(doc.start);
      return;
    }
  }
}

// ── Form event bindings ───────────────────────────────────────────────────────
function bindFormEvents() {
  document.getElementById('gt-type').addEventListener('change', () => {
    const type    = document.getElementById('gt-type').value;
    const isEmail = type === 'E-mail';
    const isOther = type === 'Other';
    document.getElementById('row-num-emails').style.display    = isEmail ? '' : 'none';
    document.getElementById('row-email-details').style.display = isEmail ? '' : 'none';
    document.getElementById('row-other-specify').style.display = isOther ? '' : 'none';
    if (isOther) document.getElementById('gt-other-specify').focus();
    saveFormToState();
    renderDocList();
    scheduleSave(true);
  });

  ['gt-date', 'gt-notes', 'gt-other-specify'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
      saveFormToState();
      renderDocList();
      scheduleSave(false);
    });
  });
}

// ── Email detail row helpers ──────────────────────────────────────────────────
function _appendEmailEntry(container, subject, sender, date_val) {
  const idx = container.children.length + 1;
  const entry = document.createElement('div');
  entry.className = 'email-entry';
  entry.innerHTML =
    `<div class="email-entry-header">` +
      `<span class="email-entry-idx">Email ${idx}</span>` +
      `<button class="btn-del-email" onclick="removeEmailEntry(this)">×</button>` +
    `</div>` +
    `<div class="email-entry-lbl">Subject</div>` +
    `<input type="text" class="email-subj" placeholder="Onderwerp..." value="${esc(subject)}">` +
    `<div class="email-entry-lbl">Sender</div>` +
    `<input type="text" class="email-sndr" placeholder="Van..." value="${esc(sender)}">` +
    `<div class="email-entry-lbl">Date</div>` +
    `<input type="text" class="email-edate" placeholder="2024-03-15" value="${esc(date_val)}">`;
  container.appendChild(entry);
  entry.querySelectorAll('input').forEach(inp => {
    inp.addEventListener('input', () => { saveFormToState(); scheduleSave(false); });
  });
}

function addEmailEntry() {
  const container = document.getElementById('email-rows-container');
  _appendEmailEntry(container, '', '', '');
  saveFormToState();
  scheduleSave(true);
}

function removeEmailEntry(btn) {
  btn.closest('.email-entry').remove();
  // Re-number remaining entries
  const container = document.getElementById('email-rows-container');
  Array.from(container.children).forEach((entry, i) => {
    const lbl = entry.querySelector('.email-entry-idx');
    if (lbl) lbl.textContent = `Email ${i + 1}`;
  });
  saveFormToState();
  scheduleSave(true);
}

// ── Pipeline email copy helpers ───────────────────────────────────────────────
function _currentPred() {
  const gtDocs = getGtDocs();
  const doc    = gtDocs.find(d => d.start === S.selectedStart);
  return doc ? predictionForDoc(doc) : null;
}

function copyAllPipelineEmails() {
  const pred = _currentPred();
  if (!pred || !pred.emails_pipeline || !pred.emails_pipeline.length) return;
  const container = document.getElementById('email-rows-container');
  container.innerHTML = '';
  for (const em of pred.emails_pipeline) {
    _appendEmailEntry(container, em.subject || '', em.sender || '', em.date || '');
  }
  // Ensure email section is visible
  document.getElementById('row-email-details').style.display = '';
  saveFormToState();
  renderDocList();
  scheduleSave(true);
}

function copyOnePipelineEmail(idx) {
  const pred = _currentPred();
  if (!pred || !pred.emails_pipeline) return;
  const em = pred.emails_pipeline[idx];
  if (!em) return;
  const container = document.getElementById('email-rows-container');
  _appendEmailEntry(container, em.subject || '', em.sender || '', em.date || '');
  document.getElementById('row-email-details').style.display = '';
  saveFormToState();
  scheduleSave(true);
}

// ── All pipeline emails panel ─────────────────────────────────────────────────
function renderAllEmailsPanel() {
  const container = document.getElementById('all-emails-list');
  if (!container) return;

  // Collect all emails from all predictions in page order
  const allEmails = [];
  const sorted = [...S.predictions].sort((a, b) => a.start_page - b.start_page);
  for (const pred of sorted) {
    for (const em of (pred.emails_pipeline || [])) {
      allEmails.push({ subject: em.subject, sender: em.sender, date: em.date, doc_code: pred.doc_code, start_page: pred.start_page });
    }
  }
  S._allPipelineEmails = allEmails;

  if (!allEmails.length) {
    container.innerHTML = '<span class="pred-none">No emails in pipeline</span>';
    return;
  }

  let html = '';
  for (let i = 0; i < allEmails.length; i++) {
    const em = allEmails[i];
    html += `<div class="all-email-row">` +
      `<button class="btn-copy-email" onclick="copyEmailFromGlobal(${i})" title="Copy to current GT doc">↓</button>` +
      `<div class="all-email-info">` +
        `<div class="all-email-subj">${esc(em.subject || '—')}</div>` +
        `<div class="all-email-meta">${esc(em.date || '')}${em.doc_code ? ' · ' + esc(em.doc_code) : ''} · p${em.start_page + 1}</div>` +
      `</div>` +
    `</div>`;
  }
  container.innerHTML = html;
}

function copyEmailFromGlobal(idx) {
  const em = (S._allPipelineEmails || [])[idx];
  if (!em) return;
  // Ensure current GT doc is E-mail type and fields are visible
  const ann = getAnnotation(S.selectedStart);
  if (!ann.type_gt) {
    ann.type_gt = 'E-mail';
    document.getElementById('gt-type').value = 'E-mail';
  }
  document.getElementById('row-num-emails').style.display = '';
  document.getElementById('row-email-details').style.display = '';
  _appendEmailEntry(document.getElementById('email-rows-container'), em.subject || '', em.sender || '', em.date || '');
  saveFormToState();
  renderDocList();
  scheduleSave(true);
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
function bindKeyboard() {
  document.addEventListener('keydown', e => {
    const tag = document.activeElement ? document.activeElement.tagName : '';

    // In text inputs: let browser handle Tab naturally (field-to-field); block other shortcuts
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    if (e.key === 'b' || e.key === 'B') {
      toggleBoundary(S.currentPage);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const n = Math.max(0, S.currentPage - 1);
      scrollToPage(n);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const n = Math.min(S.pageCount - 1, S.currentPage + 1);
      scrollToPage(n);
      return;
    }
    if (e.key === 's' || e.key === 'S') {
      doSave();
      return;
    }
    const TYPE_SHORTCUTS = { '1': 'E-mail', '2': 'Nota', '3': 'Report', '4': 'Other' };
    if (TYPE_SHORTCUTS[e.key] && S.selectedStart != null) {
      const type    = TYPE_SHORTCUTS[e.key];
      const isEmail = type === 'E-mail';
      const isOther = type === 'Other';
      const ann     = getAnnotation(S.selectedStart);
      ann.type_gt   = type;
      document.getElementById('gt-type').value                 = type;
      document.getElementById('row-num-emails').style.display    = isEmail ? '' : 'none';
      document.getElementById('row-email-details').style.display = isEmail ? '' : 'none';
      document.getElementById('row-other-specify').style.display = isOther ? '' : 'none';
      if (isOther) setTimeout(() => document.getElementById('gt-other-specify').focus(), 0);
      renderDocList();
      scheduleSave(true);
    }
  });
}

// ── Export ────────────────────────────────────────────────────────────────────
function bindExport() {
  document.getElementById('btn-export').addEventListener('click', async () => {
    const r    = await fetch('/api/export');
    const data = await r.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `annotations_${S.pdfName}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const [info, state, preds, preds_b] = await Promise.all([
    apiGet('/api/info'),
    apiGet('/api/state'),
    apiGet('/api/predictions'),
    apiGet('/api/predictions_b'),
  ]);

  S.pageCount      = info.page_count;
  S.pdfName        = info.pdf_name;
  S.pipelineMode   = info.pipeline_mode;
  S.pipelineModeB  = info.pipeline_mode_b || 'none';
  S.predictions    = preds;
  S.predictions_b  = preds_b || [];
  S.boundaries     = state.boundaries;
  S.annotations    = state.annotations;
  S.metrics        = state.metrics;
  S.metrics_b      = state.metrics_b || null;
  S.selectedStart  = S.boundaries[0] ?? 0;

  // Show "Pipe B" tab only when second pipeline is loaded
  if (S.predictions_b.length > 0) {
    document.getElementById('tab-pipeline-b').style.display = '';
    document.getElementById('tab-pipeline').textContent = 'Pipe A';
  }

  document.getElementById('tb-info').textContent =
    `${S.pdfName} — ${S.pageCount} pages`;
  document.getElementById('tb-mode').textContent = S.pipelineMode;

  buildPdfViewer();
  renderDocList();
  renderForm();
  renderMetrics();
  renderAllEmailsPanel();
  bindFormEvents();
  bindKeyboard();
  bindExport();
}

init().catch(console.error);
</script>
</body>
</html>
"""

# ── Pipeline loaders ──────────────────────────────────────────────────────────

def _load_from_cache(cache_path: Path, pdf_path: Path) -> list[dict]:
    """Load pipeline predictions from GPT-4o cache JSON using forward-fill logic."""
    # Import inside function to avoid errors when pipeline files are absent
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from pipeline_gpt4o import docs_from_cache  # noqa: F401
        from text_sorting import sort_documents      # noqa: F401
    except ImportError:
        pass  # We'll still re-derive pages from raw cache below

    with open(cache_path, encoding="utf-8") as f:
        raw_cache = json.load(f)

    # Forward-fill using contiguous runs so the same doc code appearing twice
    # (e.g. 0124 on p7 and again on p31) correctly creates two separate segments.
    from collections import defaultdict
    raw_segs: list[dict] = []  # [{code, pages, category, method}]
    last_code     = None
    unknown_count = 0

    for p in raw_cache["pages"]:
        page_0   = p["page_num"] - 1
        detected = p.get("doc_code")
        wpn      = p.get("within_doc_page")
        is_new   = (p.get("is_new_document", False) or wpn == 1) and (wpn is None or wpn == 1)
        cat      = p.get("category", "Other")

        if detected:
            # Start new segment if: different code from current, OR same code but is_new
            if not raw_segs or detected != last_code or is_new:
                raw_segs.append({"code": detected, "pages": [], "category": cat,
                                  "method": "gpt4o-stamp"})
                last_code = detected
            raw_segs[-1]["pages"].append(page_0)
        elif is_new:
            unknown_count += 1
            code = f"unknown_{unknown_count}"
            raw_segs.append({"code": code, "pages": [page_0], "category": cat,
                              "method": "gpt4o-boundary"})
            last_code = code
        else:
            if raw_segs:
                raw_segs[-1]["pages"].append(page_0)
            else:
                unknown_count += 1
                code = f"unknown_{unknown_count}"
                raw_segs.append({"code": code, "pages": [page_0], "category": cat,
                                  "method": "gpt4o-boundary"})
                last_code = code

    # Build page→segment index and collect page texts per segment
    page_to_seg_idx: dict[int, int] = {}
    seg_texts: dict[int, list[str]] = defaultdict(list)
    seg_emails_page: dict[int, list[dict]] = defaultdict(list)  # pass-1/2 per-page fallback

    for i, seg in enumerate(raw_segs):
        for pg in seg["pages"]:
            page_to_seg_idx[pg] = i

    for p in raw_cache["pages"]:
        page_0  = p["page_num"] - 1
        seg_idx = page_to_seg_idx.get(page_0)
        if seg_idx is None:
            continue
        seg_texts[seg_idx].append(p.get("text") or "")
        if p.get("email_start"):
            seg_emails_page[seg_idx].append({
                "subject": p.get("email_subject") or "",
                "sender":  p.get("email_from")    or "",
                "to":      p.get("email_to")       or "",
                "cc":      p.get("email_cc")       or "",
                "date":    p.get("email_date")     or "",
            })

    # Pass-3 email results stored in cache (most accurate — sees full document)
    emails_by_doc: dict[str, list[dict]] = raw_cache.get("emails_by_doc") or {}

    # Import email_splitter for text-based fallback
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from email_splitter import split_emails as _se
        _splitter_available = True
    except ImportError:
        _splitter_available = False

    def _pick_emails_for_seg(seg_idx: int, code: str, category: str) -> list[dict]:
        """Mirror server._pick_emails: max(pass-3, email_splitter), enrich if splitter wins."""
        if category != "E-mail":
            return []
        # GPT-4o pass-3 (preferred) → fall back to per-page pass-1/2 flags
        gpt_emails = emails_by_doc.get(code) or [
            {"subject": e["subject"], "sender": e["sender"], "to": e.get("to",""),
             "cc": e.get("cc",""), "date": e["date"]}
            for e in seg_emails_page.get(seg_idx, [])
        ]
        text_emails: list[dict] = []
        if _splitter_available:
            text = "\n\n".join(seg_texts.get(seg_idx, []))
            try:
                text_emails = _se(text, code) if text.strip() else []
            except Exception:
                text_emails = []
        if len(text_emails) <= len(gpt_emails):
            return gpt_emails or text_emails
        # Text splitter found more — enrich with GPT-4o metadata where available
        for j, em in enumerate(text_emails):
            if j >= len(gpt_emails):
                break
            g = gpt_emails[j]
            if not em.get("subject") and g.get("subject"):
                em["subject"] = g["subject"]
            if not em.get("sender") and g.get("sender"):
                em["sender"] = g["sender"]
            if not em.get("date") and g.get("date"):
                em["date"] = g["date"]
        return text_emails

    segments = []
    for i, rs in enumerate(raw_segs):
        if not rs["pages"]:
            continue
        seg: dict = {
            "doc_code":   rs["code"],
            "start_page": min(rs["pages"]),
            "end_page":   max(rs["pages"]),
            "type":       rs["category"],
            "date_raw":   "",
            "date_iso":   "",
            "method":     rs["method"],
        }
        emails = _pick_emails_for_seg(i, rs["code"], rs["category"])
        if emails:
            seg["num_emails_pipeline"] = len(emails)
            seg["emails_pipeline"] = [
                {
                    "subject": em.get("subject") or "",
                    "sender":  em.get("sender")  or "",
                    "date":    em.get("date")     or "",
                }
                for em in emails
            ]
        segments.append(seg)
    segments.sort(key=lambda s: s["start_page"])

    # Try to overlay dates from sort_documents (best effort)
    try:
        from pipeline_gpt4o import docs_from_cache as _dfc
        from text_sorting import sort_documents as _sd
        docs = _sd(_dfc(cache_path, pdf_path))
        for code, doc in docs.items():
            d = doc.get("date")
            if d:
                iso = str(d)[:10]
                for seg in segments:
                    if seg["doc_code"] == code:
                        seg["date_iso"] = iso
                        seg["date_raw"] = iso
    except Exception:
        pass

    return segments


def _docs_to_segments(docs: dict, method_override: str = "") -> list[dict]:
    """Convert a docs dict (from any pipeline) into the segment list format."""
    segments = []
    for code, doc in docs.items():
        pages = doc.get("pages", [])
        if not pages:
            continue
        # pages can be 1-based ints or the raw list depending on pipeline
        page_0_list = [p - 1 for p in pages if isinstance(p, int) and p >= 1]
        if not page_0_list:
            continue
        date_val = doc.get("date")
        date_raw = doc.get("date_raw", "")
        date_iso = ""
        if date_val is not None:
            try:
                date_iso = date_val.strftime("%Y-%m-%d")
            except Exception:
                date_iso = str(date_val)
        seg: dict = {
            "doc_code":   code,
            "start_page": min(page_0_list),
            "end_page":   max(page_0_list),
            "type":       doc.get("category", "Other"),
            "date_raw":   date_raw or "",
            "date_iso":   date_iso,
            "method":     method_override or doc.get("method", ""),
        }
        if doc.get("category") == "E-mail":
            # Try to get per-email details
            emails_list = doc.get("emails") or []
            if not emails_list:
                try:
                    from email_splitter import split_emails as _se
                    emails_list = _se(doc.get("text", ""), code)
                except Exception:
                    pass
            if emails_list:
                seg["num_emails_pipeline"] = len(emails_list)
                seg["emails_pipeline"] = [
                    {
                        "subject": em.get("subject") or "",
                        "sender":  em.get("sender")  or "",
                        "date":    em.get("date")     or "",
                    }
                    for em in emails_list
                ]
        segments.append(seg)
    return segments


def _load_ocr(pdf_path: Path, save_to: Path | None = None) -> list[dict]:
    import sys as _sys
    _sys.path.insert(0, str(pdf_path.parent))
    from pipeline_ocr import load_pdf
    from text_sorting import sort_documents
    docs = load_pdf(pdf_path, ocr_supplement=True)
    docs = sort_documents(docs)
    segs = sorted(_docs_to_segments(docs, "ocr"), key=lambda s: s["start_page"])
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump({"pipeline": "ocr", "segments": segs}, f, indent=2, default=str)
        print(f"[annotate] OCR segments saved to {save_to}")
    return segs


def _load_segments_json(path: Path) -> list[dict]:
    """Load a pre-saved segments JSON file (from --ocr auto-save or manual export)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both {"segments": [...]} wrapper and bare list
    return data.get("segments", data) if isinstance(data, dict) else data


def _load_api_key(pdf_path: Path, api_key: str) -> list[dict]:
    import sys as _sys
    _sys.path.insert(0, str(pdf_path.parent))
    from pipeline_gpt4o import load_pdf_vlm
    from text_sorting import sort_documents

    pdf_stem   = pdf_path.stem
    cache_path = pdf_path.parent / f"{pdf_stem}_gpt4o_cache.json"
    docs       = load_pdf_vlm(pdf_path, api_key=api_key, cache_path=cache_path)
    docs       = sort_documents(docs)
    segs       = sorted(_docs_to_segments(docs, "gpt4o-live"), key=lambda s: s["start_page"])

    # Now re-load from the saved cache for accurate forward-fill boundaries
    if cache_path.exists():
        try:
            segs = _load_from_cache(cache_path, pdf_path)
            G["pipeline_mode"] = "gpt4o-live"
        except Exception:
            pass

    return segs


# ── Persistence ───────────────────────────────────────────────────────────────

def _annotations_path(pdf_name: str) -> Path:
    return Path(f"annotations_{pdf_name}.json")


def _save_annotations() -> None:
    path = G["annotations_path"]
    if not path:
        return
    payload = {
        "boundaries":  G["boundaries"],
        "annotations": G["annotations"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_annotations(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bounds = data.get("boundaries", [0])
    if 0 not in bounds:
        bounds = [0] + bounds
    G["boundaries"]  = sorted(set(bounds))
    G["annotations"] = data.get("annotations", {})


# ── Date normalisation ────────────────────────────────────────────────────────

_NL_MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "augustus": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
    # English
    "january": 1, "february": 2, "march": 3, "may": 5, "june": 6,
    "july": 7, "august": 8, "october": 10,
}

def _normalise_date(s: str) -> str:
    """Convert any Dutch/English date expression to YYYY-MM-DD, or '' if not parseable."""
    import re
    s = s.strip()
    if not s:
        return ""
    # Already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    # DD-MM-YYYY or DD/MM/YYYY
    m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    # D(D) MonthName YYYY  (Dutch or English)
    m = re.fullmatch(r"(\d{1,2})\s+([a-zA-Zë]+)\s+(\d{4})", s)
    if m:
        month_str = m.group(2).lower().rstrip(".")
        month_num = _NL_MONTHS.get(month_str)
        if month_num:
            return f"{m.group(3)}-{month_num:02d}-{int(m.group(1)):02d}"
    # Try stdlib as last resort
    try:
        from datetime import datetime
        for fmt in ("%d %B %Y", "%B %d, %Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    except Exception:
        pass
    return ""


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(predictions: list | None = None) -> dict:
    boundaries  = G["boundaries"]
    if predictions is None:
        predictions = G["pipeline_predictions"]
    annotations = G["annotations"]
    page_count  = G["page_count"]

    gt_bound_set   = set(boundaries) - {0}
    pred_bound_set = {p["start_page"] for p in predictions if p["start_page"] > 0}

    tp = len(gt_bound_set & pred_bound_set)
    fp = len(pred_bound_set - gt_bound_set)
    fn = len(gt_bound_set - pred_bound_set)

    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    # Build GT docs in order
    sorted_bounds = sorted(boundaries)
    gt_docs = []
    for i, start in enumerate(sorted_bounds):
        end = sorted_bounds[i + 1] - 1 if i + 1 < len(sorted_bounds) else page_count - 1
        ann = annotations.get(str(start), {})
        gt_docs.append({"start": start, "end": end, "ann": ann})

    # Type accuracy: positional match gt[i] vs pred[i]
    type_correct    = 0
    type_annotated  = 0
    for i, doc in enumerate(gt_docs):
        ann = doc["ann"]
        if ann.get("type_gt"):
            type_annotated += 1
            pred = predictions[i] if i < len(predictions) else None
            if pred and ann["type_gt"] == pred.get("type", ""):
                type_correct += 1

    type_accuracy = type_correct / type_annotated if type_annotated > 0 else 0.0

    # Email count accuracy: compare gt num_emails vs pipeline num_emails_pipeline
    email_count_correct    = 0
    email_count_annotated  = 0
    for i, doc in enumerate(gt_docs):
        ann = doc["ann"]
        ne_gt = ann.get("num_emails_gt")
        if ann.get("type_gt") == "E-mail" and ne_gt is not None:
            email_count_annotated += 1
            pred = predictions[i] if i < len(predictions) else None
            if pred and pred.get("num_emails_pipeline") == ne_gt:
                email_count_correct += 1
    email_count_accuracy = email_count_correct / email_count_annotated if email_count_annotated > 0 else None

    # Date rate
    date_count = sum(1 for doc in gt_docs if doc["ann"].get("date_gt"))
    gt_count   = len(gt_docs)
    date_rate  = date_count / gt_count if gt_count > 0 else 0.0

    # Email-level subject / date / sender metrics
    # Only computed for docs where GT has emails_gt with at least one entry
    # and pipeline has emails_pipeline for the matched prediction.
    email_subject_tp = 0; email_subject_fp = 0; email_subject_fn = 0
    email_date_correct = 0; email_date_annotated = 0
    email_sender_correct = 0; email_sender_annotated = 0
    any_email_detail = False

    for i, doc in enumerate(gt_docs):
        ann = doc["ann"]
        emails_gt = ann.get("emails_gt") or []
        if not emails_gt:
            continue
        pred = predictions[i] if i < len(predictions) else None
        emails_pred = (pred.get("emails_pipeline") or []) if pred else []
        any_email_detail = True

        # Greedy bipartite matching on subject: each GT email matches best unmatched pred email
        used_pred: set[int] = set()
        for gt_em in emails_gt:
            gt_subj = (gt_em.get("subject") or "").strip()
            best_j, best_score = -1, -1
            for j, pred_em in enumerate(emails_pred):
                if j in used_pred:
                    continue
                pred_subj = (pred_em.get("subject") or "").strip()
                if gt_subj and pred_subj and _fuzzy_match(gt_subj, pred_subj):
                    # score: use simple partial_ratio if available, else 100 for exact
                    try:
                        from rapidfuzz import fuzz as _rf
                        score = _rf.partial_ratio(gt_subj.lower(), pred_subj.lower())
                    except ImportError:
                        score = 100
                    if score > best_score:
                        best_score = score
                        best_j = j

            if best_j >= 0:
                email_subject_tp += 1
                used_pred.add(best_j)
                # Date match for this pair
                gt_date   = _normalise_date(gt_em.get("date") or "")
                pred_date = _normalise_date((emails_pred[best_j].get("date") or ""))
                if gt_date:
                    email_date_annotated += 1
                    if gt_date == pred_date:
                        email_date_correct += 1
                # Sender match for this pair
                gt_sndr   = (gt_em.get("sender") or "").strip()
                pred_sndr = (emails_pred[best_j].get("sender") or "").strip()
                if gt_sndr:
                    email_sender_annotated += 1
                    if _fuzzy_match(gt_sndr, pred_sndr):
                        email_sender_correct += 1
            else:
                email_subject_fn += 1

        # Unmatched pred emails are false positives
        email_subject_fp += len(emails_pred) - len(used_pred)

    if any_email_detail:
        esp = email_subject_tp / (email_subject_tp + email_subject_fp) if (email_subject_tp + email_subject_fp) > 0 else 0.0
        esr = email_subject_tp / (email_subject_tp + email_subject_fn) if (email_subject_tp + email_subject_fn) > 0 else 0.0
        email_subject_f1 = 2 * esp * esr / (esp + esr) if (esp + esr) > 0 else 0.0
        email_date_accuracy   = email_date_correct   / email_date_annotated   if email_date_annotated   > 0 else None
        email_sender_accuracy = email_sender_correct / email_sender_annotated if email_sender_annotated > 0 else None
    else:
        email_subject_f1 = None
        email_date_accuracy = None
        email_sender_accuracy = None

    return {
        "boundary_precision":     P,
        "boundary_recall":        R,
        "boundary_f1":            F1,
        "boundary_tp":            tp,
        "boundary_fp":            fp,
        "boundary_fn":            fn,
        "type_accuracy":            type_accuracy,
        "type_correct":             type_correct,
        "type_annotated":           type_annotated,
        "email_count_accuracy":     email_count_accuracy,
        "email_count_correct":      email_count_correct,
        "email_count_annotated":    email_count_annotated,
        "email_subject_f1":         email_subject_f1,
        "email_date_accuracy":      email_date_accuracy,
        "email_date_correct":       email_date_correct,
        "email_date_annotated":     email_date_annotated,
        "email_sender_accuracy":    email_sender_accuracy,
        "email_sender_correct":     email_sender_correct,
        "email_sender_annotated":   email_sender_annotated,
        "date_extraction_rate":     date_rate,
        "date_count":               date_count,
        "gt_doc_count":             gt_count,
        "pipeline_doc_count":       len(predictions),
    }


def _pipeline_b_fields(predictions_b: list, doc_index: int, type_gt: str, date_gt: str) -> dict:
    """Return pipeline-B comparison fields for a single document row in the export."""
    if not predictions_b:
        return {}
    pred_b = predictions_b[doc_index] if doc_index < len(predictions_b) else None
    type_match_b = bool(type_gt and pred_b and type_gt == pred_b.get("type", ""))
    gt_iso       = _normalise_date(date_gt)
    pred_iso_b   = _normalise_date(pred_b.get("date_iso", "") if pred_b else "")
    date_match_b = bool(gt_iso and pred_iso_b and gt_iso == pred_iso_b)
    return {
        "pipeline_b_doc_code":   pred_b["doc_code"]                    if pred_b else None,
        "pipeline_b_type":       pred_b["type"]                        if pred_b else None,
        "pipeline_b_date":       pred_b["date_raw"]                    if pred_b else None,
        "pipeline_b_method":     pred_b["method"]                      if pred_b else None,
        "num_emails_pipeline_b": pred_b.get("num_emails_pipeline")     if pred_b else None,
        "emails_pipeline_b":     pred_b.get("emails_pipeline")         if pred_b else None,
        "type_match_b":          type_match_b,
        "date_match_b":          date_match_b,
    }


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/info")
def api_info():
    return jsonify({
        "page_count":       G["page_count"],
        "pdf_name":         G["pdf_name"],
        "pipeline_mode":    G["pipeline_mode"],
        "pipeline_count":   len(G["pipeline_predictions"]),
        "pipeline_mode_b":  G["pipeline_mode_b"],
        "pipeline_count_b": len(G["predictions_b"]),
    })


@app.route("/api/page/<int:n>")
def api_page(n: int):
    doc = G["fitz_doc"]
    if doc is None or n < 0 or n >= G["page_count"]:
        return jsonify({"error": "invalid page"}), 404

    page = doc[n]
    mat  = fitz.Matrix(1.5, 1.5)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    png  = pix.tobytes("png")
    b64  = base64.b64encode(png).decode()
    return jsonify({"image": f"data:image/png;base64,{b64}"})


@app.route("/api/predictions")
def api_predictions():
    return jsonify(G["pipeline_predictions"])


@app.route("/api/predictions_b")
def api_predictions_b():
    return jsonify(G["predictions_b"])


@app.route("/api/state", methods=["GET"])
def api_state_get():
    metrics   = _compute_metrics()
    metrics_b = _compute_metrics(G["predictions_b"]) if G["predictions_b"] else None
    return jsonify({
        "boundaries":  G["boundaries"],
        "annotations": G["annotations"],
        "metrics":     metrics,
        "metrics_b":   metrics_b,
    })


@app.route("/api/state", methods=["POST"])
def api_state_post():
    data = request.get_json(force=True)

    bounds = data.get("boundaries", G["boundaries"])
    if 0 not in bounds:
        bounds = [0] + bounds
    G["boundaries"]  = sorted(set(bounds))
    G["annotations"] = data.get("annotations", G["annotations"])

    _save_annotations()
    metrics   = _compute_metrics()
    metrics_b = _compute_metrics(G["predictions_b"]) if G["predictions_b"] else None
    return jsonify({"ok": True, "metrics": metrics, "metrics_b": metrics_b})


@app.route("/api/export")
def api_export():
    boundaries    = G["boundaries"]
    predictions   = G["pipeline_predictions"]
    predictions_b = G["predictions_b"]
    annotations   = G["annotations"]
    page_count    = G["page_count"]
    metrics       = _compute_metrics()
    metrics_b     = _compute_metrics(predictions_b) if predictions_b else None

    sorted_bounds = sorted(boundaries)
    gt_docs = []
    for i, start in enumerate(sorted_bounds):
        end  = sorted_bounds[i + 1] - 1 if i + 1 < len(sorted_bounds) else page_count - 1
        ann  = annotations.get(str(start), {})
        pred = predictions[i] if i < len(predictions) else None

        type_gt  = ann.get("type_gt", "")
        date_gt  = ann.get("date_gt", "")
        type_match = bool(type_gt and pred and type_gt == pred.get("type", ""))
        gt_iso   = _normalise_date(date_gt)
        pred_iso = _normalise_date(pred.get("date_iso", "") if pred else "")
        date_match = bool(gt_iso and pred_iso and gt_iso == pred_iso)

        gt_docs.append({
            "doc_index":        i,
            "start_page":       start,
            "end_page":         end,
            "type_gt":          type_gt,
            "date_gt":          date_gt,
            "notes":                   ann.get("notes", ""),
            "other_type":              ann.get("other_type", ""),
            "num_emails_gt":           ann.get("num_emails_gt"),
            "emails_gt":               ann.get("emails_gt") or [],
            "pipeline_doc_code":       pred["doc_code"]             if pred else None,
            "pipeline_type":           pred["type"]                 if pred else None,
            "pipeline_date":           pred["date_raw"]             if pred else None,
            "pipeline_method":         pred["method"]               if pred else None,
            "num_emails_pipeline":     pred.get("num_emails_pipeline") if pred else None,
            "emails_pipeline":         pred.get("emails_pipeline")  if pred else None,
            "type_match":              type_match,
            "date_match":              date_match,
            # Pipeline B comparison (if loaded)
            **_pipeline_b_fields(predictions_b, i, type_gt, date_gt),
        })

    payload = {
        "pdf":             G["pdf_name"],
        "pipeline_mode_a": G["pipeline_mode"],
        "pipeline_mode_b": G["pipeline_mode_b"],
        "date_annotated":  date.today().isoformat(),
        "documents":       gt_docs,
        "metrics_a":       metrics,
        "metrics_b":       metrics_b,
    }
    return jsonify(payload)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WOO document segmentation annotation tool"
    )
    parser.add_argument("--pdf",               required=True,  metavar="FILE", help="PDF file to annotate")
    parser.add_argument("--auto",              action="store_true",
                        help="Auto mode: GPT-4o (A) + OCR (B). Uses cached results when available.")
    parser.add_argument("--api-key",          metavar="KEY",  help="OpenAI API key — used by --auto and --api-key pipeline A mode (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--from-cache",       metavar="FILE", help="Pipeline A: load from GPT-4o cache JSON")
    parser.add_argument("--from-segments",    metavar="FILE", help="Pipeline A: load from pre-saved segments JSON")
    parser.add_argument("--ocr",              action="store_true", help="Pipeline A: run OCR pipeline (auto-saves segments JSON)")
    parser.add_argument("--compare-cache",    metavar="FILE", help="Pipeline B: load from GPT-4o cache JSON")
    parser.add_argument("--compare-segments", metavar="FILE", help="Pipeline B: load from pre-saved segments JSON")
    parser.add_argument("--compare-ocr",      action="store_true", help="Pipeline B: run OCR pipeline (auto-saves segments JSON)")
    parser.add_argument("--compare-api-key",  metavar="KEY",  help="Pipeline B: OpenAI API key (GPT-4o live)")
    parser.add_argument("--annotations",      metavar="FILE", help="Existing annotations JSON to load")
    parser.add_argument("--port",             type=int, default=5050, metavar="PORT", help="HTTP port (default: 5050)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[annotate] ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    pdf_name = pdf_path.stem

    # Open with PyMuPDF
    fitz_doc   = fitz.open(str(pdf_path))
    page_count = fitz_doc.page_count

    G["pdf_path"]   = pdf_path
    G["fitz_doc"]   = fitz_doc
    G["page_count"] = page_count
    G["pdf_name"]   = pdf_name

    # ── Load pipelines ─────────────────────────────────────────────────────────
    segments:   list[dict] = []
    mode   = "none"
    segments_b: list[dict] = []
    mode_b = "none"

    gpt4o_cache_path = pdf_path.with_name(f"{pdf_path.stem}_gpt4o_cache.json")
    ocr_segs_path    = pdf_path.with_name(f"{pdf_path.stem}_ocr_segs.json")

    def _run_ocr_cached(save_path: Path) -> list[dict]:
        if save_path.exists():
            print(f"[annotate] Loading cached OCR segments: {save_path.name}  (delete to re-run)")
            return _load_segments_json(save_path)
        print("[annotate] Running OCR pipeline…")
        import sys as _sys
        _sys.path.insert(0, str(pdf_path.parent))
        return _load_ocr(pdf_path, save_to=save_path)

    if args.auto:
        # ── Auto mode: GPT-4o as Pipeline A, OCR as Pipeline B ────────────────
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

        # Pipeline A — GPT-4o
        if gpt4o_cache_path.exists():
            print(f"[annotate] [auto] GPT-4o cache found: {gpt4o_cache_path.name}")
            try:
                segments = _load_from_cache(gpt4o_cache_path, pdf_path)
                mode = "gpt4o-cache"
            except Exception as e:
                print(f"[annotate] WARNING: Could not load GPT-4o cache: {e}")
        elif api_key:
            print("[annotate] [auto] No GPT-4o cache — running GPT-4o pipeline…")
            try:
                import sys as _sys
                _sys.path.insert(0, str(pdf_path.parent))
                segments = _load_api_key(pdf_path, api_key)
                mode = "gpt4o-live"
            except Exception as e:
                print(f"[annotate] WARNING: GPT-4o pipeline failed: {e}")
        else:
            print("[annotate] [auto] No GPT-4o cache and no API key — skipping GPT-4o.")
            print("           Set OPENAI_API_KEY or pass --api-key to run GPT-4o.")

        # Pipeline B — OCR
        try:
            segments_b = _run_ocr_cached(ocr_segs_path)
            mode_b = "ocr"
        except Exception as e:
            print(f"[annotate] WARNING: OCR pipeline failed: {e}")

        # If GPT-4o failed but OCR worked, promote OCR to Pipeline A
        if not segments and segments_b:
            segments, mode, segments_b, mode_b = segments_b, mode_b, [], "none"
            print("[annotate] [auto] GPT-4o unavailable — loaded OCR as Pipeline A only.")

    else:
        # ── Manual pipeline A ─────────────────────────────────────────────────
        if args.from_cache:
            print(f"[annotate] Loading pipeline A from GPT-4o cache: {args.from_cache}")
            try:
                segments = _load_from_cache(Path(args.from_cache).resolve(), pdf_path)
                mode = "gpt4o-cache"
            except Exception as e:
                print(f"[annotate] WARNING: Could not load cache: {e}")

        elif args.from_segments:
            print(f"[annotate] Loading pipeline A from segments: {args.from_segments}")
            try:
                segments = _load_segments_json(Path(args.from_segments).resolve())
                mode = segments[0].get("method", "segments") if segments else "segments"
            except Exception as e:
                print(f"[annotate] WARNING: Could not load segments: {e}")

        elif args.ocr:
            try:
                segments = _run_ocr_cached(ocr_segs_path)
                mode = "ocr"
            except Exception as e:
                print(f"[annotate] WARNING: OCR pipeline failed: {e}")

        elif args.api_key:
            print("[annotate] Loading pipeline A via GPT-4o live…")
            try:
                import sys as _sys
                _sys.path.insert(0, str(pdf_path.parent))
                segments = _load_api_key(pdf_path, args.api_key)
                mode = "gpt4o-live"
            except Exception as e:
                print(f"[annotate] WARNING: GPT-4o pipeline failed: {e}")

        # ── Manual pipeline B ─────────────────────────────────────────────────
        if args.compare_cache:
            print(f"[annotate] Loading pipeline B from GPT-4o cache: {args.compare_cache}")
            try:
                segments_b = _load_from_cache(Path(args.compare_cache).resolve(), pdf_path)
                mode_b = "gpt4o-cache"
            except Exception as e:
                print(f"[annotate] WARNING: Could not load pipeline B cache: {e}")

        elif args.compare_segments:
            print(f"[annotate] Loading pipeline B from segments: {args.compare_segments}")
            try:
                segments_b = _load_segments_json(Path(args.compare_segments).resolve())
                mode_b = segments_b[0].get("method", "segments") if segments_b else "segments"
            except Exception as e:
                print(f"[annotate] WARNING: Could not load pipeline B segments: {e}")

        elif args.compare_ocr:
            try:
                segments_b = _run_ocr_cached(ocr_segs_path)
                mode_b = "ocr"
            except Exception as e:
                print(f"[annotate] WARNING: Pipeline B OCR failed: {e}")

        elif args.compare_api_key:
            print("[annotate] Loading pipeline B via GPT-4o live…")
            try:
                import sys as _sys
                _sys.path.insert(0, str(pdf_path.parent))
                segments_b = _load_api_key(pdf_path, args.compare_api_key)
                mode_b = "gpt4o-live"
            except Exception as e:
                print(f"[annotate] WARNING: Pipeline B GPT-4o failed: {e}")

    G["pipeline_predictions"] = segments
    G["pipeline_mode"]        = mode
    G["predictions_b"]        = segments_b
    G["pipeline_mode_b"]      = mode_b

    print(f"[annotate] Pipeline A: {mode} — {len(segments)} segments")
    if segments_b:
        print(f"[annotate] Pipeline B: {mode_b} — {len(segments_b)} segments")

    # ── Load annotations ───────────────────────────────────────────────────────
    ann_path = Path(args.annotations).resolve() if args.annotations else _annotations_path(pdf_name)
    G["annotations_path"] = ann_path

    if ann_path.exists():
        _load_annotations(ann_path)
        print(f"[annotate] Loaded existing annotations from {ann_path.name}")
    else:
        # Ensure page 0 is always in boundaries
        G["boundaries"] = [0]

    # ── Start server ───────────────────────────────────────────────────────────
    print(f"[annotate] Serving {pdf_path.name} ({page_count} pages)")
    print(f"[annotate] Open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
