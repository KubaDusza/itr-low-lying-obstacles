"""Review which parts of ISOD's not-drivable mask are low-lying floor obstacles.

    python code/review_isod.py          # then open the URL it prints

ISOD labels every pixel drivable or not. The not-drivable class holds the walls, the
furniture standing on the floor and the objects lying on it; only the last are what this
project detects. isod_components.py splits that mask into components and proposes an answer
for each. This is where the answer is decided.

The rule being applied: **keep anything lying on the floor** -- cables, cloths, mats, shoes,
bags, a vacuum lying down. **Drop anything standing on the floor as structure** -- chairs,
tables, cabinets, bins, walls, doors.

The cut brush removes pixels within ISOD's mask to separate touching regions.
Cached SAM masks offer another way to separate them. Clicking a suggestion
keeps its intersection with the clicked component as an independent region.

Each frame is judged on its own: the 100 frames of an ISOD site are separate captures around
a room, not a slow camera move, so a decision cannot be carried from one to the next. What
the tool does instead is say which components the rule was unsure about -- outlined in
yellow -- so a frame whose proposal is plainly right takes one keystroke.

Writes data/ISOD_gt2/<site>/<i>.png, the corrected obstacle mask, plus <i>.json recording
which component ids were kept. data/ISOD_gt stays untouched.
"""
import base64
import http.server
import json
import os
from pathlib import Path
import socket
import sys
import threading
import urllib.parse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isod                                                       # noqa: E402
import isod_components as comps                                   # noqa: E402
from isod_sam_review import PROMPT_GROUPS, read_cache               # noqa: E402

OUT = os.path.join(os.path.dirname(isod.GT), "ISOD_gt2")
SAM_CACHE = Path(isod.DATA_ROOT) / "ISOD_sam_review"
SAM_CONFIDENCE = 0.45
PORT = int(os.environ.get("REVIEW_PORT", 8810))
N_FRAMES = 100

PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>ISOD obstacle review</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2e333c; --ink:#e7e9ee; --muted:#9aa3b2;
          --accent:#3b6fd4; --ok:#2f9e6e; --warn:#e8a33d; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  #side { width:230px; flex:none; border-right:1px solid var(--line); display:flex;
          flex-direction:column; background:var(--panel); }
  #side h1 { font-size:13px; margin:0; padding:12px 14px; border-bottom:1px solid var(--line); }
  #filter { padding:8px 14px; border-bottom:1px solid var(--line); color:var(--muted); }
  #sites { overflow:auto; flex:1; }
  .site { padding:6px 14px; cursor:pointer; border-bottom:1px solid #22252b; display:flex;
          justify-content:space-between; gap:8px; }
  .site:hover { background:#232832; }
  .site.sel { background:var(--accent); color:#fff; }
  .site .n { font:11px ui-monospace,Menlo,monospace; color:var(--muted); }
  .site.sel .n { color:#dbe6ff; }
  .site.done .n { color:var(--ok); }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #bar { display:flex; align-items:center; gap:10px; padding:8px 14px;
         border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  #bar b { font-weight:600; }
  .tool { padding:3px 9px; border:1px solid var(--line); border-radius:5px; cursor:pointer;
          color:var(--muted); white-space:nowrap; position:relative; }
  .tool:hover { border-color:#4a515e; color:var(--ink); }
  .tool.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .tool .k { opacity:.75; font-size:11px; margin-left:3px; }
  button.tool { background:transparent; font:inherit; }
  button.tool:disabled { opacity:.5; cursor:default; }
  #sam-bar { display:flex; align-items:center; gap:8px; padding:6px 14px;
             background:var(--panel); border-bottom:1px solid var(--line); }
  #sam-bar[hidden] { display:none; }
  #region-legend { display:flex; gap:12px; padding:6px 14px; align-items:center; flex-wrap:wrap;
                   background:var(--panel); border-bottom:1px solid var(--line); }
  #region-legend[hidden], button.tool[hidden], .tool[hidden] { display:none; }
  .swatch { display:inline-block; width:10px; height:10px; margin-right:4px; border-radius:2px;
            background:#3cc878; }
  .swatch.background { background:#648afe; }
  .swatch.check { background:transparent; border:2px solid #ffd200; }
  .tool[data-hint]:hover::after {
    content:attr(data-hint); position:absolute; left:0; top:calc(100% + 7px); width:290px;
    background:#0c0e12; color:var(--ink); border:1px solid var(--line); border-radius:6px;
    padding:7px 9px; font-size:12px; line-height:1.4; z-index:20; white-space:normal;
    box-shadow:0 6px 18px rgba(0,0,0,.45); }
  .tool.hr[data-hint]:hover::after { left:auto; right:0; }
  #note { width:150px; background:#12151a; color:var(--ink); border:1px solid var(--line);
          border-radius:5px; padding:3px 7px; font:12px ui-sans-serif,system-ui; }
  #note:focus { outline:none; border-color:var(--accent); }
  #wrap { flex:1; overflow:hidden; position:relative; background:#0d0f12; }
  canvas { position:absolute; image-rendering:pixelated; }
  #strip { display:flex; flex-wrap:wrap; gap:2px; padding:6px 14px;
           border-top:1px solid var(--line); background:var(--panel); max-height:76px;
           overflow:auto; }
  .f { width:16px; height:16px; border-radius:3px; background:#2a2f38; cursor:pointer;
       font:9px/16px ui-monospace,Menlo,monospace; text-align:center; color:#8a93a2; }
  .f.done { background:#255c43; color:#b8ecd3; }
  .f.flag { box-shadow:inset 0 0 0 2px var(--warn); }
  .f.sel { outline:2px solid var(--accent); }
  #status { padding:6px 14px; border-top:1px solid var(--line); color:var(--muted);
            background:var(--panel); display:flex; gap:16px; }
  #counts { font:12px ui-monospace,Menlo,monospace; flex:none; width:260px; }
  #msg { flex:none; width:210px; font-size:12px; white-space:nowrap; }
  #msg.warn { color:var(--warn); }
  #hints { flex:1; min-width:0; overflow:hidden; white-space:nowrap; }
  kbd { background:#262b33; border:1px solid var(--line); border-radius:4px; padding:0 4px;
        font:11px/1.6 ui-monospace,Menlo,monospace; }
  #keys { position:fixed; inset:0; background:rgba(8,9,12,.72); display:none; z-index:40;
          align-items:center; justify-content:center; }
  #keys.on { display:flex; }
  #keybox { background:var(--panel); border:1px solid var(--line); border-radius:10px;
            width:560px; max-height:80vh; overflow:auto; padding:18px 20px; }
  #keybox h2 { margin:0 0 4px; font-size:15px; }
  #keybox p { margin:0 0 14px; color:var(--muted); }
  .krow { display:flex; align-items:center; gap:12px; padding:5px 0;
          border-bottom:1px solid #22252b; }
  .krow .sp { flex:1; }
  .kbtn { min-width:62px; text-align:center; background:#262b33; border:1px solid var(--line);
          border-radius:5px; padding:3px 8px; cursor:pointer;
          font:12px ui-monospace,Menlo,monospace; }
  .kbtn.listen { background:var(--warn); color:#17191d; }
  .mbtn { min-width:52px; text-align:center; border:1px solid var(--line); border-radius:5px;
          padding:3px 8px; cursor:pointer; font-size:11px; color:var(--muted); }
  .mbtn.hold { background:#2b3a2f; color:#8fd6a8; border-color:#36513d; }
  #keyfoot { display:flex; gap:10px; margin-top:16px; }
</style>
<div id="side">
  <h1>ISOD obstacle review</h1>
  <label id="filter"><input type="checkbox" id="only"> only frames needing a look</label>
  <div id="sites"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="title">—</b>
    <span class="tool" id="cut" data-hint="Cut brush. Erases inside ISOD's mask only, to part an object from the furniture it touches so it becomes its own component. The frame is then re-split and the decisions carried onto whatever came out. Nothing outside the mask is ever affected.">cut<span class="k" data-k="cut"></span></span>
    <span class="tool" id="szbox" data-hint="Cut brush diameter, in image pixels.">size <b id="sz">4</b><span class="k" data-k="sizeDown"></span><span class="k" data-k="sizeUp"></span></span>
    <span class="tool" id="pan" data-hint="Move the view: drag with the left button, nothing is cut. Held down by default rather than toggled; the ? panel sets hold or toggle for every key. Space pans from any tool and the wheel still zooms.">pan<span class="k" data-k="pan"></span></span>
    <span class="tool" id="unsure" data-hint="Go to the next component the rule was unsure about, in this frame or the next one that has any, and zoom to it.">next to check<span class="k" data-k="unsure"></span></span>
    <span class="tool" id="keepall" data-hint="Keep every component in this frame.">keep all<span class="k" data-k="keepAll"></span></span>
    <span class="tool" id="reset" data-hint="Back to what the automatic pass proposed, discarding your toggles and cuts on this frame.">reset<span class="k" data-k="reset"></span></span>
    <span class="tool" id="hide" data-hint="Hide the overlay to see the photograph underneath. Tap to toggle, or set it to hold in the ? panel and it lasts only while the key is down.">hide<span class="k" data-k="hide"></span></span>
    <button type="button" class="tool" id="sam-toggle" aria-expanded="false">SAM regions</button>
    <span class="tool" id="fit" data-hint="Fit the frame to the window.">fit<span class="k" data-k="fit"></span></span>
    <span class="tool" id="flag" data-hint="Mark this frame to come back to, with a note if you want one. Flagged frames carry an amber ring in the strip below and the filter on the left can show only those.">flag<span class="k" data-k="flag"></span></span>
    <input id="note" placeholder="note" spellcheck="false">
    <span class="tool" id="save" data-hint="Write this frame's mask. Also happens when you step to the next frame.">save<span class="k">⌘s</span></span>
    <button class="tool" id="test-advance" type="button" hidden>Save and next</button>
    <span class="tool" id="help" data-hint="Change any shortcut.">?</span>
  </div>
  <div id="sam-bar" hidden>
    <button type="button" class="tool" id="sam-prev" aria-label="Previous SAM region">←</button>
    <button type="button" class="tool" id="sam-next" aria-label="Next SAM region">→</button>
    <span id="sam-label"></span>
    <button type="button" class="tool" id="sam-apply"></button>
  </div>
  <div id="region-legend" hidden>
    <span><i class="swatch"></i>Keep</span>
    <span><i class="swatch background"></i>Background</span>
    <span><i class="swatch check"></i>Check obstacle label</span>
    <span>Keep loose objects resting on the floor. Click to switch labels.</span>
    <a class="tool" id="reference-mask" target="_blank" rel="noopener">Saved correction</a>
    <a class="tool" id="source-mask" target="_blank" rel="noopener">Original ISOD mask</a>
    <button class="tool" id="region-prev" type="button">Previous</button>
    <button class="tool" id="region-next" type="button">Save and next</button>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="strip"></div>
  <div id="status">
    <span id="counts">—</span>
    <span id="msg"></span>
    <span id="hints"></span>
  </div>
</div>
<div id="keys"><div id="keybox">
  <h2>Shortcuts</h2>
  <p>Click a key, then press the one you want. Saved in this browser.</p>
  <div id="krows"></div>
  <div id="keyfoot">
    <span class="tool" id="kreset" data-hint="Back to the shipped defaults.">reset to defaults</span>
    <span class="tool" id="kclose">close</span>
  </div>
</div></div>
<script>
const $ = s => document.querySelector(s);
/* The same letters as the mask annotator, for the actions the two tools share: e erases,
   v pans, h hides, x clears, 0 fits, t flags, u walks to what needs attention, n/N move. */
const DEFAULTS = {cut:'e', sizeDown:'!', sizeUp:'@', pan:'v', unsure:'u', keepAll:'k',
                  reset:'x', hide:'h', fit:'0', flag:'t', next:'n', prev:'N'};
const LABEL = {cut:'cut brush', sizeDown:'smaller brush', sizeUp:'bigger brush',
               pan:'pan the view', unsure:'go to the next component to check',
               keepAll:'keep every component', reset:'reset this frame',
               hide:'hide / show the overlay', fit:'fit to window', flag:'flag this frame',
               next:'next frame', prev:'previous frame'};
const HOLDABLE = ['cut', 'pan', 'hide'];
const MODE_DEF = {pan: 'hold'};
let MODES = {};
HOLDABLE.forEach(a => MODES[a] = MODE_DEF[a] || 'toggle');
try { Object.assign(MODES, JSON.parse(localStorage.getItem('review_modes') || '{}')); } catch (e) {}
let KEYS = Object.assign({}, DEFAULTS);
try { Object.assign(KEYS, JSON.parse(localStorage.getItem('review_keys') || '{}')); } catch (e) {}

const KEEP = [60, 200, 120], DROP = [70, 110, 220], CHECK = [255, 210, 0];
let sites = [], site = null, idx = 0, frames = [], flags = {};
const params = new URLSearchParams(location.search);
const samPartition = false;
const testReview = location.pathname === '/test';
const trainingReview = samPartition && location.pathname === '/training';
const queueKind = testReview ? 'test' : trainingReview ? 'training' : '';
const queueTitle = trainingReview ? 'Training spot-check' : 'Test batch';
const practice = !samPartition && !testReview && params.get('practice') === '1';
const practiceQuery = practice ? '&practice=1' : '';
const sampleSite = params.get('site');
let sample = (params.get('sample') || '').split(',').filter(Boolean).map(Number);
let reviewQueue = [];
let reviewingSample = false;
let loadingFrame = false, cutting = false, saving = false;
let img = null, comp = null, rows = [], W = 0, H = 0;
let view = {z: 1, ox: 0, oy: 0}, hide = false, dirty = false;
let tool = null, size = 4, drawing = false, panning = false, last = null, panKey = false;
let cut = null, undo = [], held = null, listening = null, mouse = null, unsureAt = 0;
let samEdits = null, suggestions = [], suggestionAt = -1;
const cv = $('#cv'), ctx = cv.getContext('2d');
const decodeBytes = value => Uint8Array.from(atob(value), c => c.charCodeAt(0));
const decodeComponents = response => {
  const bytes = decodeBytes(response.comp);
  return response.comp16 ? new Uint16Array(bytes.buffer) : bytes;
};
const encodeBytes = value => value.some(Boolean) ?
  btoa(Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
                  v => String.fromCharCode(v)).join('')) : '';

/* ------------------------------------------------------------------ keys */
const SHIFTED = {'!':'⇧1', '@':'⇧2', '#':'⇧3'};
const keyLabel = k => k === ' ' ? 'space' : (SHIFTED[k] || k || '—');
function paintKeys() {
  document.querySelectorAll('[data-k]').forEach(e => {
    e.textContent = KEYS[e.dataset.k] ? ' ' + keyLabel(KEYS[e.dataset.k]) : '';
  });
  $('#hints').innerHTML =
    `click a component to keep / drop it · <kbd>${keyLabel(KEYS.next)}</kbd>/` +
    `<kbd>${keyLabel(KEYS.prev)}</kbd> frame · <kbd>${keyLabel(KEYS.unsure)}</kbd> next to ` +
    `check · scroll zoom · space-drag pan · <kbd>⌘z</kbd> undo`;
}
function buildKeyPanel() {
  const box = $('#krows'); box.innerHTML = '';
  Object.keys(DEFAULTS).forEach(a => {
    const r = document.createElement('div'); r.className = 'krow';
    r.innerHTML = `<span>${LABEL[a]}</span><span class="sp"></span>`;
    if (HOLDABLE.includes(a)) {
      const m = document.createElement('span');
      const paint = () => {
        m.className = 'mbtn' + (MODES[a] === 'hold' ? ' hold' : '');
        m.textContent = MODES[a];
        m.title = MODES[a] === 'hold' ? 'active only while the key is held'
                                      : 'press once, stays until you press it again';
      };
      m.onclick = () => { MODES[a] = MODES[a] === 'hold' ? 'toggle' : 'hold';
                          localStorage.setItem('review_modes', JSON.stringify(MODES)); paint(); };
      paint(); r.appendChild(m);
    }
    const b = document.createElement('span');
    b.className = 'kbtn'; b.textContent = keyLabel(KEYS[a]);
    b.onclick = () => {
      document.querySelectorAll('.kbtn').forEach(x => x.classList.remove('listen'));
      b.classList.add('listen'); b.textContent = 'press…'; listening = {action: a, el: b};
    };
    r.appendChild(b); box.appendChild(r);
  });
}
function setKey(action, k) {
  Object.keys(KEYS).forEach(a => { if (a !== action && KEYS[a] === k) KEYS[a] = ''; });
  KEYS[action] = k;
  localStorage.setItem('review_keys', JSON.stringify(KEYS));
  buildKeyPanel(); paintKeys();
}
let msgTimer = null;
function msg(text, warn) {
  $('#msg').textContent = text; $('#msg').className = warn ? 'warn' : '';
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => { $('#msg').textContent = ''; }, 4000);
}

/* ---------------------------------------------------------------- frames */
async function boot() {
  sites = await (await fetch('/api/sites')).json();
  flags = await (await fetch('/api/flags')).json();
  buildKeyPanel(); paintKeys();
  if (samPartition) {
    $('#region-legend').hidden = false;
    for (const id of ['sam-toggle', 'unsure', 'keepall', 'flag', 'note']) $('#' + id).hidden = true;
    $('#reset').dataset.hint = 'Reset this draft to the SAM labels.';
  }
  if (queueKind) {
    reviewingSample = true;
    $('#sam-toggle').hidden = true;
    $('#test-advance').hidden = !testReview;
    try {
      const response = await fetch(`/api/${queueKind}-queue`);
      if (!response.ok) throw new Error('Review queue unavailable');
      reviewQueue = await response.json();
      renderSites();
      const next = reviewQueue.find(frame => !frame.done);
      if (next) return load(next.i, false, next.site);
      finishQueue();
    } catch (e) {
      msg('Could not load review queue. Refresh to retry.', true);
    }
    return;
  }
  if (practice) {
    $('#save').hidden = true;
    $('#flag').hidden = true;
    $('#note').hidden = true;
    $('#keepall').hidden = true;
  }
  if (sample.length && sites.some(z => z.site === sampleSite) &&
      sample.every(i => Number.isInteger(i) && i >= 0 && i < sites.find(z => z.site === sampleSite).n)) {
    reviewingSample = true;
    hide = true;
    $('#hide').classList.add('on');
    $('#filter').innerHTML = samPartition ?
      'Next validation frames<br>' +
      '<a href="/validation?site=ABEC_CM" style="color:inherit">ABEC_CM</a> · ' +
      '<a href="/validation?site=HL_BM" style="color:inherit">HL_BM</a> · ' +
      '<a href="/validation?site=MEB_HW_A" style="color:inherit">MEB_HW_A</a><br>' +
      '<a href="/?site=ABEC_CM&sample=35,41,50,62,74,86,95,99" style="color:inherit">ABEC_CM next</a> · ' +
      '<a href="/?site=HL_BM&sample=1,16,29,46,59,78,89,99" style="color:inherit">HL_BM next</a> · ' +
      '<a href="/?site=MEB_HW_A&sample=21,27,31,46,57,73,88,99" style="color:inherit">MEB_HW_A next</a><br>' +
      '<a href="/?site=ABEC_CM&sample=32,44,56,68,80,92" style="color:inherit">ABEC_CM</a> · ' +
      '<a href="/?site=HL_BM&sample=8,22,38,52,68,92" style="color:inherit">HL_BM</a> · ' +
      '<a href="/?site=MEB_HW_A&sample=22,38,52,68,82,96" style="color:inherit">MEB_HW_A</a><br>' +
      '<a href="/?site=ABEC_RM&sample=12,36,60,84" style="color:inherit">ABEC_RM</a> · ' +
      '<a href="/?site=BT_BM&sample=12,36,60,84" style="color:inherit">BT_BM</a> · ' +
      '<a href="/?site=HB_MM&sample=12,36,60,84" style="color:inherit">HB_MM</a><br>' +
      '<a href="/?site=BT_LM&sample=28,76" style="color:inherit">BT_LM check</a> · ' +
      '<a href="/?site=FL_LM&sample=28,76" style="color:inherit">FL_LM check</a><br>' +
      '<a href="/?site=HDCL_HW_B&sample=28,76" style="color:inherit">HDCL_HW_B check</a><br>' +
      '<a href="/?site=BT_LM&sample=16,40,64,88" style="color:inherit">BT_LM batch</a> · ' +
      '<a href="/?site=FL_LM&sample=16,40,64,88" style="color:inherit">FL_LM batch</a><br>' +
      '<a href="/?site=HDCL_HW_B&sample=16,40,64,88" style="color:inherit">HDCL_HW_B batch</a><br>' +
      '<a href="/?site=HB_LM&sample=16,40,64,88" style="color:inherit">Latest HB_LM drafts</a><br>' +
      '<a href="/?site=HB_LM&sample=8,24,48,72" style="color:inherit">Previous batch</a><br>' +
      '<a href="/?site=HB_LM&sample=12,28,52" style="color:inherit">Adjusted prelabels</a><br>' +
      '<a href="/?site=HB_LM&sample=20,84" style="color:inherit">Paper drafts</a> · ' +
      '<a href="/?site=MEB_HW_A&sample=19" style="color:inherit">Cable draft</a>' :
      '<a href="/" style="color:inherit">Open all ISOD frames</a>';
    return openSite(sampleSite, sample[0]);
  }
  let s = sites[0].site, i = 0;
  try {                                      // come back to where this browser left off
    const at = (localStorage.getItem('review_at') || '').split('|');
    if (sites.some(z => z.site === at[0])) { s = at[0]; i = Math.max(0, +at[1] || 0); }
  } catch (e) {}
  openSite(s, i);
}
function renderSites() {
  const l = $('#sites'); l.innerHTML = '';
  if (queueKind) {
    const done = reviewQueue.filter(frame => frame.done).length;
    $('#filter').textContent = `${done}/${reviewQueue.length} ${queueKind} frames saved`;
    for (const room of new Set(reviewQueue.map(frame => frame.site))) {
      const queued = reviewQueue.filter(frame => frame.site === room);
      const saved = queued.filter(frame => frame.done).length;
      const d = document.createElement('div');
      d.className = 'site' + (room === site ? ' sel' : '') + (saved === queued.length ? ' done' : '');
      d.textContent = `${room} · ${saved}/${queued.length}`;
      d.onclick = () => load((queued.find(frame => !frame.done) || queued[0]).i, false, room);
      l.appendChild(d);
    }
    return;
  }
  sites.filter(s => !reviewingSample || s.site === sampleSite).forEach(s => {
    const d = document.createElement('div');
    const done = reviewingSample ? sample.filter(i => frames[i]?.done).length : s.done;
    const total = reviewingSample ? sample.length : s.n;
    d.className = 'site' + (s.site === site ? ' sel' : '') + (done === total ? ' done' : '');
    d.innerHTML = `<span>${s.site}</span><span class="n">${done}/${total}</span>`;
    if (!reviewingSample) d.onclick = () => openSite(s.site, 0);
    l.appendChild(d);
  });
  const sel = l.querySelector('.site.sel');
  if (sel) sel.scrollIntoView({block: 'nearest'});
}
async function openSite(s, i) {
  if (cutting || drawing || loadingFrame) return;
  if (dirty && !await save(true)) return;
  site = s;
  frames = await (await fetch(`/api/frames?site=${s}`)).json();
  if (reviewingSample) i = sample.find(frame => !frames[frame].done) ?? i;
  renderSites();
  load(i);
}
function renderStrip() {
  const only = $('#only')?.checked, l = $('#strip'); l.innerHTML = '';
  frames.forEach((f, i) => {
    if (reviewingSample && !sample.includes(i)) return;
    if (only && f.done && !flags[`${site}_${i}`] && !f.check) return;
    const d = document.createElement('div');
    d.className = 'f' + (f.done ? ' done' : '') + (i === idx ? ' sel' : '') +
                  (flags[`${site}_${i}`] ? ' flag' : '');
    d.textContent = reviewingSample || i % 10 === 0 ? i : '';
    d.title = `frame ${i}` + (f.check ? ` · ${f.check} to check` : '');
    d.onclick = () => load(i);
    l.appendChild(d);
  });
  const sel = l.querySelector('.f.sel');
  if (sel) sel.scrollIntoView({block: 'nearest'});
}
async function load(i, accept, room = site) {
  if (loadingFrame || cutting || drawing || saving) return;
  loadingFrame = true;
  try {
    /* Stepping confirms the current frame. Jumping with the mouse only saves edits. */
    if ((dirty || accept) && comp && !await save(true)) return;
    const response = await fetch(`/api/frame?site=${room}&i=${i}${practiceQuery}`);
    if (!response.ok) throw new Error('Frame unavailable');
    const r = await response.json();
    const nextFrames = room === site ? frames : await (await fetch(`/api/frames?site=${room}`)).json();
    const nextImage = new Image();
    await new Promise((resolve, reject) => {
      nextImage.onload = resolve;
      nextImage.onerror = reject;
      nextImage.src = `/rgb/${room}/${i}.png`;
    });
    if (queueKind) {
      site = room;
      frames = nextFrames;
      sample = reviewQueue.filter(frame => frame.site === room).map(frame => frame.i);
      renderSites();
    }
    idx = i; undo = []; unsureAt = 0;
    rows = r.rows; W = r.w; H = r.h;
    comp = decodeComponents(r);
    cut = r.cut ? decodeBytes(r.cut) : new Uint8Array(W * H);
    const samBytes = r.sam ? decodeBytes(r.sam) : null;
    samEdits = r.sam16 ? new Uint16Array(samBytes.buffer) : samBytes || new Uint8Array(W * H);
    suggestions = r.suggestions.map(s => ({...s, mask: decodeBytes(s.mask)}));
    suggestionAt = -1;
    updateSuggestions();
    img = nextImage;
    if (reviewingSample) hide = !samPartition;
    $('#hide').classList.toggle('on', hide);
    $('#title').textContent = `${site}  frame ${i}` +
      (reviewingSample ? ` · sample ${sample.indexOf(i) + 1}/${sample.length}` : '') +
      (samPartition ? ' · Draft review' : practice ? ' · Practice' : '');
    const fl = flags[`${site}_${i}`];
    $('#flag').classList.toggle('on', !!fl);
    $('#note').value = fl ? fl.note : '';
    if (!reviewingSample) {
      try { localStorage.setItem('review_at', `${site}|${i}`); } catch (e) {}
    }
    dirty = false; renderStrip(); fit();
    if (samPartition) {
      $('#reference-mask').href = `/reference/${site}/${i}.png`;
      $('#reference-mask').hidden = !r.reference;
      $('#source-mask').href = `/source/${site}/${i}.png`;
      $('#region-prev').disabled = queueKind ? reviewQueue[0]?.site === room && reviewQueue[0]?.i === i :
                                                 sample.indexOf(i) <= 0;
      $('#region-next').textContent = !queueKind && sample.at(-1) === i ? 'Save draft' : 'Save and next';
    }
  } catch (e) {
    msg('Could not load frame. Try opening it again.', true);
  } finally {
    loadingFrame = false;
  }
}
function fit() {
  if (!W) return;
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  view.z = Math.min(w / W, h / H);
  view.ox = (w - W * view.z) / 2; view.oy = (h - H * view.z) / 2;
  draw();
}
let overlay = document.createElement('canvas'), octx = null;
function buildOverlay() {
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx = octx || overlay.getContext('2d', {willReadFrequently: true});
  const d = octx.createImageData(W, H), colour = {};
  rows.forEach(r => colour[r.id] = r.keep ? KEEP : DROP);
  const checkFloor = samPartition ? new Set(rows.filter(r => r.fresh).map(r => r.id)) : null;
  const suggestion = suggestions[suggestionAt];
  for (let p = 0, k = 0; p < comp.length; p++, k += 4) {
    const active = suggestedPixel(suggestion, p);
    const id = comp[p];
    const c = active ? [232, 127, 255] : colour[id];
    if (!c) continue;
    const edge = samPartition && (p % W === 0 || comp[p - 1] !== id ||
      p % W === W - 1 || comp[p + 1] !== id || comp[p - W] !== id || comp[p + W] !== id);
    const tint = edge && checkFloor.has(id) ? CHECK : c;
    d.data[k] = tint[0]; d.data[k+1] = tint[1]; d.data[k+2] = tint[2];
    d.data[k+3] = edge ? 235 : active ? 175 : suggestion ? 42 : samPartition && c === DROP ? 86 : 130;
  }
  octx.putImageData(d, 0, 0);
  // a bright edge on whatever the rule was unsure about: that is what needs looking at
  octx.lineWidth = 1; octx.strokeStyle = `rgb(${CHECK.join(',')})`;
  if (!samPartition) rows.filter(r => r.fresh).forEach(r => {
    const [x0, y0, x1, y1] = r.box;
    octx.strokeRect(x0 - 1.5, y0 - 1.5, x1 - x0 + 3, y1 - y0 + 3);
  });
}
function draw() {
  if (!img) return;
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  cv.width = w; cv.height = h;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.save(); ctx.translate(view.ox, view.oy); ctx.scale(view.z, view.z);
  ctx.drawImage(img, 0, 0);
  if (!hide) { buildOverlay(); ctx.drawImage(overlay, 0, 0); }
  ctx.restore();
  if (mouse && tool === 'cut') {                 // the cursor is the brush
    const r = Math.max(2, size / 2 * view.z);
    const x = mouse[0] * view.z + view.ox, y = mouse[1] * view.z + view.oy;
    ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(0,0,0,.55)';
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
  }
  const keep = rows.filter(r => r.keep).length, check = rows.filter(r => r.fresh).length;
  $('#counts').textContent =
    `${rows.length} comp · ${keep} keep · ${rows.length - keep} drop` +
    (check ? ` · ${check} to check` : '') + (dirty ? ' · unsaved' : '');
}
const toImg = e => {
  const r = cv.getBoundingClientRect();
  return [Math.floor((e.clientX - r.left - view.ox) / view.z),
          Math.floor((e.clientY - r.top - view.oy) / view.z)];
};

/* ------------------------------------------------------------------ edit */
cv.addEventListener('mousedown', e => {
  if (loadingFrame || cutting || saving) return;
  if (e.button === 1 || panKey || tool === 'pan') {
    panning = true; last = [e.clientX, e.clientY];
    if (tool === 'pan') cv.style.cursor = 'grabbing';
    return;
  }
  const [x, y] = toImg(e);
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  if (e.button !== 0) return;
  if (testReview && hide) return msg('Press H to show the editable overlay.', true);
  if (suggestionAt >= 0 && !hide) {
    if (suggestedPixel(suggestions[suggestionAt], y * W + x)) applySuggestion();
    return;
  }
  if (tool === 'cut') { undo.push({comp: comp.slice(), cut: cut.slice(), samEdits, rows, dirty}); drawing = true;
                        paintCut(x, y); return; }
  const id = comp[y * W + x];
  if (!id) return;
  const r = rows.find(z => z.id === id);
  undo.push({keep: rows.map(z => z.keep), fresh: rows.map(z => z.fresh)});
  r.keep = !r.keep;
  if (samPartition) r.fresh = false;
  dirty = true; draw();
  if (samPartition) msg(r.keep ? 'Region kept' : 'Region marked as background');
});
window.addEventListener('mousemove', e => {
  if (panning) { view.ox += e.clientX - last[0]; view.oy += e.clientY - last[1];
                 last = [e.clientX, e.clientY]; draw(); return; }
  mouse = toImg(e);
  if (drawing) paintCut(mouse[0], mouse[1]);
  else if (tool === 'cut') draw();
});
window.addEventListener('mouseup', () => {
  if (drawing) { drawing = false; recut(); }
  if (panning && tool === 'pan') cv.style.cursor = 'grab';
  panning = false;
});
cv.addEventListener('mouseleave', () => { mouse = null; draw(); });
function paintCut(cx, cy) {
  const r = Math.max(1, size / 2);
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(H - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(W - 1, Math.ceil(cx + r)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) { cut[y * W + x] = 1; comp[y * W + x] = 0; }
  dirty = true; draw();
}
async function recut() {
  const previous = undo.at(-1);
  const result = await previewCut(cut, samEdits);
  if (!result) {
    undo.pop();
    ({comp, cut, samEdits, rows, dirty} = previous);
    draw();
    return;
  }
  const nextComp = decodeComponents(result);
  carryUnchanged(previous, nextComp, result.rows);
  if (!samPartition) {
    const oldSam = new Map(previous.rows.filter(row => row.sam).map(row => [row.sam, row.keep]));
    for (const row of result.rows) {
      if (oldSam.has(row.sam)) { row.keep = oldSam.get(row.sam); row.fresh = false; }
    }
  }
  rows = result.rows;
  comp = nextComp;
  dirty = true; draw();
  msg(`re-split into ${rows.length} components`);
}
function carryUnchanged(previous, nextComp, nextRows) {
  const oldRows = new Map(previous.rows.map(row => [row.id, row]));
  const parents = new Map();
  for (let p = 0; p < nextComp.length; p++) {
    const id = nextComp[p];
    if (!id) continue;
    const parent = previous.comp[p];
    if (!parents.has(id)) parents.set(id, parent);
    else if (parents.get(id) !== parent) parents.set(id, 0);
  }
  for (const row of nextRows) {
    const parent = oldRows.get(parents.get(row.id));
    if (parent && (samPartition || parent.area === row.area)) {
      row.keep = parent.keep;
      row.fresh = parent.fresh;
      row.why = parent.why;
    } else {
      row.fresh = true;
    }
  }
}
async function previewCut(value, sam) {
  cutting = true;
  try {
    const body = new Uint8Array(value.length + sam.byteLength);
    body.set(value);
    body.set(new Uint8Array(sam.buffer, sam.byteOffset, sam.byteLength), value.length);
    const response = await fetch(`/api/recut?site=${site}&i=${idx}&preview=1${practiceQuery}`,
                                 {method: 'POST', body});
    if (!response.ok) throw new Error('split failed');
    return await response.json();
  } catch (e) {
    msg('Split failed. Try again.', true);
    return null;
  } finally {
    cutting = false;
  }
}
function suggestedPixel(suggestion, p) {
  return suggestion && !cut[p] && !samEdits[p] &&
    (suggestion.mask[p >> 3] & (128 >> (p & 7)));
}
function updateSuggestions() {
  const suggestion = suggestions[suggestionAt];
  $('#sam-toggle').disabled = !suggestions.length;
  $('#sam-toggle').title = suggestions.length ? '' : 'No cached SAM regions for this frame';
  $('#sam-toggle').classList.toggle('on', Boolean(suggestion));
  $('#sam-toggle').setAttribute('aria-expanded', Boolean(suggestion));
  $('#sam-bar').hidden = !suggestion;
  if (!suggestion) return;
  $('#sam-label').textContent =
    `${suggestionAt + 1}/${suggestions.length} · ${suggestion.prompt} · ${suggestion.score.toFixed(2)}`;
  $('#sam-apply').textContent = suggestion.keep ? 'Keep region' : 'Mark as background';
}
function showSuggestion(index) {
  if (loadingFrame || cutting || drawing || saving || !suggestions.length) return;
  suggestionAt = index < 0 ? -1 : index % suggestions.length;
  if (suggestionAt >= 0) {
    if (tool) setTool(tool);
    hide = false;
    $('#hide').classList.remove('on');
  }
  updateSuggestions();
  draw();
}
async function applySuggestion() {
  const suggestion = suggestions[suggestionAt];
  if (!suggestion || loadingFrame || cutting || drawing || saving) return;
  const sam = samEdits.slice(), used = new Set(sam);
  let id = 1;
  while (used.has(id)) id++;
  if (id > 255) return msg('Too many SAM regions', true);
  let changed = false;
  for (let p = 0; p < sam.length; p++) {
    if (suggestedPixel(suggestion, p)) { sam[p] = id; changed = true; }
  }
  if (!changed) return msg('This region is already assigned');
  const result = await previewCut(cut, sam);
  if (!result) return;
  const decisions = new Map(rows.map(row => [row.sam ? `sam-${row.sam}` : row.id, row]));
  for (const row of result.rows) {
    if (row.sam === id) {
      row.keep = suggestion.keep;
      continue;
    }
    const prior = decisions.get(row.sam ? `sam-${row.sam}` : row.id);
    if (prior) { row.keep = prior.keep; row.fresh = prior.fresh; }
  }
  undo.push({comp, cut, samEdits, rows});
  comp = decodeComponents(result);
  rows = result.rows;
  samEdits = sam;
  dirty = true;
  showSuggestion(-1);
  msg(suggestion.keep ? 'SAM region kept' : 'SAM region marked as background');
}
function nextUnsure() {
  const list = rows.filter(r => r.fresh);
  if (!list.length) {
    const j = frames.findIndex((f, i) =>
      i > idx && f.check && (!reviewingSample || sample.includes(i)));
    if (j < 0) return msg('nothing left to check in this site');
    return load(j);
  }
  const r = list[unsureAt++ % list.length];
  const [x0, y0, x1, y1] = r.box;
  view.z = Math.max(view.z, Math.min(8, 300 / Math.max(8, x1 - x0, y1 - y0)));
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  view.ox = w / 2 - (x0 + x1) / 2 * view.z;
  view.oy = h / 2 - (y0 + y1) / 2 * view.z;
  draw();
  msg(`${r.area} px · ${r.why}`);
}

/* -------------------------------------------------------------------- ui */
function setTool(t) {
  tool = tool === t ? null : t;
  ['cut', 'pan'].forEach(x => $('#' + x).classList.toggle('on', tool === x));
  cv.style.cursor = tool === 'pan' ? 'grab' : 'default';
  draw();
}
$('#sam-toggle').onclick = () => showSuggestion(suggestionAt < 0 ? 0 : -1);
$('#sam-prev').onclick = () => showSuggestion(suggestionAt - 1 + suggestions.length);
$('#sam-next').onclick = () => showSuggestion(suggestionAt + 1);
$('#sam-apply').onclick = applySuggestion;
$('#region-prev').onclick = () => run('prev');
$('#region-next').onclick = () => !queueKind && sample.at(-1) === idx ? save() : run('next');
$('#test-advance').onclick = () => run('next');
$('#cut').onclick = () => setTool('cut');
$('#pan').onclick = () => setTool('pan');
$('#unsure').onclick = nextUnsure;
$('#keepall').onclick = () => {
  if (loadingFrame || cutting || drawing || saving) return;
  undo.push({keep: rows.map(r => r.keep)});
  rows.forEach(r => r.keep = true);
  dirty = true; draw();
};
$('#reset').onclick = async () => {
  if (loadingFrame || cutting || drawing || saving) return;
  if (practice) { dirty = false; return load(idx); }
  await fetch(`/api/reset?site=${site}&i=${idx}`, {method: 'POST'});
  frames[idx].done = false; dirty = false; load(idx); msg('back to the proposal');
};
$('#hide').onclick = () => { hide = !hide; $('#hide').classList.toggle('on', hide); draw(); };
$('#fit').onclick = fit;
$('#save').onclick = () => save();
$('#only').onchange = renderStrip;
$('#help').onclick = () => $('#keys').classList.add('on');
$('#kclose').onclick = () => $('#keys').classList.remove('on');
$('#kreset').onclick = () => {
  KEYS = Object.assign({}, DEFAULTS);
  MODES = {}; HOLDABLE.forEach(a => MODES[a] = MODE_DEF[a] || 'toggle');
  localStorage.setItem('review_keys', JSON.stringify(KEYS));
  localStorage.setItem('review_modes', JSON.stringify(MODES));
  buildKeyPanel(); paintKeys();
};
$('#keys').onclick = e => { if (e.target.id === 'keys') $('#keys').classList.remove('on'); };
/* pressing the button must not blur the note first: its change event would flag the frame
   and the click would then toggle that straight back off */
$('#flag').onmousedown = e => e.preventDefault();
$('#flag').onclick = () => setFlag(!flags[`${site}_${idx}`], $('#note').value);
$('#note').onchange = () => {
  const want = $('#note').value !== '' || !!flags[`${site}_${idx}`];
  if (want || flags[`${site}_${idx}`]) setFlag(want, $('#note').value);
};
$('#note').onkeydown = e => { if (e.key === 'Enter') $('#note').blur(); };
function setFlag(on, note) {
  if (practice) return;
  const k = `${site}_${idx}`;
  if (on) flags[k] = {note: note || ''}; else delete flags[k];
  $('#flag').classList.toggle('on', on);
  $('#note').value = on ? (note || '') : '';
  renderStrip();
  fetch(`/api/flag?site=${site}&i=${idx}`, {method: 'POST',
        body: JSON.stringify({flag: on, note: note || ''})});
}
$('#wrap').addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15, nz = Math.min(24, Math.max(0.2, view.z * k));
  view.ox = mx - (mx - view.ox) * (nz / view.z);
  view.oy = my - (my - view.oy) * (nz / view.z);
  view.z = nz; draw();
}, {passive: false});

async function save(quiet) {
  if (practice) return true;
  if (cutting || drawing || saving || !cut) return false;
  if (testReview && hide && !frames[idx].done) {
    msg('Show and verify the overlay before saving.', true);
    return false;
  }
  const keep = rows.filter(r => r.keep).map(r => r.id);
  saving = true;
  try {
    const response = await fetch(`/api/save?site=${site}&i=${idx}`,
      {method: 'POST', body: JSON.stringify({keep, cut: encodeBytes(cut), sam: encodeBytes(samEdits),
                                            sam16: samEdits.BYTES_PER_ELEMENT === 2,
                                            separateIslands: samPartition})});
    if (response.status === 409) {
      msg('Reload the updated reviewer before editing.', true);
      return false;
    }
    if (!response.ok) throw new Error('save failed');
  } catch (e) {
    msg('Save failed. Try again before leaving.', true);
    return false;
  } finally {
    saving = false;
  }
  dirty = false;
  if (samPartition) msg('Draft saved');
  frames[idx].done = true;
  if (queueKind) reviewQueue.find(frame => frame.site === site && frame.i === idx).done = true;
  frames[idx].check = rows.filter(r => r.fresh).length;
  const s = sites.find(z => z.site === site);
  if (s) s.done = frames.filter(f => f.done).length;
  renderStrip(); renderSites(); draw();
  if (!quiet) { $('#save').classList.add('on');
                setTimeout(() => $('#save').classList.remove('on'), 300); }
  return true;
}

window.addEventListener('keydown', e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && !listening) return;
  if (listening) {
    e.preventDefault();
    if (e.key !== 'Escape' && e.key.length === 1) setKey(listening.action, e.key);
    else buildKeyPanel();
    listening = null; return;
  }
  if (e.key === ' ') { panKey = true; e.preventDefault(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault();
    if (cutting || drawing || loadingFrame || saving) return;
    const u = undo.pop();
    if (!u) return;
    if (u.keep) rows.forEach((r, i) => {
      r.keep = u.keep[i];
      if (u.fresh) r.fresh = u.fresh[i];
    });
    else { ({comp, cut, samEdits, rows} = u); }
    dirty = true; draw(); return;
  }
  if (e.metaKey || e.ctrlKey) return;
  if (e.key === 'Escape') { $('#keys').classList.remove('on'); showSuggestion(-1); return; }
  const act = Object.keys(KEYS).find(a => KEYS[a] && KEYS[a] === e.key);
  if (!act) return;
  e.preventDefault();
  if (HOLDABLE.includes(act) && MODES[act] === 'hold') {
    if (e.repeat || held) return;
    held = {act: act, key: e.key, tool: tool, hide: hide};
    if (act === 'hide') { hide = true; draw(); } else setTool(act);
    return;
  }
  run(act);
});
function finishQueue() {
  $('#title').textContent = `${queueTitle} complete`;
  $(trainingReview ? '#region-next' : '#test-advance').disabled = true;
}
function run(act) {
  if (act === 'cut' || act === 'pan') setTool(act);
  else if (act === 'sizeDown') { size = Math.max(1, size - 1); $('#sz').textContent = size; draw(); }
  else if (act === 'sizeUp') { size = Math.min(40, size + 1); $('#sz').textContent = size; draw(); }
  else if (act === 'unsure') nextUnsure();
  else if (act === 'keepAll') $('#keepall').click();
  else if (act === 'reset') $('#reset').click();
  else if (act === 'hide') $('#hide').click();
  else if (act === 'fit') fit();
  else if (act === 'flag') $('#flag').click();
  else if (queueKind && (act === 'next' || act === 'prev')) {
    if (loadingFrame || cutting || drawing || saving || !comp) return;
    if (act === 'prev') {
      const previous = reviewQueue[reviewQueue.findIndex(frame => frame.site === site && frame.i === idx) - 1];
      if (previous) load(previous.i, false, previous.site);
    } else {
      save(true).then(ok => {
        if (!ok) return;
        const next = reviewQueue.find(frame => !frame.done);
        if (next) load(next.i, false, next.site);
        else {
          finishQueue();
          msg('All frames in this batch are saved.');
        }
      });
    }
  }
  else if (reviewingSample && (act === 'next' || act === 'prev')) {
    if (loadingFrame) return;
    if (!practice && hide && act === 'next') return msg('Show and verify the overlay before saving.', true);
    const next = sample[sample.indexOf(idx) + (act === 'next' ? 1 : -1)];
    if (next !== undefined) load(next, true);
    else if (act === 'next' && !practice) save(true).then(ok => { if (ok) msg('Last sample frame saved'); });
  }
  else if (act === 'next') {
    if (idx + 1 < frames.length) load(idx + 1, true);
    else { const j = sites.findIndex(z => z.site === site);
           save(true).then(ok => { if (!ok) return;
                                   if (j + 1 < sites.length) openSite(sites[j+1].site, 0);
                                   else msg('that was the last site'); }); }
  }
  else if (act === 'prev') { if (idx > 0) load(idx - 1, true); }
}
window.addEventListener('keyup', e => {
  if (e.key === ' ') panKey = false;
  if (held && e.key === held.key) {
    if (held.act === 'hide') { hide = held.hide; draw(); } else setTool(held.tool || held.act);
    held = null;
  }
});
window.addEventListener('resize', fit);
document.querySelectorAll('.tool[data-hint]').forEach(el => {
  el.addEventListener('mouseenter', () => {
    el.classList.toggle('hr', el.getBoundingClientRect().left + 300 > window.innerWidth);
  });
});
boot();
</script>
"""


FLAGS = f"{OUT}/flags.json"


def read_flags():
    try:
        return json.load(open(FLAGS))
    except Exception:
        return {}


def out_dir(site):
    d = f"{OUT}/{site}"
    os.makedirs(d, exist_ok=True)
    return d


def comp_paths(site, i):
    return (f"{comps.OUT}/{site}/{i}_comp.png", f"{comps.OUT}/{site}/{i}.json")


def base_frame(site, i):
    cpath, jpath = comp_paths(site, i)
    comp = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)
    with open(jpath) as f:
        return comp, json.load(f)


def read_edit(site, i, suffix):
    path = f"{OUT}/{site}/{i}_{suffix}.png"
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None


def resplit(site, i, cut):
    """Split the frame again with the hand cut removed from the mask."""
    _, _, label, _ = isod.frame(site, i)
    label = np.array(label).copy()
    label[cut] = 255                      # a cut pixel is treated as floor, so the mask parts
    comp, rows = comps.frame_components_from_label(label)
    return comp.astype(np.uint8), rows


def region_geometry(mask):
    ys, xs = np.nonzero(mask)
    return dict(area=len(xs), box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])


def edit_components(site, i, cut, sam):
    comp, rows = resplit(site, i, cut > 0) if cut is not None and cut.any() else base_frame(site, i)
    if sam is None or not sam.any():
        return comp, rows
    if sam.shape != comp.shape:
        raise ValueError("wrong SAM edit shape")
    if sam.itemsize == 2:
        comp = comp.astype("<u2")
    label = cv2.imread(f"{isod.DATA}/{site}/label/{i}.png", cv2.IMREAD_GRAYSCALE)
    if np.any((sam > 0) & (label != 0)):
        raise ValueError("SAM edit outside ISOD's not-drivable mask")
    regions = sam if cut is None else np.where(cut > 0, 0, sam)
    affected = set(np.unique(comp[regions > 0]))
    result = []
    for row in rows:
        if row["id"] in affected:
            remainder = (comp == row["id"]) & (regions == 0)
            if not remainder.any():
                continue
            row.update(region_geometry(remainder))
        result.append(row)
    next_id = max((row["id"] for row in rows), default=0) + 1
    for value in np.unique(regions[regions > 0]):
        if next_id > np.iinfo(comp.dtype).max:
            raise ValueError("too many regions in frame")
        region = regions == value
        comp[region] = next_id
        result.append(dict(id=next_id, sam=int(value), keep=False, fresh=False,
                           why="SAM suggestion", **region_geometry(region)))
        next_id += 1
    return comp, result


def pack_image(image):
    return base64.b64encode(image.tobytes()).decode() if image is not None else None


def separate_islands(comp, rows):
    """Give each disconnected island its own clickable region and decision."""
    result = np.zeros(comp.shape, dtype="<u2")
    islands = []
    for row in rows:
        x0, y0, x1, y1 = row["box"]
        box = np.s_[y0:y1 + 1, x0:x1 + 1]
        mask = (comp[box] == row["id"]).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        lookup = np.zeros(count, dtype="<u2")
        for island in range(1, count):
            x, y, width, height, area = map(int, stats[island])
            ident = len(islands) + 1
            lookup[island] = ident
            islands.append(dict(row, id=ident, area=area,
                                box=[x + x0, y + y0, x + x0 + width - 1, y + y0 + height - 1]))
        dest = result[box]
        inside = mask > 0
        dest[inside] = lookup[labels[inside]]
    return result, islands


def sam_suggestions(site, i, comp):
    regions = []
    structure = PROMPT_GROUPS["structure"]
    for folder in ("cache", "crop-cache"):
        path = SAM_CACHE / folder / site / f"{i}.npz"
        if path.is_file():
            regions.extend(read_cache(path, comp.shape))
    # Limit proposals to existing components so ISOD still supplies their outer boundary.
    seen = set()
    for prompt, score, mask in sorted(regions, key=lambda item: -item[1]):
        if score < SAM_CONFIDENCE or prompt in structure:
            continue
        mask &= comp > 0
        packed = np.packbits(mask)
        key = packed.tobytes()
        if mask.sum() < comps.MIN_AREA or key in seen:
            continue
        seen.add(key)
        yield dict(prompt=prompt, score=round(score, 2), mask=pack_image(packed))


def flag(rows):
    """Mark the components the rule was not sure about, so the eye goes to them.

    Carrying decisions between frames was the original plan and it does not work here: the
    100 frames of an ISOD site are not a slow camera move but separate captures around a
    room, with a mean absolute difference of ~40 grey levels between neighbours, so a small
    object's mask rarely overlaps its own mask in the next frame. What is left is to make
    each frame quick to scan, and the quickest version of that is to say which of its twelve
    components are the doubtful ones: everything the two clean rules -- reaches the top of
    the frame, or is ringed by floor -- did not settle, plus anything large enough to move
    the pixel IoU on its own.
    """
    for r in rows:
        if "prob" in r or "sam" in r:
            continue  # the learned proposal already includes its uncertainty flag
        r["fresh"] = bool(
            (not r["touches_top"] and r["enclosed"] < comps.ENCLOSED) or
            (r["keep"] and r["area"] > 20000))
    return rows


def frame_state(site, i, practice=False, *, split_islands=False):
    cut = None if practice else read_edit(site, i, "cut")
    sam = None if practice else read_edit(site, i, "sam")
    comp, rows = edit_components(site, i, cut, sam)
    saved = f"{OUT}/{site}/{i}.json"
    if not practice and os.path.exists(saved):
        with open(saved) as f:
            decisions = json.load(f)
        split_islands = split_islands or isinstance(decisions, dict)
        if split_islands:
            comp, rows = separate_islands(comp, rows)
            foreground = cv2.imread(f"{OUT}/{site}/{i}.png", cv2.IMREAD_GRAYSCALE) > 0
            keep = set(np.unique(comp[foreground]))
        else:
            keep = set(decisions)
        for r in rows:
            r["keep"] = r["id"] in keep
            r["fresh"] = False
    else:
        rows = flag(rows)
        if split_islands:
            comp, rows = separate_islands(comp, rows)
    return comp, rows, cut, sam


def write_mask(site, i, keep, edits=None, *, split_islands=False):
    cut, sam = edits if edits is not None else (read_edit(site, i, "cut"), read_edit(site, i, "sam"))
    comp, rows = edit_components(site, i, cut, sam)
    if split_islands:
        comp, rows = separate_islands(comp, rows)
    if not set(keep).issubset(row["id"] for row in rows):
        raise ValueError("unknown component in saved mask")
    folder = out_dir(site)
    mask = np.isin(comp, list(keep)).astype(np.uint8) * 255
    if not cv2.imwrite(f"{folder}/{i}.tmp.png", mask):
        raise OSError("could not save review mask")
    images = () if edits is None else (("cut", cut * np.uint8(255)), ("sam", sam))
    for name, image in images:
        if image.any() and not cv2.imwrite(f"{folder}/{i}_{name}.tmp.png", image):
            raise OSError(f"could not save {name}")
    with open(f"{folder}/{i}.tmp", "w") as f:
        decisions = sorted(keep)
        json.dump({"keep": decisions} if split_islands else decisions, f)
    for name, image in images:
        path = Path(folder) / f"{i}_{name}.png"
        if image.any():
            os.replace(f"{folder}/{i}_{name}.tmp.png", path)
        else:
            path.unlink(missing_ok=True)
    os.replace(f"{folder}/{i}.tmp.png", f"{folder}/{i}.png")
    os.replace(f"{folder}/{i}.tmp", f"{folder}/{i}.json")


class Handler(http.server.BaseHTTPRequestHandler):
    timeout = 10
    write_lock = threading.Lock()
    split_islands = False

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/test"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/test-queue":
            return self._json([
                dict(site=site, i=i, done=all((Path(OUT) / site / f"{i}{suffix}").is_file()
                                             for suffix in (".json", ".png")))
                for site in isod.load_splits()["test"] for i in range(4, N_FRAMES, 10)
            ])
        if u.path == "/api/sites":
            return self._json([dict(site=s, n=N_FRAMES,
                                    done=len([f for f in os.listdir(out_dir(s))
                                              if f.endswith(".json")]))
                               for s in isod.sites()])
        if u.path == "/api/flags":
            return self._json(read_flags())
        if u.path == "/api/frames":
            s = q["site"][0]
            out = []
            for i in range(N_FRAMES):
                done = os.path.exists(f"{out_dir(s)}/{i}.json")
                check = 0 if done else sum(
                    r["fresh"] for r in flag(json.load(open(comp_paths(s, i)[1]))))
                out.append(dict(i=i, done=done, check=check))
            return self._json(out)
        if u.path == "/api/frame":
            s, i = q["site"][0], int(q["i"][0])
            comp, rows, cut, sam = frame_state(s, i, practice=q.get("practice") == ["1"])
            return self._json(dict(w=comp.shape[1], h=comp.shape[0], rows=rows,
                                   comp=pack_image(comp), comp16=comp.itemsize == 2,
                                   cut=pack_image(cut), sam=pack_image(sam),
                                   sam16=sam is not None and sam.itemsize == 2,
                                   suggestions=[] if s in isod.load_splits()["test"] else
                                   list(sam_suggestions(s, i, comp))))
        if u.path.startswith("/rgb/"):
            site, name = u.path[len("/rgb/"):].split("/")
            p = f"{isod.DATA}/{site}/rgb/{name}"
            if not os.path.exists(p):
                return self._send(404, b"no")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "image/png")
        self._send(404, b"no")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        s, i = q["site"][0], int(q["i"][0])
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with self.write_lock:
            self.handle_edit(u.path, q, s, i, body)

    def handle_edit(self, path, q, s, i, body):
        if q.get("practice") == ["1"] and (path != "/api/recut" or q.get("preview") != ["1"]):
            return self._send(403, b"practice view")
        if path == "/api/flag":
            body = json.loads(body or b"{}")
            fl = read_flags()
            key = f"{s}_{i}"
            if body.get("flag"):
                fl[key] = {"note": body.get("note", "")}
            else:
                fl.pop(key, None)
            os.makedirs(OUT, exist_ok=True)
            json.dump(fl, open(FLAGS, "w"), indent=1, sort_keys=True)
            return self._send(200, b"ok", "text/plain")
        if path == "/api/save":
            try:
                payload = json.loads(body)
                if self.split_islands and (not isinstance(payload, dict) or not payload.get("separateIslands")):
                    return self._send(409, b"Reload the reviewer before saving")
                if isinstance(payload, list):
                    write_mask(s, i, payload)
                else:
                    shape = cv2.imread(comp_paths(s, i)[0], cv2.IMREAD_GRAYSCALE).shape
                    def decode(encoded, dtype=np.uint8):
                        if not encoded:
                            return np.zeros(shape, dtype=dtype)
                        return np.frombuffer(base64.b64decode(encoded, validate=True), dtype).reshape(shape)
                    cut = decode(payload["cut"]) > 0
                    sam_type = "<u2" if payload.get("sam16") else np.uint8
                    sam = (decode(payload["sam"], sam_type) if "sam" in payload else read_edit(s, i, "sam"))
                    if sam is None:
                        sam = np.zeros(shape, dtype=np.uint8)
                    write_mask(s, i, payload["keep"], (cut, sam), split_islands=self.split_islands)
            except (ValueError, TypeError, KeyError):
                return self._send(400, b"invalid mask")
            except OSError:
                return self._send(500, b"could not save mask")
            return self._send(200, b"ok", "text/plain")
        if path == "/api/recut":
            shape = cv2.imread(comp_paths(s, i)[0], cv2.IMREAD_GRAYSCALE).shape
            raw = np.frombuffer(body, np.uint8)
            count = np.prod(shape)
            if raw.size not in (count, 2 * count, 3 * count):
                return self._send(400, b"invalid edit")
            cut = raw[:count].reshape(shape) > 0
            sam = raw[count:].reshape(shape) if raw.size == 2 * count else read_edit(s, i, "sam")
            if raw.size == 3 * count:
                sam = raw[count:].view("<u2").reshape(shape)
            try:
                comp, rows = edit_components(s, i, cut, sam)
            except ValueError:
                return self._send(400, b"invalid edit")
            if q.get("preview") != ["1"]:
                cv2.imwrite(f"{out_dir(s)}/{i}_cut.png", cut.astype(np.uint8) * 255)
            rows = flag(rows)
            if self.split_islands:
                comp, rows = separate_islands(comp, rows)
            return self._json(dict(rows=rows, comp=pack_image(comp), comp16=comp.itemsize == 2))
        if path == "/api/reset":
            for suffix in (".json", ".png", "_cut.png", "_sam.png"):
                (Path(OUT) / s / f"{i}{suffix}").unlink(missing_ok=True)
            return self._send(200, b"ok", "text/plain")
        self._send(404, b"no")


def free_port(start):
    for p in range(start, start + 40):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", p)):
                return p
    raise SystemExit("no free port")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else free_port(PORT)
    print(f"{len(isod.sites())} sites x {N_FRAMES} frames · masks -> {OUT}\n"
          f"  http://localhost:{port}\n")
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
