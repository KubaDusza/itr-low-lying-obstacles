"""Review which parts of ISOD's not-drivable mask are low-lying floor obstacles.

    python code/review_isod.py          # then open the URL it prints

ISOD labels every pixel drivable or not. The not-drivable class holds the walls, the
furniture standing on the floor and the objects lying on it; only the last are what this
project detects. isod_components.py splits that mask into components and proposes an answer
for each. This is where the answer is decided.

The rule being applied: **keep anything lying on the floor** -- cables, cloths, mats, shoes,
bags, a vacuum lying down. **Drop anything standing on the floor as structure** -- chairs,
tables, cabinets, bins, walls, doors.

The mask is ISOD's and stays ISOD's. Nothing here adds a pixel outside it and no boundary is
redrawn; a component is kept or dropped whole. The one exception is the cut brush, which
erases *within* the mask to part an object from the furniture it touches, so that the object
becomes a component of its own and can then be kept. That is the merging failure the derived
ground truth suffers from, and cutting is the only way to fix it by hand.

Each frame is judged on its own: the 100 frames of an ISOD site are separate captures around
a room, not a slow camera move, so a decision cannot be carried from one to the next. What
the tool does instead is say which components the rule was unsure about -- outlined in
yellow -- so a frame whose proposal is plainly right takes one keystroke.

Writes data/ISOD_gt2/<site>/<i>.png, the corrected obstacle mask, plus <i>.json recording
which component ids were kept. data/ISOD_gt stays untouched.
"""
import http.server
import json
import os
import socket
import socketserver
import sys
import urllib.parse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isod                                                       # noqa: E402
import isod_components as comps                                   # noqa: E402

OUT = os.path.join(os.path.dirname(isod.GT), "ISOD_gt2")
PORT = int(os.environ.get("REVIEW_PORT", 8810))
N_FRAMES = 100
CARRY_IOU = 0.3          # overlap at which a component is the same object as last frame's

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
    <span class="tool" id="fit" data-hint="Fit the frame to the window.">fit<span class="k" data-k="fit"></span></span>
    <span class="tool" id="flag" data-hint="Mark this frame to come back to, with a note if you want one. Flagged frames carry an amber ring in the strip below and the filter on the left can show only those.">flag<span class="k" data-k="flag"></span></span>
    <input id="note" placeholder="note" spellcheck="false">
    <span class="tool" id="save" data-hint="Write this frame's mask. Also happens when you step to the next frame.">save<span class="k">⌘s</span></span>
    <span class="tool" id="help" data-hint="Change any shortcut.">?</span>
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
let img = null, comp = null, rows = [], W = 0, H = 0;
let view = {z: 1, ox: 0, oy: 0}, hide = false, dirty = false;
let tool = null, size = 4, drawing = false, panning = false, last = null, panKey = false;
let cut = null, undo = [], held = null, listening = null, mouse = null, unsureAt = 0;
const cv = $('#cv'), ctx = cv.getContext('2d');

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
  renderSites();
  let s = sites[0].site, i = 0;
  try {                                      // come back to where this browser left off
    const at = (localStorage.getItem('review_at') || '').split('|');
    if (sites.some(z => z.site === at[0])) { s = at[0]; i = Math.max(0, +at[1] || 0); }
  } catch (e) {}
  openSite(s, i);
}
function renderSites() {
  const l = $('#sites'); l.innerHTML = '';
  sites.forEach(s => {
    const d = document.createElement('div');
    d.className = 'site' + (s.site === site ? ' sel' : '') + (s.done === s.n ? ' done' : '');
    d.innerHTML = `<span>${s.site}</span><span class="n">${s.done}/${s.n}</span>`;
    d.onclick = () => openSite(s.site, 0);
    l.appendChild(d);
  });
  const sel = l.querySelector('.site.sel');
  if (sel) sel.scrollIntoView({block: 'nearest'});
}
async function openSite(s, i) {
  if (dirty) await save(true);
  site = s;
  frames = await (await fetch(`/api/frames?site=${s}`)).json();
  renderSites();
  load(i);
}
function renderStrip() {
  const only = $('#only').checked, l = $('#strip'); l.innerHTML = '';
  frames.forEach((f, i) => {
    if (only && f.done && !flags[`${site}_${i}`] && !f.check) return;
    const d = document.createElement('div');
    d.className = 'f' + (f.done ? ' done' : '') + (i === idx ? ' sel' : '') +
                  (flags[`${site}_${i}`] ? ' flag' : '');
    d.textContent = i % 10 === 0 ? i : '';
    d.title = `frame ${i}` + (f.check ? ` · ${f.check} to check` : '');
    d.onclick = () => load(i);
    l.appendChild(d);
  });
  const sel = l.querySelector('.f.sel');
  if (sel) sel.scrollIntoView({block: 'nearest'});
}
async function load(i, accept) {
  /* Stepping to the next frame counts as reviewing this one: the decision on screen is the
     decision, whether or not anything was clicked. Jumping about with the mouse does not. */
  if ((dirty || accept) && comp) await save(true);
  idx = i; undo = []; unsureAt = 0;
  const r = await (await fetch(`/api/frame?site=${site}&i=${i}`)).json();
  rows = r.rows; W = r.w; H = r.h;
  comp = Uint8Array.from(atob(r.comp), c => c.charCodeAt(0));
  cut = r.cut ? Uint8Array.from(atob(r.cut), c => c.charCodeAt(0)) : new Uint8Array(W * H);
  img = new Image();
  await new Promise(res => { img.onload = res; img.onerror = res;
                             img.src = `/rgb/${site}/${i}.png`; });
  $('#title').textContent = `${site}  frame ${i}`;
  const fl = flags[`${site}_${i}`];
  $('#flag').classList.toggle('on', !!fl);
  $('#note').value = fl ? fl.note : '';
  try { localStorage.setItem('review_at', `${site}|${i}`); } catch (e) {}
  dirty = false; renderStrip(); fit();
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
  if (overlay.width !== W) { overlay.width = W; overlay.height = H; }
  octx = octx || overlay.getContext('2d', {willReadFrequently: true});
  const d = octx.createImageData(W, H), colour = {};
  rows.forEach(r => colour[r.id] = r.keep ? KEEP : DROP);
  for (let p = 0, k = 0; p < comp.length; p++, k += 4) {
    const c = colour[comp[p]];
    if (!c) continue;
    d.data[k] = c[0]; d.data[k+1] = c[1]; d.data[k+2] = c[2]; d.data[k+3] = 130;
  }
  octx.putImageData(d, 0, 0);
  // a bright edge on whatever the rule was unsure about: that is what needs looking at
  octx.lineWidth = 1; octx.strokeStyle = `rgb(${CHECK.join(',')})`;
  rows.filter(r => r.fresh).forEach(r => {
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
  return [Math.round((e.clientX - r.left - view.ox) / view.z),
          Math.round((e.clientY - r.top - view.oy) / view.z)];
};

/* ------------------------------------------------------------------ edit */
cv.addEventListener('mousedown', e => {
  if (e.button === 1 || panKey || tool === 'pan') {
    panning = true; last = [e.clientX, e.clientY];
    if (tool === 'pan') cv.style.cursor = 'grabbing';
    return;
  }
  const [x, y] = toImg(e);
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  if (tool === 'cut') { undo.push({comp: comp.slice(), cut: cut.slice()}); drawing = true;
                        paintCut(x, y); return; }
  const id = comp[y * W + x];
  if (!id) return;
  const r = rows.find(z => z.id === id);
  undo.push({keep: rows.map(z => z.keep)});
  r.keep = !r.keep; dirty = true; draw();
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
  const r = await (await fetch(`/api/recut?site=${site}&i=${idx}`,
                               {method: 'POST', body: cut})).json();
  rows = r.rows;
  comp = Uint8Array.from(atob(r.comp), c => c.charCodeAt(0));
  dirty = true; draw();
  msg(`re-split into ${rows.length} components`);
}
function nextUnsure() {
  const list = rows.filter(r => r.fresh);
  if (!list.length) {
    const j = frames.findIndex((f, i) => i > idx && f.check);
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
$('#cut').onclick = () => setTool('cut');
$('#pan').onclick = () => setTool('pan');
$('#unsure').onclick = nextUnsure;
$('#keepall').onclick = () => { undo.push({keep: rows.map(r => r.keep)});
                                rows.forEach(r => r.keep = true); dirty = true; draw(); };
$('#reset').onclick = async () => {
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
  const keep = rows.filter(r => r.keep).map(r => r.id);
  await fetch(`/api/save?site=${site}&i=${idx}`, {method: 'POST', body: JSON.stringify(keep)});
  dirty = false;
  frames[idx].done = true;
  frames[idx].check = rows.filter(r => r.fresh).length;
  const s = sites.find(z => z.site === site);
  if (s) s.done = frames.filter(f => f.done).length;
  renderStrip(); renderSites(); draw();
  if (!quiet) { $('#save').classList.add('on');
                setTimeout(() => $('#save').classList.remove('on'), 300); }
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
    const u = undo.pop();
    if (!u) return;
    if (u.keep) rows.forEach((r, i) => r.keep = u.keep[i]);
    else { comp = u.comp; cut = u.cut; }
    dirty = true; draw(); return;
  }
  if (e.metaKey || e.ctrlKey) return;
  if (e.key === 'Escape') { $('#keys').classList.remove('on'); return; }
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
  else if (act === 'next') {
    if (idx + 1 < frames.length) load(idx + 1, true);
    else { const j = sites.findIndex(z => z.site === site);
           save(true).then(() => { if (j + 1 < sites.length) openSite(sites[j+1].site, 0);
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


def load_frame(site, i):
    """Component map and rows for a frame, with any hand cut already applied."""
    cpath, jpath = comp_paths(site, i)
    comp = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)
    rows = json.load(open(jpath))
    cut = cv2.imread(f"{out_dir(site)}/{i}_cut.png", cv2.IMREAD_GRAYSCALE)
    if cut is not None:
        comp, rows = resplit(site, i, cut > 0)
    return comp, rows, cut


def resplit(site, i, cut):
    """Split the frame again with the hand cut removed from the mask."""
    _, _, label, _ = isod.frame(site, i)
    label = np.array(label).copy()
    label[cut] = 255                      # a cut pixel is treated as floor, so the mask parts
    comp, rows = comps.frame_components_from_label(label)
    return comp.astype(np.uint8), rows


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
        r["fresh"] = bool(
            (not r["touches_top"] and r["enclosed"] < comps.ENCLOSED) or
            (r["keep"] and r["area"] > 20000))
    return rows


def frame_state(site, i):
    comp, rows, cut = load_frame(site, i)
    saved = f"{out_dir(site)}/{i}.json"
    if os.path.exists(saved):
        keep = set(json.load(open(saved)))
        for r in rows:
            r["keep"] = r["id"] in keep
            r["fresh"] = False
    else:
        rows = flag(rows)
    return comp, rows, cut


def write_mask(site, i, keep):
    comp, _, _ = load_frame(site, i)
    mask = np.isin(comp, list(keep)).astype(np.uint8) * 255
    cv2.imwrite(f"{out_dir(site)}/{i}.png", mask)
    json.dump(sorted(int(k) for k in keep), open(f"{out_dir(site)}/{i}.json", "w"))


class Handler(http.server.BaseHTTPRequestHandler):
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
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
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
                rows = json.load(open(comp_paths(s, i)[1]))
                out.append(dict(i=i, done=os.path.exists(f"{out_dir(s)}/{i}.json"),
                                check=sum(r["fresh"] for r in flag(rows))))
            return self._json(out)
        if u.path == "/api/frame":
            import base64
            s, i = q["site"][0], int(q["i"][0])
            comp, rows, cut = frame_state(s, i)
            return self._json(dict(w=comp.shape[1], h=comp.shape[0], rows=rows,
                                   comp=base64.b64encode(comp.tobytes()).decode(),
                                   cut=base64.b64encode(cut.tobytes()).decode() if cut is not None else None))
        if u.path.startswith("/rgb/"):
            site, name = u.path[len("/rgb/"):].split("/")
            p = f"{isod.DATA}/{site}/rgb/{name}"
            if not os.path.exists(p):
                return self._send(404, b"no")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "image/png")
        self._send(404, b"no")

    def do_POST(self):
        import base64
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        s, i = q["site"][0], int(q["i"][0])
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if u.path == "/api/flag":
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
        if u.path == "/api/save":
            write_mask(s, i, json.loads(body or b"[]"))
            return self._send(200, b"ok", "text/plain")
        if u.path == "/api/recut":
            raw = np.frombuffer(body, np.uint8)        # raw bytes, not base64: see recut()
            _, _, label, _ = isod.frame(s, i)
            cut = raw.reshape(np.array(label).shape)
            cv2.imwrite(f"{out_dir(s)}/{i}_cut.png", cut * 255)
            comp, rows = resplit(s, i, cut > 0)
            rows = flag(rows)
            return self._json(dict(rows=rows,
                                   comp=base64.b64encode(comp.tobytes()).decode()))
        if u.path == "/api/reset":
            for p in (f"{out_dir(s)}/{i}.json", f"{out_dir(s)}/{i}.png",
                      f"{out_dir(s)}/{i}_cut.png"):
                if os.path.exists(p):
                    os.remove(p)
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
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
