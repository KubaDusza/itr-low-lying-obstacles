"""Minimal mask annotator for the own-recordings dataset.

    python code/annotate.py            # then open the URL it prints

Two masks are written per frame:

    data/OWN_manual/<run>_<frame>.png       255 = obstacle, the format own_gt.py produces,
                                            so it overrides the automatic mask for that frame
    data/OWN_manual/<run>_<frame>_id.png    which *item* each pixel belongs to

The id map is the point of the labelling pass. The same physical objects were re-placed in run
after run -- the same four cables, the same five tape rolls -- so if every pixel carries the
item that produced it, the benchmark can be read per item: this cable is found at 0.75 m and
lost at 2.5 m, that hex key is never found at all. The item inventory lives in
data_collection/items.csv, one row per object with its set, and the pixel value is that item's
1-based position within its set. 255 means masked but not yet assigned to an item; 0 is
background.

A frame opens with its SAM 3 pre-label, each instance already a separate region, so most of the
work is pressing an item's number and clicking its blob. "carry" moves the labelled mask onto
the next frame, which is nearly identical because the camera never moved inside a run.

The reference panel holds a second frame beside it with its own mask drawn on: the same objects
at another distance, height or background. Clicking an item there copies its mask over as a
stamp to drag, scale and rotate into place, so a cable that was traced once does not have to be
traced again.

Every shortcut is remappable in the panel behind the ? button, and the bindings persist in the
browser.
"""
import csv
import http.server
import json
import os
import socket
import socketserver
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN, MAN = f"{ROOT}/data/OWN", f"{ROOT}/data/OWN_manual"
ITEMS_CSV = f"{ROOT}/data_collection/items.csv"
INDEX_CSV = f"{ROOT}/data_collection/dataset_index.csv"
UNLABELLED = 255
PORT = int(os.environ.get("ANNOT_PORT", 8765))


def item_table():
    """object set -> the items placed in it, in csv order. The index is the pixel value."""
    sets = {}
    if os.path.exists(ITEMS_CSV):
        for r in csv.DictReader(open(ITEMS_CSV)):
            row = sets.setdefault(r["object_set"], [])
            row.append(dict(idx=len(row) + 1, id=r["item_id"], label=r["label"],
                            colour=r["colour"],
                            phrases=[p for p in r["sam_phrases"].split(";") if p]))
    return sets


def run_sets():
    """run -> its object set."""
    out = {}
    if os.path.exists(INDEX_CSV):
        for r in csv.DictReader(open(INDEX_CSV)):
            out[r["run"]] = r["object_set"]
    return out

PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>mask annotator</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2e333c; --ink:#e7e9ee; --muted:#9aa3b2;
          --accent:#3b6fd4; --ok:#2f9e6e; --warn:#e8a33d; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  #side { width:262px; flex:none; border-right:1px solid var(--line); display:flex;
          flex-direction:column; background:var(--panel); }
  #side h1 { font-size:13px; margin:0; padding:12px 14px; border-bottom:1px solid var(--line); }
  #filter { padding:8px 14px 4px; color:var(--muted); }
  #filter2 { padding:0 14px 8px; border-bottom:1px solid var(--line); color:var(--muted); }
  #note { width:150px; background:#12151a; color:var(--ink); border:1px solid var(--line);
          border-radius:5px; padding:3px 7px; font:12px ui-sans-serif,system-ui; }
  #note:focus { outline:none; border-color:var(--accent); }
  .flagged { color:var(--warn); }
  #list { overflow:auto; flex:1; }
  #itembox { flex:none; border-top:1px solid var(--line); background:#191c22; }
  #itembox h2 { font-size:11px; letter-spacing:.06em; text-transform:uppercase;
                color:var(--muted); margin:0; padding:9px 14px 5px; }
  .item { display:flex; align-items:center; gap:8px; padding:4px 14px; cursor:pointer; }
  .item:hover { background:#232832; }
  .item.sel { background:#232832; box-shadow:inset 3px 0 0 var(--accent); }
  .item .sw { width:11px; height:11px; border-radius:3px; flex:none;
              border:1px solid rgba(255,255,255,.35); }
  .item .nm { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .item .n, .item .num { font:11px ui-monospace,Menlo,monospace; color:var(--muted); flex:none; }
  .item.empty .nm { color:#6b7585; }
  .item.sel .nm { color:#fff; }
  .item.todo { background:#2a1430; }
  .item.todo .nm, .item.todo .n { color:#ff00e6; }
  .sw.stripe { background:repeating-linear-gradient(45deg,#ff00e6 0 3px,#141414 3px 6px); }
  .row { padding:6px 14px; cursor:pointer; border-bottom:1px solid #22252b; display:flex;
         justify-content:space-between; gap:8px; }
  .row:hover { background:#232832; }
  .row.sel { background:var(--accent); color:#fff; }
  .row .tag { font-size:11px; color:var(--muted); flex:none; }
  .row.sel .tag { color:#dbe6ff; }
  .row.done .tag { color:var(--ok); }
  .row.done .tag.bad { color:var(--warn); }
  .bad { color:var(--warn); }
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
    content:attr(data-hint); position:absolute; left:0; top:calc(100% + 7px); width:280px;
    background:#0c0e12; color:var(--ink); border:1px solid var(--line); border-radius:6px;
    padding:7px 9px; font-size:12px; line-height:1.4; z-index:20; white-space:normal;
    box-shadow:0 6px 18px rgba(0,0,0,.45); }
  .tool.hr[data-hint]:hover::after { left:auto; right:0; }
  #mid { flex:1; display:flex; min-height:0; }
  #wrap { flex:1; overflow:hidden; position:relative; background:#0d0f12; min-width:0; }
  #ref { width:430px; flex:none; border-left:1px solid var(--line); background:var(--panel);
         display:none; flex-direction:column; min-width:0; }
  #ref.on { display:flex; }
  #refhead { display:flex; gap:7px; align-items:center; padding:7px 10px;
             border-bottom:1px solid var(--line); }
  #refsel { flex:1; min-width:0; background:#12151a; color:var(--ink); padding:3px 6px;
            border:1px solid var(--line); border-radius:5px; font:12px ui-sans-serif,system-ui; }
  #refwrap { flex:1; position:relative; overflow:hidden; background:#0d0f12; }
  #refcv { position:absolute; image-rendering:pixelated; cursor:crosshair; }
  #reffoot { padding:5px 10px; border-top:1px solid var(--line); color:var(--muted);
             font-size:11px; line-height:1.5; }
  canvas { position:absolute; image-rendering:pixelated; cursor:none; }
  #status { padding:6px 14px; border-top:1px solid var(--line); color:var(--muted);
            background:var(--panel); display:flex; gap:18px; flex-wrap:nowrap; }
  /* fixed widths and a monospace figure so the bar does not jitter as the cursor moves */
  #pos, #px, #zoom { font:12px/1.5 ui-monospace,Menlo,monospace; flex:none;
                     display:inline-block; text-align:left; }
  #pos { width:116px; } #px { width:84px; } #zoom { width:52px; }
  #under { width:150px; flex:none; overflow:hidden; white-space:nowrap; }
  #todo { width:120px; flex:none; font:12px/1.5 ui-monospace,Menlo,monospace; }
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
  .krow { display:flex; align-items:center; justify-content:space-between; gap:12px;
          padding:5px 0; border-bottom:1px solid #22252b; }
  .kbtn { min-width:62px; text-align:center; background:#262b33; border:1px solid var(--line);
          border-radius:5px; padding:3px 8px; cursor:pointer;
          font:12px ui-monospace,Menlo,monospace; }
  .kbtn.listen { background:var(--warn); color:#17191d; }
  .mbtn { min-width:52px; text-align:center; border:1px solid var(--line); border-radius:5px;
          padding:3px 8px; cursor:pointer; font-size:11px; color:var(--muted); }
  .mbtn.hold { background:#2b3a2f; color:#8fd6a8; border-color:#36513d; }
  .krow .sp { flex:1; }
  #keyfoot { display:flex; gap:10px; margin-top:16px; }
</style>
<div id="side">
  <h1>mask annotator</h1>
  <label id="filter"><input type="checkbox" id="only"> only runs needing work</label>
  <label id="filter2"><input type="checkbox" id="onlyflag"> only flagged</label>
  <div id="list"></div>
  <div id="itembox">
    <h2>items in this run</h2>
    <div id="itemlist"></div>
  </div>
</div>
<div id="main">
  <div id="bar">
    <b id="title">—</b>
    <span class="tool" data-t="brush" data-hint="Paint mask freehand. Hold and drag; the ring on the canvas is the exact size you are painting.">brush<span class="k" data-k="brush"></span></span>
    <span class="tool" data-t="eraser" data-hint="Remove mask freehand. Same ring, same size control — use a big one to wipe the automatic mask's false blobs.">eraser<span class="k" data-k="eraser"></span></span>
    <span class="tool" data-t="wand" data-hint="Flood fill by colour: click one pixel and every connected pixel of a similar colour joins the mask. One click covers a whole rag or tape roll. Tolerance decides how different a pixel may be before it stops spreading.">wand<span class="k" data-k="wand"></span></span>
    <span class="tool" data-t="poly" data-hint="Polygon: click each corner, Enter fills the shape, Esc cancels. Best for straight-edged objects like the pink rag.">polygon<span class="k" data-k="poly"></span></span>
    <span class="tool" data-t="assign" data-hint="Assign a whole blob to the selected item: click any masked pixel and every connected masked pixel takes that item's colour. This is the fast way through a SAM pre-label — press the item's number, click its blob, done.">assign<span class="k" data-k="assign"></span></span>
    <span class="tool" data-t="move" data-hint="Pick a mask up and drag it. Press on any masked pixel and that whole item's connected blob lifts off the frame and follows the mouse until you let go; [ ] scale it and ; ' rotate it while it is in the air, and Esc puts it back where it was. The same grab works on a mask just copied from the reference panel.">move<span class="k" data-k="move"></span></span>
    <span class="tool" id="refbtn" data-hint="A second frame beside this one, with its mask drawn on it: the same objects at another distance, height or background. Clicking an item there copies its mask onto this frame as a stamp you can move, scale and rotate into place, which beats redrawing a cable you have already drawn once.">reference<span class="k" data-k="ref"></span></span>
    <span class="tool" data-t="pan" data-hint="Move the view: drag with the left button, nothing is painted. Its key is held rather than toggled by default — the ? panel sets hold or toggle for every tool. Space pans from any tool, and the scroll wheel still zooms.">pan<span class="k" data-k="pan"></span></span>
    <span class="tool" id="smart" data-hint="Smart edge mode. On mouse-down the colour under the cursor is sampled, and the stroke then only affects pixels close to that colour — so a sloppy sweep along a cable paints the cable and stops at its edge. The eraser works the same way: click on floor and only floor-coloured mask is removed. Uses the tolerance setting.">smart<span class="k" data-k="smart"></span></span>
    <span class="tool" id="szbox" data-hint="Brush and eraser diameter, in image pixels.">size <b id="sz">6</b><span class="k" data-k="sizeDown"></span><span class="k" data-k="sizeUp"></span></span>
    <span class="tool" id="tolbox" data-hint="How different a pixel may be from the one you clicked before the wand stops. Raise it if the wand grabs too little, lower it if it bleeds onto the floor.">tol <b id="tol">14</b><span class="k" data-k="tolDown"></span><span class="k" data-k="tolUp"></span></span>
    <span class="tool" id="loadsam" data-hint="Replace the mask with the SAM 3 pre-label from sam3_prelabel.py, discarding your edits to this frame. This is what a frame opens with when it exists: SAM 3 was prompted with the object set that was actually placed, so the edges are usually close and the work is removing what it invented and adding what it missed.">load sam<span class="k" data-k="loadSam"></span></span>
    <span class="tool" id="loadauto" data-hint="Replace the mask with the background-subtraction mask from own_gt.py, discarding your edits to this frame. Worth a look when SAM 3 missed an object entirely: the two methods fail in different places.">load auto<span class="k" data-k="loadAuto"></span></span>
    <span class="tool" id="clear" data-hint="Empty the mask for this frame.">clear<span class="k" data-k="clear"></span></span>
    <span class="tool" id="carry" data-hint="Carry the mask to the next frame. Camera and objects do not move inside a run, so the five frames are nearly identical — with this on, moving to the next frame keeps what you drew instead of loading that frame's own mask.">carry<span class="k" data-k="carry"></span></span>
    <span class="tool" id="flag" data-hint="Mark this frame to come back to, with a note if you want one. Flagged frames carry a ⚑ in the list and the filter above it can show only those.">flag<span class="k" data-k="flag"></span></span>
    <input id="note" placeholder="note" spellcheck="false">
    <span class="tool" id="save" data-hint="Write the mask to data/OWN_manual/. Also happens automatically when you change frame.">save<span class="k">⌘s</span></span>
    <span class="tool" id="help" data-hint="Change any shortcut.">?</span>
  </div>
  <div id="mid">
    <div id="wrap"><canvas id="cv"></canvas></div>
    <div id="ref">
      <div id="refhead">
        <select id="refsel"></select>
        <span class="tool on" id="refmask" data-hint="Show or hide the reference frame's own mask.">mask</span>
        <span class="tool" id="refclose" data-hint="Close the reference panel.">&times;</span>
      </div>
      <div id="refwrap"><canvas id="refcv"></canvas></div>
      <div id="reffoot">click an item to copy its mask · drag to pan · scroll to zoom</div>
    </div>
  </div>
  <div id="status">
    <span id="pos">—</span><span id="px">0 px</span><span id="zoom">1x</span>
    <span id="under"></span>
    <span id="todo"></span>
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
const DEFAULTS = {brush:'w', eraser:'e', wand:'q', poly:'p', assign:'f', sizeDown:'!',
                  sizeUp:'@', tolDown:',', tolUp:'.', hide:'h', fit:'0', loadAuto:'a',
                  loadSam:'g', clear:'x', pan:'v', ref:'r', move:'m', flag:'t',
                  carry:'c', smart:'s', next:'n', prev:'N'};
const LABEL = {brush:'brush', eraser:'eraser', wand:'wand', poly:'polygon',
               assign:'assign blob to item', pan:'pan the view', move:'move a blob',
               flag:'flag this frame',
               ref:'reference panel',
               sizeDown:'smaller brush', sizeUp:'bigger brush', tolDown:'lower tolerance',
               tolUp:'raise tolerance', hide:'hide / show mask', fit:'fit to window',
               loadAuto:'load automatic mask', loadSam:'load SAM 3 mask',
               clear:'clear mask',
               carry:'toggle carry to next frame', smart:'toggle smart edges',
               next:'next frame', prev:'previous frame'};
/* Each of these can be a sticky toggle -- press once and it stays -- or spring-loaded: it
   lasts only while the key is held, and the previous tool comes back on release. Pan is the
   one that wants holding by default; the rest are set in the ? panel. */
const HOLDABLE = ['brush', 'eraser', 'wand', 'poly', 'assign', 'move', 'pan', 'hide'];
const MODE_DEF = {pan: 'hold'};
let MODES = {};
HOLDABLE.forEach(a => MODES[a] = MODE_DEF[a] || 'toggle');
try { Object.assign(MODES, JSON.parse(localStorage.getItem('annot_modes') || '{}')); } catch (e) {}

let KEYS = Object.assign({}, DEFAULTS);
try { Object.assign(KEYS, JSON.parse(localStorage.getItem('annot_keys') || '{}')); } catch (e) {}
/* 1-9 now pick the item to paint with, so a binding saved on a digit before that -- brush
   size lived on 1 and 2 -- goes back to its default, which is the shifted digit. */
Object.keys(KEYS).forEach(a => { if (/^[1-9]$/.test(KEYS[a])) KEYS[a] = DEFAULTS[a]; });
try { localStorage.setItem('annot_keys', JSON.stringify(KEYS)); } catch (e) {}

let frames = [], cur = -1, img = null, imgData = null;
/* The mask canvas carries the item in its colour: every stroke is painted in the selected
   item's palette colour, so no second buffer has to be kept in step with it, and saving is a
   nearest-colour lookup. UNLAB is what a mask loaded from SAM 3 or own_gt.py starts as. */
let itemSets = {}, items = [], item = 0;
const UNLAB = [190, 190, 190], UNLAB_IDX = 255;
let mask = document.createElement('canvas'), mctx = null;
let view = {z: 2, ox: 0, oy: 0}, tool = 'brush', size = 6, tol = 14, carry = true;
let drawing = false, panning = false, last = null, hide = false, dirty = false;
let smart = false, strokeRef = null;
let undo = [], poly = [], mouse = null, panKey = false, listening = null, held = null;
let moving = null;
const cv = $('#cv'), ctx = cv.getContext('2d');

/* ------------------------------------------------------------------ keys */
const SHIFTED = {'!':'⇧1', '@':'⇧2', '#':'⇧3', '$':'⇧4', '%':'⇧5'};
const keyLabel = k => k === ' ' ? 'space' : (SHIFTED[k] || k || '—');
function paintKeys() {
  document.querySelectorAll('[data-k]').forEach(e => {
    e.textContent = KEYS[e.dataset.k] ? ' ' + keyLabel(KEYS[e.dataset.k]) : '';
  });
  $('#hints').innerHTML =
    `scroll zoom · space-drag pan · <kbd>1</kbd>…<kbd>9</kbd> item · ` +
    `<kbd>${keyLabel(KEYS.fit)}</kbd> fit · ` +
    `<kbd>${keyLabel(KEYS.hide)}</kbd> hide · <kbd>${keyLabel(KEYS.next)}</kbd>/` +
    `<kbd>${keyLabel(KEYS.prev)}</kbd> frame · <kbd>⌘z</kbd> undo`;
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
                                      : 'press once, stays until you pick another';
      };
      m.onclick = () => {
        MODES[a] = MODES[a] === 'hold' ? 'toggle' : 'hold';
        localStorage.setItem('annot_modes', JSON.stringify(MODES)); paint();
      };
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
  localStorage.setItem('annot_keys', JSON.stringify(KEYS));
  buildKeyPanel(); paintKeys();
}

/* ------------------------------------------------------------------ items */
const hexRgb = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
const curColour = () => items.length ? hexRgb(items[item].colour) : UNLAB;

function nearestItem(r, g, b) {
  /* Which item a painted pixel belongs to. Strokes are painted in exact palette colours, but
     the canvas antialiases their edges, so match to the nearest colour rather than demanding
     equality. */
  let best = UNLAB_IDX, bd = (r-UNLAB[0])**2 + (g-UNLAB[1])**2 + (b-UNLAB[2])**2;
  items.forEach(it => {
    const c = hexRgb(it.colour), d = (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2;
    if (d < bd) { bd = d; best = it.idx; }
  });
  return best;
}
function renderItems(counts) {
  const l = $('#itemlist'); l.innerHTML = '';
  if (!items.length) {
    l.innerHTML = '<div class="item empty"><span class="nm">nothing was placed</span></div>';
    return;
  }
  items.forEach((it, i) => {
    const n = counts ? (counts[it.idx] || 0) : null;
    const d = document.createElement('div');
    d.className = 'item' + (i === item ? ' sel' : '') + (n === 0 ? ' empty' : '');
    d.innerHTML = `<span class="num">${i + 1}</span>` +
                  `<span class="sw" style="background:${it.colour}"></span>` +
                  `<span class="nm">${it.label}</span>` +
                  `<span class="n">${n === null ? '' : n}</span>`;
    d.onclick = () => setItem(i);
    l.appendChild(d);
  });
  if (counts && counts[UNLAB_IDX]) {
    const d = document.createElement('div');
    d.className = 'item todo';
    d.innerHTML = `<span class="num"></span><span class="sw stripe"></span>` +
                  `<span class="nm">unassigned</span><span class="n">${counts[UNLAB_IDX]}</span>`;
    l.appendChild(d);
  }
}
function setItem(i) { item = Math.max(0, Math.min(items.length - 1, i)); renderItems(lastCounts); }

/* ---------------------------------------------------------------- frames */
async function boot() {
  itemSets = await (await fetch('/api/items')).json();
  frames = await (await fetch('/api/list')).json();
  renderList();
  if (frames.length) load(0);
}
function renderList() {
  const only = $('#only').checked, l = $('#list'); l.innerHTML = '';
  const onlyFlag = $('#onlyflag').checked;
  frames.forEach((f, i) => {
    if (only && f.reliable && !f.manual) return;
    if (onlyFlag && !f.flag) return;
    const d = document.createElement('div');
    d.className = 'row' + (i === cur ? ' sel' : '') + (f.manual ? ' done' : '');
    /* The state that matters while labelling is whether the items are named, which is not
       the same as whether the frame was saved. */
    const state = !f.manual ? (f.reliable ? 'ok' : 'needs work')
                : f.todo || !f.manual_id ? 'items left' : 'labelled';
    const bad = state === 'needs work' || state === 'items left';
    const note = f.flag && f.flag.note ? ` — ${f.flag.note}` : '';
    d.innerHTML = `<span>${f.flag ? '<span class="flagged">⚑ </span>' : ''}${f.run}` +
                  `<span class="tag"> ·${f.frame}${note}</span></span>` +
                  `<span class="tag ${bad ? 'bad' : ''}">${state}</span>`;
    d.onclick = () => load(i);
    l.appendChild(d);
  });
}
async function load(i, keep) {
 try {
  if (stamp) { moving = null; dropStamp(); }
  if (dirty && cur >= 0) await save(true);
  const keepData = keep && mctx ? mctx.getImageData(0, 0, mask.width, mask.height) : null;
  cur = i; poly = []; undo = [];
  const f = frames[i];
  items = itemSets[f.oset] || []; item = 0; lastCounts = null; renderItems();
  $('#flag').classList.toggle('on', !!f.flag);
  $('#note').value = f.flag ? f.flag.note : '';
  $('#title').textContent = `${f.run}  frame ${f.frame}`;
  img = new Image();
  // onload rather than decode(): decode() never settles while the tab is in the background,
  // because Chrome defers decoding for hidden documents, and the whole loader would stall.
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = () => rej(new Error('image failed to load'));
    img.src = `/file/OWN/${f.run}/rgb/${f.frame}.png?t=${Date.now()}`;
  });
  mask.width = img.width; mask.height = img.height;
  mctx = mask.getContext('2d', {willReadFrequently: true});
  mctx.clearRect(0, 0, mask.width, mask.height);
  const tmp = document.createElement('canvas');
  tmp.width = img.width; tmp.height = img.height;
  const tctx = tmp.getContext('2d', {willReadFrequently: true});
  tctx.drawImage(img, 0, 0);
  imgData = tctx.getImageData(0, 0, img.width, img.height);
  buildGrad();
  if (keepData && keepData.width === mask.width) {
    mctx.putImageData(keepData, 0, 0); dirty = true; fit(); rescan(); renderList(); return;
  }
  await loadInto('best');
  dirty = false; fit(); renderList();
 } catch (err) { banner('load failed: ' + (err && err.message || err)); }
}
function readPng(url, w, h) {
  return new Promise(res => {
    const m = new Image();
    m.onload = () => {
      const t = document.createElement('canvas');
      t.width = w; t.height = h;
      const c = t.getContext('2d', {willReadFrequently: true});
      c.imageSmoothingEnabled = false;             // an id map must not be interpolated
      c.drawImage(m, 0, 0, w, h);
      res(c.getImageData(0, 0, w, h));
    };
    m.onerror = () => res(null);
    m.src = url + '?t=' + Date.now();
  });
}
/* The item id of every pixel of a frame, as one array: 0 background, 1..N an item of the
   set, 255 masked but unassigned. `src` picks where it comes from -- 'best' is what the
   frame should open with, 'sam' and 'auto' are the two reload buttons. Both panes read a
   frame through this, so the reference shows exactly what the main canvas would. */
async function idMapFor(f, w, h, src, fitems) {
  const out = new Uint8Array(w * h);
  const binary = d => { for (let p = 0, i = 0; i < d.data.length; i += 4, p++)
                          if (d.data[i] > 127) out[p] = UNLAB_IDX; };
  const indexed = (d, map) => { for (let p = 0, i = 0; i < d.data.length; i += 4, p++)
                                  if (d.data[i]) out[p] = map(d.data[i]); };
  let d = null;
  if (src === 'best' && f.manual_id) {
    d = await readPng(`/file/OWN_manual/${f.run}_${f.frame}_id.png`, w, h);
    if (d) indexed(d, v => v);
    return out;
  }
  if (src === 'best' && f.manual) {
    d = await readPng(`/file/OWN_manual/${f.run}_${f.frame}.png`, w, h);
    if (d) binary(d);
    return out;
  }
  if (src !== 'auto' && f.sam) {
    /* The SAM 3 pre-label, with the instances it named unambiguously already assigned: a
       phrase that belongs to exactly one item of the set and fired exactly once in this
       frame is that item. The rest is left unassigned for the click-through. */
    let meta = null;
    try { meta = await (await fetch(`/file/OWN/${f.run}/sam/${f.frame}.json`)).json(); } catch (e) {}
    const assign = {};
    if (meta) {
      const once = {};
      meta.instances.forEach(x => once[x.phrase] = (once[x.phrase] || 0) + 1);
      meta.instances.forEach((x, i) => {
        const hit = (fitems || []).filter(it => it.phrases.includes(x.phrase));
        if (hit.length === 1 && once[x.phrase] === 1) assign[i + 1] = hit[0].idx;
      });
    }
    d = await readPng(`/file/OWN/${f.run}/sam/${f.frame}_id.png`, w, h);
    if (d) indexed(d, v => assign[v] || UNLAB_IDX);
    return out;
  }
  d = await readPng(`/file/OWN/${f.run}/gt/${f.frame}.png`, w, h);
  if (d) binary(d);
  return out;
}
/* ids -> the painted mask. Items are drawn in their own colour, which is what makes the
   mask readable and what save() reads back. */
function idsToImage(ids, w, h, fitems) {
  const out = new ImageData(w, h);
  const col = {[UNLAB_IDX]: UNLAB};
  (fitems || []).forEach(it => col[it.idx] = hexRgb(it.colour));
  for (let p = 0, i = 0; p < ids.length; p++, i += 4) {
    const c = col[ids[p]];
    if (!c) continue;
    out.data[i] = c[0]; out.data[i+1] = c[1]; out.data[i+2] = c[2]; out.data[i+3] = 255;
  }
  return out;
}
async function loadInto(src) {
  const ids = await idMapFor(frames[cur], img.width, img.height, src, items);
  mctx.putImageData(idsToImage(ids, img.width, img.height, items), 0, 0);
  rescan(); draw();
}

/* ------------------------------------------------------------------ view */
function fit() {
  if (!img) return;
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  view.z = Math.max(1, Math.floor(Math.min(w / img.width, h / img.height) * 10) / 10);
  view.ox = (w - img.width * view.z) / 2; view.oy = (h - img.height * view.z) / 2;
  draw();
}
function draw() {
  if (!img || !mctx) return;
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  cv.width = w; cv.height = h;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.save(); ctx.translate(view.ox, view.oy); ctx.scale(view.z, view.z);
  ctx.drawImage(img, 0, 0);
  if (!hide) {
    ctx.globalAlpha = 0.5; ctx.drawImage(mask, 0, 0); ctx.globalAlpha = 1;
    if (warn.width === img.width) ctx.drawImage(warn, 0, 0);   // what is still unassigned
  }
  if (stamp) {
    const k = view.z * stamp.scale;
    ctx.save();
    ctx.translate(stamp.x, stamp.y); ctx.rotate(stamp.rot); ctx.scale(stamp.scale, stamp.scale);
    ctx.globalAlpha = 0.85; ctx.drawImage(stamp.cv, -stamp.w / 2, -stamp.h / 2);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 1 / k;
    ctx.setLineDash([4 / k, 3 / k]);
    ctx.strokeRect(-stamp.w / 2, -stamp.h / 2, stamp.w, stamp.h);
    ctx.setLineDash([]);
    ctx.restore();
  }
  if (poly.length) {
    ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 1 / view.z;
    ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
    poly.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
  }
  ctx.restore();
  if (mouse && tool !== 'pan') {            // the cursor IS the brush size
    const r = (tool === 'brush' || tool === 'eraser') ? Math.max(2, size / 2 * view.z) : 4;
    const x = mouse[0] * view.z + view.ox, y = mouse[1] * view.z + view.oy;
    ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(0,0,0,.55)';
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.strokeStyle = tool === 'eraser' ? '#ffffff'
                      : items.length ? items[item].colour : '#ff3b30';
    if (smart && (tool === 'brush' || tool === 'eraser')) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
  }
  $('#zoom').textContent = view.z.toFixed(1) + 'x';
}
let pxTimer = null, lastCounts = null;
let warn = document.createElement('canvas'), wctx = null;
/* Counting, the per-item tally and the unassigned overlay all need the same full pass over
   the mask, so they share one debounced scan. It is called from the edits, never from draw(),
   which would loop: the scan ends in a redraw. */
function rescan() {
  if (!mctx) return;
  clearTimeout(pxTimer);
  pxTimer = setTimeout(() => {
    const W = mask.width, H = mask.height;
    const d = mctx.getImageData(0, 0, W, H).data;
    const counts = {}, cache = new Map();
    /* Unassigned pixels are drawn back over the mask as hazard stripes at full opacity. A
       flat colour does not work here: half the items are white on a white poster, and a
       quiet grey wash over them is exactly what gets missed and saved. */
    if (warn.width !== W) { warn.width = W; warn.height = H; }
    wctx = wctx || warn.getContext('2d', {willReadFrequently: true});
    const w = wctx.createImageData(W, H);
    let n = 0, unassigned = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i+3] === 0) continue;
      n++;
      const k = (d[i] << 16) | (d[i+1] << 8) | d[i+2];
      let v = cache.get(k);
      if (v === undefined) { v = nearestItem(d[i], d[i+1], d[i+2]); cache.set(k, v); }
      counts[v] = (counts[v] || 0) + 1;
      if (v !== UNLAB_IDX) continue;
      unassigned++;
      const p = i / 4, x = p % W, y = (p / W) | 0, on = ((x + y) % 10) < 5;
      w.data[i] = on ? 255 : 20; w.data[i+1] = on ? 0 : 20; w.data[i+2] = on ? 230 : 20;
      w.data[i+3] = 255;
    }
    wctx.putImageData(w, 0, 0);
    $('#px').textContent = String(n).padStart(6) + ' px';
    $('#todo').textContent = unassigned ? `${unassigned} unassigned` : '';
    $('#todo').className = unassigned ? 'bad' : '';
    lastCounts = counts; renderItems(counts);
    draw();
  }, 120);
}
function itemUnder(p) {
  if (!mctx || !img) return '';
  const x = Math.round(p[0]), y = Math.round(p[1]);
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return '';
  const d = mctx.getImageData(x, y, 1, 1).data;
  if (d[3] === 0) return '';
  const v = nearestItem(d[0], d[1], d[2]);
  const it = items.find(i => i.idx === v);
  return it ? it.label : 'unassigned';
}
const toImg = e => {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left - view.ox) / view.z, (e.clientY - r.top - view.oy) / view.z];
};

/* ----------------------------------------------------------- smart edges */
let grad = null, gradLimit = 60, smoothData = null;
function buildGrad() {
  /* Edge strength, used as a barrier the region grow will not cross. The grey is blurred
     first because this floor is speckled and raw Sobel fires on the speckle, not on the
     object boundary. */
  const W = img.width, H = img.height, d = imgData.data;
  /* Colour matching runs on a 3x3 box blur, not the raw pixels. This floor is speckled and
     its bright specks sit within tolerance of a white cable, so matching raw colour scatters
     the mask over the floor. The cable survives a 3x3 blur; single-pixel speckle does not. */
  smoothData = new Uint8ClampedArray(d.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let r = 0, g2 = 0, b2 = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      const i = (yy * W + xx) * 4;
      r += d[i]; g2 += d[i+1]; b2 += d[i+2]; n++;
    }
    const o = (y * W + x) * 4;
    smoothData[o] = r / n; smoothData[o+1] = g2 / n; smoothData[o+2] = b2 / n; smoothData[o+3] = 255;
  }
  const g = new Float32Array(W * H), blur = new Float32Array(W * H);
  for (let i = 0, k = 0; i < d.length; i += 4, k++)
    g[k] = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += g[(y+dy)*W + x+dx];
    blur[y*W + x] = s / 9;
  }
  grad = new Float32Array(W * H);
  const sample = [];
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const k = y * W + x;
    const gx = blur[k-W+1] + 2*blur[k+1] + blur[k+W+1] - blur[k-W-1] - 2*blur[k-1] - blur[k+W-1];
    const gy = blur[k+W-1] + 2*blur[k+W] + blur[k+W+1] - blur[k-W-1] - 2*blur[k-W] - blur[k-W+1];
    grad[k] = (Math.abs(gx) + Math.abs(gy)) / 8;
    if ((x + y) % 7 === 0) sample.push(grad[k]);
  }
  sample.sort((a, b) => a - b);
  gradLimit = Math.max(12, sample[Math.floor(sample.length * 0.90)] * 1.6);
}
function colDist(i, c) {
  const d = smoothData || imgData.data;
  return Math.max(Math.abs(d[i] - c[0]), Math.abs(d[i+1] - c[1]), Math.abs(d[i+2] - c[2]));
}
function strokeSeed(cx, cy) {
  /* Colour the whole stroke will follow.

     The cursor is often on the local background — easy when the brush is wider than a 4 px
     cable — so if it is, snap to whatever else is under the brush. Taking the single most
     deviating pixel picks a specular highlight and matches nothing afterwards, so take the
     MEDIAN of the deviating pixels instead. */
  const W = img.width, H = img.height, d = smoothData || imgData.data;
  const R = Math.max(2, Math.round(size / 2));
  cx = Math.round(cx); cy = Math.round(cy);
  const here = sampleColour([cx, cy]);
  if (!here) return null;
  const lum = [], idx = [];
  for (let y = Math.max(0, cy - R); y <= Math.min(H - 1, cy + R); y++)
    for (let x = Math.max(0, cx - R); x <= Math.min(W - 1, cx + R); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > R * R) continue;
      const i = (y * W + x) * 4;
      lum.push(0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2]);
      idx.push(i);
    }
  if (lum.length < 9) return here;
  const med = lum.slice().sort((a, b) => a - b)[lum.length >> 1];
  const hereLum = 0.299 * here[0] + 0.587 * here[1] + 0.114 * here[2];
  if (Math.abs(hereLum - med) >= tol * 0.6) return here;     // the cursor is already on it
  const dev = [];
  for (let k = 0; k < lum.length; k++)
    if (Math.abs(lum[k] - med) > tol * 0.8) dev.push(idx[k]);
  // only snap when what is under the brush is a clear minority, i.e. a thin object crossing
  // it. Without this guard the brush "snaps" to the floor enclosed by a cable loop.
  if (dev.length < 4 || dev.length > lum.length * 0.35) return here;
  const pick = c => {
    const v = dev.map(i => d[i + c]).sort((a, b) => a - b);
    return v[v.length >> 1];
  };
  return [pick(0), pick(1), pick(2)];
}
function smartDab(cx, cy) {
  /* Intelligent-Paint style: sample under the cursor, then grow a CONNECTED region inside
     the brush disc, stopping at colour difference or at an edge. Connectivity is what my
     first attempt lacked — without it the brush painted every matching pixel in its
     bounding box, including floor on the far side of the object. */
  const W = img.width, H = img.height, R = Math.max(1, size / 2);
  cx = Math.round(cx); cy = Math.round(cy);
  const x0 = Math.max(1, Math.floor(cx - R)), x1 = Math.min(W - 1, Math.ceil(cx + R));
  const y0 = Math.max(1, Math.floor(cy - R)), y1 = Math.min(H - 1, Math.ceil(cy + R));
  if (x1 <= x0 || y1 <= y0) return;
  const ref = strokeRef;                      // fixed for the whole stroke
  if (!ref) return;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const inDisc = (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= R * R;
  const seen = new Uint8Array(bw * bh), keep = new Uint8Array(bw * bh);
  // start from the pixel nearest the cursor that matches the stroke colour, so drifting a
  // couple of pixels off a cable does not abandon the stroke
  let sx = -1, sy = -1, sd = 1e9;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (!inDisc(x, y) || colDist((y * W + x) * 4, ref) > tol) continue;
    const dd = (x - cx) ** 2 + (y - cy) ** 2;
    if (dd < sd) { sd = dd; sx = x; sy = y; }
  }
  if (sx < 0) return;
  const st = [[sx, sy]];
  while (st.length) {
    const [x, y] = st.pop();
    if (x < x0 || y < y0 || x > x1 || y > y1 || !inDisc(x, y)) continue;
    const b = (y - y0) * bw + (x - x0);
    if (seen[b]) continue;
    seen[b] = 1;
    const i = (y * W + x) * 4;
    if (colDist(i, ref) > tol) continue;          // different material
    if (grad[y * W + x] > gradLimit) continue;    // an edge: stop here
    keep[b] = 1;
    st.push([x-1, y]); st.push([x+1, y]); st.push([x, y-1]); st.push([x, y+1]);
  }
  // one step over the edge itself, so the boundary pixel belongs to the object
  const grown = keep.slice();
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const b = (y - y0) * bw + (x - x0);
    if (keep[b] || !inDisc(x, y)) continue;
    const n = (x > x0 && keep[b-1]) || (x < x1 && keep[b+1]) ||
              (y > y0 && keep[b-bw]) || (y < y1 && keep[b+bw]);
    if (n && colDist((y * W + x) * 4, ref) <= tol * 1.8) grown[b] = 1;
  }
  const md = mctx.getImageData(x0, y0, bw, bh);
  const er = tool === 'eraser', c = curColour();
  for (let b = 0; b < grown.length; b++) {
    if (!grown[b]) continue;
    const k = b * 4;
    if (er) md.data[k+3] = 0;
    else { md.data[k] = c[0]; md.data[k+1] = c[1]; md.data[k+2] = c[2]; md.data[k+3] = 255; }
  }
  mctx.putImageData(md, x0, y0);
}

/* ------------------------------------------------------------------ edit */
function snap() {
  undo.push(mctx.getImageData(0, 0, mask.width, mask.height));
  if (undo.length > 30) undo.shift();
}
function sampleColour(p) {
  const W = img.width, H = img.height, d = smoothData || imgData.data;
  const x = Math.round(p[0]), y = Math.round(p[1]);
  let r = 0, g = 0, b = 0, n = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
    const i = (yy * W + xx) * 4; r += d[i]; g += d[i+1]; b += d[i+2]; n++;
  }
  return n ? [r/n, g/n, b/n] : null;
}
function paint(a, b) {
  if (smart && grad && strokeRef) {                       // dab along the segment, each one a region grow
    const n = Math.max(1, Math.ceil(Math.hypot(b[0]-a[0], b[1]-a[1]) / Math.max(1, size / 3)));
    for (let i = 0; i <= n; i++)
      smartDab(a[0] + (b[0]-a[0]) * i / n, a[1] + (b[1]-a[1]) * i / n);
    return;
  }
  mctx.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
  mctx.strokeStyle = items.length ? items[item].colour : `rgb(${UNLAB.join(',')})`;
  mctx.lineWidth = size;
  mctx.lineCap = 'round'; mctx.lineJoin = 'round';
  mctx.beginPath(); mctx.moveTo(a[0], a[1]); mctx.lineTo(b[0], b[1]); mctx.stroke();
  mctx.globalCompositeOperation = 'source-over';
}
function wand(x, y) {
  x = Math.round(x); y = Math.round(y);
  const W = img.width, H = img.height, d = imgData.data;
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const s = (y * W + x) * 4, r0 = d[s], g0 = d[s+1], b0 = d[s+2];
  const seen = new Uint8Array(W * H), out = mctx.getImageData(0, 0, W, H);
  const c = curColour(), st = [[x, y]]; let n = 0;
  while (st.length && n < 80000) {
    const [cx, cy] = st.pop(), k = cy * W + cx;
    if (seen[k]) continue;
    seen[k] = 1;
    const i = k * 4;
    if (Math.abs(d[i]-r0) + Math.abs(d[i+1]-g0) + Math.abs(d[i+2]-b0) > tol * 3) continue;
    out.data[i] = c[0]; out.data[i+1] = c[1]; out.data[i+2] = c[2]; out.data[i+3] = 255; n++;
    if (cx > 0) st.push([cx-1, cy]);
    if (cx < W-1) st.push([cx+1, cy]);
    if (cy > 0) st.push([cx, cy-1]);
    if (cy < H-1) st.push([cx, cy+1]);
  }
  mctx.putImageData(out, 0, 0); dirty = true; rescan(); draw();
}
function assign(x, y) {
  /* Recolour one connected masked blob to the selected item. Connectivity is over the mask,
     not the photograph, so it does not matter what the blob looks like -- a SAM instance, a
     stroke, a wand fill -- it moves to the item in one click. */
  x = Math.round(x); y = Math.round(y);
  const W = img.width, H = img.height;
  if (x < 0 || y < 0 || x >= W || y >= H || !items.length) return;
  const out = mctx.getImageData(0, 0, W, H), d = out.data;
  if (d[(y * W + x) * 4 + 3] === 0) return;            // clicked on bare floor
  const c = curColour(), seen = new Uint8Array(W * H), st = [[x, y]];
  while (st.length) {
    const [cx, cy] = st.pop(), k = cy * W + cx;
    if (seen[k]) continue;
    seen[k] = 1;
    const i = k * 4;
    if (d[i+3] === 0) continue;                        // the blob ends here
    d[i] = c[0]; d[i+1] = c[1]; d[i+2] = c[2];
    if (cx > 0) st.push([cx-1, cy]);
    if (cx < W-1) st.push([cx+1, cy]);
    if (cy > 0) st.push([cx, cy-1]);
    if (cy < H-1) st.push([cx, cy+1]);
  }
  mctx.putImageData(out, 0, 0); dirty = true; rescan(); draw();
}
cv.addEventListener('mousedown', e => {
  if (stamp && e.button === 0 && tool !== 'move') { dropStamp(); return; }
  if (tool === 'move' && e.button === 0) {
    const q = toImg(e);
    if (!stamp && !liftAt(q[0], q[1])) return;
    moving = [stamp.x - q[0], stamp.y - q[1]];      // keep the grab point under the cursor
    draw(); return;
  }
  if (e.button === 1 || panKey || tool === 'pan') {
    panning = true; last = [e.clientX, e.clientY];
    if (tool === 'pan') cv.style.cursor = 'grabbing';
    return;
  }
  const p = toImg(e);
  if (tool === 'wand') { snap(); wand(p[0], p[1]); return; }
  if (tool === 'assign') { snap(); assign(p[0], p[1]); return; }
  if (tool === 'poly') { poly.push(p); draw(); return; }
  snap(); drawing = true; last = p;
  strokeRef = smart ? strokeSeed(p[0], p[1]) : null;
  paint(p, p); dirty = true; rescan(); draw();
});
window.addEventListener('mousemove', e => {
  if (panning) {
    view.ox += e.clientX - last[0]; view.oy += e.clientY - last[1];
    last = [e.clientX, e.clientY]; draw(); return;
  }
  const p = toImg(e);
  mouse = p;
  $('#pos').textContent = `x ${String(Math.round(p[0])).padStart(4)} y ${String(Math.round(p[1])).padStart(4)}`;
  $('#under').textContent = itemUnder(p);
  if (moving && stamp) { stamp.x = p[0] + moving[0]; stamp.y = p[1] + moving[1]; draw(); }
  else if (stamp && stamp.follow) { stamp.x = p[0]; stamp.y = p[1]; draw(); }
  if (drawing) { paint(last, p); last = p; dirty = true; rescan(); }
  draw();
});
window.addEventListener('mouseup', () => {
  drawing = false; panning = false;
  if (moving) { moving = null; dropStamp(); }       // let go and it lands
  if (tool === 'pan') cv.style.cursor = 'grab';
});
$('#wrap').addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15, nz = Math.min(24, Math.max(0.5, view.z * k));
  view.ox = mx - (mx - view.ox) * (nz / view.z);
  view.oy = my - (my - view.oy) * (nz / view.z);
  view.z = nz; draw();
}, {passive: false});

/* ------------------------------------------------------------- reference */
/* A second frame beside the one being annotated, with its mask on it. The same objects were
   re-placed run after run, so the cable drawn at 0.75 m is the cable to draw at 2.5 m: click
   it in the reference and it comes over as a stamp to move, scale and rotate into place. */
let ref = {i: -1, img: null, ids: null, mask: null, items: [], show: true,
           view: {z: 1, ox: 0, oy: 0}};
let stamp = null, refDrag = null;

function refOptions() {
  const sel = $('#refsel'); sel.innerHTML = '';
  frames.forEach((f, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = `${f.run} ·${f.frame}${f.manual_id ? '  ✓' : ''}`;
    sel.appendChild(o);
  });
  if (ref.i >= 0) sel.value = ref.i;
}
function defaultRef() {
  /* the most useful neighbour: the same objects, already labelled, in another run */
  const f = frames[cur];
  let i = frames.findIndex(g => g.oset === f.oset && g.run !== f.run && g.manual_id);
  if (i < 0) i = frames.findIndex(g => g.oset === f.oset && g.run !== f.run);
  if (i < 0) i = cur > 0 ? cur - 1 : 0;
  return i;
}
async function loadRef(i) {
  const f = frames[i];
  if (!f) return;
  ref.i = i; ref.items = itemSets[f.oset] || [];
  const im = new Image();
  await new Promise(res => { im.onload = res; im.onerror = res;
                             im.src = `/file/OWN/${f.run}/rgb/${f.frame}.png`; });
  if (!im.width) return;
  ref.img = im;
  ref.ids = await idMapFor(f, im.width, im.height, 'best', ref.items);
  const c = document.createElement('canvas');
  c.width = im.width; c.height = im.height;
  c.getContext('2d').putImageData(idsToImage(ref.ids, im.width, im.height, ref.items), 0, 0);
  ref.mask = c;
  $('#refsel').value = i;
  $('#reffoot').textContent = `${f.run} ·${f.frame} — ${f.oset}` +
    (f.oset === frames[cur].oset ? ' · same set, items carry over'
                                 : ' · different set, a copy lands unassigned');
  refFit();
}
function refFit() {
  if (!ref.img) return;
  const w = $('#refwrap').clientWidth, h = $('#refwrap').clientHeight;
  ref.view.z = Math.min(w / ref.img.width, h / ref.img.height);
  ref.view.ox = (w - ref.img.width * ref.view.z) / 2;
  ref.view.oy = (h - ref.img.height * ref.view.z) / 2;
  refDraw();
}
function refDraw() {
  const el = $('#refcv'), w = $('#refwrap').clientWidth, h = $('#refwrap').clientHeight;
  el.width = w; el.height = h;
  const c = el.getContext('2d');
  c.imageSmoothingEnabled = false;
  c.clearRect(0, 0, w, h);
  if (!ref.img) return;
  c.save(); c.translate(ref.view.ox, ref.view.oy); c.scale(ref.view.z, ref.view.z);
  c.drawImage(ref.img, 0, 0);
  if (ref.show && ref.mask) { c.globalAlpha = 0.5; c.drawImage(ref.mask, 0, 0); c.globalAlpha = 1; }
  c.restore();
}
function refOpen(on) {
  $('#ref').classList.toggle('on', on);
  $('#refbtn').classList.toggle('on', on);
  if (!on) return;
  if (ref.i < 0) { refOptions(); loadRef(defaultRef()); } else refFit();
  draw();
}
$('#refbtn').onclick = () => refOpen(!$('#ref').classList.contains('on'));
$('#refclose').onclick = () => refOpen(false);
$('#refmask').onclick = () => { ref.show = !ref.show;
                                $('#refmask').classList.toggle('on', ref.show); refDraw(); };
$('#refsel').onchange = e => loadRef(+e.target.value);
$('#refwrap').addEventListener('wheel', e => {
  e.preventDefault();
  const r = $('#refcv').getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15, nz = Math.min(24, Math.max(0.1, ref.view.z * k));
  ref.view.ox = mx - (mx - ref.view.ox) * (nz / ref.view.z);
  ref.view.oy = my - (my - ref.view.oy) * (nz / ref.view.z);
  ref.view.z = nz; refDraw();
}, {passive: false});
$('#refcv').addEventListener('mousedown', e => {
  refDrag = {x: e.clientX, y: e.clientY, moved: 0};
});
$('#refcv').addEventListener('mouseleave', () => { if (!refDrag) $('#refcv').style.cursor = 'crosshair'; });
window.addEventListener('mousemove', e => {
  if (!refDrag) return;
  const dx = e.clientX - refDrag.x, dy = e.clientY - refDrag.y;
  refDrag.moved += Math.abs(dx) + Math.abs(dy);
  refDrag.x = e.clientX; refDrag.y = e.clientY;
  if (refDrag.moved > 4) $('#refcv').style.cursor = 'grabbing';
  ref.view.ox += dx; ref.view.oy += dy; refDraw();
});
window.addEventListener('mouseup', e => {
  if (refDrag && refDrag.moved < 4) refPickAt(e);     // a click, not a drag
  if (refDrag) $('#refcv').style.cursor = 'crosshair';
  refDrag = null;
});
function refPickAt(e) {
  if (!ref.ids || !ref.img) return;
  const r = $('#refcv').getBoundingClientRect();
  const x = Math.round((e.clientX - r.left - ref.view.ox) / ref.view.z);
  const y = Math.round((e.clientY - r.top - ref.view.oy) / ref.view.z);
  const W = ref.img.width, H = ref.img.height;
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const id = ref.ids[y * W + x];
  if (id) copyFromRef(id);
}
function stampColour(idx) {
  const it = items.find(i => i.idx === idx);
  return it ? hexRgb(it.colour) : UNLAB;
}
function copyFromRef(id) {
  /* every pixel of that item in the reference, cropped to its box */
  const W = ref.img.width, H = ref.img.height;
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (ref.ids[y * W + x] !== id) continue;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (x1 < 0) return;
  // the item only carries over when both frames hold the same objects
  const same = frames[ref.i].oset === frames[cur].oset && items.some(i => i.idx === id);
  stamp = {w: x1 - x0 + 1, h: y1 - y0 + 1, src: [], x: img.width / 2, y: img.height / 2,
           scale: 1, rot: 0, follow: true, item: same ? id : UNLAB_IDX};
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++)
    if (ref.ids[y * W + x] === id) stamp.src.push((y - y0) * stamp.w + (x - x0));
  paintStamp();
  $('#hints').innerHTML = 'move the mouse to place · <kbd>click</kbd> drop · <kbd>[</kbd><kbd>]</kbd> ' +
    'scale · <kbd>;</kbd><kbd>\'</kbd> rotate · <kbd>1</kbd>…<kbd>9</kbd> item · ' +
    '<kbd>esc</kbd> cancel';
  draw();
}
function paintStamp() {
  const c = document.createElement('canvas');
  c.width = stamp.w; c.height = stamp.h;
  const cx = c.getContext('2d', {willReadFrequently: true});
  const d = cx.createImageData(stamp.w, stamp.h), col = stampColour(stamp.item);
  stamp.src.forEach(p => {
    const k = p * 4;
    d.data[k] = col[0]; d.data[k+1] = col[1]; d.data[k+2] = col[2]; d.data[k+3] = 255;
  });
  cx.putImageData(d, 0, 0);
  stamp.cv = c;
}
/* Lift the blob under the cursor off the mask and into the floating stamp, so it can be
   dragged. Only pixels of the same item travel: two items that touch stay separate. */
function liftAt(px, py) {
  const W = img.width, H = img.height;
  const x = Math.round(px), y = Math.round(py);
  if (x < 0 || y < 0 || x >= W || y >= H) return false;
  const d = mctx.getImageData(0, 0, W, H), dd = d.data;
  if (!dd[(y * W + x) * 4 + 3]) return false;
  const i0 = (y * W + x) * 4;
  const id = nearestItem(dd[i0], dd[i0+1], dd[i0+2]);
  const seen = new Uint8Array(W * H), got = [], st = [[x, y]];
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  while (st.length) {
    const [cx, cy] = st.pop(), k = cy * W + cx;
    if (cx < 0 || cy < 0 || cx >= W || cy >= H || seen[k]) continue;
    seen[k] = 1;
    const i = k * 4;
    if (!dd[i+3] || nearestItem(dd[i], dd[i+1], dd[i+2]) !== id) continue;
    got.push(k); dd[i+3] = 0;                       // it leaves the mask while in the air
    if (cx < x0) x0 = cx;
    if (cx > x1) x1 = cx;
    if (cy < y0) y0 = cy;
    if (cy > y1) y1 = cy;
    st.push([cx-1, cy]); st.push([cx+1, cy]); st.push([cx, cy-1]); st.push([cx, cy+1]);
  }
  if (!got.length) return false;
  snap();
  mctx.putImageData(d, 0, 0);
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  stamp = {w: w, h: h, src: got.map(k => ((k / W | 0) - y0) * w + (k % W - x0)),
           x: x0 + w / 2, y: y0 + h / 2, scale: 1, rot: 0, follow: false, item: id,
           home: {x: x0 + w / 2, y: y0 + h / 2}};
  paintStamp();
  return true;
}
function dropStamp() {
  if (!stamp) return;
  if (!stamp.home) snap();     // a lifted blob already snapped before it left the mask, and
                               // one undo should put a move back, not half of it
  mctx.save();
  mctx.imageSmoothingEnabled = false;
  mctx.translate(stamp.x, stamp.y);
  mctx.rotate(stamp.rot);
  mctx.scale(stamp.scale, stamp.scale);
  mctx.drawImage(stamp.cv, -stamp.w / 2, -stamp.h / 2);
  mctx.restore();
  /* rotating and scaling feathers the edge; a mask may not be soft, so threshold what was
     just laid down and snap it back to the item's exact colour */
  const rad = Math.hypot(stamp.w, stamp.h) * stamp.scale / 2 + 2;
  harden(stamp.x - rad, stamp.y - rad, stamp.x + rad, stamp.y + rad);
  stamp = null; dirty = true; rescan(); draw(); paintKeys();
}
function harden(x0, y0, x1, y1) {
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(mask.width - 1, Math.ceil(x1)); y1 = Math.min(mask.height - 1, Math.ceil(y1));
  if (x1 < x0 || y1 < y0) return;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, d = mctx.getImageData(x0, y0, w, h);
  const cache = new Map();
  for (let i = 0; i < d.data.length; i += 4) {
    const a = d.data[i+3];
    if (!a) continue;
    if (a < 128) { d.data[i+3] = 0; continue; }
    d.data[i+3] = 255;
    const k = (d.data[i] << 16) | (d.data[i+1] << 8) | d.data[i+2];
    let c = cache.get(k);
    if (!c) { c = stampColour(nearestItem(d.data[i], d.data[i+1], d.data[i+2])); cache.set(k, c); }
    d.data[i] = c[0]; d.data[i+1] = c[1]; d.data[i+2] = c[2];
  }
  mctx.putImageData(d, x0, y0);
}

/* -------------------------------------------------------------------- ui */
function setTool(t) {
  tool = t; poly = [];
  cv.style.cursor = t === 'pan' ? 'grab' : 'none';
  document.querySelectorAll('.tool[data-t]').forEach(e => e.classList.toggle('on', e.dataset.t === t));
  draw();
}
document.querySelectorAll('.tool[data-t]').forEach(el => el.onclick = () => setTool(el.dataset.t));
$('#loadauto').onclick = () => { snap(); loadInto('auto'); dirty = true; };
$('#loadsam').onclick = () => { snap(); loadInto('sam'); dirty = true; };
$('#clear').onclick = () => { snap(); mctx.clearRect(0, 0, mask.width, mask.height);
  dirty = true; rescan(); draw(); };
$('#save').onclick = () => save();
function setFlag(on, note) {
  const f = frames[cur];
  f.flag = on ? {note: note || ''} : null;
  $('#flag').classList.toggle('on', !!f.flag);
  $('#note').value = f.flag ? f.flag.note : '';
  renderList();
  fetch(`/api/flag?run=${f.run}&frame=${f.frame}`,
        {method: 'POST', body: JSON.stringify({flag: !!f.flag, note: note || ''})});
}
/* Pressing the button must not take focus off the note field: its change event would flag
   the frame first and the click would then toggle that straight back off, which looks like
   nothing happening. */
$('#flag').onmousedown = e => e.preventDefault();
$('#flag').onclick = () => setFlag(!frames[cur].flag, $('#note').value);
$('#note').onchange = () => {
  const want = $('#note').value !== '' || !!frames[cur].flag;
  if (want || frames[cur].flag) setFlag(want, $('#note').value);
};
$('#note').onkeydown = e => { if (e.key === 'Enter') $('#note').blur(); };
$('#only').onchange = renderList;
$('#onlyflag').onchange = renderList;
$('#carry').onclick = () => { carry = !carry; $('#carry').classList.toggle('on', carry);
$('#smart').classList.toggle('on', smart); };
$('#smart').onclick = () => { smart = !smart; $('#smart').classList.toggle('on', smart); draw(); };
$('#help').onclick = () => $('#keys').classList.add('on');
$('#kclose').onclick = () => $('#keys').classList.remove('on');
$('#kreset').onclick = () => {
  KEYS = Object.assign({}, DEFAULTS);
  MODES = {}; HOLDABLE.forEach(a => MODES[a] = MODE_DEF[a] || 'toggle');
  localStorage.setItem('annot_keys', JSON.stringify(KEYS));
  localStorage.setItem('annot_modes', JSON.stringify(MODES));
  buildKeyPanel(); paintKeys();
};
$('#keys').onclick = e => { if (e.target.id === 'keys') $('#keys').classList.remove('on'); };

async function save(quiet) {
  /* Two files: the 0/255 mask everything downstream already reads, and the id map that says
     which item each pixel came from. */
  const d = mctx.getImageData(0, 0, mask.width, mask.height).data;
  const bin = blank(), ids = blank(), cache = new Map();
  for (let i = 0; i < d.length; i += 4) {
    if (d[i+3] === 0) continue;
    bin.data[i] = bin.data[i+1] = bin.data[i+2] = 255;
    const k = (d[i] << 16) | (d[i+1] << 8) | d[i+2];
    let v = cache.get(k);
    if (v === undefined) { v = nearestItem(d[i], d[i+1], d[i+2]); cache.set(k, v); }
    ids.data[i] = ids.data[i+1] = ids.data[i+2] = v;
  }
  const f = frames[cur];
  await put(bin, `/api/save?run=${f.run}&frame=${f.frame}&kind=mask`);
  await put(ids, `/api/save?run=${f.run}&frame=${f.frame}&kind=id`);
  f.manual = true; f.manual_id = true; dirty = false; renderList();
  if (!quiet) { $('#save').classList.add('on'); setTimeout(() => $('#save').classList.remove('on'), 350); }
}
function blank() {
  const o = new ImageData(mask.width, mask.height);
  for (let i = 3; i < o.data.length; i += 4) o.data[i] = 255;
  return o;
}
async function put(imgdata, url) {
  const t = document.createElement('canvas');
  t.width = mask.width; t.height = mask.height;
  t.getContext('2d').putImageData(imgdata, 0, 0);
  const blob = await new Promise(r => t.toBlob(r, 'image/png'));
  await fetch(url, {method: 'POST', body: blob});
}

window.addEventListener('keydown', e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && !listening) return;
  if (listening) {                              // capturing a new shortcut
    e.preventDefault();
    if (e.key !== 'Escape' && e.key.length === 1) setKey(listening.action, e.key);
    else buildKeyPanel();
    listening = null; return;
  }
  if (stamp) {
    const step = e.shiftKey ? 10 : 1;
    if (e.key === 'Escape') {
      if (stamp.home) {                             // it came off this frame: put it back
        stamp.x = stamp.home.x; stamp.y = stamp.home.y; stamp.scale = 1; stamp.rot = 0;
        moving = null; dropStamp();
      } else { stamp = null; draw(); paintKeys(); }
      return;
    }
    if (e.key === 'Enter') { dropStamp(); return; }
    if (e.key === '[' || e.key === ']') {
      stamp.scale *= e.key === ']' ? 1.05 : 1 / 1.05; draw(); return;
    }
    if (e.key === ';' || e.key === "'") {
      stamp.rot += (e.key === "'" ? 1 : -1) * Math.PI / 60; draw(); return;
    }
    if (e.key.startsWith('Arrow')) {
      e.preventDefault(); stamp.follow = false;
      if (e.key === 'ArrowLeft') stamp.x -= step;
      if (e.key === 'ArrowRight') stamp.x += step;
      if (e.key === 'ArrowUp') stamp.y -= step;
      if (e.key === 'ArrowDown') stamp.y += step;
      draw(); return;
    }
    if (e.key >= '1' && e.key <= '9' && items[+e.key - 1]) {
      stamp.item = items[+e.key - 1].idx; paintStamp(); draw(); return;
    }
  }
  if (e.key === ' ') { panKey = true; e.preventDefault(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault();
    const u = undo.pop();
    if (u) { mctx.putImageData(u, 0, 0); dirty = true; rescan(); draw(); }
    return;
  }
  if (e.metaKey || e.ctrlKey) return;
  if (e.key >= '1' && e.key <= '9' && items.length) {     // pick the item to paint with
    e.preventDefault(); setItem(+e.key - 1); return;
  }
  if (e.key === 'Enter' && tool === 'poly' && poly.length > 2) {
    snap(); mctx.fillStyle = items.length ? items[item].colour : `rgb(${UNLAB.join(',')})`;
    mctx.beginPath();
    mctx.moveTo(poly[0][0], poly[0][1]);
    poly.slice(1).forEach(p => mctx.lineTo(p[0], p[1]));
    mctx.closePath(); mctx.fill(); poly = []; dirty = true; rescan(); draw(); return;
  }
  if (e.key === 'Escape') { poly = []; $('#keys').classList.remove('on'); draw(); return; }
  const act = Object.keys(KEYS).find(a => KEYS[a] && KEYS[a] === e.key);
  if (!act) return;
  e.preventDefault();
  if (HOLDABLE.includes(act) && MODES[act] === 'hold') {
    if (e.repeat || held) return;                 // autorepeat is not a second press
    held = {act: act, key: e.key, tool: tool, hide: hide};
    if (act === 'hide') { hide = true; draw(); } else setTool(act);
    return;
  }
  if (['brush', 'eraser', 'wand', 'poly', 'assign', 'move', 'pan'].includes(act)) setTool(act);
  else if (act === 'sizeDown') { size = Math.max(1, size - 1); $('#sz').textContent = size; draw(); }
  else if (act === 'sizeUp') { size = Math.min(120, size + 1); $('#sz').textContent = size; draw(); }
  else if (act === 'tolDown') { tol = Math.max(2, tol - 3); $('#tol').textContent = tol; }
  else if (act === 'tolUp') { tol = Math.min(120, tol + 3); $('#tol').textContent = tol; }
  else if (act === 'hide') { hide = !hide; draw(); }
  else if (act === 'fit') fit();
  else if (act === 'loadAuto') $('#loadauto').click();
  else if (act === 'loadSam') $('#loadsam').click();
  else if (act === 'clear') $('#clear').click();
  else if (act === 'ref') $('#refbtn').click();
  else if (act === 'flag') $('#flag').click();
  else if (act === 'carry') $('#carry').click();
  else if (act === 'smart') $('#smart').click();
  else if (act === 'next') { if (cur + 1 < frames.length) load(cur + 1, carry); }
  else if (act === 'prev') { if (cur > 0) load(cur - 1, carry); }
});
window.addEventListener('keyup', e => {
  if (e.key === ' ') panKey = false;
  if (held && e.key === held.key) {
    if (held.act === 'hide') { hide = held.hide; draw(); } else setTool(held.tool);
    held = null;
  }
});
window.addEventListener('resize', () => { draw(); if ($('#ref').classList.contains('on')) refFit(); });
function banner(msg) {
  const b = document.createElement('div');
  b.style.cssText = 'position:fixed;bottom:0;left:0;right:0;background:#7a1f1f;color:#fff;' +
                    'padding:6px 12px;z-index:99;font:12px ui-monospace,Menlo,monospace';
  b.textContent = msg;
  document.body.appendChild(b);
}
window.addEventListener('error', ev => banner('error: ' + ev.message));
window.addEventListener('unhandledrejection', ev => banner('error: ' + (ev.reason && ev.reason.message || ev.reason)));
cv.addEventListener('mouseleave', () => { mouse = null; draw(); });
document.querySelectorAll('.tool[data-hint]').forEach(el => {   // keep hints on screen
  el.addEventListener('mouseenter', () => {
    el.classList.toggle('hr', el.getBoundingClientRect().left + 300 > window.innerWidth);
  });
});

buildKeyPanel(); paintKeys();
$('#carry').classList.toggle('on', carry);
$('#smart').classList.toggle('on', smart);
boot(); setTool('brush');
</script>
"""


FLAGS = f"{MAN}/flags.json"


def read_flags():
    """<run>_<frame> -> {"note": ...}, the frames marked to come back to."""
    try:
        return json.load(open(FLAGS))
    except Exception:
        return {}


def unassigned(path):
    """Does this saved id map still hold pixels nobody put a name to? Items are 1..N and
    UNLABELLED is 255, so the maximum answers it."""
    if not os.path.exists(path):
        return False
    try:
        from PIL import Image
        return Image.open(path).convert("L").getextrema()[1] == UNLABELLED
    except Exception:
        return False


def frame_list():
    meta = {}
    p = f"{OWN}/gt_summary.json"
    if os.path.exists(p):
        meta = {r["run"]: r for r in json.load(open(p))}
    sets = run_sets()
    flags = read_flags()
    out = []
    for run in sorted(os.listdir(OWN)):
        d = f"{OWN}/{run}/rgb"
        if not os.path.isdir(d):
            continue
        rel = bool(meta.get(run, {}).get("reliable", True))
        oset = sets.get(run, run.split("_")[2] if len(run.split("_")) > 2 else "")
        for f in sorted(os.listdir(d), key=lambda x: int(x[:-4])):
            i = f[:-4]
            out.append(dict(run=run, frame=i, reliable=rel, oset=oset,
                            flag=flags.get(f"{run}_{i}"),
                            manual=os.path.exists(f"{MAN}/{run}_{i}.png"),
                            manual_id=os.path.exists(f"{MAN}/{run}_{i}_id.png"),
                            todo=unassigned(f"{MAN}/{run}_{i}_id.png"),
                            sam=os.path.exists(f"{OWN}/{run}/sam/{i}.png")))
    out.sort(key=lambda r: (r["reliable"], r["run"], int(r["frame"])))   # needs-work first
    return out


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

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/list":
            return self._send(200, json.dumps(frame_list()).encode(), "application/json")
        if u.path == "/api/flags":
            return self._send(200, json.dumps(read_flags()).encode(), "application/json")
        if u.path == "/api/items":
            return self._send(200, json.dumps(item_table()).encode(), "application/json")
        if u.path.startswith("/file/"):
            rel = urllib.parse.unquote(u.path[len("/file/"):])
            base = f"{ROOT}/data"
            full = os.path.normpath(os.path.join(base, rel))
            if not full.startswith(base) or not os.path.exists(full):
                return self._send(404, b"no")
            ctype = "application/json" if full.endswith(".json") else "image/png"
            with open(full, "rb") as f:
                return self._send(200, f.read(), ctype)
        self._send(404, b"no")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/api/flag" and "run" in q and "frame" in q:
            run, frame = q["run"][0], q["frame"][0]
            if "/" in run or "/" in frame:
                return self._send(400, b"bad")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            flags = read_flags()
            key = f"{run}_{frame}"
            if body.get("flag"):
                flags[key] = {"note": body.get("note", "")}
            else:
                flags.pop(key, None)
            os.makedirs(MAN, exist_ok=True)
            json.dump(flags, open(FLAGS, "w"), indent=1, sort_keys=True)
            return self._send(200, b"ok", "text/plain")
        if u.path != "/api/save" or "run" not in q or "frame" not in q:
            return self._send(400, b"bad")
        run, frame = q["run"][0], q["frame"][0]
        kind = q.get("kind", ["mask"])[0]
        if "/" in run or "/" in frame or kind not in ("mask", "id"):
            return self._send(400, b"bad")
        os.makedirs(MAN, exist_ok=True)
        data = self.rfile.read(int(self.headers["Content-Length"]))
        name = f"{run}_{frame}.png" if kind == "mask" else f"{run}_{frame}_id.png"
        with open(f"{MAN}/{name}", "wb") as f:
            f.write(data)
        print(f"saved {name} ({len(data)} bytes)")
        self._send(200, b"ok", "text/plain")


def free_port(start):
    for p in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)):
                return p
    raise SystemExit("no free port")


if __name__ == "__main__":
    os.makedirs(MAN, exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else free_port(PORT)
    n = len(frame_list())
    print(f"{n} frames · masks -> {MAN}\n  http://localhost:{port}\n")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as s:
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
