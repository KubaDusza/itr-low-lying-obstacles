"""Minimal mask annotator for the own-recordings dataset.

    python code/annotate.py            # then open the URL it prints

Masks are written to data/OWN_manual/<run>_<frame>.png (255 = obstacle), the same format
own_gt.py produces, so anything saved here overrides the automatic mask for that frame.

The point of this tool rather than a general one: a frame opens with its automatic mask
already loaded so you correct rather than draw, the wand fills a whole rag in one click, and
"carry" moves the mask you just drew onto the next frame, which is nearly identical because
the camera never moved inside a run.

Every shortcut is remappable in the panel behind the ? button, and the bindings persist in the
browser.
"""
import http.server
import json
import os
import socket
import socketserver
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN, MAN = f"{ROOT}/data/OWN", f"{ROOT}/data/OWN_manual"
PORT = int(os.environ.get("ANNOT_PORT", 8765))

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
  #filter { padding:8px 14px; border-bottom:1px solid var(--line); color:var(--muted); }
  #list { overflow:auto; flex:1; }
  .row { padding:6px 14px; cursor:pointer; border-bottom:1px solid #22252b; display:flex;
         justify-content:space-between; gap:8px; }
  .row:hover { background:#232832; }
  .row.sel { background:var(--accent); color:#fff; }
  .row .tag { font-size:11px; color:var(--muted); flex:none; }
  .row.sel .tag { color:#dbe6ff; }
  .row.done .tag::after { content:" saved"; color:var(--ok); }
  .row.sel.done .tag::after { color:#bdf5db; }
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
  #wrap { flex:1; overflow:hidden; position:relative; background:#0d0f12; }
  canvas { position:absolute; image-rendering:pixelated; cursor:none; }
  #status { padding:6px 14px; border-top:1px solid var(--line); color:var(--muted);
            background:var(--panel); display:flex; gap:18px; flex-wrap:nowrap; }
  /* fixed widths and a monospace figure so the bar does not jitter as the cursor moves */
  #pos, #px, #zoom { font:12px/1.5 ui-monospace,Menlo,monospace; flex:none;
                     display:inline-block; text-align:left; }
  #pos { width:116px; } #px { width:84px; } #zoom { width:52px; }
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
  #keyfoot { display:flex; gap:10px; margin-top:16px; }
</style>
<div id="side">
  <h1>mask annotator</h1>
  <label id="filter"><input type="checkbox" id="only"> only runs needing work</label>
  <div id="list"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="title">—</b>
    <span class="tool" data-t="brush" data-hint="Paint mask freehand. Hold and drag; the ring on the canvas is the exact size you are painting.">brush<span class="k" data-k="brush"></span></span>
    <span class="tool" data-t="eraser" data-hint="Remove mask freehand. Same ring, same size control — use a big one to wipe the automatic mask's false blobs.">eraser<span class="k" data-k="eraser"></span></span>
    <span class="tool" data-t="wand" data-hint="Flood fill by colour: click one pixel and every connected pixel of a similar colour joins the mask. One click covers a whole rag or tape roll. Tolerance decides how different a pixel may be before it stops spreading.">wand<span class="k" data-k="wand"></span></span>
    <span class="tool" data-t="poly" data-hint="Polygon: click each corner, Enter fills the shape, Esc cancels. Best for straight-edged objects like the pink rag.">polygon<span class="k" data-k="poly"></span></span>
    <span class="tool" id="smart" data-hint="Smart edge mode. On mouse-down the colour under the cursor is sampled, and the stroke then only affects pixels close to that colour — so a sloppy sweep along a cable paints the cable and stops at its edge. The eraser works the same way: click on floor and only floor-coloured mask is removed. Uses the tolerance setting.">smart<span class="k" data-k="smart"></span></span>
    <span class="tool" id="szbox" data-hint="Brush and eraser diameter, in image pixels.">size <b id="sz">6</b><span class="k" data-k="sizeDown"></span><span class="k" data-k="sizeUp"></span></span>
    <span class="tool" id="tolbox" data-hint="How different a pixel may be from the one you clicked before the wand stops. Raise it if the wand grabs too little, lower it if it bleeds onto the floor.">tol <b id="tol">14</b><span class="k" data-k="tolDown"></span><span class="k" data-k="tolUp"></span></span>
    <span class="tool" id="loadsam" data-hint="Replace the mask with the SAM 3 pre-label from sam3_prelabel.py, discarding your edits to this frame. This is what a frame opens with when it exists: SAM 3 was prompted with the object set that was actually placed, so the edges are usually close and the work is removing what it invented and adding what it missed.">load sam<span class="k" data-k="loadSam"></span></span>
    <span class="tool" id="loadauto" data-hint="Replace the mask with the background-subtraction mask from own_gt.py, discarding your edits to this frame. Worth a look when SAM 3 missed an object entirely: the two methods fail in different places.">load auto<span class="k" data-k="loadAuto"></span></span>
    <span class="tool" id="clear" data-hint="Empty the mask for this frame.">clear<span class="k" data-k="clear"></span></span>
    <span class="tool" id="carry" data-hint="Carry the mask to the next frame. Camera and objects do not move inside a run, so the five frames are nearly identical — with this on, moving to the next frame keeps what you drew instead of loading that frame's own mask.">carry<span class="k" data-k="carry"></span></span>
    <span class="tool" id="save" data-hint="Write the mask to data/OWN_manual/. Also happens automatically when you change frame.">save<span class="k">⌘s</span></span>
    <span class="tool" id="help" data-hint="Change any shortcut.">?</span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="status">
    <span id="pos">—</span><span id="px">0 px</span><span id="zoom">1x</span>
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
const DEFAULTS = {brush:'w', eraser:'e', wand:'q', poly:'p', sizeDown:'[', sizeUp:']',
                  tolDown:',', tolUp:'.', hide:'h', fit:'0', loadAuto:'a', loadSam:'g',
                  clear:'x',
                  carry:'c', smart:'s', next:'n', prev:'N'};
const LABEL = {brush:'brush', eraser:'eraser', wand:'wand', poly:'polygon',
               sizeDown:'smaller brush', sizeUp:'bigger brush', tolDown:'lower tolerance',
               tolUp:'raise tolerance', hide:'hide / show mask', fit:'fit to window',
               loadAuto:'load automatic mask', loadSam:'load SAM 3 mask',
               clear:'clear mask',
               carry:'toggle carry to next frame', smart:'toggle smart edges',
               next:'next frame', prev:'previous frame'};
let KEYS = Object.assign({}, DEFAULTS);
try { Object.assign(KEYS, JSON.parse(localStorage.getItem('annot_keys') || '{}')); } catch (e) {}

let frames = [], cur = -1, img = null, imgData = null;
let mask = document.createElement('canvas'), mctx = null;
let view = {z: 2, ox: 0, oy: 0}, tool = 'brush', size = 6, tol = 14, carry = true;
let drawing = false, panning = false, last = null, hide = false, dirty = false;
let smart = true, strokeRef = null;
let undo = [], poly = [], mouse = null, panKey = false, listening = null;
const cv = $('#cv'), ctx = cv.getContext('2d');

/* ------------------------------------------------------------------ keys */
const keyLabel = k => k === ' ' ? 'space' : (k || '—');
function paintKeys() {
  document.querySelectorAll('[data-k]').forEach(e => {
    e.textContent = KEYS[e.dataset.k] ? ' ' + keyLabel(KEYS[e.dataset.k]) : '';
  });
  $('#hints').innerHTML =
    `scroll zoom · space-drag pan · <kbd>${keyLabel(KEYS.fit)}</kbd> fit · ` +
    `<kbd>${keyLabel(KEYS.hide)}</kbd> hide · <kbd>${keyLabel(KEYS.next)}</kbd>/` +
    `<kbd>${keyLabel(KEYS.prev)}</kbd> frame · <kbd>⌘z</kbd> undo`;
}
function buildKeyPanel() {
  const box = $('#krows'); box.innerHTML = '';
  Object.keys(DEFAULTS).forEach(a => {
    const r = document.createElement('div'); r.className = 'krow';
    r.innerHTML = `<span>${LABEL[a]}</span>`;
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

/* ---------------------------------------------------------------- frames */
async function boot() {
  frames = await (await fetch('/api/list')).json();
  renderList();
  if (frames.length) load(0);
}
function renderList() {
  const only = $('#only').checked, l = $('#list'); l.innerHTML = '';
  frames.forEach((f, i) => {
    if (only && f.reliable && !f.manual) return;
    const d = document.createElement('div');
    d.className = 'row' + (i === cur ? ' sel' : '') + (f.manual ? ' done' : '');
    d.innerHTML = `<span>${f.run}<span class="tag"> ·${f.frame}</span></span>` +
                  `<span class="tag ${f.reliable ? '' : 'bad'}">${f.reliable ? 'ok' : 'needs work'}</span>`;
    d.onclick = () => load(i);
    l.appendChild(d);
  });
}
async function load(i, keep) {
 try {
  if (dirty && cur >= 0) await save(true);
  const keepData = keep && mctx ? mctx.getImageData(0, 0, mask.width, mask.height) : null;
  cur = i; poly = []; undo = [];
  const f = frames[i];
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
    mctx.putImageData(keepData, 0, 0); dirty = true; fit(); renderList(); return;
  }
  await loadMask(f.manual ? `/file/OWN_manual/${f.run}_${f.frame}.png`
                 : f.sam  ? `/file/OWN/${f.run}/sam/${f.frame}.png`
                          : `/file/OWN/${f.run}/gt/${f.frame}.png`);
  dirty = false; fit(); renderList();
 } catch (err) { banner('load failed: ' + (err && err.message || err)); }
}
function loadMask(url) {
  return new Promise(res => {
    const m = new Image();
    m.onload = () => {
      const t = document.createElement('canvas');
      t.width = img.width; t.height = img.height;
      const c = t.getContext('2d', {willReadFrequently: true});
      c.drawImage(m, 0, 0, img.width, img.height);
      const d = c.getImageData(0, 0, img.width, img.height);
      const out = mctx.createImageData(img.width, img.height);
      for (let i = 0; i < d.data.length; i += 4)
        if (d.data[i] > 127) { out.data[i] = 255; out.data[i+3] = 255; }
      mctx.putImageData(out, 0, 0); draw(); res();
    };
    m.onerror = () => { draw(); res(); };
    m.src = url + '?t=' + Date.now();
  });
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
  if (!hide) { ctx.globalAlpha = 0.5; ctx.drawImage(mask, 0, 0); ctx.globalAlpha = 1; }
  if (poly.length) {
    ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 1 / view.z;
    ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
    poly.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
  }
  ctx.restore();
  if (mouse) {                              // the cursor IS the brush size
    const r = (tool === 'brush' || tool === 'eraser') ? Math.max(2, size / 2 * view.z) : 4;
    const x = mouse[0] * view.z + view.ox, y = mouse[1] * view.z + view.oy;
    ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(0,0,0,.55)';
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = tool === 'eraser' ? '#ffffff' : '#ff3b30';
    if (smart && (tool === 'brush' || tool === 'eraser')) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
  }
  $('#zoom').textContent = view.z.toFixed(1) + 'x';
  countPx();
}
let pxTimer = null;
function countPx() {
  if (!mctx) return;
  clearTimeout(pxTimer);
  pxTimer = setTimeout(() => {
    const d = mctx.getImageData(0, 0, mask.width, mask.height).data;
    let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    $('#px').textContent = String(n).padStart(6) + ' px';
  }, 120);
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
  const er = tool === 'eraser';
  for (let b = 0; b < grown.length; b++) {
    if (!grown[b]) continue;
    const k = b * 4;
    if (er) md.data[k+3] = 0; else { md.data[k] = 255; md.data[k+3] = 255; }
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
  mctx.strokeStyle = '#ff0000'; mctx.lineWidth = size;
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
  const st = [[x, y]]; let n = 0;
  while (st.length && n < 80000) {
    const [cx, cy] = st.pop(), k = cy * W + cx;
    if (seen[k]) continue;
    seen[k] = 1;
    const i = k * 4;
    if (Math.abs(d[i]-r0) + Math.abs(d[i+1]-g0) + Math.abs(d[i+2]-b0) > tol * 3) continue;
    out.data[i] = 255; out.data[i+3] = 255; n++;
    if (cx > 0) st.push([cx-1, cy]);
    if (cx < W-1) st.push([cx+1, cy]);
    if (cy > 0) st.push([cx, cy-1]);
    if (cy < H-1) st.push([cx, cy+1]);
  }
  mctx.putImageData(out, 0, 0); dirty = true; draw();
}
cv.addEventListener('mousedown', e => {
  if (e.button === 1 || panKey) { panning = true; last = [e.clientX, e.clientY]; return; }
  const p = toImg(e);
  if (tool === 'wand') { snap(); wand(p[0], p[1]); return; }
  if (tool === 'poly') { poly.push(p); draw(); return; }
  snap(); drawing = true; last = p;
  strokeRef = smart ? strokeSeed(p[0], p[1]) : null;
  paint(p, p); dirty = true; draw();
});
window.addEventListener('mousemove', e => {
  if (panning) {
    view.ox += e.clientX - last[0]; view.oy += e.clientY - last[1];
    last = [e.clientX, e.clientY]; draw(); return;
  }
  const p = toImg(e);
  mouse = p;
  $('#pos').textContent = `x ${String(Math.round(p[0])).padStart(4)} y ${String(Math.round(p[1])).padStart(4)}`;
  if (drawing) { paint(last, p); last = p; dirty = true; }
  draw();
});
window.addEventListener('mouseup', () => { drawing = false; panning = false; });
$('#wrap').addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15, nz = Math.min(24, Math.max(0.5, view.z * k));
  view.ox = mx - (mx - view.ox) * (nz / view.z);
  view.oy = my - (my - view.oy) * (nz / view.z);
  view.z = nz; draw();
}, {passive: false});

/* -------------------------------------------------------------------- ui */
function setTool(t) {
  tool = t; poly = [];
  document.querySelectorAll('.tool[data-t]').forEach(e => e.classList.toggle('on', e.dataset.t === t));
  draw();
}
document.querySelectorAll('.tool[data-t]').forEach(el => el.onclick = () => setTool(el.dataset.t));
$('#loadauto').onclick = () => {
  snap(); loadMask(`/file/OWN/${frames[cur].run}/gt/${frames[cur].frame}.png`); dirty = true;
};
$('#loadsam').onclick = () => {
  snap(); loadMask(`/file/OWN/${frames[cur].run}/sam/${frames[cur].frame}.png`); dirty = true;
};
$('#clear').onclick = () => { snap(); mctx.clearRect(0, 0, mask.width, mask.height); dirty = true; draw(); };
$('#save').onclick = () => save();
$('#only').onchange = renderList;
$('#carry').onclick = () => { carry = !carry; $('#carry').classList.toggle('on', carry);
$('#smart').classList.toggle('on', smart); };
$('#smart').onclick = () => { smart = !smart; $('#smart').classList.toggle('on', smart); draw(); };
$('#help').onclick = () => $('#keys').classList.add('on');
$('#kclose').onclick = () => $('#keys').classList.remove('on');
$('#kreset').onclick = () => {
  KEYS = Object.assign({}, DEFAULTS);
  localStorage.setItem('annot_keys', JSON.stringify(KEYS)); buildKeyPanel(); paintKeys();
};
$('#keys').onclick = e => { if (e.target.id === 'keys') $('#keys').classList.remove('on'); };

async function save(quiet) {
  const t = document.createElement('canvas');
  t.width = mask.width; t.height = mask.height;
  const c = t.getContext('2d');
  const d = mctx.getImageData(0, 0, mask.width, mask.height);
  const o = c.createImageData(t.width, t.height);
  for (let i = 0; i < d.data.length; i += 4) {
    const v = d.data[i+3] > 0 ? 255 : 0;
    o.data[i] = o.data[i+1] = o.data[i+2] = v; o.data[i+3] = 255;
  }
  c.putImageData(o, 0, 0);
  const blob = await new Promise(r => t.toBlob(r, 'image/png'));
  const f = frames[cur];
  await fetch(`/api/save?run=${f.run}&frame=${f.frame}`, {method: 'POST', body: blob});
  f.manual = true; dirty = false; renderList();
  if (!quiet) { $('#save').classList.add('on'); setTimeout(() => $('#save').classList.remove('on'), 350); }
}

window.addEventListener('keydown', e => {
  if (listening) {                              // capturing a new shortcut
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
    if (u) { mctx.putImageData(u, 0, 0); dirty = true; draw(); }
    return;
  }
  if (e.metaKey || e.ctrlKey) return;
  if (e.key === 'Enter' && tool === 'poly' && poly.length > 2) {
    snap(); mctx.fillStyle = '#ff0000'; mctx.beginPath();
    mctx.moveTo(poly[0][0], poly[0][1]);
    poly.slice(1).forEach(p => mctx.lineTo(p[0], p[1]));
    mctx.closePath(); mctx.fill(); poly = []; dirty = true; draw(); return;
  }
  if (e.key === 'Escape') { poly = []; $('#keys').classList.remove('on'); draw(); return; }
  const act = Object.keys(KEYS).find(a => KEYS[a] && KEYS[a] === e.key);
  if (!act) return;
  e.preventDefault();
  if (['brush', 'eraser', 'wand', 'poly'].includes(act)) setTool(act);
  else if (act === 'sizeDown') { size = Math.max(1, size - 1); $('#sz').textContent = size; draw(); }
  else if (act === 'sizeUp') { size = Math.min(120, size + 1); $('#sz').textContent = size; draw(); }
  else if (act === 'tolDown') { tol = Math.max(2, tol - 3); $('#tol').textContent = tol; }
  else if (act === 'tolUp') { tol = Math.min(120, tol + 3); $('#tol').textContent = tol; }
  else if (act === 'hide') { hide = !hide; draw(); }
  else if (act === 'fit') fit();
  else if (act === 'loadAuto') $('#loadauto').click();
  else if (act === 'loadSam') $('#loadsam').click();
  else if (act === 'clear') $('#clear').click();
  else if (act === 'carry') $('#carry').click();
  else if (act === 'smart') $('#smart').click();
  else if (act === 'next') { if (cur + 1 < frames.length) load(cur + 1, carry); }
  else if (act === 'prev') { if (cur > 0) load(cur - 1, carry); }
});
window.addEventListener('keyup', e => { if (e.key === ' ') panKey = false; });
window.addEventListener('resize', () => draw());
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


def frame_list():
    meta = {}
    p = f"{OWN}/gt_summary.json"
    if os.path.exists(p):
        meta = {r["run"]: r for r in json.load(open(p))}
    out = []
    for run in sorted(os.listdir(OWN)):
        d = f"{OWN}/{run}/rgb"
        if not os.path.isdir(d):
            continue
        rel = bool(meta.get(run, {}).get("reliable", True))
        for f in sorted(os.listdir(d), key=lambda x: int(x[:-4])):
            i = f[:-4]
            out.append(dict(run=run, frame=i, reliable=rel,
                            manual=os.path.exists(f"{MAN}/{run}_{i}.png"),
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
        if u.path.startswith("/file/"):
            rel = urllib.parse.unquote(u.path[len("/file/"):])
            base = f"{ROOT}/data"
            full = os.path.normpath(os.path.join(base, rel))
            if not full.startswith(base) or not os.path.exists(full):
                return self._send(404, b"no")
            with open(full, "rb") as f:
                return self._send(200, f.read(), "image/png")
        self._send(404, b"no")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path != "/api/save" or "run" not in q or "frame" not in q:
            return self._send(400, b"bad")
        run, frame = q["run"][0], q["frame"][0]
        if "/" in run or "/" in frame:
            return self._send(400, b"bad")
        os.makedirs(MAN, exist_ok=True)
        data = self.rfile.read(int(self.headers["Content-Length"]))
        with open(f"{MAN}/{run}_{frame}.png", "wb") as f:
            f.write(data)
        print(f"saved {run}_{frame}.png ({len(data)} bytes)")
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
