"""Minimal mask annotator for the own-recordings dataset.

    python code/annotate.py            # then open http://localhost:8765

Masks are written to data/OWN_manual/<run>_<frame>.png (255 = obstacle), which is the same
format own_gt.py produces, so anything saved here overrides the automatic mask for that frame.

The point of this tool rather than a general one: you can load the automatic mask and fix it
instead of drawing from scratch, and the wand fills a whole rag or tape roll in one click.

Tools      b brush · e eraser · w wand (flood fill by colour) · p polygon (Enter closes)
Size       [ and ]                              Wand tolerance  , and .
View       scroll zoom · space+drag or middle-drag pan · 0 fit · h hide mask
Edit       ctrl+z undo · a load automatic mask · x clear
Frames     n next · N previous · ctrl+s save (also saves when you change frame)
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
          --accent:#3b6fd4; --ok:#2f9e6e; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  #side { width:260px; flex:none; border-right:1px solid var(--line); display:flex;
          flex-direction:column; background:var(--panel); }
  #side h1 { font-size:13px; margin:0; padding:12px 14px; border-bottom:1px solid var(--line);
             letter-spacing:.02em; }
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
  .bad { color:#e8a33d; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #bar { display:flex; align-items:center; gap:14px; padding:8px 14px;
         border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  #bar b { font-weight:600; }
  .tool { padding:3px 9px; border:1px solid var(--line); border-radius:5px; cursor:pointer;
          color:var(--muted); }
  .tool.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  #wrap { flex:1; overflow:hidden; position:relative; background:#0d0f12; }
  canvas { position:absolute; image-rendering:pixelated; cursor:crosshair; }
  #status { padding:6px 14px; border-top:1px solid var(--line); color:var(--muted);
            background:var(--panel); display:flex; gap:18px; }
  kbd { background:#262b33; border:1px solid var(--line); border-radius:4px; padding:0 4px; }
</style>
<div id="side">
  <h1>mask annotator</h1>
  <label id="filter"><input type="checkbox" id="only"> only runs needing work</label>
  <div id="list"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="title">—</b>
    <span class="tool" data-t="brush">brush <kbd>b</kbd></span>
    <span class="tool" data-t="eraser">eraser <kbd>e</kbd></span>
    <span class="tool" data-t="wand">wand <kbd>w</kbd></span>
    <span class="tool" data-t="poly">polygon <kbd>p</kbd></span>
    <span>size <b id="sz">6</b> <kbd>[</kbd><kbd>]</kbd></span>
    <span>tol <b id="tol">26</b> <kbd>,</kbd><kbd>.</kbd></span>
    <span class="tool" id="loadauto">load auto <kbd>a</kbd></span>
    <span class="tool" id="clear">clear <kbd>x</kbd></span>
    <span class="tool" id="save">save <kbd>⌘s</kbd></span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="status">
    <span id="pos">—</span><span id="px">0 px</span><span id="zoom">1x</span>
    <span>scroll zoom · space-drag pan · <kbd>0</kbd> fit · <kbd>h</kbd> hide · <kbd>n</kbd>/<kbd>N</kbd> frame · <kbd>⌘z</kbd> undo</span>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
let frames = [], cur = -1, img = null, imgData = null;
let mask = document.createElement('canvas'), mctx = null;
let view = {z: 2, ox: 0, oy: 0}, tool = 'brush', size = 6, tol = 26;
let drawing = false, panning = false, last = null, hide = false, dirty = false;
let undo = [], poly = [];
const cv = $('#cv'), ctx = cv.getContext('2d');

async function boot() {
  frames = await (await fetch('/api/list')).json();
  renderList();
  if (frames.length) load(0);
}
function renderList() {
  const only = $('#only').checked;
  const l = $('#list'); l.innerHTML = '';
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
async function load(i) {
  if (dirty && cur >= 0) await save(true);
  cur = i; poly = []; undo = [];
  const f = frames[i];
  $('#title').textContent = `${f.run}  frame ${f.frame}`;
  img = new Image();
  img.src = `/file/OWN/${f.run}/rgb/${f.frame}.png?t=${Date.now()}`;
  await img.decode();
  mask.width = img.width; mask.height = img.height;
  mctx = mask.getContext('2d', {willReadFrequently: true});
  mctx.clearRect(0, 0, mask.width, mask.height);
  const tmp = document.createElement('canvas');
  tmp.width = img.width; tmp.height = img.height;
  const tctx = tmp.getContext('2d', {willReadFrequently: true});
  tctx.drawImage(img, 0, 0);
  imgData = tctx.getImageData(0, 0, img.width, img.height);
  const src = f.manual ? `/file/OWN_manual/${f.run}_${f.frame}.png`
                       : `/file/OWN/${f.run}/gt/${f.frame}.png`;
  await loadMask(src);
  dirty = false; fit(); renderList();
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
      for (let i = 0; i < d.data.length; i += 4) {
        if (d.data[i] > 127) { out.data[i] = 255; out.data[i+3] = 255; }
      }
      mctx.putImageData(out, 0, 0); draw(); res();
    };
    m.onerror = () => { draw(); res(); };
    m.src = url + '?t=' + Date.now();
  });
}
function fit() {
  if (!img) return;
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  view.z = Math.max(1, Math.floor(Math.min(w / img.width, h / img.height) * 10) / 10);
  view.ox = (w - img.width * view.z) / 2; view.oy = (h - img.height * view.z) / 2;
  draw();
}
function draw() {
  if (!img || !mctx) return;                 // nothing loaded yet
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  cv.width = w; cv.height = h; cv.style.left = '0px'; cv.style.top = '0px';
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.save(); ctx.translate(view.ox, view.oy); ctx.scale(view.z, view.z);
  ctx.drawImage(img, 0, 0);
  if (!hide) { ctx.globalAlpha = 0.5; ctx.drawImage(mask, 0, 0); ctx.globalAlpha = 1; }
  if (poly.length) {
    ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 1 / view.z;
    ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
    poly.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.stroke();
    poly.forEach(p => ctx.fillRect(p[0] - 1 / view.z, p[1] - 1 / view.z, 2 / view.z, 2 / view.z));
  }
  ctx.restore();
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
    $('#px').textContent = n + ' px';
  }, 120);
}
const toImg = e => {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left - view.ox) / view.z, (e.clientY - r.top - view.oy) / view.z];
};
function snap() { undo.push(mctx.getImageData(0, 0, mask.width, mask.height)); if (undo.length > 30) undo.shift(); }
function paint(a, b) {
  mctx.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
  mctx.strokeStyle = '#ff0000'; mctx.fillStyle = '#ff0000';
  mctx.lineWidth = size; mctx.lineCap = 'round'; mctx.lineJoin = 'round';
  mctx.beginPath(); mctx.moveTo(a[0], a[1]); mctx.lineTo(b[0], b[1]); mctx.stroke();
  mctx.globalCompositeOperation = 'source-over';
}
function wand(x, y) {
  x = Math.round(x); y = Math.round(y);
  const W = img.width, H = img.height, d = imgData.data;
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const at = (px, py) => (py * W + px) * 4;
  const s = at(x, y), r0 = d[s], g0 = d[s+1], b0 = d[s+2];
  const seen = new Uint8Array(W * H), out = mctx.createImageData(W, H);
  const cur = mctx.getImageData(0, 0, W, H);
  out.data.set(cur.data);
  const st = [[x, y]]; let n = 0;
  while (st.length && n < 60000) {
    const [cx, cy] = st.pop(), k = cy * W + cx;
    if (seen[k]) continue; seen[k] = 1;
    const i = k * 4;
    if (Math.abs(d[i]-r0) + Math.abs(d[i+1]-g0) + Math.abs(d[i+2]-b0) > tol * 3) continue;
    out.data[i] = 255; out.data[i+3] = 255; n++;
    if (cx > 0) st.push([cx-1, cy]); if (cx < W-1) st.push([cx+1, cy]);
    if (cy > 0) st.push([cx, cy-1]); if (cy < H-1) st.push([cx, cy+1]);
  }
  mctx.putImageData(out, 0, 0); dirty = true; draw();
}
cv.addEventListener('mousedown', e => {
  if (e.button === 1 || e.shiftKey || panKey) { panning = true; last = [e.clientX, e.clientY]; return; }
  const p = toImg(e);
  if (tool === 'wand') { snap(); wand(p[0], p[1]); return; }
  if (tool === 'poly') { poly.push(p); draw(); return; }
  snap(); drawing = true; last = p; paint(p, p); dirty = true; draw();
});
window.addEventListener('mousemove', e => {
  if (panning) { view.ox += e.clientX - last[0]; view.oy += e.clientY - last[1];
                 last = [e.clientX, e.clientY]; draw(); return; }
  const p = toImg(e);
  $('#pos').textContent = `x ${Math.round(p[0])}  y ${Math.round(p[1])}`;
  if (!drawing) return;
  paint(last, p); last = p; dirty = true; draw();
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
let panKey = false;
document.querySelectorAll('.tool[data-t]').forEach(el => el.onclick = () => setTool(el.dataset.t));
function setTool(t) {
  tool = t; poly = [];
  document.querySelectorAll('.tool[data-t]').forEach(e => e.classList.toggle('on', e.dataset.t === t));
  draw();
}
$('#loadauto').onclick = () => { snap(); loadMask(`/file/OWN/${frames[cur].run}/gt/${frames[cur].frame}.png`); dirty = true; };
$('#clear').onclick = () => { snap(); mctx.clearRect(0, 0, mask.width, mask.height); dirty = true; draw(); };
$('#save').onclick = () => save();
$('#only').onchange = renderList;
async function save(quiet) {
  const t = document.createElement('canvas');
  t.width = mask.width; t.height = mask.height;
  const c = t.getContext('2d');
  c.fillStyle = '#000'; c.fillRect(0, 0, t.width, t.height);
  const d = mctx.getImageData(0, 0, mask.width, mask.height), o = c.getImageData(0, 0, t.width, t.height);
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
  if (e.key === ' ') { panKey = true; e.preventDefault(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault(); const u = undo.pop(); if (u) { mctx.putImageData(u, 0, 0); dirty = true; draw(); } return; }
  if (e.key === 'Enter' && tool === 'poly' && poly.length > 2) {
    snap(); mctx.fillStyle = '#ff0000'; mctx.beginPath();
    mctx.moveTo(poly[0][0], poly[0][1]); poly.slice(1).forEach(p => mctx.lineTo(p[0], p[1]));
    mctx.closePath(); mctx.fill(); poly = []; dirty = true; draw(); return; }
  if (e.key === 'Escape') { poly = []; draw(); return; }
  const k = e.key;
  if (k === 'b') setTool('brush'); else if (k === 'e') setTool('eraser');
  else if (k === 'w') setTool('wand'); else if (k === 'p') setTool('poly');
  else if (k === '[') { size = Math.max(1, size - 1); $('#sz').textContent = size; }
  else if (k === ']') { size = Math.min(80, size + 1); $('#sz').textContent = size; }
  else if (k === ',') { tol = Math.max(2, tol - 3); $('#tol').textContent = tol; }
  else if (k === '.') { tol = Math.min(120, tol + 3); $('#tol').textContent = tol; }
  else if (k === 'h') { hide = !hide; draw(); }
  else if (k === '0') fit();
  else if (k === 'a') $('#loadauto').click();
  else if (k === 'x') $('#clear').click();
  else if (k === 'n') { if (cur + 1 < frames.length) load(cur + 1); }
  else if (k === 'N') { if (cur > 0) load(cur - 1); }
});
window.addEventListener('keyup', e => { if (e.key === ' ') panKey = false; });
window.addEventListener('resize', () => draw());
window.addEventListener('error', e => {        // surface failures instead of dying silently
  const b = document.createElement('div');
  b.style.cssText = 'position:fixed;bottom:0;left:0;right:0;background:#7a1f1f;color:#fff;padding:6px 12px;z-index:9;font:12px monospace';
  b.textContent = 'error: ' + e.message;
  document.body.appendChild(b);
});
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
                            manual=os.path.exists(f"{MAN}/{run}_{i}.png")))
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
