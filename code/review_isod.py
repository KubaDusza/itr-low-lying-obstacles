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
          --accent:#3b6fd4; --keep:#2f9e6e; --drop:#4a5cd4; --warn:#e8a33d; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  #side { width:220px; flex:none; border-right:1px solid var(--line); display:flex;
          flex-direction:column; background:var(--panel); }
  #side h1 { font-size:13px; margin:0; padding:12px 14px; border-bottom:1px solid var(--line); }
  #sites { overflow:auto; flex:1; }
  .site { padding:6px 14px; cursor:pointer; border-bottom:1px solid #22252b; display:flex;
          justify-content:space-between; gap:8px; }
  .site:hover { background:#232832; }
  .site.sel { background:var(--accent); color:#fff; }
  .site .n { font:11px ui-monospace,Menlo,monospace; color:var(--muted); }
  .site.sel .n { color:#dbe6ff; }
  .site.done .n { color:var(--keep); }
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
  #wrap { flex:1; overflow:hidden; position:relative; background:#0d0f12; }
  canvas { position:absolute; image-rendering:pixelated; }
  #strip { display:flex; flex-wrap:wrap; gap:2px; padding:6px 14px;
           border-top:1px solid var(--line); background:var(--panel); max-height:72px;
           overflow:auto; }
  .f { width:16px; height:16px; border-radius:3px; background:#2a2f38; cursor:pointer;
       font:9px/16px ui-monospace,Menlo,monospace; text-align:center; color:#8a93a2; }
  .f.done { background:#255c43; color:#b8ecd3; }
  .f.sel { outline:2px solid var(--accent); }
  #status { padding:6px 14px; border-top:1px solid var(--line); color:var(--muted);
            background:var(--panel); display:flex; gap:18px; }
  #counts { font:12px ui-monospace,Menlo,monospace; }
  kbd { background:#262b33; border:1px solid var(--line); border-radius:4px; padding:0 4px;
        font:11px/1.6 ui-monospace,Menlo,monospace; }
</style>
<div id="side">
  <h1>ISOD obstacle review</h1>
  <div id="sites"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="title">—</b>
    <span class="tool" id="cut" data-hint="Cut brush. Erases inside ISOD's mask only, to part an object from the furniture it touches so it becomes its own component. The frame is then re-split and the decisions carried over. Nothing outside the mask is ever affected.">cut<span class="k">e</span></span>
    <span class="tool" id="szbox" data-hint="Cut brush diameter.">size <b id="sz">4</b><span class="k">[ ]</span></span>
    <span class="tool" id="keepall" data-hint="Keep every component in this frame.">keep all<span class="k">k</span></span>
    <span class="tool" id="reset" data-hint="Back to what the automatic pass proposed for this frame, discarding your toggles and cuts.">reset<span class="k">x</span></span>
    <span class="tool" id="hide" data-hint="Hide the overlay to see the photograph.">hide<span class="k">h</span></span>
    <span class="tool" id="fit" data-hint="Fit the frame to the window.">fit<span class="k">0</span></span>
    <span class="tool" id="save" data-hint="Write this frame's mask. Also happens when you change frame.">save<span class="k">⌘s</span></span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="strip"></div>
  <div id="status">
    <span id="counts">—</span>
    <span id="hints">click a component to keep / drop it ·
      <kbd>n</kbd>/<kbd>N</kbd> frame · scroll zoom · space-drag pan · <kbd>⌘z</kbd> undo</span>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const KEEP = [60, 200, 120], DROP = [70, 110, 220], NEW = [255, 210, 0];

let sites = [], site = null, idx = 0, frames = [];
let img = null, comp = null, rows = [], W = 0, H = 0;
let view = {z: 1, ox: 0, oy: 0}, hide = false, dirty = false;
let cutting = false, size = 4, drawing = false, panning = false, last = null, panKey = false;
let cut = null, undo = [];
const cv = $('#cv'), ctx = cv.getContext('2d');

async function boot() {
  sites = await (await fetch('/api/sites')).json();
  renderSites();
  openSite(sites[0].site, 0);
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
}
async function openSite(s, i) {
  if (dirty) await save(true);
  site = s;
  frames = (await (await fetch(`/api/frames?site=${s}`)).json());
  renderStrip();
  renderSites();
  load(i);
}
function renderStrip() {
  const l = $('#strip'); l.innerHTML = '';
  frames.forEach((f, i) => {
    const d = document.createElement('div');
    d.className = 'f' + (f.done ? ' done' : '') + (i === idx ? ' sel' : '');
    d.textContent = i % 10 === 0 ? i : '';
    d.title = `frame ${i}`;
    d.onclick = () => load(i);
    l.appendChild(d);
  });
}
async function load(i) {
  if (dirty) await save(true);
  idx = i; undo = [];
  const r = await (await fetch(`/api/frame?site=${site}&i=${i}`)).json();
  rows = r.rows; W = r.w; H = r.h;
  comp = Uint8Array.from(atob(r.comp), c => c.charCodeAt(0));
  cut = r.cut ? Uint8Array.from(atob(r.cut), c => c.charCodeAt(0)) : new Uint8Array(W * H);
  img = new Image();
  await new Promise(res => { img.onload = res; img.onerror = res;
                             img.src = `/rgb/${site}/${i}.png`; });
  $('#title').textContent = `${site}  frame ${i}`;
  dirty = false; renderStrip(); fit();
}
function fit() {
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  view.z = Math.min(w / W, h / H);
  view.ox = (w - W * view.z) / 2; view.oy = (h - H * view.z) / 2;
  draw();
}
let overlay = document.createElement('canvas'), octx = null;
function buildOverlay() {
  if (overlay.width !== W) { overlay.width = W; overlay.height = H; }
  octx = octx || overlay.getContext('2d', {willReadFrequently: true});
  const d = octx.createImageData(W, H);
  const colour = {};
  rows.forEach(r => colour[r.id] = r.keep ? KEEP : DROP);
  for (let p = 0, k = 0; p < comp.length; p++, k += 4) {
    const c = colour[comp[p]];
    if (!c) continue;
    d.data[k] = c[0]; d.data[k+1] = c[1]; d.data[k+2] = c[2]; d.data[k+3] = 130;
  }
  octx.putImageData(d, 0, 0);
  // a bright edge on whatever the rule was unsure about: that is what needs looking at
  const edge = octx;
  edge.lineWidth = 1; edge.strokeStyle = `rgb(${NEW.join(',')})`;
  rows.filter(r => r.fresh).forEach(r => {
    const [x0, y0, x1, y1] = r.box;
    edge.strokeRect(x0 - 1.5, y0 - 1.5, x1 - x0 + 3, y1 - y0 + 3);
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
  const keep = rows.filter(r => r.keep).length, fresh = rows.filter(r => r.fresh).length;
  $('#counts').textContent =
    `${rows.length} components · ${keep} keep · ${rows.length - keep} drop` +
    (fresh ? ` · ${fresh} to check` : '') + (dirty ? ' · unsaved' : '');
}
const toImg = e => {
  const r = cv.getBoundingClientRect();
  return [Math.round((e.clientX - r.left - view.ox) / view.z),
          Math.round((e.clientY - r.top - view.oy) / view.z)];
};
cv.addEventListener('mousedown', e => {
  if (e.button === 1 || panKey) { panning = true; last = [e.clientX, e.clientY]; return; }
  const [x, y] = toImg(e);
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  if (cutting) { undo.push(comp.slice()); drawing = true; paintCut(x, y); return; }
  const id = comp[y * W + x];
  if (!id) return;
  const r = rows.find(z => z.id === id);
  undo.push(rows.map(z => z.keep));
  r.keep = !r.keep; dirty = true; draw();
});
window.addEventListener('mousemove', e => {
  if (panning) { view.ox += e.clientX - last[0]; view.oy += e.clientY - last[1];
                 last = [e.clientX, e.clientY]; draw(); return; }
  if (!drawing) return;
  const [x, y] = toImg(e);
  paintCut(x, y);
});
window.addEventListener('mouseup', () => {
  if (drawing) { drawing = false; recut(); }
  panning = false;
});
function paintCut(cx, cy) {
  const r = Math.max(1, size / 2);
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(H - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(W - 1, Math.ceil(cx + r)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) { cut[y * W + x] = 1; comp[y * W + x] = 0; }
  dirty = true; draw();
}
async function recut() {
  /* the cut changed the shape of the mask, so the frame is split again on the server and
     the decisions carried onto whatever components came out */
  const body = btoa(String.fromCharCode(...cut));
  const r = await (await fetch(`/api/recut?site=${site}&i=${idx}`, {method: 'POST', body})).json();
  rows = r.rows;
  comp = Uint8Array.from(atob(r.comp), c => c.charCodeAt(0));
  dirty = true; draw();
}
$('#wrap').addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15, nz = Math.min(24, Math.max(0.2, view.z * k));
  view.ox = mx - (mx - view.ox) * (nz / view.z);
  view.oy = my - (my - view.oy) * (nz / view.z);
  view.z = nz; draw();
}, {passive: false});

$('#cut').onclick = () => { cutting = !cutting; $('#cut').classList.toggle('on', cutting);
                            cv.style.cursor = cutting ? 'crosshair' : 'default'; };
$('#keepall').onclick = () => { undo.push(rows.map(r => r.keep));
                                rows.forEach(r => r.keep = true); dirty = true; draw(); };
$('#reset').onclick = async () => {
  await fetch(`/api/reset?site=${site}&i=${idx}`, {method: 'POST'});
  dirty = false; load(idx);
};
$('#hide').onclick = () => { hide = !hide; $('#hide').classList.toggle('on', hide); draw(); };
$('#fit').onclick = fit;
$('#save').onclick = () => save();

async function save(quiet) {
  const keep = rows.filter(r => r.keep).map(r => r.id);
  await fetch(`/api/save?site=${site}&i=${idx}`, {method: 'POST', body: JSON.stringify(keep)});
  dirty = false;
  frames[idx].done = true;
  const s = sites.find(z => z.site === site);
  if (s) s.done = frames.filter(f => f.done).length;
  renderStrip(); renderSites(); draw();
  if (!quiet) { $('#save').classList.add('on'); setTimeout(() => $('#save').classList.remove('on'), 300); }
}
window.addEventListener('keydown', async e => {
  if (e.key === ' ') { panKey = true; e.preventDefault(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault();
    const u = undo.pop();
    if (Array.isArray(u)) { rows.forEach((r, i) => r.keep = u[i]); dirty = true; draw(); }
    else if (u) { comp = u; dirty = true; draw(); }
    return;
  }
  if (e.metaKey || e.ctrlKey) return;
  if (e.key === 'n') { if (idx + 1 < frames.length) load(idx + 1);
                       else { const j = sites.findIndex(z => z.site === site);
                              if (j + 1 < sites.length) openSite(sites[j+1].site, 0); } }
  else if (e.key === 'N') { if (idx > 0) load(idx - 1); }
  else if (e.key === 'e') $('#cut').click();
  else if (e.key === 'k') $('#keepall').click();
  else if (e.key === 'x') $('#reset').click();
  else if (e.key === 'h') $('#hide').click();
  else if (e.key === '0') fit();
  else if (e.key === '[') { size = Math.max(1, size - 1); $('#sz').textContent = size; }
  else if (e.key === ']') { size = Math.min(40, size + 1); $('#sz').textContent = size; }
});
window.addEventListener('keyup', e => { if (e.key === ' ') panKey = false; });
window.addEventListener('resize', fit);
boot();
</script>
"""


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
        if u.path == "/api/frames":
            s = q["site"][0]
            return self._json([dict(i=i, done=os.path.exists(f"{out_dir(s)}/{i}.json"))
                               for i in range(N_FRAMES)])
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
        if u.path == "/api/save":
            write_mask(s, i, json.loads(body or b"[]"))
            return self._send(200, b"ok", "text/plain")
        if u.path == "/api/recut":
            raw = np.frombuffer(base64.b64decode(body), np.uint8)
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
