"""Review object inventories for a fixed RGB-only SAM prompt development subset."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sam_prompt_v2"
DEVELOPMENT = [("ABEC_MM", 4), ("BT_LM", 34), ("FL_LM", 54), ("HB_G", 64),
               ("HB_LM", 8), ("HB_MM", 94), ("HDCL_HW_B", 40), ("MEB_HW_C", 34)]


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/ISOD_reference_train_v1/manifest.json").read_text())
    reviewed = {(r["site"], r["frame"]): r for r in manifest["frames"]
                if r["review_status"] == "human_reviewed"}
    assert len(reviewed) == 70 and all(key in reviewed for key in DEVELOPMENT)
    sites = {r["site"] for r in manifest["frames"]} - {s for s, _ in DEVELOPMENT}
    heldout = sorted(key for key in reviewed if key[0] in sites)

    def record(key):
        site, frame = key
        source = ROOT / "data/ISOD/data" / site / "rgb" / f"{frame}.png"
        reference = ROOT / "data/ISOD_reference_train_v1" / site / f"{frame}.png"
        assert source.is_file() and reference.is_file()
        return dict(id=f"{site}/{frame}", site=site, frame=frame,
                    rgb=str(source.relative_to(ROOT)),
                    reference=str(reference.relative_to(ROOT)),
                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())

    old = json.loads((ROOT / "data/ISOD_sam_train_v1/manifest.json").read_text())
    protocol = dict(status="awaiting_object_inventory", development=[record(k) for k in DEVELOPMENT],
                    heldout=[record(k) for k in heldout],
                    reserved_reviewed_frames=70-len(DEVELOPMENT)-len(heldout),
                    input="Full RGB image and text phrases",
                    reference_role="Scoring and human review only",
                    model=old["inference"]["model"], revision=old["inference"]["revision"],
                    fixed_confidence=old["confidence"], fixed_mask_probability=0.5,
                    combining="Union of accepted instance masks",
                    tune="Text phrases only", baseline_prompts=old["prompts"],
                    candidate_generic_prompts=["object on the floor", "small object on the floor",
                                               "low lying object on the floor", "loose item on the floor"],
                    candidate_specific_form="Observed object name, with and without 'on the floor'")
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Existing protocol differs"
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    (OUT / "development_frames.json").write_text(json.dumps(DEVELOPMENT, indent=2) + "\n")
    (OUT / "heldout_frames.json").write_text(json.dumps(heldout, indent=2) + "\n")
    return protocol


class Handler(BaseHTTPRequestHandler):
    protocol = None
    save_lock = threading.Lock()

    def send(self, status, body, content_type="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        frames = self.protocol["development"]
        if route == "/":
            self.send(200, Path(__file__).with_name("sam_prompt_inventory.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/review":
            path = OUT / "object_inventory.json"
            saved = json.loads(path.read_text()) if path.exists() else {}
            self.send(200, json.dumps(dict(frames=frames, inventory=saved)))
        elif route.startswith("/image/"):
            parts = route.split("/")
            row = next((r for r in frames if len(parts) == 5 and
                        r["site"] == parts[3] and str(r["frame"]) == parts[4]), None)
            if row is None or parts[2] not in ("rgb", "reference"):
                self.send(404, "Unknown image", "text/plain")
                return
            self.send(200, (ROOT / row[parts[2]]).read_bytes(), "image/png")
        else:
            self.send(404, "Unknown route", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/inventory":
            self.send(404, "Unknown route", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size < 1 or size > 20000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            ident = data.get("id")
            if ident not in {r["id"] for r in self.protocol["development"]}:
                raise ValueError("Frame is outside the development subset")
            fields = {key: data.get(key, "") for key in ("objects", "exclude", "notes")}
            if not all(isinstance(value, str) and len(value) <= 5000 for value in fields.values()):
                raise ValueError("Invalid text")
            if not fields["objects"].strip():
                raise ValueError("List the floor objects, or write 'none' for an empty scene")
            path = OUT / "object_inventory.json"
            with self.save_lock:
                saved = json.loads(path.read_text()) if path.exists() else {}
                saved[ident] = fields
                with tempfile.NamedTemporaryFile("w", dir=OUT, delete=False) as file:
                    json.dump(saved, file, indent=2)
                    file.write("\n")
                    temporary = file.name
                os.replace(temporary, path)
            self.send(200, json.dumps(dict(saved=True, count=len(saved))))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send(400, json.dumps(dict(error=str(error))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8874)
    args = parser.parse_args()
    Handler.protocol = setup()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Prompt inventory review at http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
