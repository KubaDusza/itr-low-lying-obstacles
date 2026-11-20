"""Build a visual review of every frozen ISOD training target."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ISOD_reference_train_v1"
OUT = ROOT / "results/frozen_v1/qualitative/ground_truth_builder/gallery"


def image_url(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def main():
    cv2.setNumThreads(1)
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    frames = []
    for row in sorted(manifest["frames"], key=lambda r: (r["site"], r["frame"])):
        site, frame = row["site"], row["frame"]
        mask_path = SOURCE / row["path"]
        assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == row["sha256"], mask_path
        rgb_path = ROOT / "data/ISOD/data" / site / "rgb" / f"{frame}.png"
        rgb = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None or rgb.shape[:2] != mask.shape:
            raise ValueError(f"Missing or misaligned input for {site}/{frame}")
        assert set(np.unique(mask)) <= {0, 255}, mask_path
        fg = mask > 0
        overlay = rgb.copy()
        overlay[fg] = np.rint(rgb[fg] * 0.42 + np.array([242, 153, 51]) * 0.58).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 190, 80), 1)
        stem = f"{site}_{frame}"
        overlay_path = OUT / "overlays" / f"{stem}.jpg"
        preview_path = OUT / "previews" / f"{stem}.jpg"
        preview = cv2.resize(np.concatenate([rgb, overlay], axis=1), (960, 317), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(dict(site=site, frame=frame, status=row["review_status"],
                           rgb=image_url(rgb_path), mask=image_url(mask_path),
                           overlay=image_url(overlay_path), preview=image_url(preview_path),
                           foreground_pixels=int(fg.sum())))
    assert len(frames) == 1400
    assert sum(r["status"] == "isod_sam_prelabel" for r in frames) == 1330
    sites = sorted({r["site"] for r in frames})
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    for site in sites:
        automatic = [r for r in frames if r["site"] == site and r["status"] == "isod_sam_prelabel"]
        selected = [automatic[i] for i in np.linspace(0, len(automatic) - 1, 12, dtype=int)]
        canvas = Image.new("RGB", (1440, 1720), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 14), f"{site} — automatic ground-truth proposals", fill="black", font=font)
        draw.text((18, 46), "12 evenly spaced frames. Each pair shows RGB on the left and its frozen target on the right.",
                  fill="#444444", font=font)
        for i, row in enumerate(selected):
            x, y = (i % 2) * 720, 96 + (i // 2) * 268
            draw.text((x + 12, y), f"{row['site']} / {row['frame']}", fill="black", font=font)
            with Image.open(OUT / row["preview"]) as picture:
                picture = picture.resize((708, 234), Image.Resampling.LANCZOS)
                canvas.paste(picture, (x + 6, y + 27))
        canvas.save(OUT / "sheets" / f"{site}.jpg", quality=91)
    (OUT / "frames.json").write_text(json.dumps(frames, separators=(",", ":")) + "\n")
    template = Path(__file__).with_name("gt_gallery.html").read_text()
    (OUT / "index.html").write_text(template.replace("__FRAME_DATA__", json.dumps(frames)))
    print(f"Verified and prepared {len(frames)} frames across {len(sites)} training locations.")
    print("Default review contains all 1,330 automatically generated masks.")
    print("14 contact sheets contain 168 evenly spaced automatic examples.")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
