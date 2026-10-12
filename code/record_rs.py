"""Capture RealSense RGB-D frames, or convert a recorded .bag, into the ISOD layout.

Depth is aligned to colour, kept as uint16 millimetres with 0 = invalid, centre-cropped to
ISOD's aspect ratio and resized to 640x422, so eval.py / analyze_objects.py run unchanged.

  # convert a .bag recorded in realsense-viewer, sampling every 15th frame
  python record_rs.py --bag B-01.bag --scene B-01 --every 15 --frames 5

  # live, fixed mount: lock exposure so "low contrast" means the object, not the camera
  python record_rs.py --live --scene B-01 --frames 5 --interval 1.5 --lock-exposure

Writes <out>/<scene>/{rgb,depth}/<i>.png and <out>/<scene>/intrinsics.json, and prints the
invalid-depth rate (ISOD is about 1.3%; much higher means the depth arm is handicapped for
reasons unrelated to the research question).

Needs pyrealsense2, which has no macOS-arm64 wheel: run this on the Linux GPU/lab machines
(pip install pyrealsense2) or on Windows. See ../data_collection/PLAN.md.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

SIZE = (640, 422)  # (w, h), ISOD's frame size


def to_isod(rgb, depth):
    """Centre-crop to ISOD's aspect ratio, then resize. Depth is never interpolated."""
    h, w = depth.shape
    ar = SIZE[0] / SIZE[1]
    if w / h > ar:
        cw = int(round(h * ar))
        x0 = (w - cw) // 2
        rgb, depth = rgb[:, x0:x0 + cw], depth[:, x0:x0 + cw]
    else:
        ch = int(round(w / ar))
        y0 = (h - ch) // 2
        rgb, depth = rgb[y0:y0 + ch], depth[y0:y0 + ch]
    return (cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA),
            cv2.resize(depth, SIZE, interpolation=cv2.INTER_NEAREST))


def start(a):
    pipe, cfg = rs.pipeline(), rs.config()
    if a.bag:
        rs.config.enable_device_from_file(cfg, a.bag, repeat_playback=False)
    else:
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    if a.bag:
        profile.get_device().as_playback().set_real_time(False)  # do not drop frames
    dev = profile.get_device()
    sensor = dev.first_depth_sensor()
    scale = sensor.get_depth_scale()  # metres per unit; 0.001 on a D4xx by default
    if not a.bag and a.lock_exposure:
        for s in (sensor, dev.first_color_sensor()):
            if s.supports(rs.option.enable_auto_exposure):
                s.set_option(rs.option.enable_auto_exposure, 0)
        c = dev.first_color_sensor()
        if c.supports(rs.option.enable_auto_white_balance):
            c.set_option(rs.option.enable_auto_white_balance, 0)
    return pipe, profile, scale


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="recorded .bag file to convert")
    g.add_argument("--live", action="store_true", help="capture from the connected camera")
    ap.add_argument("--scene", required=True, help="scene id, e.g. B-01")
    ap.add_argument("--out", default="data/OWN")
    ap.add_argument("--frames", type=int, default=5, help="frames to keep")
    ap.add_argument("--every", type=int, default=15, help="--bag: keep every Nth frame")
    ap.add_argument("--interval", type=float, default=1.5, help="--live: seconds between frames")
    ap.add_argument("--warmup", type=int, default=30, help="--live: frames to discard first")
    ap.add_argument("--lock-exposure", action="store_true",
                    help="--live: disable auto exposure and auto white balance (pass B)")
    a = ap.parse_args()

    root = f"{a.out}/{a.scene}"
    for sub in ("rgb", "depth"):
        os.makedirs(f"{root}/{sub}", exist_ok=True)

    pipe, profile, scale = start(a)
    align = rs.align(rs.stream.color)
    print(f"depth scale {scale} m/unit", "(x1000 -> mm)" if abs(scale - 0.001) > 1e-9 else "")

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    with open(f"{root}/intrinsics.json", "w") as f:
        json.dump({"source_width": intr.width, "source_height": intr.height,
                   "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                   "model": str(intr.model), "coeffs": list(intr.coeffs),
                   "depth_scale_m": scale, "saved_as": "centre-cropped to 640x422",
                   "note": "intrinsics are for the source resolution, before crop and resize"},
                  f, indent=1)

    kept, seen, invalid, t_last = 0, 0, [], 0.0
    try:
        while kept < a.frames:
            try:
                fs = pipe.wait_for_frames(5000)
            except RuntimeError:
                print("stream ended")
                break
            seen += 1
            if a.live:
                if seen <= a.warmup or time.time() - t_last < a.interval:
                    continue
                t_last = time.time()
            elif seen % a.every:
                continue

            fs = align.process(fs)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            depth = np.asanyarray(d.get_data())
            if abs(scale - 0.001) > 1e-9:  # store millimetres whatever the sensor reports
                depth = np.clip(depth.astype(np.float64) * scale * 1000, 0, 65535).astype(np.uint16)
            bgr, depth = to_isod(np.asanyarray(c.get_data()), depth)
            cv2.imwrite(f"{root}/rgb/{kept}.png", bgr)
            cv2.imwrite(f"{root}/depth/{kept}.png", depth)
            invalid.append(float((depth == 0).mean()))
            print(f"  {a.scene}/{kept}: median {int(np.median(depth[depth > 0]))} mm, "
                  f"invalid {invalid[-1]:.2%}")
            kept += 1
    finally:
        pipe.stop()

    if invalid:
        print(f"{kept} frames -> {root}  mean invalid depth {np.mean(invalid):.2%} "
              f"(ISOD: ~1.3%)")


if __name__ == "__main__":
    main()
