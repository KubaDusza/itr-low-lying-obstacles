"""Turn the RealSense .db3 recordings into an ISOD-shaped dataset.

Runs on the laptop: rs-convert (Homebrew librealsense) writes colour PNGs and raw depth,
then depth is aligned to colour here in numpy, because pyrealsense2 has no macOS-arm64 wheel.
The calibration below was read once off a recording with pyrealsense2 on the GPU server.

  python db3_to_frames.py                      # all runs in data/raw
  python db3_to_frames.py h52_floor_cables_075 # one run

Writes data/OWN/<run>/rgb/<i>.png (640x422 RGB) and depth/<i>.png (640x422 uint16 mm,
0 = invalid), the same layout ISOD uses, so eval.py and analyze_objects.py work unchanged.
"""
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = f"{ROOT}/data/raw", f"{ROOT}/data/OWN"
TMP = "/tmp/rs_grab/_convert"
SIZE = (640, 422)
N_FRAMES = 5
WINDOW = ("1.0", "1.6")          # seconds into the clip; the camera has settled by then

# intrinsics/extrinsics of this D435 at 848x480, both streams, read with pyrealsense2
D = dict(fx=423.55145263671875, fy=423.55145263671875, ppx=419.07257080078125, ppy=239.80633544921875)
C = dict(fx=608.1510620117188, fy=606.9262084960938, ppx=432.7842102050781, ppy=242.90274047851562)
R = np.array([0.9999849796295166, -0.0052919997833669186, -0.0013599999947473407,
              0.005284000188112259, 0.999970018863678, -0.005745000205934048,
              0.0013909999979659915, 0.0057379999198019505, 0.999983012676239]).reshape(3, 3).T
T = np.array([0.015064997598528862, 0.00015599869948346168, 3.400971763767302e-05])
W, H = 848, 480


def align(depth_m):
    """Depth in the depth sensor's frame -> depth sampled on the colour image grid.

    Same forward projection librealsense's rs2::align does: deproject, rigid transform,
    reproject, keep the nearest hit. The colour stream has 1.44x the focal length, so each
    depth pixel lands on about 2x2 colour pixels; splatting that 2x2 block is what keeps the
    result hole-free (a bare forward map leaves a grid of gaps).
    """
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u]
    pts = np.stack([(u - D["ppx"]) / D["fx"] * z, (v - D["ppy"]) / D["fy"] * z, z])
    x, y, zc = R @ pts + T[:, None]
    ok = zc > 0
    x, y, zc = x[ok], y[ok], zc[ok]
    uu = x / zc * C["fx"] + C["ppx"]
    vv = y / zc * C["fy"] + C["ppy"]
    out = np.full((H, W), np.inf, np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            ui = np.floor(uu).astype(np.int32) + du
            vi = np.floor(vv).astype(np.int32) + dv
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            np.minimum.at(out, (vi[m], ui[m]), zc[m])   # z-buffer: nearest surface wins
    out[np.isinf(out)] = 0
    return out


def to_isod(rgb, depth_mm):
    """Centre-crop to ISOD's 640:422 aspect, then resize. Depth is never interpolated."""
    h, w = depth_mm.shape
    ar = SIZE[0] / SIZE[1]
    cw = int(round(h * ar))
    x0 = (w - cw) // 2
    rgb, depth_mm = rgb[:, x0:x0 + cw], depth_mm[:, x0:x0 + cw]
    return (cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA),
            cv2.resize(depth_mm, SIZE, interpolation=cv2.INTER_NEAREST))


def convert(run):
    tmp = f"{TMP}/{run}"
    subprocess.run(["rm", "-rf", tmp], check=False)
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["rs-convert", "-i", f"{RAW}/{run}.db3", "-p", f"{tmp}/f", "-b", f"{tmp}/raw",
                    "-s", WINDOW[0], "-e", WINDOW[1]],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    colour = sorted(glob.glob(f"{tmp}/f_Color_1*.png"))
    raw = sorted(glob.glob(f"{tmp}/raw_Depth_*.bin"))
    if not colour or not raw:
        print(f"  {run}: NO FRAMES"); return 0
    n = min(len(colour), len(raw))
    pick = np.linspace(0, n - 1, min(N_FRAMES, n)).round().astype(int)
    for sub in ("rgb", "depth"):
        os.makedirs(f"{OUT}/{run}/{sub}", exist_ok=True)
    for i, k in enumerate(pick):
        bgr = cv2.imread(colour[k])
        d = np.fromfile(raw[k], np.float32).reshape(H, W)              # metres
        mm = np.clip(align(d) * 1000, 0, 65535).astype(np.uint16)
        bgr, mm = to_isod(bgr, mm)
        cv2.imwrite(f"{OUT}/{run}/rgb/{i}.png", bgr)
        cv2.imwrite(f"{OUT}/{run}/depth/{i}.png", mm)
    subprocess.run(["rm", "-rf", tmp], check=False)
    return len(pick)


def main():
    runs = sys.argv[1:] or sorted(
        os.path.basename(p)[:-4] for p in glob.glob(f"{RAW}/*.db3")
        if not os.path.basename(p).startswith("pilot"))
    total = 0
    for r in runs:
        k = convert(r)
        total += k
        print(f"  {r}: {k} frames", flush=True)
    print(f"{total} frames from {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    main()
