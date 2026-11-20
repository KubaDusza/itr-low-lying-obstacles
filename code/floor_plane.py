"""Inverse-depth floor fitting shared by the depth encoders and baseline."""

import numpy as np


def fit_floor_plane(inv, mask, trim=2.5, iters=4, step=4):
    """Robust affine fit of inverse depth over `mask`; returns the fitted 1/z image.

    For a pinhole camera looking at a plane, inverse depth is an *exactly* affine function
    of the pixel coordinates, whatever the focal length and principal point are: with
    X = z((u-cx)/fx, (v-cy)/fy, 1) and a plane n.X = d, one gets
    1/z = (n0(u-cx)/fx + n1(v-cy)/fy + n2)/d. So the floor can be recovered with three
    numbers and no intrinsics, which matters because ISOD ships none.
    """
    h, w = inv.shape
    u, v = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32),
                       np.linspace(-1, 1, h, dtype=np.float32))
    us, vs, bs = u[mask][::step], v[mask][::step], inv[mask][::step]
    if bs.size < 200:
        return None
    A = np.stack([us, vs, np.ones_like(us)], 1)
    c = np.zeros(3, np.float32)
    for _ in range(iters):
        c = np.linalg.lstsq(A, bs, rcond=None)[0]
        r = np.abs(A @ c - bs)
        keep = r < max(1e-9, trim * float(np.median(r)))
        if keep.sum() < 200:
            break
        A, bs = A[keep], bs[keep]
    return c[0] * u + c[1] * v + c[2]


def floor_gap(raw, st, seed_from=0.1):
    """Metres by which each pixel's surface sits in front of the fitted floor plane.

    The plane is fitted blind, on the valid pixels of the lower part of the image, which
    is floor in this dataset and in our own recordings. Along a pixel's ray the floor is
    at z_plane and the measurement is at z, so z_plane - z is how far the surface is in
    front of the floor: zero on the floor, positive on anything lying on it, and in
    metres, which makes it comparable between near and far objects. That is HHA's "height
    above ground" channel, and it is also exactly the quantity the dataset's own geometric
    baseline thresholds.

    Returned together with the validity mask, because above the horizon the plane is
    behind the camera and the quantity is meaningless.
    """
    d = raw.astype(np.float32) / 1000.0
    valid = (raw > 0) & (d > st["near_m"]) & (d < st["far_m"])
    inv = st["near_m"] / np.maximum(d, 1e-3)
    h, w = raw.shape
    _, v = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32),
                       np.linspace(-1, 1, h, dtype=np.float32))
    pl = fit_floor_plane(inv, valid & (v > seed_from))
    if pl is None:
        return np.zeros_like(inv), np.zeros_like(valid)
    ok = valid & (pl > st["near_m"] / st["far_m"])
    z_plane = st["near_m"] / np.maximum(pl, 1e-3)
    return np.where(ok, np.clip(z_plane - d, -0.5, 0.5), 0.0).astype(np.float32), ok
