"""Shared pieces of the modality experiment: depth encodings, dataset, model builders.

One U-Net + MobileNetV2 is trained on several inputs with everything else held fixed, so
that a difference between two arms is a difference of input and not of anything else:

  rgb      3 channels, the pretrained first convolution untouched
  depth    1 channel of inverse depth, the 3 pretrained filters summed (FuseNet's case i)
  depth3   the same channel tripled, so the pretrained conv1 is untouched as well; this
           removes "you butchered the first layer" as an explanation for the depth result
  geo      3 geometric channels in the spirit of HHA, the published way to hand depth to
           an RGB-pretrained network (see encode_geo)
  rgbd     4 channels, early fusion: RGB filters kept exactly, the depth filter zero
  rgbgeo   6 channels, the same with the geometric encoding
  fuse     mid-level fusion: a second MobileNetV2 reads depth and its features are added
           into the RGB encoder's features at each stage, through a zero-initialised gate
  fusegeo  the same with the geometric encoding

Every fusion arm is built so that at step 0 it computes exactly what the RGB arm computes
(zero depth filter / zero gate), which is TernausNetV2's initialisation. That is what makes
a paired RGB vs. RGB-D difference mean something: at a given seed the two runs start from
identical weights and see identical batches, so the difference is the depth input and
nothing else. `diagnose.py check` verifies that rather than asserting it.
"""
import json
import os
import random

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from floor_plane import fit_floor_plane, floor_gap

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (640, 448)        # multiple of 32 for the encoder; ISOD frames are 640x422

# what each arm reads, how many input channels that is, and how the first layer is built
ARMS = {
    "rgb":     dict(ch=3, reads=("rgb",)),
    "depth":   dict(ch=1, reads=("dep",)),
    "depth3":  dict(ch=3, reads=("dep3",)),
    "geo":     dict(ch=3, reads=("geo",)),
    "rgbd":    dict(ch=4, reads=("rgb", "dep"), expand=1),
    "rgbgeo":  dict(ch=6, reads=("rgb", "geo"), expand=3),
    "fuse":    dict(ch=3, reads=("rgb", "dep"), two_stream=1),
    "fusegeo": dict(ch=3, reads=("rgb", "geo"), two_stream=3),
}
MODALITIES = tuple(ARMS)
USES_DEPTH = tuple(k for k, v in ARMS.items() if v["reads"] != ("rgb",))


# ---------------------------------------------------------------- depth encodings

def encode_depth(raw, st):
    """uint16 millimetres -> one standardised channel of clipped inverse depth.

    Inverse depth, because a stereo camera's error grows with z^2: 1/z is the quantity
    whose noise is roughly constant, and it keeps the near field (where the obstacles are)
    spread out instead of squeezed into the top of the range. Clipped at the 1st and 99th
    percentile of the training sites, so a few wild readings cannot set the scale. A
    missing reading (0) means "no return", which on a stereo camera happens on far, dark
    or specular surfaces, so it is mapped to the far end rather than to 0 mm, which would
    read as an obstacle directly in front of the camera.
    """
    d = raw.astype(np.float32) / 1000.0
    inv = st["near_m"] / np.clip(d, st["near_m"], st["far_m"])
    inv[raw == 0] = st["near_m"] / st["far_m"]
    return (inv - st["mean"]) / st["std"]


def encode_geo(raw, st):
    """Three geometric channels, HHA's idea without needing intrinsics or an IMU.

    HHA (Gupta et al. 2014) is horizontal disparity, height above ground and the angle of
    the surface normal with gravity. It exists because raw depth poured into an
    RGB-pretrained network wastes most of what depth knows, so it is the right way to ask
    whether our depth arm was simply fed badly. The same three quantities, measured here
    without intrinsics:

      0  inverse depth           (disparity, as in the single-channel arm)
      1  gap to the floor plane  (height above the ground, in metres)
      2  gradient of that gap    (how the surface tilts away from the floor, which is what
                                  the normal-angle channel carries)
    """
    inv = encode_depth(raw, st) * st["std"] + st["mean"]        # raw clipped inverse depth
    gap, _ = floor_gap(raw, st)
    grad = cv2.magnitude(cv2.Sobel(gap, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(gap, cv2.CV_32F, 0, 1, 3))
    # identity until depth_stats.py has measured the three channels on the training sites
    g = st.get("geo", {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]})
    out = np.stack([inv, gap, grad]).astype(np.float32)
    return (out - np.array(g["mean"], np.float32)[:, None, None]) / \
        np.array(g["std"], np.float32)[:, None, None]


def load_depth_stats(root):
    with open(f"{root}/depth_stats.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- dataset

class ISOD(Dataset):
    """Every arm gets the same loader, which always returns every input.

    The model decides what to look at. That is what keeps the batches identical across
    arms at a given seed, and it costs one extra PNG read and one plane fit per sample.

    target="floor" swaps the obstacle mask for ISOD's drivable-floor label. It is not part
    of the comparison: it is the sufficiency check. Depth is known to be good at finding
    the floor plane (the dataset's own geometric method scores 82.7 mIoU on it), so a
    depth-only network that reaches a high score there but fails on obstacles has shown
    that the architecture and the schedule are not what is failing.
    """

    def __init__(self, root, split, stats, augment=False, target="obstacle", limit=0):
        with open(f"{root}/splits.json") as f:
            self.sites = json.load(f)[split]
        self.root, self.augment, self.stats, self.target = root, augment, stats, target
        self.items = [(s, i) for s in self.sites for i in range(100)]
        if limit:
            self.items = self.items[::max(1, len(self.items) // limit)][:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        b = f"{self.root}/ISOD/data/{site}"
        rgb = cv2.cvtColor(cv2.imread(f"{b}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(f"{b}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
        label = cv2.imread(f"{b}/label/{i}.png", 0)
        gt = cv2.imread(f"{self.root}/ISOD_gt/{site}/{i}.png", 0) > 0
        ignore = (label == 0) & ~gt                      # walls and furniture
        if self.target == "floor":
            gt, ignore = label == 255, np.zeros_like(ignore)
        if self.augment and np.random.rand() < 0.5:
            rgb, raw, gt, ignore = rgb[:, ::-1], raw[:, ::-1], gt[:, ::-1], ignore[:, ::-1]
        rgb_t, dep_t, geo_t = to_tensors(rgb, raw, self.stats)
        return (rgb_t, dep_t, geo_t,
                t(cv2.resize(gt.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)).float(),
                t(1 - cv2.resize(ignore.astype(np.uint8), SIZE,
                                 interpolation=cv2.INTER_NEAREST)).float())


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def to_tensors(rgb, raw, stats, size=SIZE):
    """RGB, depth and the geometric encoding, as model inputs.

    Shared by training and by inference on any folder, so our own recordings go through
    exactly the same encoding as ISOD did.
    """
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    raw = cv2.resize(raw, size, interpolation=cv2.INTER_NEAREST)   # never interpolate holes
    x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return t(x), t(encode_depth(raw, stats))[None], t(encode_geo(raw, stats))


def loaders(root, stats, batch, seed, target="obstacle", limit=0, workers=8):
    """Identical batches, in identical order, for every arm at a given seed."""
    g = torch.Generator()
    g.manual_seed(seed)

    def init(w):
        np.random.seed(seed * 1000 + w)
        random.seed(seed * 1000 + w)

    tr = DataLoader(ISOD(root, "train", stats, augment=True, target=target, limit=limit),
                    batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                    generator=g, worker_init_fn=init)
    va = DataLoader(ISOD(root, "val" if not limit else "train", stats, target=target,
                         limit=limit), batch_size=batch, num_workers=max(2, workers // 2))
    return tr, va


# ---------------------------------------------------------------- models

def expand_first_conv(net, extra=1):
    """3-channel first conv -> 3+extra, keeping the pretrained filters and zeroing the rest.

    Done by hand rather than through smp's `in_channels=4`, which tiles the RGB filters
    into the new channels and rescales all of them by 3/4 -- after which the RGB and RGB-D
    arms would no longer start from the same function.
    """
    for mod in net.encoder.modules():
        for name, child in mod.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new = nn.Conv2d(3 + extra, child.out_channels, child.kernel_size,
                                child.stride, child.padding, child.dilation,
                                child.groups, child.bias is not None)
                with torch.no_grad():
                    new.weight.zero_()
                    new.weight[:, :3] = child.weight
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(mod, name, new)
                return new
    raise RuntimeError("no 3-channel convolution found in the encoder")


class MidFusion(nn.Module):
    """FuseNet's mid-level fusion at MobileNetV2 scale, with a zero-initialised gate.

    A second encoder reads the depth input; at every stage its features are added into the
    RGB features, scaled by one learnable number per stage. The gates start at zero, so
    the model begins as the RGB model and whatever it does with depth it had to learn. The
    learned gate values are a direct readout of how much depth the model wanted, which the
    4-channel arm cannot give.
    """

    def __init__(self, encoder="mobilenet_v2", pretrained=True, dep_ch=1):
        super().__init__()
        w = "imagenet" if pretrained else None
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=w, in_channels=3, classes=1)
        self.dep = smp.encoders.get_encoder(encoder, in_channels=dep_ch, depth=5, weights=w)
        self.gate = nn.Parameter(torch.zeros(len(self.net.encoder.out_channels) - 1))

    def forward(self, rgb, dep):
        fr, fd = self.net.encoder(rgb), self.dep(dep)
        f = [fr[0]] + [r + self.gate[i] * d for i, (r, d) in enumerate(zip(fr[1:], fd[1:]))]
        return self.net.segmentation_head(self.net.decoder(f))


def build(modality, encoder="mobilenet_v2", pretrained=True):
    spec = ARMS[modality]
    if "two_stream" in spec:
        return MidFusion(encoder, pretrained, spec["two_stream"])
    w = "imagenet" if pretrained else None
    # built at 3 channels and expanded by hand, so the RNG draws match the RGB arm exactly
    base = 3 if "expand" in spec else spec["ch"]
    net = smp.Unet(encoder_name=encoder, encoder_weights=w, in_channels=base, classes=1)
    if "expand" in spec:
        expand_first_conv(net, spec["expand"])
    return net


def pick(modality, rgb, dep, geo, blind=False):
    """The tensors this arm feeds its encoder(s).

    blind=True replaces every depth-derived input with the training mean (zero after
    standardisation), which leaves the architecture and the parameter count untouched and
    removes the information. It is the control for "the fusion arm only won because it is
    a bigger model".
    """
    if blind:
        dep, geo = torch.zeros_like(dep), torch.zeros_like(geo)
    avail = {"rgb": rgb, "dep": dep, "dep3": dep.expand(-1, 3, -1, -1), "geo": geo}
    return [avail[k] for k in ARMS[modality]["reads"]]


def forward(net, modality, rgb, dep, geo, blind=False):
    xs = pick(modality, rgb, dep, geo, blind)
    if "two_stream" in ARMS[modality]:
        return net(*xs)
    return net(torch.cat(xs, 1) if len(xs) > 1 else xs[0])


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_name(modality, seed, scratch=False, blind=False, tag=""):
    return (f"{modality}{'_scratch' if scratch else ''}{'_blind' if blind else ''}"
            f"{('_' + tag) if tag else ''}_s{seed}")


def checkpoint(root, name):
    return f"{root}/runs/modal/{name}.pt"


def load(root, name, modality, device, pretrained=False, encoder="mobilenet_v2"):
    net = build(modality, encoder, pretrained=pretrained)
    net.load_state_dict(torch.load(checkpoint(root, name), map_location=device))
    return net.to(device).eval()


def masked_loss(logits, y, w):
    """Same loss as train_seg.py: BCE + soft dice, with walls and furniture dropped."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], y, reduction="none")
    bce = (bce * w).sum() / w.sum().clamp(min=1)
    p, tgt = torch.sigmoid(logits[:, 0]) * w, y * w
    dice = 1 - (2 * (p * tgt).sum() + 1) / ((p + tgt).sum() + 1)
    return bce + dice
