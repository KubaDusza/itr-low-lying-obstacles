"""Train ESANet on ISOD obstacle masks with RGB, depth or RGB-D input.

What this script is for
-----------------------
Our research question: does adding depth to RGB help a network find small obstacles on the
floor? To test it fairly we train the SAME network family (ESANet, Seichter et al., ICRA 2021)
three times, changing only the input: RGB, depth, or both.

What comes from ESANet, unchanged
---------------------------------
The network, the data augmentation, the normalisation and the loss are imported from the
authors' own code (https://github.com/TUI-NICR/ESANet), which must be cloned separately and
passed with --esanet. All their default settings (learning rate, batch size, ...) are read from
their own argument parser, so nothing is retyped (and possibly mistyped) here.

What this script adds for our task
----------------------------------
1. Reading ISOD with our existing split (splits.json) and obstacle masks (ISOD_gt/).
2. Turning each frame into the 3-value label ESANet expects:
       0 = void (walls and furniture: ignored by the loss and the metrics)
       1 = background (floor and everything else that is not an obstacle)
       2 = obstacle
3. Depth mean/std and class weights computed on the TRAINING sites only.
4. Keeping the checkpoint with the best validation OBSTACLE IoU (ESANet would pick the best
   mean IoU over all classes), and saving only that one file.
5. Writing test masks at ISOD's native 640x422 size and scoring them with isod.py, the same
   scoring code used for every other method in the project.

How to run
----------
  python train_esanet.py --esanet <ESANet folder> --pretrained_dir <folder with r34_NBt1D.pth> \
      --root <isod_work> --out <approved storage folder> --modality rgbd --seed 0
  Add --smoke to only check that everything runs (2 epochs of 3 batches, 10 test frames).

Outputs, in <out>/esanet_r34nbt1d_<modality>_s<seed>/
  config.json     every setting of the run, code versions, GPU
  history.csv     one row per epoch: learning rate, train/val loss, val IoU/precision/recall, time
  best.pt         weights of the epoch with the best validation obstacle IoU
  pred_test/      predicted test masks, <site>/<i>.png, 255 = obstacle
  test_eval.json  test IoU / precision / recall, speed, peak GPU memory
"""
import argparse
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import isod  # our own helpers (code/isod.py): the scoring functions used for every method

# The network input size. Same as in the report and as the U-Net runs (train_seg.SIZE).
# Both sides must be divisible by 32 because the encoder halves the image 5 times.
HEIGHT = 448
WIDTH = 640
# ISOD frames are 640 wide and 422 high. Test masks are written back at this size.
NATIVE_HEIGHT = 422
NATIVE_WIDTH = 640
# Values of the label image given to ESANet.
VOID = 0
BACKGROUND = 1
OBSTACLE = 2


# =============================================================================================
# 1. Reading one ISOD frame
# =============================================================================================
def read_frame(root, site, i):
    """Read one frame and check that its files are what we expect.

    Returns
      rgb           uint8, height x width x 3, in RGB order
      depth         uint16, height x width, millimetres, 0 = the camera had no reading
      floor_label   uint8, height x width, ISOD's hand label: 255 = drivable floor, 0 = not
      obstacle      bool, height x width, our derived obstacle mask (ISOD_gt, 255 = obstacle)
    """
    folder = f"{root}/ISOD/data/{site}"
    # IMREAD_UNCHANGED: read the file exactly as stored. Without it OpenCV would silently turn
    # 16-bit depth into 8-bit and any image into 3 channels, and we could not detect problems.
    bgr = cv2.imread(f"{folder}/rgb/{i}.png", cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(f"{folder}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
    floor_label = cv2.imread(f"{folder}/label/{i}.png", cv2.IMREAD_UNCHANGED)
    obstacle_png = cv2.imread(f"{root}/ISOD_gt/{site}/{i}.png", cv2.IMREAD_UNCHANGED)

    name = f"{site}/{i}"
    for image in (bgr, depth, floor_label, obstacle_png):
        assert image is not None, f"{name}: a file is missing or unreadable"
    assert bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, f"{name}: rgb is not 8-bit colour"
    assert depth.dtype == np.uint16 and depth.ndim == 2, f"{name}: depth is not 16-bit (mm)"
    assert floor_label.ndim == 2 and obstacle_png.ndim == 2, f"{name}: a label is not single-channel"
    same_size = bgr.shape[:2] == depth.shape == floor_label.shape == obstacle_png.shape
    assert same_size, f"{name}: rgb, depth and labels have different sizes"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV stores colour as BGR; networks expect RGB
    obstacle = obstacle_png > 0
    return rgb, depth, floor_label, obstacle


def make_esanet_label(floor_label, obstacle):
    """Build the 0/1/2 label from ISOD's floor label and our obstacle mask.

    The rule is the same as in the U-Net runs (train_modal.py): a pixel is ignored (VOID) when
    ISOD says "not drivable" and it is not one of our obstacles, i.e. walls and furniture.
    They are not our objects, so predicting them should neither help nor hurt.
    """
    label = np.full(floor_label.shape, BACKGROUND, dtype=np.uint8)
    walls_and_furniture = (floor_label == 0) & (obstacle == False)
    label[walls_and_furniture] = VOID
    label[obstacle] = OBSTACLE
    return label


# =============================================================================================
# 2. The dataset object that PyTorch's DataLoader asks for
# =============================================================================================
class ISODForESANet(Dataset):
    """PyTorch needs an object that can say how many samples there are (__len__) and give
    sample number k (__getitem__). The DataLoader then batches and shuffles them for us.
    The `preprocessor` is ESANet's own chain of transforms (resize/crop/flip/normalise)."""

    def __init__(self, root, sites, preprocessor):
        self.root = root
        self.preprocessor = preprocessor
        self.frames = []  # list of (site, frame number); every ISOD site has frames 0..99
        for site in sites:
            for i in range(100):
                self.frames.append((site, i))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        site, i = self.frames[k]
        rgb, depth, floor_label, obstacle = read_frame(self.root, site, i)
        sample = {
            "image": rgb,
            "depth": depth.astype(np.float32),  # still millimetres; ESANet normalises it later
            "label": make_esanet_label(floor_label, obstacle),
        }
        return self.preprocessor(sample)


# =============================================================================================
# 3. Statistics computed on the training sites only
# =============================================================================================
def compute_train_statistics(root, train_sites):
    """Depth mean/std and class weights, from the TRAINING frames only.

    Using validation or test frames here would leak information about them into training.

    Depth: mean and standard deviation over valid pixels (depth > 0), in millimetres. This is
    what ESANet does for raw depth; invalid pixels are later kept at exactly 0.

    Class weights ("median frequency", ESANet's default): obstacles cover well under 1 % of the
    pixels, so an unweighted loss would let the network ignore them. For each class c:
        frequency[c] = pixels of class c / total pixels of the frames in which c appears
        weight[c]    = median(frequency) / frequency[c]
    so the rarer class gets the larger weight. VOID is left out, as in ESANet.
    """
    depth_sum = 0.0
    depth_sum_of_squares = 0.0
    depth_count = 0
    pixels_per_class = np.zeros(3)               # index 0 = VOID, 1 = BACKGROUND, 2 = OBSTACLE
    pixels_of_frames_containing_class = np.zeros(3)

    for site in train_sites:
        for i in range(100):
            _, depth, floor_label, obstacle = read_frame(root, site, i)
            label = make_esanet_label(floor_label, obstacle)

            valid_depth = depth[depth > 0].astype(np.float64)  # float64: the sums get very large
            depth_sum += valid_depth.sum()
            depth_sum_of_squares += (valid_depth ** 2).sum()
            depth_count += valid_depth.size

            counts = np.bincount(label.ravel(), minlength=3)  # pixels of value 0, 1 and 2
            pixels_per_class += counts
            for c in range(3):
                if counts[c] > 0:
                    pixels_of_frames_containing_class[c] += label.size

    depth_mean = depth_sum / depth_count
    depth_variance = depth_sum_of_squares / depth_count - depth_mean ** 2
    frequency = pixels_per_class[1:] / pixels_of_frames_containing_class[1:]  # skip VOID
    class_weights = np.median(frequency) / frequency  # [background weight, obstacle weight]

    return {
        "depth_mean": float(depth_mean),
        "depth_std": float(np.sqrt(depth_variance)),
        "class_weights": class_weights.tolist(),
        "pixels_per_class": pixels_per_class.tolist(),
        "train_sites": sorted(train_sites),
    }


def load_or_compute_train_statistics(root, train_sites, out_folder):
    """The statistics take a few minutes, so they are saved once and reused by every run in
    the same --out folder. That also guarantees all runs use identical numbers."""
    path = f"{out_folder}/isod_train_stats.json"
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)
        assert stats["train_sites"] == sorted(train_sites), f"{path} was made from another split"
        return stats
    print("Computing depth mean/std and class weights on the training sites ...", flush=True)
    stats = compute_train_statistics(root, train_sites)
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# =============================================================================================
# 4. Running the network
# =============================================================================================
def run_network(model, batch, modality, device):
    """ESANet's RGB-D model takes two inputs; the single-modality models take one."""
    if modality == "rgbd":
        return model(batch["image"].to(device), batch["depth"].to(device))
    if modality == "rgb":
        return model(batch["image"].to(device))
    return model(batch["depth"].to(device))


def predicted_obstacles(logits):
    """The network outputs 2 scores per pixel: index 0 = background, index 1 = obstacle.
    (They are shifted by one from our label values because VOID has no output.)
    The predicted class is the one with the higher score."""
    return logits.argmax(dim=1) == 1


def train_one_epoch(model, loader, loss_function, optimizer, modality, device, max_batches):
    model.train()  # training mode: e.g. batch-norm statistics are updated
    losses = []
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        # ESANet supervises the full-resolution output and three smaller side outputs
        # (1/8, 1/16, 1/32 of the size). ESANet's preprocessing already made the matching
        # smaller labels in batch["label_down"]; the order must be 8, 16, 32 like the outputs.
        targets = [batch["label"].to(device)]
        for rate in sorted(batch["label_down"]):
            targets.append(batch["label_down"][rate].to(device))

        optimizer.zero_grad()
        outputs = run_network(model, batch, modality, device)  # 4 outputs in training mode
        loss = sum(loss_function(outputs, targets))  # ESANet returns one loss per output
        if not torch.isfinite(loss):
            raise SystemExit(f"Loss became NaN/inf at batch {batch_number}: stopping.")
        loss.backward()   # compute the gradients
        optimizer.step()  # update the weights
        losses.append(loss.item())
    return float(np.mean(losses))


def validate(model, loader, loss_function, modality, device, max_batches):
    """Validation loss, and obstacle IoU / precision / recall with VOID pixels excluded.
    Pixel counts are summed over all frames before dividing, like isod.totals()."""
    model.eval()  # evaluation mode: one output only, batch-norm statistics frozen
    losses = []
    intersection = union = predicted = true = 0
    with torch.no_grad():  # no gradients needed: faster and uses less memory
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            label = batch["label"].to(device)
            logits = run_network(model, batch, modality, device)
            losses.append(loss_function([logits], [label])[0].item())  # same weighting as training

            prediction = predicted_obstacles(logits) & (label != VOID)
            truth = label == OBSTACLE
            intersection += int((prediction & truth).sum())
            union += int((prediction | truth).sum())
            predicted += int(prediction.sum())
            true += int(truth.sum())
    iou = intersection / max(1, union)
    precision = intersection / max(1, predicted)
    recall = intersection / max(1, true)
    return float(np.mean(losses)), iou, precision, recall


def predict_and_score_test(model, test_set, modality, device, out_folder, n_frames):
    """Predict every test frame, save the mask at native size, and score it with isod.py.

    Scoring uses isod.ignore_region / isod.score / isod.totals: exactly what eval.py does for
    YOLO, U-Net, DeepLabV3+ and the depth baseline, so all methods are measured the same way."""
    model.eval()
    rows = []
    start = time.time()
    with torch.no_grad():
        for k in range(n_frames):
            site, i = test_set.frames[k]
            sample = test_set[k]
            # The network expects a batch: add a batch dimension of size 1 with [None].
            batch = {"image": sample["image"][None], "depth": sample["depth"][None]}
            logits = run_network(model, batch, modality, device)
            # Resize the scores (not the hard mask) back to 640x422, then pick the class.
            logits = F.interpolate(logits, size=(NATIVE_HEIGHT, NATIVE_WIDTH),
                                   mode="bilinear", align_corners=False)
            mask = predicted_obstacles(logits)[0].cpu().numpy().astype(np.uint8)  # 1 = obstacle

            os.makedirs(f"{out_folder}/{site}", exist_ok=True)
            Image.fromarray(mask * 255).save(f"{out_folder}/{site}/{i}.png")

            _, _, floor_label, obstacle = read_frame(test_set.root, site, i)
            gt = obstacle.astype(np.uint8)
            rows.append(isod.score(mask, gt, ignore=isod.ignore_region(floor_label, gt)))
    seconds = time.time() - start
    return isod.totals(rows), n_frames / seconds


# =============================================================================================
# 5. Small helpers for reproducibility
# =============================================================================================
def set_seeds(seed):
    """Fix every random number generator we use, so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    """The DataLoader loads images in several parallel worker processes. ESANet's augmentation
    uses numpy's random numbers, and without this every worker would produce the same "random"
    crops and flips. PyTorch gives each worker its own seed; we pass it on to numpy."""
    np.random.seed(torch.initial_seed() % 2**32)


def git_revision(folder):
    """Short commit id of a git checkout, with '+dirty' if it has uncommitted changes.
    Stored in config.json so we always know which code produced a result."""
    commit = subprocess.run(["git", "-C", folder, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(["git", "-C", folder, "status", "--porcelain"],
                             capture_output=True, text=True).stdout.strip()
    if changes:
        return commit + "+dirty"
    return commit


# =============================================================================================
# 6. Main: one training run from start to finish
# =============================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esanet", required=True, help="folder of the TUI-NICR/ESANet clone")
    parser.add_argument("--pretrained_dir", default=None,
                        help="folder containing r34_NBt1D.pth (ImageNet weights); only --smoke may omit it")
    parser.add_argument("--root", required=True, help="isod_work: ISOD/data, ISOD_gt, splits.json")
    parser.add_argument("--out", required=True, help="approved storage folder (NOT the home disk)")
    parser.add_argument("--modality", required=True, choices=["rgb", "depth", "rgbd"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8, help="parallel image-loading processes")
    parser.add_argument("--batch", type=int, default=None,
                        help="override ESANet's batch size of 8 (only for CPU smoke tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="only check the pipeline: 2 epochs x 3 batches, 10 test frames")
    our_args = parser.parse_args()

    if our_args.pretrained_dir is None and not our_args.smoke:
        raise SystemExit("--pretrained_dir is required for real runs (ImageNet encoder weights).")
    epochs = our_args.epochs
    max_batches = None
    n_test_frames = 300
    if our_args.smoke:
        epochs, max_batches, n_test_frames = 2, 3, 10

    # --- ESANet's code and its default settings -------------------------------------------
    # ESANet is not an installable package. Adding its folder to sys.path lets Python find its
    # "src" package, so the imports below load the authors' unmodified code.
    sys.path.insert(0, our_args.esanet)
    from src.args import ArgumentParserRGBDSegmentation
    from src.build_model import build_model
    from src.preprocessing import get_preprocessor
    from src.utils import CrossEntropyLoss2d

    esanet_flags = ["--modality", our_args.modality, "--height", str(HEIGHT),
                    "--width", str(WIDTH), "--epochs", str(epochs),
                    "--raw_depth"]  # ISOD depth is raw sensor depth with 0 = no reading
    if our_args.pretrained_dir is None:
        esanet_flags.append("--no_imagenet_pretraining")
    else:
        esanet_flags += ["--pretrained_dir", our_args.pretrained_dir]
    esanet_parser = ArgumentParserRGBDSegmentation()
    esanet_parser.set_common_args()
    esanet_args = esanet_parser.parse_args(esanet_flags)  # all other values: ESANet defaults
    if our_args.batch is not None:
        esanet_args.batch_size = our_args.batch

    # --- output folder ------------------------------------------------------------------------
    run_name = f"esanet_r34nbt1d_{our_args.modality}_s{our_args.seed}"
    if our_args.smoke:
        run_name += "_smoke"
    run_folder = f"{our_args.out}/{run_name}"
    if os.path.exists(f"{run_folder}/history.csv") and not our_args.smoke:
        raise SystemExit(f"{run_folder} already contains a run; use another --out or --seed.")
    os.makedirs(run_folder, exist_ok=True)

    set_seeds(our_args.seed)

    # --- data -------------------------------------------------------------------------------
    with open(f"{our_args.root}/splits.json") as f:
        splits = json.load(f)
    overlap = set(splits["train"]) & (set(splits["val"]) | set(splits["test"]))
    assert not overlap, f"sites in both train and val/test: {overlap}"
    stats = load_or_compute_train_statistics(our_args.root, splits["train"], our_args.out)
    print("Training statistics:", {k: v for k, v in stats.items() if k != "train_sites"}, flush=True)

    # ESANet's transforms. Training: random rescale 1.0-1.4, random 448x640 crop, small colour
    # changes, random flip, normalisation. Validation/test: resize to 448x640, normalisation.
    def esanet_preprocessor(phase):
        return get_preprocessor(depth_mean=stats["depth_mean"], depth_std=stats["depth_std"],
                                depth_mode="raw", height=HEIGHT, width=WIDTH, phase=phase,
                                train_random_rescale=(esanet_args.aug_scale_min,
                                                      esanet_args.aug_scale_max))

    train_set = ISODForESANet(our_args.root, splits["train"], esanet_preprocessor("train"))
    val_set = ISODForESANet(our_args.root, splits["val"], esanet_preprocessor("test"))
    test_set = ISODForESANet(our_args.root, splits["test"], esanet_preprocessor("test"))
    # The generator fixes the shuffling order from the seed; drop_last avoids a tiny last batch.
    train_loader = DataLoader(train_set, batch_size=esanet_args.batch_size, shuffle=True,
                              drop_last=True, num_workers=our_args.workers,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(our_args.seed))
    val_loader = DataLoader(val_set, batch_size=esanet_args.batch_size,
                            num_workers=our_args.workers)

    # --- model, loss, optimiser, learning-rate schedule (all as in ESANet's train.py) --------
    # build_model prints the whole network (hundreds of lines); we hide that print.
    with contextlib.redirect_stdout(io.StringIO()):
        model, device = build_model(esanet_args, n_classes=2)  # background and obstacle
    # Weighted cross-entropy; ESANet's version skips VOID pixels (label 0) automatically.
    loss_function = CrossEntropyLoss2d(weight=np.array(stats["class_weights"], dtype=np.float32),
                                       device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=esanet_args.lr,
                                momentum=esanet_args.momentum,
                                weight_decay=esanet_args.weight_decay, nesterov=True)
    # One-cycle schedule: the learning rate rises from lr/25 to lr during the first 10 % of
    # the epochs, then decreases smoothly to almost 0. ESANet updates it once per EPOCH.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=esanet_args.lr, total_steps=epochs, div_factor=25, pct_start=0.1,
        anneal_strategy="cos", final_div_factor=1e4)

    n_parameters = sum(p.numel() for p in model.parameters())
    config = {
        "our_args": vars(our_args),
        "esanet_args": vars(esanet_args),
        "train_stats": stats,
        "n_parameters": n_parameters,
        "selection_rule": "best validation obstacle IoU (void excluded), at 448x640",
        "our_git": git_revision(os.path.dirname(os.path.abspath(__file__))),
        "esanet_git": git_revision(our_args.esanet),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(f"{run_folder}/config.json", "w") as f:
        json.dump(config, f, indent=1, default=str)

    # --- training loop -------------------------------------------------------------------------
    history = open(f"{run_folder}/history.csv", "w")
    history.write("epoch,lr,train_loss,val_loss,val_IoU,val_precision,val_recall,seconds\n")
    best_val_iou = -1.0
    start = time.time()
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer,
                                     our_args.modality, device, max_batches)
        scheduler.step()
        val_loss, val_iou, val_precision, val_recall = validate(
            model, val_loader, loss_function, our_args.modality, device, max_batches)

        seconds = time.time() - start
        history.write(f"{epoch},{lr:.6g},{train_loss:.5f},{val_loss:.5f},{val_iou:.5f},"
                      f"{val_precision:.5f},{val_recall:.5f},{seconds:.1f}\n")
        history.flush()  # write to disk now, so the file is useful even if the run crashes
        print(f"epoch {epoch}/{epochs}  lr={lr:.4g}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_IoU={val_iou:.4f}", flush=True)

        # Keep only the best epoch so far (overwrite), chosen on VALIDATION data only.
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_IoU": float(val_iou)},
                       f"{run_folder}/best.pt")
    history.close()
    train_hours = (time.time() - start) / 3600

    # --- test: once, with the checkpoint chosen on validation ------------------------------
    checkpoint = torch.load(f"{run_folder}/best.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_scores, frames_per_second = predict_and_score_test(
        model, test_set, our_args.modality, device, f"{run_folder}/pred_test", n_test_frames)

    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 2**20
    result = {
        "best_epoch": checkpoint["epoch"],
        "best_val_IoU": checkpoint["val_IoU"],
        "test": test_scores,  # only 10 frames in a smoke test: not a result
        "n_parameters": n_parameters,
        "frames_per_second_incl_io": frames_per_second,
        "train_hours": train_hours,
        "peak_gpu_memory_MiB": peak_memory,
    }
    with open(f"{run_folder}/test_eval.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
