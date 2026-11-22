"""Predict all 300 ISOD test frames with frozen, already-selected checkpoints.

ESANet RGB/D reference (GTE-trained, weights in ~/itr_weights/esanet_gte), U-Net, longer
YOLOv8n-seg and YOLOv8n box detectors. No training, no threshold changes. Box detections are
exported at confidence 0.001 and filled at 0.25 later, exactly as for GTE-Floor.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import cv2
import numpy as np

PROD = Path('/scratch/work/duszaj1/itr_20261121/production')
EXT = Path('/scratch/work/duszaj1/itr_extended_20261122')
GTEW = Path.home() / 'itr_weights/esanet_gte'
SITES = ['HDCL_HW_A', 'MEB_CM', 'MEB_HW_B']


def frames(work):
    return [(s, i, work / 'ISOD/data' / s / 'rgb' / f'{i}.png') for s in SITES for i in range(100)]


def save(out, site, i, mask):
    p = out / site / f'{i}.png'
    p.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(p), mask.astype(np.uint8) * 255)


def esanet(work, out, env):
    spec = importlib.util.spec_from_file_location('controller', work / 'code/overnight_sam_v2.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    dataset = work / 'esanet_reference'
    subprocess.run([sys.executable, 'code/esanet/export_isod.py', '--out', str(dataset),
                    '--train-masks', str(work / 'ISOD_reference_train_v1'),
                    '--val-masks', str(work / 'ISOD_reference_val_v1'),
                    '--test-masks', str(work / 'ISOD_reference_test_v1'),
                    '--depth-stats', str(PROD / 'depth_stats.json')], cwd=work, env=env, check=True)
    controller.unlabeled_test(work, dataset, work / 'esanet_test_all')
    for name, modality in [(f'esanet_full_{m}_reference_s{s}', k) for m, k in (('rgb', 'rgb'), ('depth', 'depth'))
                           for s in range(3)]:
        dest = out / name
        if len(list(dest.glob('*/*.png'))) == 300:
            continue
        cmd = [sys.executable, str(work / 'code/esanet/predict_esanet.py')] + controller.COMMON + [
            '--modality', modality, '--dataset_dir', str(work / 'esanet_test_all'),
            '--ckpt_path', str(GTEW / f'{name}.pth'), '--pred_split', 'test', '--out', str(dest)]
        subprocess.run(cmd, cwd=work / 'esanet', env=env, check=True)
        assert len(list(dest.glob('*/*.png'))) == 300, name


def unet(work, out):
    import segmentation_models_pytorch as smp
    import torch
    sys.path.insert(0, str(EXT / 'helpers'))
    from train_seg import rgb_tensor
    for s in range(3):
        name = f'unet_logged_reference_s{s}'
        net = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, classes=1)
        net.load_state_dict(torch.load(EXT / 'runs' / name / 'best.pt', map_location='cpu', weights_only=True))
        net = net.cuda().eval()
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            for site, i, path in frames(work):
                rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                pred = torch.sigmoid(net(rgb_tensor(rgb)[None].cuda())[0, 0]) > 0.5
                save(out / name, site, i, cv2.resize(pred.cpu().numpy().astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST) > 0)


def yolo(work, out):
    from ultralytics import YOLO
    paths = frames(work)
    for s in range(3):
        name = f'yoloseg_long_reference_s{s}'
        model = YOLO(str(EXT / 'runs' / name / 'weights/selected.pt'))
        for (site, i, path) in paths:
            r = model.predict(str(path), conf=0.25, imgsz=640, retina_masks=True, device=0, verbose=False)[0]
            mask = np.zeros(r.orig_shape, bool) if r.masks is None else r.masks.data.any(dim=0).cpu().numpy()
            assert mask.shape == r.orig_shape
            save(out / name, site, i, mask)
    for kind in ('reference', 'automatic'):
        for s in range(3):
            name = f'yolov8n_detect_{kind}_s{s}'
            model = YOLO(str(EXT / 'runs' / name / 'weights/best.pt'))
            records = []
            for (site, i, path) in paths:
                r = model.predict(str(path), device=0, imgsz=640, conf=.001, iou=.7, max_det=300, verbose=False)[0]
                records.append(dict(site=site, frame=i, shape=list(r.orig_shape), boxes=r.boxes.xyxy.cpu().tolist(),
                                    scores=r.boxes.conf.cpu().tolist()))
            (out / name).mkdir(parents=True, exist_ok=True)
            (out / name / 'isod_boxes.json').write_text(json.dumps(dict(confidence_export_threshold=.001,
                                                                       frames=records)) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    work = Path(os.environ['SLURM_TMPDIR']) / 'itr'
    work.mkdir()
    for name in ('source.tar.gz', 'data.tar.gz'):
        with tarfile.open(PROD / name) as tar:
            tar.extractall(work, filter='data')
    shutil.copytree(PROD / 'code', work / 'code', dirs_exist_ok=True)
    env = os.environ.copy()
    env.update(ITR_DATA=str(work), PYTHONPATH=str(work / 'code') + ':' + str(PROD / 'deps'), HF_HUB_OFFLINE='1')
    a.out.mkdir(parents=True, exist_ok=True)
    esanet(work, a.out, env)
    unet(work, a.out)
    yolo(work, a.out)
    (a.out / 'DONE').write_text('300-frame ISOD test predictions written\n')


if __name__ == '__main__':
    main()
