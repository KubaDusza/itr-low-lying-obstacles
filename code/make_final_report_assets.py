"""Final report assets: main results table, appendix score table, cloth figure with
all-seed counts, and RGB-baseline curves using the longer YOLOv8n-seg runs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import make_report_revision_assets as base
from make_report_revision_assets import ROOT, R, F, P, NATIVE, rows, gray, save, COLORS
from floor_plane import floor_gap

NUM = json.loads((ROOT / 'results/results_report_20261122/report_numbers.json').read_text())


def fmt(d, sd=False, digits=3):
    if d is None:
        return '--'
    s = f"{d['mean']:.{digits}f}"
    if sd and d.get('sd') is not None:
        s += f"$\\pm${d['sd']:.{digits}f}"
    return s


ROWS = [  # label, key, group
    ('ESANet RGB', 'ESANet RGB / ref', 'Reference labels (from ISOD annotations)'),
    ('ESANet D', 'ESANet D / ref', 'Reference labels (from ISOD annotations)'),
    ('ESANet RGB-D', 'ESANet RGB-D / ref', 'Reference labels (from ISOD annotations)'),
    ('U-Net RGB', 'U-Net / ref', 'Reference labels (from ISOD annotations)'),
    ('YOLOv8n-seg RGB', 'YOLOv8n-seg / ref', 'Reference labels (from ISOD annotations)'),
    ('YOLOv8n boxes RGB', 'YOLOv8n boxes / ref', 'Reference labels (from ISOD annotations)'),
    ('ESANet RGB', 'ESANet RGB / SAM', 'Automatic SAM 3 labels'),
    ('ESANet D', 'ESANet D / SAM', 'Automatic SAM 3 labels'),
    ('ESANet RGB-D', 'ESANet RGB-D / SAM', 'Automatic SAM 3 labels'),
    ('U-Net RGB', 'U-Net / SAM', 'Automatic SAM 3 labels'),
    ('YOLOv8n-seg RGB', 'YOLOv8n-seg / SAM', 'Automatic SAM 3 labels'),
    ('YOLOv8n boxes RGB', 'YOLOv8n boxes / SAM', 'Automatic SAM 3 labels'),
    ('SAM 3, three prompts', 'SAM 3 direct', 'Direct methods'),
    ('Floor-plane fit (D)', 'Floor plane', 'Direct methods'),
    ('RGB-D, depth removed', 'RGB-D / ref, depth removed', 'Depth intervention (RGB-D, reference)'),
    ('RGB-D, other scene\'s depth', 'RGB-D / ref, depth from other recording', 'Depth intervention (RGB-D, reference)'),
]


def main_table():
    L = [r'\begin{table*}[!t]',
         r'\caption{ISOD and GTE-Floor results (three-seed means for trained models).}',
         r'\label{tab:main}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4.5pt}',
         r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & \multicolumn{4}{c}{ISOD test (300 frames)} & \multicolumn{4}{c}{GTE-Floor (320 frames)}\\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r'Method & Pixel IoU & Obj.\ recall & Comp.\ F1 & False reg./fr. & Pixel IoU & Obj.\ recall & Comp.\ F1 & False reg./fr.\\']
    group = None
    for label, key, g in ROWS:
        if g.startswith('Depth intervention'):
            continue
        if g != group:
            L.append(r'\midrule' if group is None else r'\addlinespace[2pt]')
            L.append(rf'\multicolumn{{9}}{{l}}{{\emph{{{g}}}}}\\')
            group = g
        i, t = NUM['isod300'].get(key), NUM['gte'][key]
        cells = []
        for src in (i, t):
            for m in ('pixel_iou', 'object_recall', 'component_f1', 'false_regions_per_frame'):
                d = src.get(m) if src else None
                cells.append(fmt(d, digits=2 if m == 'false_regions_per_frame' else 3) if d else '--')
        L.append(rf'\quad {label} & ' + ' & '.join(cells) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (P / 'main_results_table.tex').write_text('\n'.join(L) + '\n')


def full_table():
    L = [r'\begin{longtable}{llrrrr}',
         r'\caption{All scores with training-seed standard deviation.}\label{tab:full-scores}\\',
         r'\toprule Dataset & Method & Pixel IoU & Object recall & Component F1 & False regions/frame\\\midrule\endfirsthead',
         r'\toprule Dataset & Method & Pixel IoU & Object recall & Component F1 & False regions/frame\\\midrule\endhead']
    for ds, name in (('isod300', 'ISOD'), ('gte', 'GTE')):
        for label, key, g in ROWS:
            src = NUM[ds].get(key)
            if not src:
                continue
            tag = {'Reference labels (from ISOD annotations)': 'ref.', 'Automatic SAM 3 labels': 'SAM', 'Untrained': '', }.get(g, '')
            cells = [fmt(src.get(m), sd=True, digits=2 if m == 'false_regions_per_frame' else 3) if src.get(m) else '--'
                     for m in ('pixel_iou', 'object_recall', 'component_f1', 'false_regions_per_frame')]
            L.append(f"{name} & {label}{', ' + tag if tag else ''} & " + ' & '.join(cells) + r'\\')
        if ds == 'isod300':
            L.append(r'\midrule')
    L += [r'\bottomrule\end{longtable}']
    (P / 'full_scores_table.tex').write_text('\n'.join(L) + '\n')


def cloth_main(seed=2):
    sources = json.loads((R / 'summary.json').read_text())['source_runs']
    objects = rows(R / 'objects.csv')
    obs = {(r['model'], r['recording'], r['frame'], r['id']): float(r['coverage']) >= .5 for r in objects}
    definitions = json.loads((ROOT / 'results/gte_instance_proposals_20261121/manifest.json').read_text())['frames']
    lookup = {(r['recording'], r['frame']): r for r in definitions}
    settings = json.loads((ROOT / 'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m']
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('black')
    model_of = lambda mode, s: f'rgbd_reference210_s{s}' if mode == 'rgbd' else f'esanet_full_{mode}_reference_s{s}'

    def box_union(recording, frame, s, shape):
        rr = next(r for r in json.loads((NATIVE / f'runs/yolov8n_detect_reference_e60_s{s}/gte_boxes.json').read_text())['frames']
                  if r['image'] == f'{recording}_{frame}')
        u = np.zeros(shape, bool)
        for (x0, y0, x1, y1), c in zip(rr['boxes_xyxy'], rr['scores']):
            if c >= .25:
                u[max(0, int(np.floor(y0))):int(np.ceil(y1)), max(0, int(np.floor(x0))):int(np.ceil(x1))] = True
        return u, rr

    fig, axes = plt.subplots(2, 6, figsize=(6.8, 2.12))
    for row, recording in enumerate(['h52_floor_rags_075', 'h52_poster_rags_075']):
        rgb = np.array(Image.open(ROOT / 'data/OWN' / recording / 'rgb/2.png').convert('RGB'))
        gt = gray(ROOT / 'data/OWN_manual' / f'{recording}_2.png') > 0
        ids = gray(ROOT / 'results/gte_instance_proposals_20261121' / lookup[recording, 2]['path'])
        gap, valid = floor_gap(gray(ROOT / 'data/OWN' / recording / 'depth/2.png'), settings, seed_from=0)
        axes[row, 0].imshow(rgb); axes[row, 0].contour(gt, levels=[.5], colors='cyan', linewidths=.4)
        art = axes[row, 1].imshow(np.ma.masked_where(~valid, gap * 1000), cmap=cmap, vmin=-50, vmax=50)
        axes[row, 1].contour(gt, levels=[.5], colors='yellow', linewidths=.35)
        for col, mode in enumerate(['rgb', 'depth', 'rgbd'], 2):
            pred = gray(ROOT / sources[model_of(mode, seed)]['predictions'] / recording / '2.png') > 0
            painted = rgb.astype(float) * .55
            painted[pred & gt] = [40, 185, 95]; painted[pred & ~gt] = [235, 70, 60]; painted[~pred & gt] = [45, 130, 230]
            axes[row, col].imshow(painted.astype(np.uint8)); axes[row, col].contour(gt, levels=[.5], colors='yellow', linewidths=.35)
            counts = []
            for ident in [1, 2, 3, 4]:
                yy, xx = np.where(ids == ident)
                axes[row, col].text(float(np.mean(xx)), float(np.min(yy)) - 5, str(ident), fontsize=5.7, color='yellow', ha='center')
                counts.append(sum(obs[model_of(mode, s), recording, str(f), str(ident)] for s in range(3) for f in range(5)))
            axes[row, col].text(.5, -.07, '  '.join(f'{i}:{c}' for i, c in zip([1, 2, 3, 4], counts)),
                                transform=axes[row, col].transAxes, ha='center', va='top', fontsize=5.8)
        u, rr = box_union(recording, 2, seed, gt.shape)
        axes[row, 5].imshow(rgb)
        for (x0, y0, x1, y1), c in zip(rr['boxes_xyxy'], rr['scores']):
            if c >= .25:
                axes[row, 5].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=.7, ec='#f46c24'))
        counts = []
        for ident in [1, 2, 3, 4]:
            n = 0
            for s in range(3):
                for f in range(5):
                    idf = gray(ROOT / 'results/gte_instance_proposals_20261121' / lookup[recording, f]['path'])
                    uu, _ = box_union(recording, f, s, idf.shape)
                    n += np.mean(uu[idf == ident]) >= .5
            counts.append(int(n))
        axes[row, 5].text(.5, -.07, '  '.join(f'{i}:{c}' for i, c in zip([1, 2, 3, 4], counts)),
                          transform=axes[row, 5].transAxes, ha='center', va='top', fontsize=5.8)
        axes[row, 0].text(0, 1.13, 'Bare floor' if row == 0 else 'White poster', transform=axes[row, 0].transAxes,
                          ha='left', fontsize=6.8, clip_on=False)
    for ax, title in zip(axes[0], ['RGB + reference', 'Depth residual (mm)', 'ESANet RGB', 'ESANet D', 'ESANet RGB-D', 'YOLOv8n boxes']):
        ax.set_title(title, fontsize=6.7, pad=19)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=.01, right=.995, bottom=.15, top=.84, wspace=.02, hspace=.5)
    fig.canvas.draw(); pos = axes[1, 1].get_position()
    cax = fig.add_axes([pos.x0, .045, pos.width, .012])
    cb = fig.colorbar(art, cax=cax, orientation='horizontal', ticks=[-50, 0, 50]); cb.ax.tick_params(labelsize=5.5, pad=1)
    save(fig, 'cloth_main')


def rgb_baseline_curves():
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.5))
    for source, title, color in [('reference', 'Reference', COLORS['reference']), ('sam', 'SAM 3', COLORS['automatic'])]:
        unet, seg, validation = [], [], []
        for seed in range(3):
            run = ROOT / f'results/extended_training_20261122/runs/unet_logged_{source}_s{seed}'
            h = rows(run / 'history.csv'); assert len(h) == 15
            ep = np.array([int(r['epoch']) for r in h])
            unet.append(np.array([[float(r[k]) for k in ['train_loss', 'val_loss', 'val_IoU']] for r in h]))
            run = ROOT / f'results/extended_training_20261122/runs/yoloseg_e60_{source}_s{seed}'
            h = rows(run / 'results.csv'); assert len(h) == 60
            seg.append(np.array([[float(r[k]) for k in ['train/seg_loss', 'val/seg_loss']] for r in h]))
            sel = json.loads((run / 'selection.json').read_text())
            def epoch(name):
                return len(h) if name == 'last.pt' else int(name[5:-3]) + 1
            points = {epoch(k): v for k, v in sel['val_iou'].items()}
            validation.append(points)
            xs = sorted(points)
            axes[1, 2].plot(xs, [points[x] for x in xs], color=color, lw=.6, alpha=.28)
            best = epoch(sel['best'])
            axes[1, 2].plot(best, sel['val_iou'][sel['best']], marker='o', ms=3.5, mfc='none', color=color)
        for col in range(3):
            a = np.array(unet)
            for y in a: axes[0, col].plot(ep, y[:, col], color=color, alpha=.25, lw=.6)
            axes[0, col].plot(ep, a[:, :, col].mean(0), color=color, lw=1, label=title)
        a = np.array(seg)
        for col in range(2):
            for y in a: axes[1, col].plot(np.arange(1, 61), y[:, col], color=color, alpha=.25, lw=.6)
            axes[1, col].plot(np.arange(1, 61), a[:, :, col].mean(0), color=color, lw=1)
        xs = sorted(set.intersection(*(set(v) for v in validation)))
        axes[1, 2].plot(xs, np.array([[v[x] for x in xs] for v in validation]).mean(0), color=color, lw=1)
    titles = ['U-Net: training loss', 'U-Net: validation loss', 'U-Net: validation IoU', 'YOLOv8n-seg: training mask loss',
              'YOLOv8n-seg: validation mask loss', 'YOLOv8n-seg: validation obstacle IoU']
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=8.5); ax.set_xlabel('Epoch'); ax.grid(axis='y', color='.88', lw=.5); ax.spines[['top', 'right']].set_visible(False)
    for r in [0, 1]:
        lim = max(axes[r, 0].dataLim.ymax, axes[r, 1].dataLim.ymax) * 1.04
        for c in [0, 1]: axes[r, c].set_ylim(0, lim)
        axes[r, 2].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=.7); save(fig, 'appendix_rgb_baseline_curves')


def native_curves():
    fig, axes = plt.subplots(2, 4, figsize=(8.2, 4.5))
    fields = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/mAP50-95(B)']
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3)]
    for source in ['reference', 'automatic']:
        runs = []
        for seed in range(3):
            run = ROOT / f'results/extended_training_20261122/runs/yolov8n_detect_{source}_e60_s{seed}'
            h = rows(run / 'results.csv'); assert len(h) == 60
            runs.append(np.array([[float(r[k]) for k in fields] for r in h]))
        a = np.array(runs); ep = np.arange(1, 61)
        for col, pos in enumerate(positions):
            ax = axes[pos]
            for y in a:
                ax.plot(ep, y[:, col], color=COLORS[source], lw=.6, alpha=.25)
            ax.plot(ep, a[:, :, col].mean(0), color=COLORS[source], lw=1, label='Reference' if source == 'reference' else 'SAM')
    for pos, title in zip(positions, ['Train: box loss', 'Train: classification loss', 'Train: distribution loss', 'Validation: box loss',
                                      'Validation: classification loss', 'Validation: distribution loss', 'Validation box AP50:95']):
        ax = axes[pos]; ax.set_title(title, fontsize=8.3); ax.set_xlabel('Epoch'); ax.grid(axis='y', color='.88', lw=.5); ax.spines[['top', 'right']].set_visible(False)
    for c in range(3):
        lim = max(axes[0, c].dataLim.ymax, axes[1, c].dataLim.ymax) * 1.04
        for r in range(2):
            axes[r, c].set_ylim(0, lim)
    axes[0, 3].set_ylim(0, 1); axes[1, 3].axis('off')
    axes[0, 0].legend(frameon=False, fontsize=7.5); fig.tight_layout(pad=.7); save(fig, 'appendix_native_curves')


def teacher_examples():
    """SAM 3 label examples (development and validation frames), same panels as before, plain footer."""
    from matplotlib.backends.backend_pdf import PdfPages
    choices = [('development', 'BT_BM', 12), ('development', 'BT_LM', 34), ('validation', 'HL_BM', 87), ('validation', 'MEB_HW_A', 44)]
    base = {'development': 'ISOD_reference_train_v1', 'validation': 'ISOD_reference_val_v1'}
    with PdfPages(F / 'automatic_teacher_examples.pdf') as pdf:
        for start in (0, 2):
            fig, axes = plt.subplots(2, 3, figsize=(10, 5.1))
            for row, (phase, site, frame) in enumerate(choices[start:start + 2]):
                rgb = np.array(Image.open(ROOT / 'data/ISOD/data' / site / 'rgb' / f'{frame}.png').convert('RGB'))
                gt = np.array(Image.open(ROOT / 'data' / base[phase] / site / f'{frame}.png')) > 0
                pred = np.array(Image.open(ROOT / 'data/ISOD_sam_all_v2' / site / f'{frame}.png')) > 0
                ref = rgb.astype(float) * .55; ref[gt] = [20, 205, 205]
                pt = rgb.astype(float) * .55; pt[pred & gt] = [40, 185, 95]; pt[pred & ~gt] = [235, 70, 60]; pt[~pred & gt] = [45, 130, 230]
                for col, im in enumerate([rgb, ref.astype(np.uint8), pt.astype(np.uint8)]):
                    axes[row, col].imshow(im)
                axes[row, 0].set_ylabel(f'{phase}\n{site} / {frame}', fontsize=9)
            for ax, t in zip(axes[0], ['RGB', 'Reviewed reference', 'SAM 3 label']):
                ax.set_title(t, fontsize=10)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])
            fig.text(.5, .02, 'Green: correct; red: extra; blue: missed.', ha='center', fontsize=9)
            fig.subplots_adjust(left=.085, right=.99, top=.94, bottom=.07, wspace=.04, hspace=.10)
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    from make_detection_report_figures import pipeline
    main_table(); full_table(); rgb_baseline_curves(); native_curves(); cloth_main()
    teacher_examples(); pipeline()
    print("final assets written")
