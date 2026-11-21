"""Same capture configuration and frame index at all three nominal distances."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from floor_plane import floor_gap

ROOT=Path(__file__).resolve().parents[1]


def gray(path):
    a=np.array(Image.open(path))
    return a[...,0] if a.ndim==3 else a


def main():
    out=ROOT/'results/evaluation_v2_20261121/figures';out.mkdir(exist_ok=True)
    sources=json.loads((out.parent/'gte_full_diagnostic/summary.json').read_text())['source_runs']
    matched='rgbd_reference210_s0' in sources
    settings=json.loads((ROOT/'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m']
    fig,axes=plt.subplots(3,6,figsize=(17,6.5));records=[]
    depth_map=plt.get_cmap('viridis').copy();depth_map.set_bad('black')
    gap_map=plt.get_cmap('coolwarm').copy();gap_map.set_bad('black')
    for row,distance in enumerate(['075','150','250']):
        run=f'h52_floor_rags_{distance}';frame=2
        rgb=np.array(Image.open(ROOT/'data/OWN'/run/'rgb/2.png').convert('RGB'))
        raw=gray(ROOT/'data/OWN'/run/'depth/2.png');gt=gray(ROOT/'data/OWN_manual'/f'{run}_2.png')>0
        gap,valid=floor_gap(raw,settings,seed_from=0)
        axes[row,0].imshow(rgb);axes[row,0].contour(gt,levels=[.5],colors=['cyan'],linewidths=.6)
        depth_image=axes[row,1].imshow(np.ma.masked_where(raw==0,raw/1000),cmap=depth_map,vmin=.4,vmax=3)
        gap_image=axes[row,2].imshow(np.ma.masked_where(~valid,gap*1000),cmap=gap_map,vmin=-50,vmax=50)
        axes[row,0].set_ylabel(f'{int(distance)/100:g} m',fontsize=10)
        for col,mode in enumerate(['rgb','depth','rgbd'],3):
            key='rgbd_reference210_s0' if mode=='rgbd' and 'rgbd_reference210_s0' in sources else f'esanet_full_{mode}_reference_s0'
            source=sources[key]['predictions']
            pred=gray(ROOT/source/run/'2.png')>0
            painted=rgb.astype(float)*.5
            painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
            axes[row,col].imshow(painted.astype(np.uint8))
        records.append(dict(recording=run,frame=frame,nominal_floor_distance_m=int(distance)/100,
            target_pixels=int(gt.sum()),target_depth_valid_fraction=float((raw[gt]>0).mean()),
            prediction_seed=0,all_metrics_use_full_frames=True))
    for ax,title in zip(axes[0],['RGB + reference outline','Raw depth (m)','Floor-fit residual (mm)','ESANet RGB','ESANet D','ESANet RGB-D']):ax.set_title(title,fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.035,right=.965,top=.91,bottom=.13,wspace=.035,hspace=.05)
    for column,artist in [(1,depth_image),(2,gap_image)]:
        position=axes[2,column].get_position()
        cax=fig.add_axes([position.x0,.10,position.width,.012])
        c=fig.colorbar(artist,cax=cax,orientation='horizontal');c.ax.tick_params(labelsize=7,pad=1)
    fig.suptitle('Cloths/tissues at the three floor marks: same 52 cm camera configuration, frame 2',fontsize=12)
    cohort='Matched-runtime seed 0 examples.' if matched else 'Frozen-v1 seed 0 examples; matched-runtime repeats remain separate.'
    fig.text(.5,.035,'Same full-frame scale in every row; no zoom with distance. Black depth pixels are unavailable. Residual is camera-axis depth difference, not physical height.\nPrediction colors: green correct foreground, red extra foreground, blue missed foreground. '+cohort,ha='center',fontsize=8)
    fig.savefig(out/'depth_range_examples.png',dpi=200);fig.savefig(out/'depth_range_examples.svg');plt.close(fig)
    (out/'depth_range_examples.json').write_text(json.dumps(dict(selection='Fixed object set, capture configuration and frame index across all three distances; not selected for a model win',
        depth_scale_m=[.4,3],residual_scale_mm=[-50,50],records=records,physical_item_heights_not_measured=True),indent=2)+'\n')
    print('Saved matched RGB/depth/residual/prediction examples at three distances')


if __name__=='__main__':main()
