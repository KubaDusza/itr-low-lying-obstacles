"""Supporting qualitative cases selected from the complete matched-unit table."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

from floor_plane import floor_gap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/results_report_20261122'
ANALYSIS=ROOT/'results/triton_20261121/production/analysis_final'


def gray(p):
    a=np.array(Image.open(p));return a[...,0] if a.ndim==3 else a


def main():
    plt.rcParams.update({'font.size':9, 'font.family':'serif', 'font.serif':['STIXGeneral'], 'pdf.fonttype':42})
    units=list(csv.DictReader((OUT/'matched_units.csv').open()))
    observations=list(csv.DictReader((ANALYSIS/'gte_full_diagnostic/objects.csv').open()))
    inventory={r['item_id']:r['label'] for r in csv.DictReader((ROOT/'data_collection/items.csv').open())}
    definitions=json.loads((ROOT/'results/gte_instance_proposals_20261121/manifest.json').read_text())['frames']
    lookup={(r['recording'],r['frame']):r for r in definitions}
    sources=json.loads((ANALYSIS/'gte_full_diagnostic/summary.json').read_text())['source_runs']
    settings=json.loads((ROOT/'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m']
    colors=plt.get_cmap('coolwarm').copy();colors.set_bad('black')
    rawcolors=plt.get_cmap('viridis').copy();rawcolors.set_bad('black')
    def scene(recording,ident):
        rgb=np.array(Image.open(ROOT/'data/OWN'/recording/'rgb/2.png').convert('RGB'))
        gt=gray(ROOT/'data/OWN_manual'/f'{recording}_2.png')>0
        row=lookup[recording,2];ids=gray(ROOT/'results/gte_instance_proposals_20261121'/row['path'])
        raw=gray(ROOT/'data/OWN'/recording/'depth/2.png');gap,valid=floor_gap(raw,settings,seed_from=0)
        return rgb,gt,ids==ident,raw,gap,valid
    def predicted(recording,mode,rgb,gt):
        name='rgbd_reference210_s0' if mode=='rgbd' else f'esanet_full_{mode}_reference_s0'
        pred=gray(ROOT/sources[name]['predictions']/recording/'2.png')>0
        painted=rgb.astype(float)*.55;painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
        return painted.astype(np.uint8)
    def clean(axes):
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
    selected=[]
    # Stable deterministic tie-breaking uses input CSV recording/ID order.
    gain=max(units,key=lambda r:float(r['rgbd_minus_rgb']))
    recovered=max([r for r in units if float(r['rgbd_localization_rate'])>=.8 and float(r['rgb_localization_rate'])<=.2],
                  key=lambda r:float(r['rgbd_coverage_rate'])-float(r['rgb_coverage_rate']))
    loss=min(units,key=lambda r:float(r['rgbd_minus_rgb']))
    with PdfPages(OUT/'qualitative_cases.pdf') as pdf:
        for page,(kind,r) in enumerate([('Coverage and localization gain',recovered),
                                      ('Boundary gain: both inputs cover the target',gain),
                                      ('Fusion failure',loss)],1):
            rgb,gt,target,raw,gap,valid=scene(r['recording'],int(r['id']))
            fig,axes=plt.subplots(2,3,figsize=(10,6.4))
            axes[0,0].imshow(rgb);axes[0,0].contour(gt,levels=[.5],colors='cyan',linewidths=.6);axes[0,0].contour(target,levels=[.5],colors='yellow',linewidths=1)
            im1=axes[0,1].imshow(np.ma.masked_where(raw==0,raw/1000),cmap=rawcolors,vmin=.4,vmax=3)
            im2=axes[0,2].imshow(np.ma.masked_where(~valid,gap*1000),cmap=colors,vmin=-50,vmax=50)
            for col,mode in enumerate(['rgb','depth','rgbd']):
                axes[1,col].imshow(predicted(r['recording'],mode,rgb,gt));axes[1,col].contour(target,levels=[.5],colors='yellow',linewidths=.8)
            for ax,title in zip(axes.flat,['RGB; selected target in yellow','Raw depth (m)','Floor residual (mm)','RGB prediction','D prediction','RGB-D prediction']):ax.set_title(title,fontsize=10)
            for ax,artist in [(axes[0,1],im1),(axes[0,2],im2)]:fig.colorbar(artist,ax=ax,orientation='horizontal',fraction=.05,pad=.02)
            rates=' / '.join(f'{m.upper()} {100*float(r[m+"_localization_rate"]):.1f}%' for m in ['rgb','depth','rgbd'])
            covered=' / '.join(f'{m.upper()} {100*float(r[m+"_coverage_rate"]):.1f}%' for m in ['rgb','depth','rgbd'])
            shown=[]
            for mode in ['rgb','depth','rgbd']:
                model='rgbd_reference210_s0' if mode=='rgbd' else f'esanet_full_{mode}_reference_s0'
                x=next(o for o in observations if o['model']==model and o['recording']==r['recording'] and o['id']==r['id'] and o['frame']=='2')
                shown.append(f'{mode.upper()} {float(x["adjusted_iou"]):.3f}')
            height,bg,objset,dist=r['recording'].split('_')
            fig.suptitle(f'{kind}: {inventory[r["parent_item_id"]]}\n{height[1:]} cm camera, {bg}, {int(dist)/100:g} m',fontsize=12)
            fig.text(.5,.025,f'Across 3 seeds × 5 repeats, coverage recall: {covered}.\nLocalization recall (adjusted IoU ≥0.5): {rates}. Shown view adjusted IoU: {" / ".join(shown)}.\nOutcome-selected from 262 reference-unit/recording pairs; not evidence of frequency. Images: seed 0, frame 2.\nGreen: correct foreground; blue: missed; red: extra. Full-frame diagnostic; recording props included.\nResidual is camera-axis difference, not height. Channel scales are fixed. Coverage alone does not establish localization.',ha='center',fontsize=8)
            clean(axes);fig.subplots_adjust(left=.02,right=.99,top=.85,bottom=.20,wspace=.04,hspace=.35)
            pdf.savefig(fig);fig.savefig(OUT/f'qualitative_case_{page}.png',dpi=150);plt.close(fig)
            selected.append(dict(selection=kind,**r,frame=2,display_seed=0,
                selection_rule='Maximum coverage gain among RGB-D localization >=0.8 and RGB <=0.2' if r is recovered else 'Maximum/minimum localization difference; coverage and shown-view overlap disclosed'))
        # Predetermined counterexample: matched same inventory item and capture geometry.
        chosen=[next(r for r in units if r['recording']==f'h52_{bg}_rags_075' and r['parent_item_id']=='rag_white_flat') for bg in ['floor','poster']]
        fig,axes=plt.subplots(2,5,figsize=(11,5.8))
        for row,r in enumerate(chosen):
            rgb,gt,target,raw,gap,valid=scene(r['recording'],int(r['id']))
            axes[row,0].imshow(rgb);axes[row,0].contour(target,levels=[.5],colors='yellow',linewidths=1)
            axes[row,1].imshow(np.ma.masked_where(~valid,gap*1000),cmap=colors,vmin=-50,vmax=50)
            for col,mode in enumerate(['rgb','depth','rgbd'],2):axes[row,col].imshow(predicted(r['recording'],mode,rgb,gt))
            axes[row,0].set_ylabel(r['recording'].split('_')[1],fontsize=10)
            rates=' / '.join(f'{m.upper()} {100*float(r[m+"_localization_rate"]):.0f}%' for m in ['rgb','depth','rgbd'])
            fig.text(.5,.91-row*.37,'Localization: '+rates,ha='center',fontsize=9)
        for ax,title in zip(axes[0],['RGB; target in yellow','Residual ±50 mm','RGB','D','RGB-D']):ax.set_title(title,fontsize=9)
        fig.suptitle('Matched floor/poster example: flat white paper towel, 0.75 m, 52 cm camera',fontsize=12)
        fig.text(.5,.025,'Top: bare floor. Bottom: white poster. Same item, nominal range and camera configuration; scene context also changes.\nRates use 3 seeds × 5 repeats per background. Images: seed 0, frame 2. Residual scale identical; black is unavailable.\nFusion does not rescue this low-contrast, flat target on the poster. Colors are prediction agreement with all reference targets, not only the highlighted item.',ha='center',fontsize=9)
        clean(axes);fig.subplots_adjust(left=.035,right=.99,top=.86,bottom=.16,wspace=.03,hspace=.25)
        pdf.savefig(fig);fig.savefig(OUT/'qualitative_case_4.png',dpi=150);plt.close(fig)
        selected.append(dict(selection='Fixed low-contrast counterexample, not an inferred causal contrast effect',frames=chosen,frame=2,display_seed=0))
    (OUT/'qualitative_cases_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('Created four supporting pages: recovered target, boundary gain, fusion loss, matched low-contrast failure')


if __name__=='__main__':main()
