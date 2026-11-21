"""Render a small instance-conversion review from existing RGB-only SAM caches."""
from pathlib import Path
import colorsys
import html
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/instance_review_20261121'
FRAMES=[('ABEC_MM',4),('BT_BM',12),('BT_LM',34),('FL_LM',40),
        ('HB_G',64),('HB_MM',94),('HDCL_HW_B',40),('MEB_HW_C',34)]
PROMPTS=['small object on the floor','cable on the floor','things on the floor']


def main():
    OUT.mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    records=[]
    for site,frame in FRAMES:
        cache=ROOT/'results/sam_prompt_v2/dev_cache'/site/f'{frame}.npz'
        with np.load(cache,allow_pickle=False) as data:
            h,w=map(int,data['shape'])
            candidates=[]
            for i,(phrase,score) in enumerate(zip(data['prompts'],data['scores'])):
                if phrase not in PROMPTS or score<.6:continue
                mask=np.unpackbits(data['masks'][i])[:h*w].reshape(h,w).astype(bool)
                if mask.any():candidates.append(dict(mask=mask,prompt=str(phrase),score=float(score),cache_index=i))
        # Conservative proposed duplicate grouping. The human review checks this
        # conversion; it is not yet a frozen training-instance export.
        groups=[]
        for item in sorted(candidates,key=lambda r:-r['score']):
            for group in groups:
                overlap=int((item['mask']&group['mask']).sum())
                union=int((item['mask']|group['mask']).sum())
                if overlap/union>=.8:
                    group['mask']|=item['mask'];group['sources'].append(item);break
            else:groups.append(dict(mask=item['mask'].copy(),sources=[item]))
        original=np.zeros((h,w),bool);converted=original.copy()
        for item in candidates:original|=item['mask']
        for group in groups:converted|=group['mask']
        assert np.array_equal(original,converted)
        rgb=np.array(Image.open(ROOT/'data/ISOD/data'/site/'rgb'/f'{frame}.png').convert('RGB'))
        assert rgb.shape[:2]==(h,w)
        overlay=rgb.astype(float).copy()
        colours=[tuple(round(255*v) for v in colorsys.hsv_to_rgb((i*.618)%1,.75,1)) for i in range(len(groups))]
        for colour,group in zip(colours,groups):overlay[group['mask']]=.5*overlay[group['mask']]+.5*np.array(colour)
        image=Image.fromarray(overlay.astype('uint8'));draw=ImageDraw.Draw(image)
        thumbnails=Image.new('RGB',(w,((len(groups)+3)//4)*160 or 40),'white')
        td=ImageDraw.Draw(thumbnails);objects=[]
        for i,(group,colour) in enumerate(zip(groups,colours),1):
            ys,xs=np.nonzero(group['mask']);box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
            draw.rectangle(box,outline=colour,width=2)
            draw.text((box[0],max(0,box[1]-18)),str(i),font=font,fill='white',stroke_width=2,stroke_fill='black')
            isolated=rgb.copy();isolated[~group['mask']]=(isolated[~group['mask']]*.18).astype('uint8')
            pad=8;b=[max(0,box[0]-pad),max(0,box[1]-pad),min(w,box[2]+pad),min(h,box[3]+pad)]
            crop=Image.fromarray(isolated).crop(b);crop.thumbnail((150,125))
            x=((i-1)%4)*160;y=((i-1)//4)*160
            thumbnails.paste(crop,(x,y+24));td.text((x+3,y+2),f'Mask {i}',fill='black',font=font)
            objects.append(dict(id=i,box=box,pixels=int(group['mask'].sum()),
                                sources=[{k:v for k,v in s.items() if k!='mask'} for s in group['sources']]))
        stem=f'{site}_{frame}'
        panel=Image.new('RGB',(2*w,h+30),'white');panel.paste(Image.fromarray(rgb),(0,30));panel.paste(image,(w,30))
        d=ImageDraw.Draw(panel);d.text((4,4),'Original RGB',fill='black',font=font);d.text((w+4,4),'Proposed individual masks',fill='black',font=font)
        panel.save(OUT/f'{stem}.png');thumbnails.save(OUT/f'{stem}_objects.png')
        records.append(dict(site=site,frame=frame,raw_candidates=len(candidates),proposed_instances=len(groups),
                            union_preserved=True,objects=objects,panel=f'{stem}.png',tiles=f'{stem}_objects.png'))
    manifest=dict(status='awaiting_human_instance_review_not_training_labels',prompts=PROMPTS,
                  confidence=.6,proposed_duplicate_iou=.8,uses_reference_masks=False,
                  source='existing 30-frame training development cache; no additional SAM inference',frames=records)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Instance review: 8 examples</title>
<style>body{font:17px/1.5 system-ui;max-width:1280px;margin:auto;padding:24px;color:#161616;background:white}img{max-width:100%;height:auto}article{margin:50px 0;border-top:1px solid #bbb}p{max-width:85ch}details{margin:12px 0}summary{cursor:pointer}</style>
<h1>Check separate objects — 8 examples</h1><p>No drawing or saving is needed here. Check whether each numbered mask contains one floor object. Look for merged objects, duplicate masks, missed items and background. Masks may be imperfect at the edges.</p>
<p>Reply in chat with the example number and mask numbers that need changes. If everything is acceptable, say “all eight look good”. This is development feedback before instance-level training; your previous masks remain unchanged.</p>'''
    for i,r in enumerate(records,1):
        page+=f'<article><h2>{i}. {html.escape(r["site"])} / {r["frame"]} — {r["proposed_instances"]} masks</h2><a href="{r["panel"]}"><img src="{r["panel"]}" alt="RGB and numbered candidate masks"></a><p>Individual masks, with surrounding pixels darkened:</p><img src="{r["tiles"]}" alt="Each proposed mask shown separately"></article>'
    (OUT/'index.html').write_text(page)
    print(json.dumps({'review':str(OUT/'index.html'),'frames':len(records),'unions_preserved':all(r['union_preserved'] for r in records),'instances':sum(r['proposed_instances'] for r in records)}))


if __name__=='__main__':main()
