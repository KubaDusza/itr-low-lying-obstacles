"""Build a compact review of the completed 30/300-frame SAM labeling study."""
import html
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):return json.loads(path.read_text())


def main():
    base=Path('results/sam_prompt_v2');out=base/'study30'
    protocol=load(base/'expanded_protocol.json');frozen=load(out/'frozen_recipes.json');validation=load(out/'validation_scores.json')
    selected=frozen['preferred_development'];chosen=frozen['recipes'][selected]
    names={'baseline':'Original 16 phrases','broad_pair':'Two broad phrases','positive':'Selected positive union',
           'subtract':'Positive union minus exclusions','veto':'Whole-instance exclusion'}
    cv2.setNumThreads(1)
    title_font=ImageFont.load_default(size=15)
    cards=[]
    for split,rows in [('development',protocol['development']),('validation',protocol['validation'])]:
        sr=validation['results'] if split=='validation' else {k:v['development'] for k,v in frozen['recipes'].items()}
        lookup={name:{(r['site'],r['frame']):r for r in score['per_frame']} for name,score in sr.items()}
        for row in rows:
            s,f=row['site'],row['frame'];rgb=cv2.imread(f'data/ISOD/data/{s}/rgb/{f}.png');gt=cv2.imread(row['reference'],0)>0
            if rgb is None:raise FileNotFoundError(f'{s}/{f}')
            reference=rgb.copy();cnt,_=cv2.findContours(gt.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(reference,cnt,-1,(230,215,4),1)
            pictures=[('RGB + reference',reference)]
            for name in ('baseline',selected):
                mask=cv2.imread(str(out/f'{split}_masks'/name/s/f'{f}.png'),0)
                if mask is None:raise FileNotFoundError(f'{split} {name} {s}/{f}')
                pred=mask>0;image=(rgb*.4).astype(np.uint8)
                image[gt&pred]=(125,183,34);image[gt&~pred]=(39,174,239);image[~gt&pred]=(166,63,210)
                metric=lookup[name][s,f]['iou'];pictures.append((f'{names[name]} | IoU {metric:.3f}',image))
            panel=Image.new('RGB',(1200,290),'white');draw=ImageDraw.Draw(panel)
            for i,(title,image) in enumerate(pictures):
                thumb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));thumb.thumbnail((392,259));panel.paste(thumb,(i*400,26));draw.text((i*400+3,4),title,fill='black',font=title_font)
            dest=out/'examples'/split/s/f'{f}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);panel.save(dest,quality=90)
            old=lookup['baseline'][s,f]['iou'];new=lookup[selected][s,f]['iou']
            cards.append(dict(split=split,site=s,frame=f,iou=new,baseline=old,change=new-old,path=str(dest.relative_to(out))))
    table=[]
    for name,record in frozen['recipes'].items():
        dev=record['development'];val=validation['results'][name]
        table.append(f'<tr><th>{html.escape(names[name])}{" ★" if name==selected else ""}</th><td>{dev["mean_frame_iou"]:.3f}</td><td>{val["mean_frame_iou"]:.3f}</td><td>{val["pooled_iou"]:.3f}</td><td>{val["precision"]:.3f}</td><td>{val["recall"]:.3f}</td></tr>')
    r=chosen['recipe'];phrases=''.join('<li>'+html.escape(p)+'</li>' for p in r['positive']);negative='; '.join(r['negative']) or 'None'
    development_scores=load(out/'development_scores.json')
    ablations=[]
    for record in development_scores['ranking']:
        identifier=record['name']
        if identifier=='compact_union' or identifier.startswith('compact_union|') and identifier.endswith('|subtract|0.5') and identifier.split('|')[1] in ('furniture','structure','containers','relations','unsafe_size','all_safe'):
            label='No exclusions' if identifier=='compact_union' else identifier.split('|')[1].replace('_',' ')
            ablations.append(f'<tr><th>{html.escape(label)}</th><td>{record["mean_frame_iou"]:.3f}</td><td>{record["tp"]:,}</td><td>{record["fp"]:,}</td></tr>')
    per_site=''.join(f'<tr><th>{s}</th><td>{v["frames"]}</td><td>{v["mean_frame_iou"]:.3f}</td><td>{v["pooled_iou"]:.3f}</td></tr>' for s,v in validation['results'][selected]['per_site'].items())
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM labeling study: 30 development / 300 validation</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1250px;margin:auto;padding:24px;color:#171717;background:white}h1{font-size:30px}p{max-width:95ch}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{text-align:left;border-bottom:1px solid #ccc;padding:9px}td{font-variant-numeric:tabular-nums}select{font:inherit;padding:8px;margin:0 12px 12px 0}article{margin:24px 0 38px}article img{width:100%;height:auto}article h3{margin-bottom:5px}small{color:#555}a{color:#174e88}.controls{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #bbb}
</style>'''
    page+=f'<h1>RGB-only SAM labeling: 30 development / 300 validation</h1><p>Vocabulary was built from RGB context and masked snippets of 30 human-reviewed training frames, sampled with seed 20261120 and spread across all 14 training locations. Their reference masks simulate manual painting. SAM receives the full RGB image and text. Model weights stay fixed; no ISOD source masks, reference masks or depth enter inference.</p><p>{development_scores["count"]} development recipes were compared, including broad and specific names, unions, negative-mask subtraction, and whole-instance veto. Confidence remains 0.60 and mask probability 0.50.</p>'
    page+='<p>★ marks the primary recipe selected before validation. The simpler recipe is preferred when development mean IoU is within 0.005 of the highest shortlisted score. Validation contains 300 reviewed frames from three separate locations. These references were used by the previous pipeline and detector checkpoint selection; this is validation, not a new untouched final test.</p>'
    page+='<table><tr><th>Recipe</th><th>Dev mean IoU</th><th>Validation mean IoU</th><th>Validation pooled IoU</th><th>Precision</th><th>Recall</th></tr>'+''.join(table)+'</table><p>Mean image IoU gives each image equal weight. An empty reference with an empty prediction scores 1; any false detection in an empty-reference image scores 0. Pooled IoU combines pixels. Per-location scores make the correlation between nearby frames visible.</p>'
    page+=f'<h2>Primary recipe</h2><ul>{phrases}</ul><p>Exclusions: {html.escape(negative)}. Combination: {html.escape(r["mode"])}. Instance-overlap threshold where applicable: {r["overlap"]}.</p><p><a href="frozen_recipes.json">All exact recipes and development selection</a> · <a href="validation_scores.json">All validation metrics</a> · <a href="development_scores.json">Full development comparisons</a></p>'
    page+='<h2>What exclusion masks did on development</h2><p>These rows keep the compact positive union fixed and subtract each negative group. Relation phrases cover objects described as on a desk or wall, and hanging cables/cloth. In this sample they also detect many valid floor objects; subtracting them removes correct labels. Size exclusions also remove valid targets.</p><table><tr><th>Negative group</th><th>Mean image IoU</th><th>Correct obstacle pixels retained</th><th>Extra pixels retained</th></tr>'+''.join(ablations)+'</table><p><a href="../study30_config.json">Complete candidate vocabulary and negative groups</a> · <a href="inventory_evidence.json">Visual inventory and pilot overlap</a></p>'
    page+='<table><tr><th>Validation location</th><th>Frames</th><th>Mean IoU</th><th>Pooled IoU</th></tr>'+per_site+'</table>'
    page+='<h2>All 330 examples</h2><p>Cyan outlines mark the reference on RGB. Green = correct obstacle; amber = missed; magenta = extra. Click an image for full resolution.</p><div class="controls"><select id="split"><option value="validation">Validation: 300 frames</option><option value="development">Development: 30 frames</option></select><select id="site"><option value="all">All locations</option></select><select id="sort"><option value="worst">Lowest IoU first</option><option value="gain">Largest improvement first</option><option value="loss">Largest regression first</option><option value="id">Location / frame order</option></select><span id="count"></span></div><div id="examples"></div>'
    page+='<script>const rows='+json.dumps(cards)+';const split=document.querySelector("#split"),site=document.querySelector("#site"),sort=document.querySelector("#sort");function refreshSites(){site.replaceChildren(new Option("All locations","all"),...Array.from(new Set(rows.filter(r=>r.split===split.value).map(r=>r.site))).sort().map(s=>new Option(s,s)));render()}function render(){let filtered=rows.filter(r=>r.split===split.value&&(site.value==="all"||r.site===site.value));filtered.sort((a,b)=>sort.value==="worst"?a.iou-b.iou:sort.value==="gain"?b.change-a.change:sort.value==="loss"?a.change-b.change:a.site.localeCompare(b.site)||a.frame-b.frame);document.querySelector("#count").textContent=filtered.length+" images";document.querySelector("#examples").replaceChildren(...filtered.map(r=>{let article=document.createElement("article"),title=document.createElement("h3"),a=document.createElement("a"),img=document.createElement("img");title.textContent=r.site+" / "+r.frame+" — original "+r.baseline.toFixed(3)+", selected "+r.iou.toFixed(3);a.href=r.path;img.src=r.path;img.loading="lazy";img.alt=r.site+" / "+r.frame+" RGB, reference, original and selected SAM masks";a.append(img);article.append(title,a);return article}))}split.onchange=refreshSites;site.onchange=render;sort.onchange=render;refreshSites();</script>'
    (out/'REVIEW.html').write_text(page)
    selected_record=dict(model=load(base/'study30_config.json')['model'],revision=load(base/'study30_config.json')['revision'],
                         **r,confidence=.6,mask_probability=.5,manual_development_frames=30,
                         selection='Frozen on development before the 300-frame validation check',
                         frozen_recipes_sha256=hashlib.sha256((out/'frozen_recipes.json').read_bytes()).hexdigest())
    pipeline_path=out/'selected_pipeline.json'
    if pipeline_path.exists():
        existing=load(pipeline_path)
        for field in ('positive','negative','mode','overlap','confidence','mask_probability','model','revision'):
            assert existing[field]==selected_record[field], f'Existing frozen pipeline differs: {field}'
    else:
        pipeline_path.write_text(json.dumps(selected_record,indent=2)+'\n')
    print(out/'REVIEW.html')


if __name__=='__main__':
    main()
