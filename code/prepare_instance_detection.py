"""Export box-supervised detection targets from the saved reference-instance IDs.

One ID is one target, even when it has disconnected visible pieces. Unresolved
regions/groups are retained as declared hazard units, not certified individuals.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def boxes_from_ids(ids):
    boxes=[]
    for ident,slices in enumerate(ndimage.find_objects(ids),1):
        if slices is None:continue
        yy,xx=slices
        boxes.append(dict(id=ident,xyxy=[xx.start,yy.start,xx.stop,yy.stop],pixels=int((ids[yy,xx]==ident).sum())))
    return boxes


def write_labels(path,boxes,shape):
    h,w=shape;lines=[]
    for b in boxes:
        x0,y0,x1,y1=b['xyxy'];assert 0<=x0<x1<=w and 0<=y0<y1<=h
        values=[(x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h]
        assert all(0<=v<=1 for v in values) and values[2]>0 and values[3]>0
        lines.append('0 '+' '.join(f'{v:.9f}' for v in values))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+('\n' if lines else ''))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);base=a.root/'results/reference_instance_pool_reviewed_20261121';manifest=json.loads((base/'manifest.json').read_text())
    records=[];counts=Counter();units=Counter()
    for r in manifest['frames']:
        split=r['dataset_split'];path=base/r['path'];assert digest(path)==r['sha256']
        ids=np.array(Image.open(path));assert ids.ndim==2
        binary=a.root/'data'/('ISOD_reference_train_reviewed_candidate_20261121' if split=='train' else 'ISOD_reference_val_v1')/r['site']/f"{r['frame']}.png"
        assert np.array_equal(ids>0,np.array(Image.open(binary))>0)
        boxes=boxes_from_ids(ids);attrs={x['id']:x for x in r['objects']}
        assert {b['id'] for b in boxes}==set(attrs)
        for b in boxes:b['unit']=attrs[b['id']]['unit'];units[b['unit']]+=1
        label=a.out/'reference'/f'labels/{split}'/f"{r['site']}_{r['frame']}.txt";write_labels(label,boxes,ids.shape)
        records.append(dict(split=split,site=r['site'],frame=r['frame'],image=f"ISOD/data/{r['site']}/rgb/{r['frame']}.png",
            label=str(label.relative_to(a.out)),label_sha256=digest(label),source_ids_sha256=r['sha256'],source_binary_sha256=digest(binary),
            boxes=boxes,shape=list(ids.shape)));counts[split]+=1
    assert counts==dict(train=1400,val=300)
    # Check the two explicit human corrections against the actual training IDs.
    constraints=json.loads((a.root/'results/gt_quality_review_20261121/instance_constraints/decisions.json').read_text())
    group_checks=[]
    for c in constraints['constraints']:
        rr=next(r for r in manifest['frames'] if r['site']==c['site'] and r['frame']==c['frame'])
        ids=np.array(Image.open(base/rr['path']));mask=np.array(Image.open(a.root/'results/gt_quality_review_20261121'/c['mask']))>0
        assigned=np.unique(ids[mask]);assert len(assigned)==1 and assigned[0]>0
        group_checks.append(dict(site=c['site'],frame=c['frame'],single_instance_id=int(assigned[0]),pixels=int(mask.sum())))
    gbase=a.root/'results/gte_instance_proposals_20261121';gm=json.loads((gbase/'manifest.json').read_text());g=[]
    for r in gm['frames']:
        ids=np.array(Image.open(gbase/r['path']));ignored=[]
        for obj in r['objects']:
            if obj['unit']=='uncertain_reference_fragment':ids[ids==obj['id']]=0;ignored.append(obj['id'])
        boxes=boxes_from_ids(ids);path=a.out/'gte/labels/test'/f"{r['recording']}_{r['frame']}.txt";write_labels(path,boxes,ids.shape)
        g.append(dict(split='gte',recording=r['recording'],frame=r['frame'],image=f"OWN/{r['recording']}/rgb/{r['frame']}.png",
            label=str(path.relative_to(a.out)),label_sha256=digest(path),boxes=boxes,shape=list(ids.shape),ignored_reference_ids=ignored))
    assert len(g)==320
    result=dict(status='Box-training export from assisted reference instances; groups remain explicit',task='detect',class_names={0:'floor_hazard'},
        source_manifest_sha256=digest(base/'manifest.json'),reference_training_version='reviewed_candidate_20261121: 85 human-reviewed binary masks + 1315 assisted',
        source_instance_quality='Assisted units; not all verified physical individuals',counts=dict(counts),unit_counts=dict(units),
        disconnected_parts_one_box=True,no_small_target_filter=True,confirmed_group_checks=group_checks,frames=records,gte=g,
        gte_domain='Full-frame diagnostic with annotated item/groups; recording props unannotated, not operational AP')
    (a.out/'reference_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'REFERENCE_EXPORT_VERIFIED').write_text('1400 train + 300 shared validation + 320 GTE; both human grouping constraints passed\n')
    print(json.dumps(dict(frames=dict(counts),units=dict(units),gte_frames=len(g),human_groups=group_checks)))


if __name__=='__main__':main()
