"""Export independent automatic and SAM-assisted reference instance proposals.

Proposal datasets only: unresolved clusters are explicit and never silently
counted as verified physical objects. Automatic geometry never receives GT.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from refine_sam_instances import load_groups, refine, safety_checks, PROMPTS, SMALL, BROAD, CABLE

REFERENCE_NAMES={'cable','cable bundle','cloth','crayon','doormat','electrical wire',
 'extension cord','paper towel','piece of paper','power cord','power strip','rag',
 'shoe','slipper','small hand tool','tennis racket','toy block','LEGO brick','toy brick',
 'plastic block','building block','chalk','tissue','towel','sock','flip flop','cord',
 'paper','phone case','sunglasses','glasses','small bottle','banana peel','crumpled paper',
 'toy','fabric','small box'}
REFERENCE_PROMPTS=set(PROMPTS)|REFERENCE_NAMES|{p+' on the floor' for p in REFERENCE_NAMES}


def automatic_instances(groups):
    kept,events=refine(groups)
    # Remaining strong containment is ambiguous granularity, not two approved
    # objects. Retain its union as one explicit, reviewable hazard cluster.
    parents=list(range(len(kept)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(kept):
        for j in range(i+1,len(kept)):
            b=kept[j];overlap=int((a['mask']&b['mask']).sum())
            if overlap/min(int(a['mask'].sum()),int(b['mask'].sum()))>=.9:
                parents[root(j)]=root(i)
    components={}
    for i in range(len(kept)):components.setdefault(root(i),[]).append(kept[i])
    outputs=[]
    for members in components.values():
        mask=np.logical_or.reduce([g['mask'] for g in members])
        outputs.append(dict(mask=mask,review_ids=sorted({i for g in members for i in g['review_ids']}),
                            sources=[s for g in members for s in g['sources']],
                            unit='unresolved_cluster' if len(members)>1 else 'proposed_individual'))
    return outputs,events


def partition_reference(gt, proposals):
    labels=np.zeros(gt.shape,np.int32);units=[]
    # Smaller SAM regions establish separate seeds before larger masks.
    for proposal in sorted(proposals,key=lambda g:(int(g['mask'].sum()),-max(s['score'] for s in g['sources']))):
        overlap=proposal['mask']&gt
        fraction=float(overlap.sum()/proposal['mask'].sum())
        if fraction<.5:continue
        seed=overlap&(labels==0)
        if not seed.any():continue
        ident=len(units)+1;labels[seed]=ident
        units.append(dict(id=ident,unit=proposal['unit'],sam_reference_overlap=fraction,
                          sam_seed_ids=proposal['review_ids'],seed_pixels=int(seed.sum())))
    # Seeded components grow only through accepted foreground. Unseeded ones
    # become explicit unresolved regions, not an invented physical object.
    components,n=ndimage.label(gt,structure=np.ones((3,3),np.uint8))
    for component in range(1,n+1):
        selected=components==component
        if not (labels[selected]>0).any():
            ident=len(units)+1;labels[selected]=ident
            units.append(dict(id=ident,unit='unseeded_reference_region',seed_pixels=0))
    frontier=deque((int(y),int(x)) for y,x in zip(*np.nonzero(labels)))
    h,w=gt.shape
    while frontier:
        y,x=frontier.popleft();ident=labels[y,x]
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            yy,xx=y+dy,x+dx
            if 0<=yy<h and 0<=xx<w and gt[yy,xx] and labels[yy,xx]==0:
                labels[yy,xx]=ident;frontier.append((yy,xx))
    assert np.array_equal(labels>0,gt)
    assert len(units)<65535
    return labels.astype(np.uint16),units


def checks():
    safety_checks()
    gt=np.zeros((12,20),bool);gt[2:9,2:10]=True;gt[3:8,14:18]=True
    a=np.zeros_like(gt);a[3:5,3:5]=True
    p=dict(mask=a,unit='proposed_individual',review_ids=[1],sources=[dict(score=.9)])
    labels,units=partition_reference(gt,[p])
    assert np.array_equal(labels>0,gt) and len(units)==2
    assert units[-1]['unit']=='unseeded_reference_region'
    b=np.zeros_like(gt);b[6:8,7:9]=True
    labels,units=partition_reference(gt,[p,dict(p,mask=b)])
    assert len(units)==3 and np.array_equal(labels>0,gt)
    empty,units=partition_reference(np.zeros_like(gt),[])
    assert not empty.any() and not units
    def group(i,m,prompt):
        return dict(id=i,mask=m,sources=[dict(prompt=prompt,score=.9)],review_ids=[i])
    parent=np.zeros_like(gt);parent[2:9,2:10]=True
    pieces=[]
    for y,x in [(3,3),(3,7),(6,3)]:
        m=np.zeros_like(gt);m[y:y+1,x:x+1]=True;pieces.append(m)
    groups=[group(1,parent,BROAD)]+[group(i+2,m,SMALL) for i,m in enumerate(pieces)]
    resolved,_=automatic_instances(groups)
    assert len(resolved)==1 and resolved[0]['unit']=='unresolved_cluster'
    groups[0]['sources'].append(dict(prompt=CABLE,score=.9))
    resolved,_=automatic_instances(groups)
    assert len(resolved)==1 and np.array_equal(resolved[0]['mask'],parent)
    diagonal=np.eye(12,20,dtype=bool)
    assert len(automatic_instances([group(1,diagonal,SMALL),group(2,np.fliplr(diagonal),SMALL)])[0])==2


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();checks()
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.root/'results/sam_prompt_v2/expanded_protocol.json').read_text())
    signature=hashlib.sha256(Path(__file__).read_bytes()+
        Path(__file__).with_name('refine_sam_instances.py').read_bytes()+
        json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    signature_path=args.out/'export_signature.txt'
    if signature_path.exists():
        assert signature_path.read_text().strip()==signature,'Export code or protocol changed; use a new output directory'
    signature_path.write_text(signature+'\n')
    record_dir=args.out/'records';record_dir.mkdir(exist_ok=True)
    records=[]
    for split,cache_name in [('development','dev_cache'),('validation','validation_cache')]:
        for row in protocol[split]:
            site,frame=row['site'],row['frame'];key=f'{site}/{frame}'
            record_path=record_dir/site/f'{frame}.json'
            if record_path.exists():
                saved=json.loads(record_path.read_text())
                assert saved['reference']['source_sha256']==hashlib.sha256((args.root/row['reference']).read_bytes()).hexdigest()
                for branch in ('automatic','reference'):
                    assert saved[branch]['sha256']==hashlib.sha256((args.out/saved[branch]['path']).read_bytes()).hexdigest()
                records.append(saved)
                continue
            cache=args.root/'results/sam_prompt_v2'/cache_name/site/f'{frame}.npz'
            groups,raw=load_groups(cache)
            auto,events=automatic_instances(groups)
            packed=[];auto_units=[]
            for ident,g in enumerate(auto,1):
                ys,xs=np.nonzero(g['mask']);packed.append(np.packbits(g['mask'].reshape(-1)))
                auto_units.append(dict(id=ident,unit=g['unit'],review_ids=g['review_ids'],
                    bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                    area_pixels=int(g['mask'].sum()),confidence=max(s['score'] for s in g['sources']),
                    prompts=sorted({s['prompt'] for s in g['sources']})))
            # The automatic output is finished before reading reference pixels.
            aout=args.out/'automatic'/site/f'{frame}.npz';aout.parent.mkdir(parents=True,exist_ok=True)
            with np.load(cache,allow_pickle=False) as c:shape=c['shape'].copy()
            teacher_union=np.zeros(tuple(shape),bool)
            instance_union=np.zeros(tuple(shape),bool)
            for g in groups:teacher_union|=g['mask']
            for g in auto:instance_union|=g['mask']
            assert not (instance_union&~teacher_union).any()
            packed=np.stack(packed) if packed else np.zeros((0,(int(np.prod(shape))+7)//8),np.uint8)
            np.savez_compressed(aout,shape=shape,masks=packed)
            reference=args.root/row['reference'];gt=np.array(Image.open(reference).convert('L'))>0
            assert tuple(shape)==gt.shape
            richer,_=load_groups(cache,REFERENCE_PROMPTS)
            ref_seeds,ref_events=automatic_instances(richer)
            labels,ref_units=partition_reference(gt,ref_seeds)
            rout=args.out/'reference'/site/f'{frame}.png';rout.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(labels).save(rout)
            needs=[]
            if any(g['unit']!='proposed_individual' for g in auto):needs.append('automatic_unresolved_cluster')
            if any(g['unit']=='unseeded_reference_region' for g in ref_units):needs.append('reference_missing_SAM_seed')
            if any(g['unit']=='unresolved_cluster' for g in ref_units):needs.append('reference_unresolved_cluster')
            if events['ambiguous']:needs.append('ambiguous_parent_relationship')
            records.append(dict(site=site,frame=frame,split=split,
                dataset_split='train' if split=='development' else 'val',
                source_cache=str(cache.relative_to(args.root)),
                automatic=dict(uses_reference_masks=False,path=str(aout.relative_to(args.out)),objects=auto_units,
                               teacher_union_preserved=bool(np.array_equal(teacher_union,instance_union)),
                               teacher_pixels_removed_by_parent_resolution=int((teacher_union&~instance_union).sum()),
                               parent_events=events,sha256=hashlib.sha256(aout.read_bytes()).hexdigest()),
                reference=dict(uses_reference_masks=True,path=str(rout.relative_to(args.out)),objects=ref_units,
                               binary_union_preserved=True,source_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                               sha256=hashlib.sha256(rout.read_bytes()).hexdigest()),needs_review=needs,
                human_instance_review_complete=False))
            record_path.parent.mkdir(parents=True,exist_ok=True)
            pending=record_path.with_suffix('.tmp')
            pending.write_text(json.dumps(records[-1],indent=2)+'\n');pending.replace(record_path)
            progress=dict(completed_frames=len(records),total_frames=len(protocol['development'])+len(protocol['validation']),
                          last_frame=key,status='exporting unapproved proposals')
            (args.out/'progress.json').write_text(json.dumps(progress)+'\n')
            if len(records)%25==0:print(json.dumps(progress),flush=True)
    manifest=dict(status='instance proposals, not approved training or evaluation references',
        frame_count=len(records),automatic_prompts=sorted(PROMPTS),reference_seed_vocabulary=sorted(REFERENCE_PROMPTS),
        safety_checks_passed=12,original_binary_labels_modified=False,
        export_signature=signature,
        flags_are_triage_not_complete_quality_validation=True,frames=records)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    queue=[dict(site=r['site'],frame=r['frame'],split=r['split'],reasons=r['needs_review']) for r in records if r['needs_review']]
    (args.out/'review_queue.json').write_text(json.dumps(queue,indent=2)+'\n')
    (args.out/'EXPORT_INTEGRITY_VERIFIED').write_text('Export integrity verified; instance quality remains unapproved.\n')
    (args.out/'progress.json').write_text(json.dumps(dict(completed_frames=len(records),total_frames=len(records),
        status='export complete; instance quality unapproved'))+'\n')
    print(json.dumps(dict(frames=len(records),flagged_frames=len(queue),
        automatic_instances=sum(len(r['automatic']['objects']) for r in records),
        reference_instances_or_unresolved_regions=sum(len(r['reference']['objects']) for r in records),
        all_reference_unions_preserved=True)))


if __name__=='__main__':main()
