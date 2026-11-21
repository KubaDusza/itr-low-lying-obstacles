"""Development-only mask deduplication and broad-parent detection; no GT input.

Preserves the original review IDs. Does not change accepted semantic masks or
write training labels. New parent removals remain review proposals.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROMPTS={'small object on the floor','cable on the floor','things on the floor'}
SMALL='small object on the floor'
BROAD='things on the floor'
CABLE='cable on the floor'


def area(mask):return int(mask.sum())


def load_groups(path, prompts=None):
    prompts=PROMPTS if prompts is None else set(prompts)
    with np.load(path,allow_pickle=False) as data:
        h,w=map(int,data['shape']);candidates=[]
        for i,(phrase,score) in enumerate(zip(data['prompts'],data['scores'])):
            if phrase not in prompts or score<.6:continue
            mask=np.unpackbits(data['masks'][i])[:h*w].reshape(h,w).astype(bool)
            if mask.any():candidates.append(dict(mask=mask,prompt=str(phrase),score=float(score),cache_index=i))
    groups=[]
    for item in sorted(candidates,key=lambda r:-r['score']):
        for group in groups:
            if area(item['mask']&group['mask'])/area(item['mask']|group['mask'])>=.8:
                group['mask']|=item['mask'];group['sources'].append(item);break
        else:groups.append(dict(id=len(groups)+1,mask=item['mask'].copy(),sources=[item],review_ids=[len(groups)+1]))
    return groups,len(candidates)


def refine(groups):
    groups=[dict(g,mask=g['mask'].copy(),sources=list(g['sources']),review_ids=list(g['review_ids'])) for g in groups]
    duplicates=[]
    # Compare actual mask overlap, not boxes: nearby blocks and crossed cables
    # can have overlapping boxes without being duplicate object masks.
    for i,g in enumerate(groups):
        if g is None:continue
        for j in range(i+1,len(groups)):
            other=groups[j]
            if other is None:continue
            ratio=min(area(g['mask']),area(other['mask']))/max(area(g['mask']),area(other['mask']))
            iou=area(g['mask']&other['mask'])/area(g['mask']|other['mask'])
            if iou>=.70 and ratio>=.80:
                duplicates.append(dict(kept_id=g['id'],merged_id=other['id'],mask_iou=iou))
                g['mask']|=other['mask'];g['sources']+=other['sources'];g['review_ids']+=other['review_ids']
                groups[j]=None
    groups=[g for g in groups if g is not None]
    parents=[];uncertain=[]
    for parent in groups:
        phrases={s['prompt'] for s in parent['sources']}
        if BROAD not in phrases:continue
        p=parent['mask'];pa=area(p);children=[]
        for child in groups:
            if child is parent or SMALL not in {s['prompt'] for s in child['sources']}:continue
            ca=area(child['mask'])
            if ca<=.5*pa and area(p&child['mask'])/ca>=.9:children.append(child)
        # Independent children, not three near-duplicates or nested fragments.
        independent=[]
        for child in sorted(children,key=lambda g:-max(s['score'] for s in g['sources'])):
            if all(area(child['mask']&other['mask'])/min(area(child['mask']),area(other['mask']))<=.15 for other in independent):
                independent.append(child)
        if len(independent)<3:continue
        child_union=np.logical_or.reduce([g['mask'] for g in independent])
        coverage=area(p&child_union)/pa
        record=dict(parent_id=parent['id'],review_ids=parent['review_ids'],
                    child_ids=[g['id'] for g in independent],parent_covered_fraction=coverage,
                    residual_pixels=area(p&~child_union))
        if CABLE in phrases:
            uncertain.append(dict(record,reason='whole cable protected from fragment counting'))
        elif coverage<.85:
            uncertain.append(dict(record,reason='children leave too much of the parent unresolved'))
        else:parents.append(record)
    removed={p['parent_id'] for p in parents}
    kept=[g for g in groups if g['id'] not in removed]
    return kept,dict(duplicates=duplicates,parent_proposals=parents,ambiguous=uncertain)


def safety_checks():
    def mask(x0,y0,x1,y1):
        a=np.zeros((40,60),bool);a[y0:y1,x0:x1]=True;return a
    def g(i,m,p):return dict(id=i,mask=m,sources=[dict(prompt=p,score=.9)],review_ids=[i])
    # Disjoint neighbouring objects must survive even if their boxes could overlap.
    a=mask(2,2,12,12);b=mask(15,2,25,12)
    assert len(refine([g(1,a,SMALL),g(2,b,SMALL)])[0])==2
    diagonal=np.eye(40,60,dtype=bool)
    crossing=np.fliplr(diagonal)
    assert len(refine([g(1,diagonal,SMALL),g(2,crossing,SMALL)])[0])==2
    assert len(refine([g(1,a,SMALL),g(2,a,BROAD)])[0])==1
    pieces=[mask(2,2,12,12),mask(12,2,22,12),mask(22,2,32,12)]
    parent=np.logical_or.reduce(pieces)
    children=[g(i+2,m,SMALL) for i,m in enumerate(pieces)]
    assert refine([g(1,parent,BROAD)]+children)[1]['parent_proposals'][0]['parent_id']==1
    cable=g(1,parent,BROAD);cable['sources'].append(dict(prompt=CABLE,score=.9))
    assert not refine([cable]+children)[1]['parent_proposals']
    assert not refine([g(1,mask(0,0,60,40),BROAD)]+children)[1]['parent_proposals']


def main():
    safety_checks();rows=[]
    for path in sorted((ROOT/'results/sam_prompt_v2/dev_cache').glob('*/*.npz')):
        groups,raw=load_groups(path);kept,changes=refine(groups)
        rows.append(dict(site=path.parent.name,frame=int(path.stem),raw_candidates=raw,
                         previous_instances=len(groups),proposed_instances=len(kept),**changes))
    # Human-confirmed cases are regression assertions, not special-case rules.
    for site,frame,ident in [('BT_LM',34,20),('BT_BM',12,15)]:
        row=next(r for r in rows if (r['site'],r['frame'])==(site,frame))
        assert any(ident in r['review_ids'] for r in row['parent_proposals']),(site,frame)
    result=dict(status='development proposal, not deployed to training; current review IDs unchanged',
                uses_ground_truth=False,frames=len(rows),
                parameters=dict(duplicate_iou=.70,min_duplicate_area_ratio=.80,min_children=3,
                                child_containment=.90,max_child_parent_area_ratio=.5,
                                max_child_mutual_overlap=.15,min_parent_coverage=.85),
                safety_checks_passed=6,confirmed_parent_regressions_passed=2,
                ready_for_training=False,
                ambiguous_relationships='Raw parent/child proposals remain available for review; they must not both be exported as final training instances',
                previous_instances=sum(r['previous_instances'] for r in rows),
                proposed_instances=sum(r['proposed_instances'] for r in rows),
                duplicate_merges=sum(len(r['duplicates']) for r in rows),
                parent_proposals=sum(len(r['parent_proposals']) for r in rows),
                ambiguous=sum(len(r['ambiguous']) for r in rows),results=rows)
    out=ROOT/'results/instance_review_20261121/refinement_development.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))


if __name__=='__main__':main()
