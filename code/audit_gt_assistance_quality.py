"""Blind replay of the frozen GT-assistance rule against saved human masks.

Human answers are opened only after proposal generation. This is a retrospective
development audit, not independent test performance: test cache coverage is
explicitly absent. The new random human review is tracked separately.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def ratios(c):
    return dict(**c,iou=c['inter']/c['union'] if c['union'] else None,
        precision=c['inter']/c['pred'] if c['pred'] else None,
        recall=c['inter']/c['gt'] if c['gt'] else None)


def counts(p,g):return dict(inter=int((p&g).sum()),union=int((p|g).sum()),pred=int(p.sum()),gt=int(g.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    os.environ['ITR_DATA']=str(a.root/'data');os.environ['ITR_REVIEW_DRAFTS']=str(a.root/'data/EMPTY_REPLAY_DRAFTS')
    import review_isod_draft as draft
    import isod
    import isod_components as comps
    assert not Path(draft.review.OUT).exists() and not draft.REFERENCE.exists()
    cv2.setNumThreads(1)
    def no_saved_edits(*args,**kwargs):raise AssertionError('Proposal attempted to read a saved human edit')
    draft.review.read_edit=no_saved_edits
    frames=json.loads((a.root/'audit_frames.json').read_text());results=[]
    original_base=draft.review.base_frame
    for row in frames:
        site,i=row['site'],row['frame']
        comp,regions,_=draft.frame_proposal(site,i)
        assisted=np.isin(comp,[r['id'] for r in regions if r['keep']])
        # Cached component JSONs contain old classifier scores. Check that the
        # final SAM pipeline does not depend on those decision fields.
        def sanitized(s,j):
            cc,rs=original_base(s,j)
            for r in rs:
                r.pop('prob',None);r['keep']=comps.propose(r)[0];r['fresh']=False;r['why']='geometric audit control'
            return cc,rs
        draft.review.base_frame=sanitized
        try:
            control,parts,_=draft.frame_proposal(site,i)
            assert np.array_equal(assisted,np.isin(control,[r['id'] for r in parts if r['keep']])),(site,i,'legacy classifier decision dependence')
        finally:draft.review.base_frame=original_base
        label=cv2.imread(str(a.root/'data/ISOD/data'/site/'label'/f'{i}.png'),0)
        simple=isod.obstacles(label)>0
        base,rs=original_base(site,i);geometric=np.isin(base,[r['id'] for r in rs if comps.propose(r)[0]])
        # Open the manually corrected answer only after all automatic outputs.
        answer=a.root/row['answer_path'];assert hashlib.sha256(answer.read_bytes()).hexdigest()==row['answer_sha256']
        gt=cv2.imread(str(answer),0)>0
        destination=a.out/'predictions'/site/f'{i}.png';destination.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(destination),assisted.astype(np.uint8)*255)
        methods={name:ratios(counts(pred,gt)) for name,pred in [('heuristic_v0',simple),('geometry_only',geometric),('sam_assisted',assisted)]}
        n,cc,stats,_=cv2.connectedComponentsWithStats(gt.astype(np.uint8),connectivity=8)
        region_audit=[dict(area_pixels=int(stats[k,cv2.CC_STAT_AREA]),coverage=float(assisted[cc==k].mean()))
                      for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=20]
        results.append(dict(**row,methods=methods,legacy_classifier_decisions_independent=True,
                            reference_connected_regions=region_audit,
                            connected_regions_are_not_verified_physical_objects=True,
                            prediction_path=str(destination.relative_to(a.out))))
        if len(results)%25==0:
            print(json.dumps(dict(completed=len(results),expected=len(frames))),flush=True)
    summaries={}
    for group in ('train','val','source-full-crayon','cache','all_available'):
        selected=[r for r in results if group=='all_available' or r['dataset_split']==group or r['cache_family']==group]
        if not selected:continue
        methods={}
        for method in ['heuristic_v0','geometry_only','sam_assisted']:
            pooled={k:sum(r['methods'][method][k] for r in selected) for k in ['inter','union','pred','gt']}
            vals=[r['methods'][method]['iou'] for r in selected if r['methods'][method]['iou'] is not None]
            methods[method]=dict(pooled=ratios(pooled),mean_image_iou=float(np.mean(vals)),
                                images_below_iou_50=sum(v<.5 for v in vals),images_below_iou_80=sum(v<.8 for v in vals))
        summaries[group]=dict(frames=len(selected),methods=methods)
    reproduction=[]
    extra=a.root/'reproduction_frames.json'
    for row in json.loads(extra.read_text()) if extra.exists() else []:
        comp,regions,_=draft.frame_proposal(row['site'],row['frame'])
        predicted=np.isin(comp,[r['id'] for r in regions if r['keep']])
        expected=a.root/row['expected_path'];assert hashlib.sha256(expected.read_bytes()).hexdigest()==row['expected_sha256']
        frozen=cv2.imread(str(expected),0)>0
        reproduction.append(dict(site=row['site'],frame=row['frame'],changed_pixels=int((predicted!=frozen).sum()),
                                 exact=np.array_equal(predicted,frozen)))
    report=dict(status='completed retrospective quality audit; not a clean held-out result',frames=len(results),
        code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
                     ['review_isod_draft.py','review_isod.py','isod_sam_review.py','isod_components.py']},
        answer_masks_read_after_proposal_generation=True,saved_human_edits_not_read_by_pipeline=True,
        old_classifier_decision_fields_removed_without_changing_predictions=True,
        existing_test_frames_without_replay_cache=30,development_and_validation_previously_used_for_tuning=True,
        frozen_assisted_label_reproduction=reproduction,
        frozen_assisted_label_reproduction_all_exact=all(r['exact'] for r in reproduction) if reproduction else None,
        summaries=summaries,frames_detail=results)
    (a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'COMPLETE').write_text('Retrospective GT-assistance replay completed; fresh human spot-check remains separate\n')
    print(json.dumps(summaries))


if __name__=='__main__':main()
