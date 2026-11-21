"""Check a complete assisted reference instance pool without approving quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root,base):
    manifest=json.loads((base/'manifest.json').read_text());rows=manifest['frames']
    assert len(rows)==1700 and Counter(r['dataset_split'] for r in rows)==dict(train=1400,val=300)
    assert len({(r['site'],r['frame']) for r in rows})==1700
    units=Counter();source_status=Counter();uncertain=[]
    for r in rows:
        source=root/r['source_reference'];cache=root/r['source_cache'];target=base/r['path']
        assert sha(source)==r['source_sha256'] and sha(cache)==r['source_cache_sha256'] and sha(target)==r['sha256']
        gt=np.array(Image.open(source).convert('L'))>0;ids=np.array(Image.open(target))
        assert np.array_equal(ids>0,gt),(r['site'],r['frame'])
        assert set(np.unique(ids))-{0}=={o['id'] for o in r['objects']}
        for o in r['objects']:
            assert int((ids==o['id']).sum())==o['area_pixels']
            assert o['category_verified'] is False
            units[o['unit']]+=1
        source_status[r['source_review_status']]+=1
        pending=[o['id'] for o in r['objects'] if o['unit'] not in ('proposed_individual','human_confirmed_individual')]
        if pending:uncertain.append(dict(site=r['site'],frame=r['frame'],dataset_split=r['dataset_split'],uncertain_ids=pending))
    return dict(status='integrity verified; instance quality remains unapproved',frames=len(rows),
        dataset_splits=dict(Counter(r['dataset_split'] for r in rows)),reference_source_status=dict(source_status),
        units=dict(units),frames_with_uncertain_units=len(uncertain),review_queue=uncertain,
        all_reference_unions_preserved=True,frozen_binary_training_masks_modified=False,
        unattended_teacher_labels_modified=False,eligible_for_training_without_quality_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proposals',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.proposals)
    (a.proposals/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.proposals/'INTEGRITY_VERIFIED').write_text('Pixel/source integrity verified; instance-quality approval required\n')
    print(json.dumps({k:v for k,v in result.items() if k!='review_queue'}))
