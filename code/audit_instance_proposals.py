"""Read-only integrity/triage audit; passing does not approve instance quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from refine_sam_instances import PROMPTS


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, base):
    protocol = json.loads((root/'results/sam_prompt_v2/expanded_protocol.json').read_text())
    expected = {(r['site'], r['frame']): (split, r) for split in ('development','validation') for r in protocol[split]}
    records = [json.loads(p.read_text()) for p in sorted((base/'records').glob('*/*.json'))]
    seen = set(); units = {b:Counter() for b in ('automatic','reference')}; flags = Counter(); changed=[]
    for r in records:
        key = (r['site'],r['frame']); assert key in expected and key not in seen; seen.add(key)
        split, definition = expected[key]
        assert r['split']==split and r['dataset_split']==('train' if split=='development' else 'val')
        assert not r['automatic']['uses_reference_masks'] and r['reference']['uses_reference_masks']
        assert r['human_instance_review_complete'] is False
        for branch in ('automatic','reference'):
            assert digest(base/r[branch]['path'])==r[branch]['sha256'], key
            units[branch].update(o['unit'] for o in r[branch]['objects'])
        source = root/definition['reference']; assert digest(source)==r['reference']['source_sha256']
        gt = np.array(Image.open(source).convert('L'))>0
        ids = np.array(Image.open(base/r['reference']['path']))
        assert np.array_equal(ids>0,gt), key
        assert set(np.unique(ids))-{0} == {o['id'] for o in r['reference']['objects']}, key
        with np.load(base/r['automatic']['path'],allow_pickle=False) as data:
            shape = tuple(map(int,data['shape'])); packed = data['masks']
            assert shape==gt.shape and len(packed)==len(r['automatic']['objects']), key
            auto = np.unpackbits(packed,axis=1)[:,:gt.size].reshape((-1,*shape)).astype(bool)
        union = auto.any(axis=0) if len(auto) else np.zeros(shape,bool)
        for mask, obj in zip(auto,r['automatic']['objects']):
            y,x=np.nonzero(mask);assert len(x)==obj['area_pixels'] and len(x)>0
            assert obj['bbox_xyxy']==[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            assert set(obj['prompts'])<=PROMPTS
        with np.load(root/r['source_cache'],allow_pickle=False) as raw:
            selected=np.isin(raw['prompts'],list(PROMPTS)) & (raw['scores']>=.6)
            teacher_packed=np.bitwise_or.reduce(raw['masks'][selected],axis=0) if selected.any() else np.zeros((gt.size+7)//8,np.uint8)
            teacher=np.unpackbits(teacher_packed)[:gt.size].reshape(shape).astype(bool)
        assert not (union&~teacher).any(),key
        removed=int((teacher&~union).sum())
        assert removed==r['automatic']['teacher_pixels_removed_by_parent_resolution']
        assert (removed==0)==r['automatic']['teacher_union_preserved']
        if removed:changed.append(dict(site=key[0],frame=key[1],dataset_split=r['dataset_split'],removed_pixels=removed))
        flags.update(r['needs_review'])
    complete=seen==set(expected)
    if complete:
        manifest=json.loads((base/'manifest.json').read_text())
        assert len(manifest['frames'])==len(records) and (base/'EXPORT_INTEGRITY_VERIFIED').exists()
        assert {(r['site'],r['frame']) for r in manifest['frames']}==seen
    return dict(status='integrity passed; quality unapproved',complete_export=complete,
        completed_frames=len(records),expected_frames=len(expected),
        dataset_splits=dict(Counter(r['dataset_split'] for r in records)),
        units={k:dict(v) for k,v in units.items()},flags=dict(flags),
        all_reference_unions_preserved=True,automatic_pixels_added_outside_teacher=0,
        automatic_teacher_union_changed_frames=changed,
        original_reference_or_training_labels_modified=False,
        unflagged_does_not_mean_human_approved=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--proposals',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.proposals);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='automatic_teacher_union_changed_frames'}))
