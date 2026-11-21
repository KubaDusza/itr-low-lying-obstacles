"""Build reference-only instance proposals from already accepted binary masks.

Uses richer cached annotation-assistance prompts, never the unattended teacher
branch. Unresolved regions stay foreground and retain unknown instance counts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from export_instance_proposals import REFERENCE_PROMPTS, automatic_instances, partition_reference
from refine_sam_instances import load_groups


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),Path(__file__).with_name('export_instance_proposals.py'),Path(__file__).with_name('refine_sam_instances.py')]
    signature=hashlib.sha256(b''.join(x.read_bytes() for x in source_files)).hexdigest()
    sign=a.out/'signature.txt'
    if sign.exists():assert sign.read_text().strip()==signature,'Code changed; use a fresh output directory'
    sign.write_text(signature+'\n'); records=[]; begin=time.monotonic()
    for split,dataset in [('train','ISOD_reference_train_v1'),('val','ISOD_reference_val_v1')]:
        manifest=json.loads((a.root/'data'/dataset/'manifest.json').read_text())
        assert manifest['split']==split
        for row in manifest['frames']:
            site,frame=row['site'],row['frame'];record_path=a.out/'records'/split/site/f'{frame}.json'
            reference=a.root/'data'/dataset/row['path']
            assert digest(reference)==row['sha256']
            if record_path.exists():
                saved=json.loads(record_path.read_text());assert saved['source_sha256']==row['sha256']
                assert digest(a.out/saved['path'])==saved['sha256'];records.append(saved);continue
            choices=[a.root/'results/sam_prompt_v2'/('dev_cache' if split=='train' else 'validation_cache')/site/f'{frame}.npz',
                     a.root/'data/ISOD_sam_review/source-full-crayon'/site/f'{frame}.npz']
            cache=next((x for x in choices if x.exists()),None)
            if cache is None:raise FileNotFoundError(f'No reference-assistance cache: {site}/{frame}')
            groups,_=load_groups(cache,REFERENCE_PROMPTS);seeds,events=automatic_instances(groups)
            gt=np.array(Image.open(reference).convert('L'))>0
            labels,units=partition_reference(gt,seeds)
            for unit in units:
                evidence=[s for seed in seeds if set(unit.get('sam_seed_ids',[]))&set(seed['review_ids']) for s in seed['sources']]
                unit['sam_prompt_hints']=sorted({s['prompt'] for s in evidence})
                unit['sam_confidence_hint']=max((s['score'] for s in evidence),default=None)
                unit['category_verified']=False
                unit['area_pixels']=int((labels==unit['id']).sum())
            target=a.out/'ids'/split/site/f'{frame}.png';target.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(labels).save(target)
            record=dict(site=site,frame=frame,dataset_split=split,path=str(target.relative_to(a.out)),
                sha256=digest(target),source_reference=f'data/{dataset}/{row["path"]}',source_sha256=row['sha256'],
                source_review_status=row['review_status'],source_cache=str(cache.relative_to(a.root)),source_cache_sha256=digest(cache),
                uses_accepted_reference_mask=True,binary_union_preserved=True,
                objects=units,parent_events=events,human_instance_review_complete=False)
            record_path.parent.mkdir(parents=True,exist_ok=True);tmp=record_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(record_path);records.append(record)
            progress=dict(completed_frames=len(records),expected_frames=1700,elapsed_seconds=round(time.monotonic()-begin),status='reference-only proposals, unapproved instance quality')
            (a.out/'progress.json').write_text(json.dumps(progress)+'\n')
            if len(records)%50==0:print(json.dumps(progress),flush=True)
    assert Counter(r['dataset_split'] for r in records)==dict(train=1400,val=300)
    result=dict(status='reference-only instance proposals; not new unattended SAM labels',signature=signature,
        frame_count=len(records),all_binary_reference_unions_preserved=True,original_binary_masks_modified=False,
        training_labels_promoted=False,frames=records)
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'COMPLETE').write_text('1700 reference-only proposals; quality gate remains open\n')
    print(json.dumps(dict(frames=len(records),units=dict(Counter(o['unit'] for r in records for o in r['objects'])))))


if __name__=='__main__':main()
