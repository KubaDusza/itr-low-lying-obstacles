"""Version human corrections and confirmed instance groups; preserve old cohorts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from export_instance_proposals import REFERENCE_PROMPTS, automatic_instances, partition_reference
from refine_sam_instances import load_groups

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def merge_group(labels,units,selected,name):
    before=labels>0;assert selected.any() and not (selected&~before).any()
    affected=set(map(int,np.unique(labels[selected])));ident=int(labels.max())+1
    result=labels.copy();result[selected]=ident
    current=[]
    for unit in units:
        if not (result==unit['id']).any():continue
        unit=dict(unit)
        if unit['id'] in affected:unit['instance_extent_partially_reassigned_by_human']=True
        current.append(unit)
    current.append(dict(id=ident,unit='human_confirmed_individual',common_name=name,
        user_instance_group_confirmed=True,disconnected_visible_parts_allowed=True,category_verified=False))
    dense=np.zeros_like(result);final=[]
    for new_id,unit in enumerate(current,1):
        mask=result==unit['id'];dense[mask]=new_id
        final.append(dict(unit,id=new_id,area_pixels=int(mask.sum())))
    assert np.array_equal(dense>0,before) and len(np.unique(dense[selected]))==1
    return dense,final


def binary_candidate():
    source=ROOT/'data/ISOD_reference_train_v1';dest=ROOT/'data/ISOD_reference_train_reviewed_candidate_20261121'
    queue=json.loads((ROOT/'results/gt_quality_review_20261121/queue.json').read_text())['frames']
    assert len(queue)==16
    if dest.exists():
        existing=json.loads((dest/'manifest.json').read_text())
        by_key={(r['site'],r['frame']):r for r in existing['frames']}
        for r in queue:
            current=ROOT/'data/ISOD_gt_quality_review_20261121'/r['site']/f'{r["frame"]}.png'
            assert by_key[r['site'],r['frame']]['sha256']==sha(current),'New review edits require a new immutable candidate version'
        return dest
    with tempfile.TemporaryDirectory(dir=ROOT/'data',prefix='.quality-candidate-') as temp:
        stage=Path(temp)/'dataset';shutil.copytree(source,stage)
        manifest=json.loads((stage/'manifest.json').read_text());by_key={(r['site'],r['frame']):r for r in manifest['frames']}
        for reviewed in queue:
            site,i=reviewed['site'],reviewed['frame'];folder=ROOT/'data/ISOD_gt_quality_review_20261121'/site
            meta=json.loads((folder/f'{i}_quality.json').read_text());mask=folder/f'{i}.png'
            assert sha(mask)==meta['reviewed_sha256']
            row=by_key[site,i];assert row['sha256']==meta['source_sha256']
            shutil.copy2(mask,stage/row['path'])
            row.update(previous_reference_sha256=row['sha256'],previous_review_status=row['review_status'],
                sha256=sha(mask),review_status='human_quality_reviewed',source=str(mask.relative_to(ROOT/'data')))
        count=Counter(r['review_status'] for r in manifest['frames'])
        manifest.update(label_version='reviewed-candidate-20261121',source_version='ISOD_reference_train_v1',
            corrected_or_reconfirmed_frames=16,review_status_counts=dict(count),
            fully_human_annotated=False,promoted_to_running_training=False,
            supervision_description='Human-reviewed plus assisted training references; imperfect labels accepted, not gold test ground truth')
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');stage.rename(dest)
    return dest


def instance_overlay(binary):
    base=ROOT/'results/reference_instance_pool_20261121'
    if not (base/'INTEGRITY_VERIFIED').exists():return dict(status='waiting for complete reference-only instance pool')
    out=ROOT/'results/reference_instance_pool_reviewed_20261121'
    if out.exists():return dict(status='already written',path=str(out.relative_to(ROOT)))
    shutil.copytree(base,out)
    manifest=json.loads((out/'manifest.json').read_text());index={(r['site'],r['frame']):r for r in manifest['frames']}
    queue=json.loads((ROOT/'results/gt_quality_review_20261121/queue.json').read_text())['frames']
    constraints_base=ROOT/'results/gt_quality_review_20261121'
    constraints=json.loads((constraints_base/'instance_constraints/decisions.json').read_text())['constraints']
    for reviewed in queue:
        site,i=reviewed['site'],reviewed['frame'];row=index[site,i]
        cache=ROOT/row['source_cache'];groups,_=load_groups(cache,REFERENCE_PROMPTS);seeds,events=automatic_instances(groups)
        source=binary/row['site']/f'{i}.png';gt=np.array(Image.open(source).convert('L'))>0
        labels,units=partition_reference(gt,seeds)
        for constraint in [c for c in constraints if (c['site'],c['frame'])==(site,i)]:
            path=constraints_base/constraint['mask'];assert sha(path)==constraint['mask_sha256']
            selected=np.array(Image.open(path))>0
            labels,units=merge_group(labels,units,selected,constraint['common_name'])
        for unit in units:
            unit['area_pixels']=int((labels==unit['id']).sum())
            unit.setdefault('category_verified',False)
        target=out/row['path'];Image.fromarray(labels).save(target)
        row.update(sha256=sha(target),source_reference=str(source.relative_to(ROOT)),source_sha256=sha(source),
                   source_review_status='human_quality_reviewed',objects=units,parent_events=events,
                   human_binary_review_complete=True,human_instance_review_complete=False)
        record=out/'records'/row['dataset_split']/site/f'{i}.json';record.write_text(json.dumps(row,indent=2)+'\n')
        assert np.array_equal(labels>0,gt)
    manifest.update(status='Assisted reference-instance candidate with 16 human binary corrections and two confirmed object groups',
        corrected_binary_frames=16,user_confirmed_instance_groups=2,training_started=False,
        parent_manifest_sha256=sha(base/'manifest.json'),overlay_script_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # The parent audit describes old inputs; retain its provenance under its own name.
    if (out/'audit.json').exists():(out/'audit.json').rename(out/'parent_audit.json')
    audit=dict(frames=len(manifest['frames']),corrected_binary_frames=16,user_confirmed_instance_groups=2,
               all_revised_frame_unions_preserved=True,unchanged_frames_reused=len(manifest['frames'])-16,
               original_binary_or_instance_datasets_modified=False,instance_quality_is_assisted_not_fully_verified=True)
    (out/'review_overlay_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return dict(status='completed',**audit)


if __name__=='__main__':
    binary=binary_candidate();overlay=instance_overlay(binary)
    m=json.loads((binary/'manifest.json').read_text())
    result=dict(binary_candidate=str(binary.relative_to(ROOT)),review_status_counts=m['review_status_counts'],instance_overlay=overlay)
    (ROOT/'results/gt_quality_review_20261121/applied_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
