"""Separate disjoint small fasteners without changing accepted foreground.

The four fasteners share a class. Their exact subtype/identity across recordings
is unknown. If four visible components cannot be established, retain a flagged
cluster rather than invent individual pieces or keep both granularities.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]


def main():
    legend=defaultdict(list)
    with (ROOT/'data_collection/items.csv').open() as f:
        for row in csv.DictReader(f):legend[row['object_set']].append(row)
    out=ROOT/'results/gte_instance_proposals_20261121';out.mkdir(exist_ok=True)
    records=[]
    for path in sorted((ROOT/'data/OWN_manual').glob('*_id.png')):
        run,frame=path.stem[:-3].rsplit('_',1);category=run.split('_')[2]
        old=np.array(Image.open(path).convert('L'))
        gt=np.array(Image.open(path.with_name(path.name.replace('_id.png','.png'))).convert('L'))>0
        labels=np.zeros(old.shape,np.uint16);objects=[];flags=[]
        for ident,item in enumerate(legend.get(category,[]),1):
            selected=(old==ident)&gt
            if not selected.any():continue
            if item['item_id']=='scr_small':
                components,n=ndimage.label(selected,structure=np.ones((3,3),np.uint8))
                if n==4:
                    masks=[components==k for k in range(1,n+1)]
                    unit='proposed_individual_fastener'
                else:
                    masks=[selected];unit='unresolved_fastener_cluster'
                    flags.append(dict(item=item['item_id'],visible_components=n,expected_individuals=4))
            else:
                masks=[selected];unit='hazard_cluster' if item['item_id']=='scr_pile' else 'existing_named_item'
            for mask in masks:
                new_id=len(objects)+1;labels[mask]=new_id
                objects.append(dict(id=new_id,parent_item_id=item['item_id'],item_name=item['label'],
                                    training_class='floor_obstacle',unit=unit,pixels=int(mask.sum()),
                                    physical_subtype=None if item['item_id']=='scr_small' else item['label'],
                                    cross_recording_piece_identity=None if item['item_id']=='scr_small' else item['item_id']))
        assert np.array_equal(labels>0,gt),path
        target=out/'ids'/f'{run}_{frame}.png';target.parent.mkdir(exist_ok=True)
        Image.fromarray(labels).save(target)
        records.append(dict(recording=run,frame=int(frame),path=str(target.relative_to(out)),objects=objects,
                            needs_review=flags,binary_union_preserved=True,
                            source_id_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            source_binary_sha256=hashlib.sha256(path.with_name(path.name.replace('_id.png','.png')).read_bytes()).hexdigest()))
    manifest=dict(status='proposed instance references; new fastener splits require RGB check',
                  frames=records,frame_count=len(records),original_masks_modified=False,
                  split_small_fastener_frames=sum(any(o['unit']=='proposed_individual_fastener' for o in r['objects']) for r in records),
                  unresolved_fastener_frames=sum(bool(r['needs_review']) for r in records))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'review_queue.json').write_text(json.dumps([dict(recording=r['recording'],frame=r['frame'],issues=r['needs_review']) for r in records if r['needs_review']],indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='frames'}))


if __name__=='__main__':main()
