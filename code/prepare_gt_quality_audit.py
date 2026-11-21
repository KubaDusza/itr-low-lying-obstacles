"""Package only automatic GT-assistance inputs and a separate human answer key."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    files={};frames=[];reproduction=[]
    def inputs(site,i):
        for suffix in [f'{i}.json',f'{i}_comp.png']:
            name=f'data/ISOD_comp/{site}/{suffix}';files[name]=ROOT/name
        name=f'data/ISOD/data/{site}/label/{i}.png';files[name]=ROOT/name
        family=next(f for f in ['source-full-crayon','source-scale-1','cache'] if (ROOT/f'data/ISOD_sam_review/{f}/{site}/{i}.npz').exists())
        for f in [family,'crayon-pilot']:
            name=f'data/ISOD_sam_review/{f}/{site}/{i}.npz'
            if (ROOT/name).exists():files[name]=ROOT/name
        return family
    for dataset,split in [('ISOD_reference_train_v1','train'),('ISOD_reference_val_v1','val')]:
        for r in json.loads((ROOT/'data'/dataset/'manifest.json').read_text())['frames']:
            if r['review_status']!='human_reviewed':continue
            site,i=r['site'],r['frame'];source=ROOT/'data'/dataset/r['path']
            answer=f'answers/{split}/{site}/{i}.png';files[answer]=source
            frames.append(dict(site=site,frame=i,dataset_split=split,cache_family=inputs(site,i),
                               answer_path=answer,answer_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    for r in json.loads((ROOT/'results/gt_quality_review_20261121/queue.json').read_text())['frames']:
        if r['review_status']=='human_reviewed':continue
        site,i=r['site'],r['frame'];inputs(site,i);source=ROOT/r['source_reference']
        path=f'frozen_assisted/{site}/{i}.png';files[path]=source
        reproduction.append(dict(site=site,frame=i,expected_path=path,expected_sha256=r['sha256']))
    assert len(frames)==370
    for name in ['audit_gt_assistance_quality.py','review_isod_draft.py','review_isod.py','isod_sam_review.py','isod_components.py','isod.py']:
        files['code/'+name]=ROOT/'code'/name
    with tempfile.TemporaryDirectory(prefix='itr-gt-audit-inputs-') as temp:
        for name,value in [('audit_frames.json',frames),('reproduction_frames.json',reproduction)]:
            path=Path(temp)/name;path.write_text(json.dumps(value,indent=2)+'\n');files[name]=path
        with tarfile.open(a.out,'w') as tar:
            for name,path in sorted(files.items()):tar.add(path,arcname=name)
    print(json.dumps(dict(reviewed_answer_frames=len(frames),reproduction_frames=len(reproduction),bundle_bytes=a.out.stat().st_size)))


if __name__=='__main__':main()
