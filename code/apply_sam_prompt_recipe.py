"""Apply a frozen RGB-only SAM mask recipe; this command never reads reference masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--recipe',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();cv2.setNumThreads(1)
    config=json.loads(args.config.read_text());recipe=json.loads(args.recipe.read_text())
    positives=set(recipe['positive']);negatives=set(recipe['negative']);confidence=recipe['confidence']
    assert positives|negatives <= set(config['inference_prompts'])
    manifest=json.loads((args.cache/'manifest.json').read_text())
    assert manifest['model']==recipe['model'] and manifest['revision']==recipe['revision']
    records=[]
    for site,frame in config['frames']:
        cache=args.cache/site/f'{frame}.npz'
        with np.load(cache) as data:
            shape=tuple(data['shape']);bits=int(np.prod(shape));candidates=[];veto=np.zeros(shape,bool)
            for phrase,score,packed in zip(data['prompts'],data['scores'],data['masks'],strict=True):
                if score<confidence:continue
                if phrase not in positives and phrase not in negatives:continue
                mask=np.unpackbits(packed,count=bits).reshape(shape)>0
                if phrase in positives:candidates.append(mask)
                if phrase in negatives:veto|=mask
        prediction=np.zeros(shape,bool)
        for mask in candidates:
            if recipe['mode']=='veto' and negatives:
                if np.count_nonzero(mask&veto)>=recipe['overlap']*np.count_nonzero(mask):continue
            prediction|=mask
        if recipe['mode']=='subtract':prediction &= ~veto
        path=args.out/site/f'{frame}.png';path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp.png');assert cv2.imwrite(str(temporary),prediction.astype(np.uint8)*255);os.replace(temporary,path)
        records.append(dict(site=site,frame=frame,path=str(path.relative_to(args.out)),shape=list(map(int,shape)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),foreground_pixels=int(prediction.sum())))
    result=dict(input='Cached full-RGB SAM predictions and fixed text recipe only',reference_masks_read=False,
                recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),cache_provenance=manifest,frames=records,complete=True)
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Generated',len(records),'masks without reference masks:',args.out)


if __name__=='__main__':main()
