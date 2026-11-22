"""Cache RGB-only SAM candidates, reusing image features for all text prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    import torch
    import transformers
    from transformers import Sam3Model, Sam3Processor
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    c = json.loads(args.config.read_text())
    image_subdir = c.get('image_subdir', 'ISOD/data')
    phrases = c['inference_prompts']
    provenance = dict(model=c['model'], revision=c['revision'], transformers=transformers.__version__,
                      prompts=phrases, detection_threshold=.1, mask_threshold=.5,
                      input='full RGB and text only', feature_reuse=True,
                      config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = args.cache/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Existing cache has different provenance')
    manifest.write_text(json.dumps(provenance, indent=2)+'\n')
    model = Sam3Model.from_pretrained(c['model'], revision=c['revision'], local_files_only=True,
                                     dtype=torch.bfloat16).cuda().eval()
    processor = Sam3Processor.from_pretrained(c['model'], revision=c['revision'], local_files_only=True)
    begin = time.monotonic()
    verified = False
    with torch.inference_mode():
        for start in range(0, len(c['frames']), args.batch):
            pending = [(s,f) for s,f in c['frames'][start:start+args.batch]
                       if not (args.cache/s/f'{f}.npz').exists()]
            if not pending:
                continue
            images=[]
            for site,frame in pending:
                image=cv2.imread(str(args.root/image_subdir/site/'rgb'/f'{frame}.png'))
                if image is None:
                    raise FileNotFoundError(f'{site}/{frame}')
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_inputs=processor(images=images,text=[phrases[0]]*len(images),return_tensors='pt').to('cuda')
            vision=model.get_vision_features(image_inputs['pixel_values'])
            sizes=image_inputs['original_sizes'].tolist()
            candidates=[[] for _ in images]
            for j,phrase in enumerate(phrases):
                token=processor(text=[phrase]*len(images),return_tensors='pt').to('cuda')
                outputs=model(vision_embeds=vision,input_ids=token['input_ids'],attention_mask=token['attention_mask'])
                if not verified:
                    regular=model(**image_inputs)
                    assert torch.equal(outputs.pred_masks,regular.pred_masks), 'Feature reuse changed logits'
                    assert torch.equal(outputs.pred_logits,regular.pred_logits), 'Feature reuse changed confidence'
                    (args.cache/'feature_reuse_check.json').write_text(json.dumps({
                        'first_batch':pending, 'phrase':phrase, 'mask_logits_exact':True,
                        'confidence_logits_exact':True})+'\n')
                    verified=True
                    del regular
                results=processor.post_process_instance_segmentation(outputs,threshold=.1,mask_threshold=.5,target_sizes=sizes)
                for dest,result in zip(candidates,results,strict=True):
                    masks=result['masks'].cpu().numpy()
                    scores=result['scores'].float().cpu().numpy()
                    for mask,score in zip(masks,scores,strict=True):
                        dest.append((phrase,float(score),np.packbits(mask.reshape(-1)>0)))
                del outputs,results
                if (j+1)%30==0:
                    print(f'Batch {start+1}-{start+len(pending)}/{len(c["frames"])}: {j+1}/{len(phrases)} prompts; {time.monotonic()-begin:.1f}s',flush=True)
            for (site,frame),image,found in zip(pending,images,candidates,strict=True):
                path=args.cache/site/f'{frame}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.tmp.npz')
                packed=np.array([m for _,_,m in found],dtype=np.uint8).reshape(len(found),-1) if found else np.empty((0,(image.shape[0]*image.shape[1]+7)//8),np.uint8)
                np.savez_compressed(temporary,shape=image.shape[:2],prompts=np.array([p for p,_,_ in found],dtype=str),
                                    scores=np.array([s for _,s,_ in found],dtype=np.float32),masks=packed)
                os.replace(temporary,path)
            print(f'Cached {min(start+args.batch,len(c["frames"]))}/{len(c["frames"])} frames; elapsed {time.monotonic()-begin:.1f}s',flush=True)
            del vision,image_inputs,candidates
    (args.cache/'COMPLETE').write_text(str(len(c['frames']))+'\n')


if __name__=='__main__':
    main()
