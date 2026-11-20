"""Select RGB-only prompt recipes on 30 references, then evaluate frozen recipes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text())


class Study:
    def __init__(self, frames, references, cache, confidence=.6):
        self.frames=frames
        self.gt=[];self.shapes=[];self.instances=[];self.unions=[]
        for row in references:
            site,frame=row['site'],row['frame']
            path=Path(row['reference'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['reference_sha256']
            image=cv2.imread(str(path),0)
            self.shapes.append(image.shape)
            self.gt.append(np.packbits((image>0).reshape(-1)))
            with np.load(cache/site/f'{frame}.npz') as z:
                assert tuple(z['shape'])==image.shape
                scores=z['scores'];prompts=z['prompts'];masks=z['masks']
                by=defaultdict(lambda:np.zeros_like(self.gt[-1]));instances=[]
                for phrase,score,mask in zip(prompts,scores,masks,strict=True):
                    if score>=confidence:
                        phrase=str(phrase);by[phrase]|=mask
                        instances.append((phrase,mask,int(np.bitwise_count(mask).sum())))
                self.unions.append(dict(by));self.instances.append(instances)
        assert len({x.size for x in self.gt})==1
        self.gt=np.stack(self.gt)
        self.areas=np.bitwise_count(self.gt).sum(axis=1)
        self._positive={};self._negative={}

    def union(self,phrases):
        return np.stack([np.bitwise_or.reduce([by[p] for p in phrases if p in by],axis=0)
                         if any(p in by for p in phrases) else np.zeros_like(gt)
                         for by,gt in zip(self.unions,self.gt,strict=True)])

    def predict(self,recipe):
        key=tuple(recipe['positive'])
        if key not in self._positive:self._positive[key]=self.union(key)
        pos=self._positive[key]
        negatives=tuple(recipe.get('negative',[]))
        if not negatives:return pos
        if negatives not in self._negative:self._negative[negatives]=self.union(negatives)
        neg=self._negative[negatives]
        mode=recipe.get('mode','subtract')
        if mode=='subtract':return pos&~neg
        if mode=='veto':
            out=np.zeros_like(pos);phrases=set(key);fraction=recipe.get('overlap',.5)
            for row,entries in enumerate(self.instances):
                for phrase,mask,area in entries:
                    if phrase in phrases and int(np.bitwise_count(mask&neg[row]).sum()) < fraction*area:
                        out[row]|=mask
            return out
        raise ValueError(mode)

    def evaluate(self,recipe,save_to=None):
        pred=self.predict(recipe)
        tp=np.bitwise_count(pred&self.gt).sum(axis=1).astype(np.int64)
        fp=np.bitwise_count(pred&~self.gt).sum(axis=1).astype(np.int64)
        fn=np.bitwise_count(~pred&self.gt).sum(axis=1).astype(np.int64)
        union=tp+fp+fn
        iou=np.divide(tp,union,out=np.ones(len(tp),float),where=union>0)
        per_site={}
        for site in sorted({s for s,f in self.frames}):
            ids=[i for i,(s,f) in enumerate(self.frames) if s==site]
            a,b,c=int(tp[ids].sum()),int(fp[ids].sum()),int(fn[ids].sum())
            per_site[site]=dict(mean_frame_iou=float(iou[ids].mean()),pooled_iou=a/max(1,a+b+c),frames=len(ids))
        totals=tp.sum(),fp.sum(),fn.sum();a,b,c=map(int,totals)
        result=dict(mean_frame_iou=float(iou.mean()),mean_site_iou=float(np.mean([v['mean_frame_iou'] for v in per_site.values()])),
                    foreground_mean_iou=float(iou[self.areas>0].mean()),pooled_iou=a/max(1,a+b+c),
                    precision=a/max(1,a+b),recall=a/max(1,a+c),tp=a,fp=b,fn=c,
                    empty_frames=int((self.areas==0).sum()),empty_frames_with_prediction=int(((self.areas==0)&(fp>0)).sum()),
                    per_site=per_site,
                    per_frame=[dict(site=s,frame=f,iou=float(v),tp=int(x),fp=int(y),fn=int(z))
                               for (s,f),v,x,y,z in zip(self.frames,iou,tp,fp,fn,strict=True)])
        if save_to:
            for (site,frame),packed,shape in zip(self.frames,pred,self.shapes,strict=True):
                dest=save_to/site/f'{frame}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                mask=np.unpackbits(packed,count=int(np.prod(shape))).reshape(shape).astype(np.uint8)*255
                assert cv2.imwrite(str(dest),mask)
        return result


def key(result):
    return result['mean_frame_iou'],result['pooled_iou']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['develop','validate'])
    p.add_argument('--directory',type=Path,default=Path('results/sam_prompt_v2'))
    args=p.parse_args();base=args.directory
    c=load_json(base/'study30_config.json');protocol=load_json(base/'expanded_protocol.json')
    output=base/'study30';output.mkdir(exist_ok=True)
    started=time.monotonic()
    if args.command=='develop':
        study=Study(c['frames'],protocol['development'],base/'dev_cache')
        positive=list(dict.fromkeys(c['positive_broad']+c['positive_names']+c['positive_floor']+c['positive_category']+c['baseline_prompts']))
        def recipe(phrases,negative=(),mode='union',overlap=.5):
            return dict(positive=list(dict.fromkeys(phrases)),negative=list(negative),mode=mode,overlap=overlap)
        recipes={};scores={}
        def add(name,r):
            recipes[name]=r;scores[name]=study.evaluate(r);return scores[name]
        for i,phrase in enumerate(positive):add(f'single_{i:03d}',recipe([phrase]))
        broad_ids=[f'single_{positive.index(p):03d}' for p in c['positive_broad']]
        best_broad=max(broad_ids,key=lambda n:key(scores[n]))
        predefined={'baseline':c['baseline_prompts'],'broad_pair':['loose item on the floor','small objects on the floor'],
                    'all_broad':c['positive_broad'],'all_names':c['positive_names'],
                    'all_names_on_floor':c['positive_floor'],'categories':c['positive_category'],
                    'broad_and_floor_names':c['positive_broad']+c['positive_floor'],
                    'everything_positive':positive}
        for name,phrases in predefined.items():add(name,recipe(phrases))
        chosen=recipes[best_broad]['positive'].copy();current=scores[best_broad];steps=[]
        while len(chosen)<6:
            candidates=[(phrase,study.evaluate(recipe(chosen+[phrase]))) for phrase in positive if phrase not in chosen]
            phrase,result=max(candidates,key=lambda x:key(x[1]))
            if result['mean_frame_iou']-current['mean_frame_iou']<.002:break
            chosen.append(phrase);current=result;steps.append(dict(added=phrase,mean_frame_iou=current['mean_frame_iou']))
        add('compact_union',recipe(chosen))
        broad_union=recipes[best_broad]['positive']+c['positive_floor']
        add('best_broad_and_floor_names',recipe(broad_union))
        # Test subtraction and whole-instance veto against multiple interpretable positive sets.
        top_names=sorted(scores,key=lambda n:key(scores[n]),reverse=True)
        seeds=[];seen=set()
        for name in top_names:
            identity=tuple(recipes[name]['positive'])
            if identity not in seen:
                seeds.append(name);seen.add(identity)
            if len(seeds)==4:break
        for name in ('all_names_on_floor','everything_positive','broad_pair'):
            if name not in seeds:seeds.append(name)
        ng=c['negative_groups']
        pools={**ng,'furniture_structure':ng['furniture']+ng['structure'],
               'furniture_structure_containers':ng['furniture']+ng['structure']+ng['containers'],
               'all_safe':ng['furniture']+ng['structure']+ng['containers']+ng['relations'],
               'all_including_size':[v for group in ng.values() for v in group]}
        for group in ng.values():
            for phrase in group:pools['single:'+phrase]=[phrase]
        for seed in seeds:
            for pool,negative in pools.items():
                for mode,fraction in [('subtract',.5),('veto',.5),('veto',.8)]:
                    name=f'{seed}|{pool}|{mode}|{fraction}'
                    add(name,recipe(recipes[seed]['positive'],negative,mode,fraction))
        positives=[name for name,r in recipes.items() if not r['negative']]
        subtract=[name for name,r in recipes.items() if r['negative'] and r['mode']=='subtract']
        veto=[name for name,r in recipes.items() if r['negative'] and r['mode']=='veto']
        winners={kind:max(names,key=lambda n:key(scores[n])) for kind,names in [('positive',positives),('subtract',subtract),('veto',veto)]}
        ids=['baseline','broad_pair',winners['positive'],winners['subtract'],winners['veto']]
        shortlist={}
        for name in dict.fromkeys(ids):
            alias='baseline' if name=='baseline' else 'broad_pair' if name=='broad_pair' else next(k for k,v in winners.items() if v==name)
            shortlist[alias]=dict(development_id=name,recipe=recipes[name],development=study.evaluate(recipes[name],output/'development_masks'/alias))
        highest=max(record['development']['mean_frame_iou'] for record in shortlist.values())
        competitive=[name for name,record in shortlist.items() if record['development']['mean_frame_iou']>=highest-.005]
        preferred=min(competitive,key=lambda name:(len(shortlist[name]['recipe']['positive'])+len(shortlist[name]['recipe']['negative']),
                                                  shortlist[name]['recipe']['mode']!='union',-shortlist[name]['development']['mean_frame_iou']))
        frozen=dict(created_from='30 development references only',selection_metric='Mean image IoU; pooled IoU breaks ties; empty reference/prediction pair IoU=1',
                    confidence=.6,mask_probability=.5,greedy_max_phrases=6,greedy_min_gain=.002,
                    simplicity_tolerance=.005,recipes=shortlist,preferred_development=preferred,
                    development_config_sha256=hashlib.sha256((base/'study30_config.json').read_bytes()).hexdigest())
        frozen_path=output/'frozen_recipes.json'
        if frozen_path.exists() and load_json(frozen_path)!=frozen:raise ValueError('Refusing to overwrite differing frozen selection')
        frozen_path.write_text(json.dumps(frozen,indent=2)+'\n')
        ranking=sorted(scores,key=lambda n:key(scores[n]),reverse=True)
        summary=dict(count=len(scores),greedy_steps=steps,ranking=[dict(name=n,recipe=recipes[n],**{k:v for k,v in scores[n].items() if k!='per_frame'}) for n in ranking])
        (output/'development_scores.json').write_text(json.dumps(summary,indent=2)+'\n')
        vc=dict(model=c['model'],revision=c['revision'],frames=[[r['site'],r['frame']] for r in protocol['validation']],
                inference_prompts=list(dict.fromkeys(p for record in shortlist.values() for field in ('positive','negative') for p in record['recipe'][field])),
                frozen_recipes_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest())
        (base/'validation_config.json').write_text(json.dumps(vc,indent=2)+'\n')
        print('Tested',len(scores),'recipes; validation requires',len(vc['inference_prompts']),'phrases')
        for name,record in shortlist.items():print(name,round(record['development']['mean_frame_iou'],4),record['recipe'])
    else:
        frozen=load_json(output/'frozen_recipes.json');vc=load_json(base/'validation_config.json')
        assert hashlib.sha256((output/'frozen_recipes.json').read_bytes()).hexdigest()==vc['frozen_recipes_sha256']
        study=Study(vc['frames'],protocol['validation'],base/'validation_cache')
        results={name:dict(recipe=record['recipe'],**study.evaluate(record['recipe'],output/'validation_masks'/name)) for name,record in frozen['recipes'].items()}
        for name,record in results.items():
            if record['recipe']['negative']:
                union_recipe={**record['recipe'],'negative':[],'mode':'union'}
                unfiltered=study.evaluate(union_recipe)
                record['negative_effect']=dict(correct_obstacle_pixels_removed=unfiltered['tp']-record['tp'],
                                               false_positive_pixels_removed=unfiltered['fp']-record['fp'],
                                               unfiltered_mean_frame_iou=unfiltered['mean_frame_iou'],
                                               unfiltered_pooled_iou=unfiltered['pooled_iou'])
        result=dict(frames=300,selection_frozen_before_validation=True,frozen_recipes_sha256=vc['frozen_recipes_sha256'],
                    preferred_development=frozen['preferred_development'],results=results,
                    note='Validation was used by the previous pipeline and detector checkpoint selection. It is not an untouched final test.')
        (output/'validation_scores.json').write_text(json.dumps(result,indent=2)+'\n')
        for name,r in results.items():print(name,{k:round(r[k],4) for k in ('mean_frame_iou','pooled_iou','precision','recall')},r['per_site'])
    print('Seconds',round(time.monotonic()-started,1))


if __name__=='__main__':main()
