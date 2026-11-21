"""Human quality review of frozen binary references, saved in a new namespace."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.parse import parse_qs, urlparse

ROOT=Path(__file__).resolve().parents[1]
os.environ['ITR_REVIEW_DRAFTS']=os.environ.get('ITR_GT_QUALITY_OUTPUT',str(ROOT/'data/ISOD_gt_quality_review_20261121'))
import cv2
import numpy as np
import review_isod_draft as draft
review=draft.review
cv2.setNumThreads(1)
QUEUE=ROOT/'results/gt_quality_review_20261121/queue.json'
INITIAL={};LOCK=threading.Lock()

review.PAGE=review.PAGE.replace("const queueKind = testReview ? 'test' : trainingReview ? 'training' : '';",
    "const qualityReview = ['/', '/quality'].includes(location.pathname);\nconst queueKind = qualityReview ? 'quality' : testReview ? 'test' : trainingReview ? 'training' : '';")
review.PAGE=review.PAGE.replace("const queueTitle = trainingReview ? 'Training spot-check' : 'Test batch';",
    "const queueTitle = qualityReview ? 'Ground-truth quality review' : trainingReview ? 'Training spot-check' : 'Test batch';")
review.PAGE=review.PAGE.replace("['sam-toggle', 'unsure', 'keepall', 'flag', 'note']", "['sam-toggle', 'unsure', 'keepall']")
review.PAGE=review.PAGE.replace('${done}/${reviewQueue.length} ${queueKind} frames saved', '${done}/${reviewQueue.length} frames reviewed · binary GT')
review.PAGE=review.PAGE.replace('<div id="wrap"><canvas', '<div id="quality-reason" style="padding:8px 14px;border-bottom:1px solid #2e333c;color:#e7e9ee;font-size:13px"></div><div id="wrap"><canvas')
review.PAGE=review.PAGE.replace('rows = r.rows; W = r.w; H = r.h;',
    "rows = r.rows; W = r.w; H = r.h;\n    if (qualityReview) $('#quality-reason').textContent = r.review_reason + ' · Green: keep. Blue: exclude. Save and next confirms this frame.';")
review.PAGE=review.PAGE.replace("(samPartition ? ' · Draft review'", "(samPartition ? ' · Binary GT quality review'")


def queue():return json.loads(QUEUE.read_text())['frames']


def definition(site,frame):return next((r for r in queue() if (r['site'],r['frame'])==(site,frame)),None)


def initial_state(site,frame):
    key=(site,frame)
    with LOCK:
        if key in INITIAL:return INITIAL[key]
        row=definition(site,frame);assert row is not None
        frozen=ROOT/row['source_reference'];assert hashlib.sha256(frozen.read_bytes()).hexdigest()==row['sha256']
        fg=cv2.imread(str(frozen),0)>0
        label=cv2.imread(str(Path(review.isod.DATA)/site/'label'/f'{frame}.png'),0)
        assert fg.shape==label.shape and not (fg & (label!=0)).any()
        proto,_,_=draft.frame_proposal(site,frame)
        # Split clickable SAM regions by the frozen decision, retaining every
        # pixel. Loading and confirming an unchanged frame must be pixel exact.
        sam=np.zeros(label.shape,np.uint16);ident=0
        for source_id in np.unique(proto):
            region=(proto==source_id)&(label==0)
            for keep in (False,True):
                part=region&(fg if keep else ~fg)
                n,cc=cv2.connectedComponents(part.astype(np.uint8),connectivity=8)
                for k in range(1,n):
                    ident+=1;assert ident<65535;sam[cc==k]=ident
        comp,rows=review.edit_components(site,frame,None,sam)
        comp,rows=review.separate_islands(comp,rows)
        keep=set(np.unique(comp[fg]));keep.discard(0)
        for r in rows:r.update(keep=r['id'] in keep,fresh=False)
        assert np.array_equal(np.isin(comp,list(keep)),fg)
        INITIAL[key]=(comp,rows,None,sam)
        return INITIAL[key]


def saved(site,frame):
    base=Path(review.OUT)/site
    return all((base/f'{frame}{suffix}').exists() for suffix in ('.png','.json','_quality.json'))


class Handler(draft.Demo):
    def _send(self,code,body,ctype='application/octet-stream'):
        self.last_status=code
        return super()._send(code,body,ctype)

    def do_GET(self):
        route=urlparse(self.path);q=parse_qs(route.query)
        if route.path in ('/','/quality'):return self._send(200,review.PAGE.encode(),'text/html; charset=utf-8')
        if route.path=='/api/quality-queue':
            return self._json([dict(site=r['site'],i=r['frame'],done=saved(r['site'],r['frame'])) for r in queue()])
        if route.path=='/api/frames':
            site=q['site'][0]
            return self._json([dict(i=i,done=saved(site,i),check=0) for i in range(100)])
        if route.path=='/api/frame':
            site,frame=q['site'][0],int(q['i'][0]);row=definition(site,frame)
            if row is None:return self._send(404,b'Frame outside review batch')
            if (Path(review.OUT)/site/f'{frame}.json').exists():
                comp,rows,cut,sam=review.frame_state(site,frame,split_islands=True)
            else:comp,rows,cut,sam=initial_state(site,frame)
            return self._json(dict(w=comp.shape[1],h=comp.shape[0],rows=rows,comp=review.pack_image(comp),
                comp16=True,cut=review.pack_image(cut),sam=review.pack_image(sam),sam16=True,suggestions=[],
                reference=(draft.REFERENCE/site/f'{frame}.png').exists(),review_reason=row['reason']))
        return super().do_GET()

    def handle_edit(self,path,q,site,frame,body):
        row=definition(site,frame)
        if row is None:return self._send(403,b'Frame outside review batch')
        folder=Path(review.OUT)/site
        if path in ('/api/save','/api/reset') and (folder/f'{frame}.json').exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup=Path(review.OUT)/'versions'/site/str(frame)/stamp;backup.mkdir(parents=True)
            for suffix in ('.png','.json','_sam.png','_cut.png','_quality.json'):
                source=folder/f'{frame}{suffix}'
                if source.exists():shutil.copy2(source,backup/source.name)
        super().handle_edit(path,q,site,frame,body)
        if self.last_status==200 and path=='/api/save':
            mask=cv2.imread(str(folder/f'{frame}.png'),0)>0
            original=cv2.imread(str(ROOT/row['source_reference']),0)>0
            meta=dict(site=site,frame=frame,reviewed_at=datetime.now(timezone.utc).isoformat(),
                status='human_quality_reviewed',source_reference=row['source_reference'],source_sha256=row['sha256'],
                reviewed_sha256=hashlib.sha256((folder/f'{frame}.png').read_bytes()).hexdigest(),
                changed_pixels=int((mask!=original).sum()),selection_group=row['selection_group'],
                instance_counts_verified=False,automatically_promoted_to_training=False)
            tmp=folder/f'{frame}_quality.tmp';tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(folder/f'{frame}_quality.json')
        if self.last_status==200 and path=='/api/reset':(folder/f'{frame}_quality.json').unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8813);a=p.parse_args()
    with review.http.server.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        print(f'GT quality review: http://127.0.0.1:{a.port}/quality',flush=True)
        server.serve_forever()
