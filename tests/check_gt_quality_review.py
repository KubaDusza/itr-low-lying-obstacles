"""Real HTTP save/reload check against the isolated port-8814 test editor."""
import base64
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8814'
TEST=Path('/private/tmp/itr-gt-quality-save-test')


def get(path):return json.loads(urllib.request.urlopen(BASE+path,timeout=30).read())


def post(path,payload=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload or {}).encode(),method='POST')
    assert urllib.request.urlopen(req,timeout=30).status==200


def mask(frame):
    comp=np.frombuffer(base64.b64decode(frame['comp']),dtype='<u2').reshape(frame['h'],frame['w'])
    return comp,np.isin(comp,[r['id'] for r in frame['rows'] if r['keep']])


queue=get('/api/quality-queue');assert len(queue)==16
first,second=queue[:2];site,i=first['site'],first['i'];url=f'?site={site}&i={i}'
source=next(r for r in json.loads((ROOT/'results/gt_quality_review_20261121/queue.json').read_text())['frames'] if r['site']==site and r['frame']==i)
original=np.array(Image.open(ROOT/source['source_reference']).convert('L'))>0
frame=get('/api/frame'+url);comp,prelabel=mask(frame);assert np.array_equal(prelabel,original)
payload=dict(keep=[r['id'] for r in frame['rows'] if r['keep']],cut=None,sam=frame['sam'],sam16=True,separateIslands=True)
post('/api/save'+url,payload)
saved=TEST/site/f'{i}.png';assert np.array_equal(np.array(Image.open(saved))>0,original)
reloaded=get('/api/frame'+url);assert np.array_equal(mask(reloaded)[1],original)
chosen=next(r['id'] for r in frame['rows'] if r['keep']);payload['keep'].remove(chosen)
post('/api/save'+url,payload)
get(f'/api/frame?site={second["site"]}&i={second["i"]}')
back=get('/api/frame'+url);assert np.array_equal(mask(back)[1],original & (comp!=chosen))
assert next(r for r in get('/api/quality-queue') if r['site']==site and r['i']==i)['done']
time.sleep(.05)
meta=json.loads((TEST/site/f'{i}_quality.json').read_text());assert meta['changed_pixels']==int((original & (comp==chosen)).sum())
assert list((TEST/'versions'/site/str(i)).iterdir())
post('/api/reset'+url)
assert not next(r for r in get('/api/quality-queue') if r['site']==site and r['i']==i)['done']
production=json.loads(urllib.request.urlopen('http://127.0.0.1:8813/api/quality-queue',timeout=5).read())
assert not any(r['done'] for r in production),'Test modified production review progress'
html=urllib.request.urlopen(BASE+'/quality').read().decode();script=html.split('<script>',1)[1].split('</script>',1)[0]
Path('/private/tmp/itr-gt-quality-client.js').write_text(script)
report=dict(frame_count=16,initial_frozen_mask_exact=True,unchanged_save_exact=True,edited_save_persisted=True,
    next_then_back_retains_edits=True,saved_progress_persists=True,reset_clears_review_status=True,
    prior_edit_versions_retained=True,production_annotations_modified_by_test=False,test_namespace=str(TEST))
(ROOT/'results/gt_quality_review_20261121/editor_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
