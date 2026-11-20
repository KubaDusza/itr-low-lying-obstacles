from pathlib import Path
import json,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[1]
child=r'''
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/code')
import overnight_sam_v2 as m
base=Path(sys.argv[2]);m.SHARED=base/'shared';m.HOST=sys.argv[3];m.DEADLINE=time.time()+30
work=base/m.HOST;work.mkdir(exist_ok=True);(work/'data.READY').touch()
m.prepare=lambda *a,**kw:None
m.memory_free=lambda *a:(24576,0)
original_sleep=time.sleep
m.time.sleep=lambda n:original_sleep(min(n,.01))
def execute(work,cfg,job,smoke=False):
    attempt=base/'executions'/job['id'];attempt.mkdir(parents=True,exist_ok=True)
    (attempt/(m.HOST+'.'+str(os.getpid()))).write_text('executed')
    out=work/job['id'];out.mkdir(exist_ok=True);(out/'scores.json').write_text('{}')
    original_sleep(.02)
    return out
m.execute=execute
m.worker(work,{'allowed':['unet'],'require_idle':False})
'''
with tempfile.TemporaryDirectory(prefix='itr-queue-check-') as temp:
    base=Path(temp);shared=base/'shared';(shared/'jobs').mkdir(parents=True);(shared/'SMOKES_PASSED').touch()
    jobs=[dict(id=f'job_{i}',kind='unet',seed=i,epochs=15) for i in range(15)]
    (shared/'jobs.json').write_text(json.dumps(jobs))
    stale=shared/'jobs/job_0';stale.mkdir();(stale/'owner.json').write_text(json.dumps({'host':'worker_a','pid':99999999}))
    children=[subprocess.Popen([sys.executable,'-c',child,str(root),str(base),host],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for host in ('worker_a','worker_b')]
    for process in children:
        stdout,stderr=process.communicate(timeout=35)
        assert process.returncode==0,(stdout,stderr)
    for job in jobs:
        assert (shared/'jobs'/job['id']/'done.json').is_file(),job
        assert len(list((base/'executions'/job['id']).iterdir()))==1,job
    assert list((shared/'jobs').glob('job_0.interrupted.*'))
    print('PASS: two competing workers executed all 15 jobs exactly once; the interrupted local claim was recovered.')
