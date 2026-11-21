"""Slurm-owned RGB-D continuation with scratch-local data and persistent resume state."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    label_source = "sam" if a.gate or a.index % 2 else "reference"
    job = dict(id="real_data_gate" if a.gate else f"rgbd_{label_source}210_s{a.index//2}",
               kind="rgbd", seed=0 if a.gate else a.index//2, epochs=2 if a.gate else 500,
               label_source=label_source)
    persistent = a.base / "runs" / job["id"]
    persistent.mkdir(parents=True, exist_ok=True)
    if (persistent / "VERIFIED").exists():
        return
    work = Path(os.environ["SLURM_TMPDIR"]) / "itr"
    work.mkdir(exist_ok=True)
    assert shutil.disk_usage(work).free > 12 * 1024**3, "Insufficient node-local temporary storage"
    for archive in ("source.tar.gz", "data.tar.gz"):
        with tarfile.open(a.base / archive) as stream:
            stream.extractall(work, filter="data")
    # Latest reviewed adapters take precedence over the server snapshot.
    shutil.copytree(a.base / "code", work / "code", dirs_exist_ok=True)
    spec = importlib.util.spec_from_file_location("controller", work / "code/overnight_sam_v2.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    provenance = work / "provenance"
    provenance.mkdir()
    controller.SHARED = provenance
    masks = work / ("ISOD_reference_train_v1" if label_source == "reference" else "ISOD_sam_train_v2")
    controller.link(masks, provenance / "ISOD_sam_train_v2")
    controller.write(provenance / "recipe.json", dict(label_source=label_source,
                     teacher_recipe=json.loads((a.base / "teacher_recipe.json").read_text()) if label_source == "sam" else None))
    for name, count in ((masks.name,1400),("ISOD_reference_val_v1",300),("ISOD_reference_test_v1",30)):
        manifest=json.loads((work/name/"manifest.json").read_text())
        assert len(manifest['frames']) == count
        for row in manifest['frames']:
            assert hashlib.sha256((work/name/row['path']).read_bytes()).hexdigest() == row['sha256']
    cfg = dict(python210=sys.executable, python214=sys.executable,
               gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], require_idle=False,
               extra_pythonpath=str(a.base/"deps"), extra_pythonpath214=str(a.base/"deps"))
    env = controller.env_for(work, cfg, job['seed'])
    dataset = work / "esanet_data_full_sam_v2"
    command = [sys.executable,"code/esanet/export_isod.py","--out",str(dataset),
               "--train-masks",str(masks),"--val-masks",str(work/"ISOD_reference_val_v1"),
               "--test-masks",str(work/"ISOD_reference_test_v1"),
               "--depth-stats",str(a.base/"depth_stats.json"),
               "--gte",str(work/"OWN"),"--gte-out",str(work/"esanet_gte")]
    subprocess.run(command,cwd=work,env=env,check=True)
    controller.unlabeled_test(work,dataset,work/"esanet_test_all")
    # Restore a compatible checkpoint, if a previous allocation stopped.
    previous = persistent / "resume.pth"
    fit = work / "runs" / job['id'] / "fit_run"
    if previous.exists():
        dest=fit/"isod/checkpoints_resume";dest.mkdir(parents=True)
        shutil.copy2(previous,dest/"ckpt_latest.pth")
        # Selection may refer to an older best checkpoint.
        if (persistent/"best_resume.pth").exists():
            import torch
            ck=torch.load(previous,map_location="cpu",weights_only=False)
            shutil.copy2(persistent/"best_resume.pth",dest/f"ckpt_epoch_{ck['best_miou_epoch']}.pth")
    done=threading.Event()
    def persist_history():
        for pattern in ('*.csv','*.json'):
            for source in fit.rglob(pattern):
                target=persistent/'history'/source.relative_to(fit)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    def persist_checkpoint():
        import torch
        last_copy=0
        while not done.wait(30):
            if (work/'status.json').exists():
                shutil.copy2(work/'status.json',persistent/'progress.json')
            persist_history()
            log=work/'runs'/job['id']/'training.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-8192))
                    (persistent/'training_tail.txt').write_bytes(stream.read())
            if time.time()-last_copy<180:continue
            candidates=list(fit.glob("isod/checkpoints_*/ckpt_latest.pth"))
            if not candidates:continue
            source=max(candidates,key=lambda x:x.stat().st_mtime)
            try:
                before=source.stat()
                if time.time()-before.st_mtime<5:continue
                tmp=persistent/"resume.partial"
                shutil.copy2(source,tmp)
                ck=torch.load(tmp,map_location='cpu',weights_only=False)
                if source.stat().st_mtime_ns!=before.st_mtime_ns:tmp.unlink();continue
                best=source.parent/f"ckpt_epoch_{ck['best_miou_epoch']}.pth"
                if best.exists():
                    selected_state=torch.load(best,map_location='cpu',weights_only=False)
                    selected_state={k:v for k,v in selected_state.items() if k not in
                                    ('optimizer','optimizer_state_dict','lr_scheduler','scheduler')}
                    torch.save(selected_state,persistent/"best_resume.partial")
                    (persistent/"best_resume.partial").replace(persistent/"best_resume.pth")
                tmp.replace(previous)
                last_copy=time.time()
                for older in source.parent.glob('ckpt_epoch_*.pth'):
                    if older != best and time.time()-older.stat().st_mtime>10:
                        older.unlink()
            except Exception as exc:
                (persistent/'checkpoint_copy_error.txt').write_text(repr(exc))
    monitor=threading.Thread(target=persist_checkpoint,daemon=True);monitor.start()
    try:
        output=controller.execute(work,cfg,job)
        import torch
        latest=max(fit.glob("isod/checkpoints_*/ckpt_latest.pth"),key=lambda x:x.stat().st_mtime)
        state=torch.load(latest,map_location='cpu',weights_only=False)
        selected=latest.parent/f"ckpt_epoch_{state['best_miou_epoch']}.pth"
        state=torch.load(selected,map_location='cpu',weights_only=False)
        compact={k:v for k,v in state.items() if k not in ('optimizer','optimizer_state_dict','lr_scheduler','scheduler')}
        torch.save(compact,persistent/'selected_model.pth')
        if a.gate:
            reserve=12*(persistent/'selected_model.pth').stat().st_size+512*1024**2
            assert shutil.disk_usage(persistent).free > reserve, "Insufficient quota for six final models and two active resume checkpoints"
        for name in ('pred','gte'):
            shutil.copytree(output/name,persistent/name,dirs_exist_ok=True)
        for source in output.glob('*.json'):shutil.copy2(source,persistent/source.name)
        for source in output.glob('*.log'):shutil.copy2(source,persistent/source.name)
        persist_history()
        shutil.copy2(work/'status.json',persistent/'status.json')
        controller.write(persistent/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(),job=job,slurm_job=os.environ['SLURM_JOB_ID'],
                          input_archives=json.loads((a.base/'checksums.json').read_text()),
                          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          controller_sha256=hashlib.sha256((work/'code/overnight_sam_v2.py').read_bytes()).hexdigest(),
                          checkpoint_policy='periodic latest plus best model; final selected model; CSV histories retained'))
        (persistent/'VERIFIED').touch()
        (persistent/'FAILED.json').unlink(missing_ok=True)
        for name in ('resume.pth','best_resume.pth','resume.partial'):
            (persistent/name).unlink(missing_ok=True)
    except BaseException as exc:
        (persistent/'FAILED.json').write_text(json.dumps(dict(error=repr(exc),job=job)))
        for source in (work/'runs'/job['id']).glob('*.log'):shutil.copy2(source,persistent/source.name)
        raise
    finally:
        done.set();monitor.join()


if __name__ == '__main__':
    main()
