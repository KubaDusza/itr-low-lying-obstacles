"""Slurm-owned scoring and plots after training and inference dependencies finish."""
import argparse
import csv
from datetime import datetime,timezone
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    out=a.base/('analysis_gate' if a.gate else 'analysis_final');out.mkdir(exist_ok=True)
    if (out/'COMPLETE').exists():return
    if not a.gate:
        for seed in range(3):
            for source in ('reference','sam'):assert (a.base/'runs'/f'rgbd_{source}210_s{seed}'/'VERIFIED').exists()
            assert (a.base/'depth_ablations'/f'rgbd_reference210_s{seed}'/'VERIFIED').exists()
    work=Path(os.environ['SLURM_TMPDIR'])/'analysis';work.mkdir()
    with tarfile.open(a.base/'analysis-inputs.tar') as tar:tar.extractall(work,filter='data')
    raw=work/'raw';raw.mkdir()
    with tarfile.open(a.base/'data.tar.gz') as tar:
        members=[m for m in tar.getmembers() if Path(m.name).parts and Path(m.name).parts[0] in ('OWN','OWN_manual')]
        assert members;tar.extractall(raw,members=members,filter='data')
    for name in ('OWN','OWN_manual'):(work/'data'/name).symlink_to(raw/name)
    target=work/'results/triton_20261121/production';target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(a.base)
    (work/'results/evaluation_v2_20261121').symlink_to(out)
    commands=[['evaluate_gte_hazards.py','--root',str(work),'--out',str(out/'gte_full_diagnostic')],
              ['plot_hazard_analysis.py','--skip-instance-qa'],['make_depth_range_examples.py']]
    for args in commands:
        with (out/(args[0]+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(work/'code'/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT,check=True)
    sources=json.loads((out/'gte_full_diagnostic/summary.json').read_text())['source_runs']
    grouped=defaultdict(list)
    for r in csv.DictReader((out/'gte_full_diagnostic/frames.csv').open()):grouped[r['model'],r['recording']].append(r)
    recordings=[]
    for (model,recording),frames in grouped.items():
        assert len(frames)==5
        sums={k:sum(int(r[k]) for r in frames) for k in ('pixel_inter','pixel_union','tp50','fn50','fp50')}
        denominator=sums['tp50']+sums['fn50']
        recordings.append(dict(model=model,recording=recording,repeat_frames=5,
            **{k:frames[0][k] for k in ('camera_height_cm','background','object_set','nominal_distance_m')},
            pixel_iou=sums['pixel_inter']/sums['pixel_union'] if sums['pixel_union'] else None,
            localization_recall=sums['tp50']/denominator if denominator else None,
            false_components_per_frame=sums['fp50']/5,reference_units_per_recording=denominator/5))
    with (out/'recording_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(recordings[0]));writer.writeheader();writer.writerows(recordings)
    rows=[];counts={}
    for mode in ('rgb','depth','rgbd'):
        seeds=[]
        for seed in range(3):
            ref=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}'
            teacher=f'rgbd_sam210_s{seed}' if mode=='rgbd' else f'{mode}_sam_v2_s{seed}'
            candidates=[teacher+'_recovery_glaurung',teacher] if mode=='rgb' and seed==0 else [teacher]
            automatic=next((name for name in candidates if name in sources),None)
            if ref not in sources or automatic is None:continue
            seeds.append(seed)
            def metrics(name):
                r=sources[name];op=r['pooled']['operating_points']['50']
                return dict(pixel_iou=r['legacy_full_frame_pixel']['iou'],localization_recall=op['recall'],
                            false_components_per_frame=op['fp']/r['frames'],component_f1=op['f1'])
            before,after=metrics(ref),metrics(automatic)
            for metric in before:rows.append(dict(modality=mode,seed=seed,reference=ref,automatic=automatic,
                metric=metric,reference_value=before[metric],automatic_value=after[metric],change=after[metric]-before[metric]))
        counts[mode]=seeds
    if rows:
        with (out/'paired_supervision.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(updated_at=datetime.now(timezone.utc).isoformat(),pipeline_gate=a.gate,
        full_RGBD_training_and_depth_interventions_required=not a.gate,
        matched_runtime_reference_modalities_complete=all(f'rgbd_reference210_s{s}' in sources for s in range(3)),
        automatic_supervision_paired_seeds=counts,
        missing_RGB_seed0_is_not_replaced_with_an_old_teacher_version=0 not in counts['rgb'],
        domain='full-frame diagnostic; recording props included; no cropped-input experiment claimed',
        labels='Frozen-v1 binary training sources; new manual corrections retained separately for future jobs',
        component_reference_status='ID-aware GTE references with proposed fastener separation; uncertainty explicit',
        source_run_count=len(sources),slurm_job=os.environ['SLURM_JOB_ID'],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_archive_sha256=hashlib.sha256((a.base/'analysis-inputs.tar').read_bytes()).hexdigest())
    (out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'COMPLETE').write_text('Scoring, paired source table and figures completed with explicit cohort counts\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
