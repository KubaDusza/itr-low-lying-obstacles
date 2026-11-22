"""Write a short hardware-cost table from measured, verified inference runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/extended_training_20261122/cost';out=ROOT/'results_report'
lines=[r'\begin{table}[H]',r'\caption{End-to-end latency (ms).}\label{tab:inference-cost}',r'\centering\small',r'\begin{tabular}{lrr}',r'\toprule Hardware & YOLOv8n boxes & SAM 3 (three prompts)\\\midrule']
summary={}
for device in ['cpu','cuda']:
 d=json.loads((base/f'{device}.json').read_text());assert d['complete'] and (base/f'{device}.VERIFIED').exists()
 yolo=next(r for r in d['rows'] if r['model'].startswith('YOLO'))
 sam=next(r for r in d['rows'] if r['model'].startswith('SAM'))
 hardware='Xeon Gold 6230, 8 threads' if device=='cpu' else 'H200 MIG (71 GiB)'
 assert 'H200 MIG' in sam['hardware'] if device=='cuda' else True
 cells=[f'{r["median_ms"]:.2f}' for r in [yolo,sam]]
 best=min(yolo['median_ms'],sam['median_ms'])
 cells=[r'\textbf{'+v+'}' if r['median_ms']==best else v for r,v in zip([yolo,sam],cells)]
 lines.append(hardware+' & '+' & '.join(cells)+r'\\')
 summary[device]=dict(sam_to_yolo_latency=sam['median_ms']/yolo['median_ms'],hardware=yolo['hardware'])
lines +=[r'\bottomrule\end{tabular}',r'\end{table}'];(out/'inference_cost_table.tex').write_text('\n'.join(lines)+'\n');(base/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
