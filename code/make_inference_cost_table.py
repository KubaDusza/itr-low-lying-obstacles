"""Write a short hardware-cost table from measured, verified inference runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/extended_training_20261122/cost';out=ROOT/'results_report'
lines=[r'\begin{table}[H]',r'\caption{Single-image inference cost.}\label{tab:inference-cost}',r'\centering\small',r'\begin{tabular}{llrrr}',r'\toprule Hardware & Pipeline & Precision & Median latency (ms) & Peak memory (GiB)\\\midrule']
summary={}
for device in ['cpu','cuda']:
 d=json.loads((base/f'{device}.json').read_text());assert d['complete'] and (base/f'{device}.VERIFIED').exists()
 for r in d['rows']:
  title='YOLOv8n boxes' if r['model'].startswith('YOLO') else 'SAM 3, three prompts'
  hardware='Server CPU, 8 threads' if device=='cpu' else 'H200 MIG (71 GiB)'
  assert 'H200 MIG' in r['hardware'] if device=='cuda' else True
  precision={'float32':'FP32','float16':'FP16','bfloat16':'BF16'}[r['precision']]
  cells=[]
  for key,digits in [('median_ms',2),('peak_memory_GiB',3)]:
   value=f'{r[key]:.{digits}f}'
   if round(r[key],digits)==min(round(x[key],digits) for x in d['rows']):value=r'\textbf{'+value+'}'
   cells.append(value)
  lines.append(hardware+' & '+title+' & '+precision+' & '+' & '.join(cells)+r'\\')
 summary[device]=dict(sam_to_yolo_latency=d['rows'][1]['median_ms']/d['rows'][0]['median_ms'],hardware=d['rows'][0]['hardware'])
lines +=[r'\bottomrule\end{tabular}',r'\par\smallskip{\footnotesize Bold: lower cost on the same hardware.}',r'\end{table}'];(out/'inference_cost_table.tex').write_text('\n'.join(lines)+'\n');(base/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
