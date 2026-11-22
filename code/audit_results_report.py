"""Verify the current 36-run report and its complete two-dataset evaluation."""
import argparse,csv,hashlib,json,math,re,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results_report';EXT=ROOT/'results/extended_training_20261122'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expanded(p):
 t=p.read_text()
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expanded(p.parent/(m[1] if m[1].endswith('.tex') else m[1]+'.tex')),t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--visual-reviewed',action='store_true');a=ap.parse_args()
 text=expanded(REPORT/'REPORT.tex');main_source=(REPORT/'REPORT.tex').read_text();original=(ROOT/'methodology_report/REPORT.tex').read_text()
 def survey(t):return t[t.index(r'\section{Introduction}'):t.index(r'\section{Methodology}')]
 assert survey(main_source)==survey(original)
 pdf=pymupdf.open(REPORT/'REPORT.pdf');texts=[p.get_text() for p in pdf]
 locations={}
 for name,pattern in [('results',r'IV\.\s*RESULTS'),('conclusion',r'CONCLUSION\s+AND\s+FUTURE\s+WORK'),('references',r'(?m)^REFERENCES$'),('appendix',r'APPENDIX\s+A\b')]:
  pages=[i+1 for i,t in enumerate(texts) if re.search(pattern,t,0 if name=='references' else re.I)];assert pages,(name,'heading not found');locations[name]=pages[0]
 assert locations['results'] in [4,5] and locations['conclusion']<=6,locations
 assert locations['references']>=locations['conclusion'] and locations['appendix']>locations['references']
 assert not re.search(r'Overfull|undefined|LaTeX Warning',(REPORT/'REPORT.log').read_text())
 fonts=subprocess.check_output(['pdffonts',str(REPORT/'REPORT.pdf')],text=True);assert 'Type 3' not in fonts and 'Type3' not in fonts
 labels=re.findall(r'\\label\{((?:fig|tab):[^}]+)\}',text);assert len(labels)==len(set(labels))
 for label in labels:assert re.search(r'\\(?:ref|autoref)\{'+re.escape(label)+r'\}',text),label
 assert 'older SAM-label versions' not in text and r'\frac{TP}' not in text
 manifest=json.loads((ROOT/'data/ISOD_reference_test_v2/manifest.json').read_text());frames=manifest['frames'];assert len(frames)==300
 assert len({(r['site'],r['frame']) for r in frames})==300
 assert sorted(Counter(r['site'] for r in frames).values())==[100,100,100]
 assert Counter(r['review_status'] for r in frames)==Counter(human_reviewed=30,isod_sam_prelabel=270)
 cohorts={};run_names=[]
 for source in ['reference','sam']:
  for seed in range(3):
   for family,epochs,name in [('U-Net',15,f'unet_logged_{source}_s{seed}'),('YOLOv8n-seg',60,f'yoloseg_e60_{source}_s{seed}')]:
    p=EXT/'runs'/name;assert (p/'VERIFIED').exists(),name
    done=json.loads((p/'completion.json').read_text());assert done['complete'] and done['epochs']==epochs
    h=list(csv.DictReader((p/('history.csv' if family=='U-Net' else 'results.csv')).open()));assert len(h)==epochs
    if family=='U-Net':assert all(math.isfinite(float(r['val_loss'])) for r in h)
    assert len(list((p/'gte').glob('*/*.png')))==320
    pred=p/('pred300' if source=='sam' and family=='U-Net' else 'pred')
    if source=='sam' or family=='YOLOv8n-seg':assert len(list(pred.glob('*/*.png')))==300
    run_names.append(name);cohorts.setdefault(family,[]).append(name)
 for source in ['reference','automatic']:
  for seed in range(3):
   name=f'yolov8n_detect_{source}_e60_s{seed}';p=EXT/'runs'/name
   assert (p/'VERIFIED').exists();assert len(list(csv.DictReader((p/'results.csv').open())))==60
   assert len(json.loads((p/'gte_boxes.json').read_text())['frames'])==320
   assert len(json.loads((p/'isod_boxes.json').read_text())['frames'])==300
   run_names.append(name);cohorts.setdefault('YOLOv8n boxes',[]).append(name)
 training=json.loads((ROOT/'results/results_report_20261122/revision/training_audit.json').read_text());assert len(training['ESANet'])==18 and all(r['epochs']==500 for r in training['ESANet'])
 numbers=json.loads((ROOT/'results/results_report_20261122/report_numbers.json').read_text())
 learned=[f'{family} / {source}' for family in ['ESANet RGB','ESANet D','ESANet RGB-D','U-Net','YOLOv8n-seg','YOLOv8n boxes'] for source in ['ref','SAM']]
 for dataset in ['gte','isod300']:
  for arm in learned:
   row=numbers[dataset][arm]
   for metric in ['pixel_iou','object_recall','component_f1','false_regions_per_frame']:
    assert len(row[metric]['values'])==3 and all(math.isfinite(x) for x in row[metric]['values']),(dataset,arm,metric)
 scores=json.loads((ROOT/'results/test_references_20261122/isod_test300_scores.json').read_text());assert len(scores)==38
 assert len({r['objects'] for r in scores.values()})==1
 for device in ['cpu','cuda']:assert (EXT/'cost'/f'{device}.VERIFIED').exists()
 audit=dict(updated_at=datetime.now(timezone.utc).isoformat(),combined_pages=len(pdf),section_pages=locations,format='IEEEtran conference',original_survey_verbatim=True,
  learned_runs=36,seeds=[0,1,2],model_label_pairs=12,isod_test_frames=300,isod_reference_status=dict(Counter(r['review_status'] for r in frames)),gte_frames=320,
  schedules=dict(ESANet=500,UNet=15,YOLO_seg=60,YOLO_boxes=60),completed_added_rgb_runs=run_names,
  checkpoint_selection=dict(ESANet='validation obstacle IoU',UNet='validation obstacle IoU',YOLO_seg='validation obstacle IoU among saved checkpoints',YOLO_boxes='validation fitness: .9 AP50:95 + .1 AP50'),
  no_undefined_references=True,no_overfull_boxes=True,no_type3_fonts=True,all_figures_tables_referenced=True,figure_table_labels=labels,visual_review_completed=a.visual_reviewed,
  limits=['270 assisted ISOD test masks,30 reviewed overrides','GTE one room and repeated static frames','Unannotated recording props','ESANet capacities differ','Server timings,not robot timings'],
  submission_status='Not submitted',pdf_sha256=sha(REPORT/'REPORT.pdf'),main_tex_sha256=sha(REPORT/'REPORT.tex'),results_tex_sha256=sha(REPORT/'results_section.tex'))
 (REPORT/'rubric_check.json').write_text(json.dumps(audit,indent=2)+'\n')
 p=REPORT/'source_assets.json';assets=json.loads(p.read_text());assets['approved_source_tex_sha256']=sha(ROOT/'methodology_report/REPORT.tex')
 for row in assets.get('assets',[]):
  row['source_sha256']=sha(ROOT/row['source']);row['destination_sha256']=sha(REPORT/row['destination']);row['sha256']=row['destination_sha256']
 used=set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text))
 assets['generated_figures']=[dict(destination=str(p),sha256=sha(REPORT/p)) for p in sorted(used)];assets['updated_at']=audit['updated_at'];p.write_text(json.dumps(assets,indent=2)+'\n')
 print(json.dumps(dict(pages=len(pdf),sections=locations,learned_runs=36,referenced_figures_tables=len(labels),pdf_sha256=audit['pdf_sha256'])))
if __name__=='__main__':main()
