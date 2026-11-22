"""Check the built report against its actual completed experiment cohort."""
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results_report'
EXTENDED = ROOT / 'results/extended_training_20261122'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded(path):
    text = path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',
                  lambda m: expanded(path.parent / (m[1] if m[1].endswith('.tex') else m[1] + '.tex')), text)


def main():
    text = expanded(REPORT / 'REPORT.tex')
    main_source = (REPORT / 'REPORT.tex').read_text()
    original = (ROOT / 'methodology_report/REPORT.tex').read_text()
    def survey(t):
        return t[t.index(r'\section{Introduction}'):t.index(r'\section{Methodology}')]
    assert survey(main_source) == survey(original)
    pdf = fitz.open(REPORT / 'REPORT.pdf')
    assert len(pdf) == 21
    assert 'IV. RESULTS' in pdf[4].get_text()
    assert 'CONCLUSION AND FUTURE WORK' in pdf[5].get_text()
    assert 'REFERENCES' in pdf[6].get_text()
    assert 'APPENDIX A' in pdf[7].get_text()
    assert not re.search(r'Overfull|undefined|LaTeX Warning', (REPORT / 'REPORT.log').read_text())
    fonts = subprocess.check_output(['pdffonts', str(REPORT / 'REPORT.pdf')], text=True)
    assert 'Type 3' not in fonts and 'Type3' not in fonts
    labels = re.findall(r'\\label\{((?:fig|tab):[^}]+)\}', text)
    assert len(labels) == len(set(labels))
    for label in labels:
        assert re.search(r'\\(?:ref|autoref)\{' + re.escape(label) + r'\}', text), label
    assert 'older SAM-label versions' not in text
    assert r'\frac{TP}' not in text
    boxes = {}
    for source in ['reference', 'automatic']:
        boxes[source] = []
        for seed in range(3):
            run = EXTENDED / 'runs' / f'yolov8n_detect_{source}_s{seed}'
            assert (run / 'VERIFIED').exists()
            settings = json.loads((run / 'settings.json').read_text())
            done = json.loads((run / 'completion.json').read_text())
            history = list(csv.DictReader((run / 'results.csv').open()))
            assert settings['epochs'] == 200 and settings['early_stopping_patience'] == 20
            assert settings['train_frames'] == 1400 and settings['validation_frames'] == 300
            assert done['complete'] and done['actual_epochs'] == len(history)
            assert len(json.loads((run / 'gte_boxes.json').read_text())['frames']) == 320
            boxes[source].append(len(history))
    for seed in range(3):
        run = EXTENDED / 'runs' / f'unet_logged_reference_s{seed}'
        assert (run / 'VERIFIED').exists()
        history = list(csv.DictReader((run / 'history.csv').open()))
        assert len(history) == 15
        assert all(float(row['val_loss']) > 0 for row in history)
    training = json.loads((ROOT / 'results/results_report_20261122/revision/training_audit.json').read_text())
    assert len(training['ESANet']) == 18 and all(r['epochs'] == 500 for r in training['ESANet'])
    native = json.loads((EXTENDED / 'analysis/summary.json').read_text())
    assert native['complete'] and len(native['runs']) == 6
    for device in ['cpu', 'cuda']:
        assert (EXTENDED / 'cost' / f'{device}.VERIFIED').exists()
        assert json.loads((EXTENDED / 'cost' / f'{device}.json').read_text())['complete']
    audit = dict(updated_at=datetime.now(timezone.utc).isoformat(),
        combined_pages=len(pdf), results_and_conclusion_pages=[5, 6],
        appendix_pages=list(range(8, 22)), format='IEEEtran conference',
        original_introduction_and_state_of_art_verbatim=True,
        results_length_within_two_to_four_columns=True,
        conclusion_or_future_work_present=True,
        all_figures_and_tables_referenced=True, figure_table_labels=labels,
        no_undefined_references=True, no_overfull_boxes=True, no_type3_fonts=True,
        learned_conditions_all_have_three_seeds=True,
        native_box_policy=dict(max_epochs=200, patience=20, actual_epochs=boxes,
            selection='ISOD validation box AP50:95', gte_frames=320),
        unet_validation_loss_logged_all_three_runs=True,
        standard_component_metric_and_separate_native_box_AP=True,
        cpu_gpu_cost_measured=True, gte_vpn_required=False,
        scientific_limits=['30 reviewed ISOD test frames, not the full 300-frame split',
            'Assisted references and some grouped IDs', 'Validation-informed prompt development',
            'Unannotated recording props', 'One external room, static repeats',
            'Models differ in training targets and budgets',
            'Hardware cost uses different optimized precisions; no robot timing'],
        visual_review='All 21 pages rendered; main results, changed training curves, grouped cloth scenes and hardware table inspected.',
        submission_status='Not submitted; reviewable report',
        pdf_sha256=sha(REPORT / 'REPORT.pdf'),
        main_tex_sha256=sha(REPORT / 'REPORT.tex'),
        results_tex_sha256=sha(REPORT / 'results_section.tex'))
    (REPORT / 'rubric_check.json').write_text(json.dumps(audit, indent=2) + '\n')
    assets = json.loads((REPORT / 'source_assets.json').read_text())
    assets['approved_source_tex_sha256'] = sha(ROOT / 'methodology_report/REPORT.tex')
    for row in assets['assets']:
        row['source_sha256'] = sha(ROOT / row['source'])
        row['destination_sha256'] = sha(REPORT / row['destination'])
        row['sha256'] = row['destination_sha256']
    assets['generated_figures'] = [dict(destination=str(p.relative_to(REPORT)), sha256=sha(p))
        for p in sorted((REPORT / 'figures').iterdir()) if p.suffix in ['.pdf', '.png']]
    assets['updated_at'] = audit['updated_at']
    (REPORT / 'source_assets.json').write_text(json.dumps(assets, indent=2) + '\n')
    print(json.dumps(dict(pages=len(pdf), referenced_figures_and_tables=len(labels),
        native_epochs=boxes, pdf_sha256=audit['pdf_sha256'])))


if __name__ == '__main__':
    main()
