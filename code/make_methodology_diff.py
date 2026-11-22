"""Inline text diff against the saved methodology submission; no PDF annotations."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'methodology_report/REPORT.tex'
NEW = ROOT / 'results_report/REPORT.tex'
OUT = ROOT / 'results_report'
TMP = ROOT / 'tmp/pdfs/methodology_diff'
TMP.mkdir(parents=True, exist_ok=True)


def section(path):
    text = path.read_text().split(r'\section{Methodology}', 1)[1]
    return text.split(r'\input{results_section.tex}', 1)[0].split(r'\IEEEtriggeratref', 1)[0].strip()


def plain(tex):
    # Retain caption text; exclude the pictures themselves from a text diff.
    def figure_text(match):
        body = match[0]
        start = body.find(r'\caption{')
        if start < 0:
            return ''
        start += len(r'\caption{')
        depth, end = 1, start
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        return '\n\nFigure caption: ' + body[start:end - 1] + '\n\n'
    tex = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', figure_text, tex, flags=re.S)
    tex = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\cite\{([^}]+)\}', lambda m: '[citation: ' + m[1] + ']', tex)
    tex = re.sub(r'\\ref\{([^}]+)\}', lambda m: '[' + m[1] + ']', tex)
    tex = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}',
                 'IoU = TP/(TP+FP+FN); precision = TP/(TP+FP); recall = TP/(TP+FN).', tex, flags=re.S)
    text = subprocess.run(['pandoc', '-f', 'latex', '-t', 'plain', '--wrap=none'],
                          input=tex, text=True, capture_output=True, check=True).stdout
    text = re.sub(r'^\s*\[\]\s*$', '', text, flags=re.M)
    text = re.sub(r'\s*\{#[^}]+\}', '', text)
    text = re.sub(r'^\s*[- ]{8,}\s*$', '', text, flags=re.M)
    return text.strip()


def escape(text):
    for a, b in {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
                 '\u2013': '-', '\u2014': '--', '≥': '>=', '×': 'x',
                 '⁻': '-', '⁴': '4', '³': '3', '²': '2'}.items():
        text = text.replace(a, b)
    text = unicodedata.normalize('NFKC', text)
    special = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
               '#': r'\#', '$': r'\$', '%': r'\%', '&': r'\&',
               '_': r'\_', '^': r'\textasciicircum{}', '~': r'\textasciitilde{}',
               '<': r'\textless{}', '>': r'\textgreater{}'}
    return ''.join(special.get(c, c) for c in text)


old, new = section(OLD), section(NEW)
(OUT / 'methodology_changes.diff').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True),
    fromfile='Saved methodology (2026-10-25): Section III',
    tofile='Current results report: Section III')))

old_text, new_text = plain(old), plain(new)
tokens = lambda t: re.findall(r'\n\s*\n|\S+', t)
a, b = tokens(old_text), tokens(new_text)
parts = []
for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    for words, color in ([(a[i:j], None)] if op == 'equal' else
                         ([(a[i:j], 'deleted')] if op == 'delete' else
                          [(b[k:l], 'added')] if op == 'insert' else
                          [(a[i:j], 'deleted'), (b[k:l], 'added')])):
        for word in words:
            if word.isspace():
                parts.append('\n\n')
            elif color:
                parts.append(r'\colorbox{' + color + '}{' + escape(word) + '} ')
            else:
                parts.append(escape(word) + ' ')
header = r'''\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=0.65in]{geometry}
\usepackage{xcolor}
\usepackage{microtype}
\definecolor{added}{RGB}{206,240,207}
\definecolor{deleted}{RGB}{255,216,216}
\setlength{\fboxsep}{1pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\emergencystretch=3em
\begin{document}
{\large Methodology: inline changes}\par
Saved October 25 methodology compared with the current results report.
\colorbox{deleted}{Deleted text} \quad \colorbox{added}{Added text}.
Unchanged text is black. Section III text, captions and table entries are compared;
pictures are omitted. Training and evaluation details also appear in Appendices B--C
of the current report. The source diff preserves the exact TeX edits.
\medskip\hrule\medskip
\small
'''
(TMP / 'methodology_changes.tex').write_text(header + ''.join(parts) + '\n\\end{document}\n')
subprocess.run(['/Library/TeX/texbin/pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                '-output-directory=' + str(TMP), str(TMP / 'methodology_changes.tex')],
               stdout=(TMP / 'build.log').open('w'), check=True)
(OUT / 'methodology_changes.pdf').write_bytes((TMP / 'methodology_changes.pdf').read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'methodology_changes_sources.json').write_text(json.dumps({
    'baseline': str(OLD.relative_to(ROOT)), 'baseline_sha256': sha(OLD),
    'baseline_pdf_sha256': sha(ROOT / 'methodology_report/REPORT.pdf'),
    'current': str(NEW.relative_to(ROOT)), 'current_sha256': sha(NEW),
    'scope': 'Methodology section only, including captions and table entries; image contents omitted',
    'survey_unchanged': OLD.read_text().split(r'\section{Methodology}', 1)[0].split(r'\section{Introduction}', 1)[1]
       == NEW.read_text().split(r'\section{Methodology}', 1)[0].split(r'\section{Introduction}', 1)[1],
}, indent=2) + '\n')
print('Wrote inline PDF and exact methodology source diff.')
