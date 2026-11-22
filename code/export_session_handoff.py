"""Export visible conversation and execution records, without private reasoning."""
import collections,csv,gzip,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/Users/jakubdusza/.codex/sessions/2026/10/25/rollout-2026-10-25T00-01-33-01a16fee-f301-7451-9440-7ddccd54b485.jsonl')
OUT=ROOT/'results/session_handoff_20261122';OUT.mkdir(exist_ok=True)

def sanitize(s):
 s=re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+','[image data omitted]',s)
 s=re.sub(r'("data"\s*:\s*")[A-Za-z0-9+/=]{2048,}("\s*[,}])',r'\1[image/binary data omitted]\2',s)
 s=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b','[credential redacted]',s)
 s=re.sub(r'(?i)(Authorization[\s"\x27:=]+Bearer\s+)[A-Za-z0-9._-]+',r'\1[redacted]',s)
 return s

def main():
 counts=collections.Counter();types=collections.Counter();compactions=[];revisions=[];digest=hashlib.sha256();size=SOURCE.stat().st_size;read=0;first=last=None
 with SOURCE.open('rb') as src,gzip.open(OUT/'visible_conversation.jsonl.gz','wt',encoding='utf-8') as messages,gzip.open(OUT/'execution_records.jsonl.gz','wt',encoding='utf-8') as actions,(OUT/'CONVERSATION.md').open('w') as human:
  human.write('# Preserved visible conversation\n\nChronological export from the local append-only session log. Private reasoning, system/developer instructions and image binary data are excluded. Large execution records are in execution_records.jsonl.gz.\n\n')
  while read<size:
   line=src.readline();read+=len(line);digest.update(line)
   try:d=json.loads(line)
   except json.JSONDecodeError:continue
   stamp=d.get('timestamp');first=first or stamp;last=stamp;p=d.get('payload',{});kind=d.get('type')
   if kind=='compacted':
    compactions.append(dict(timestamp=stamp,ordinal=d.get('ordinal'),window_number=p.get('window_number')));continue
   if kind!='response_item':continue
   typ=p.get('type');types[typ]+=1
   if typ=='message' and p.get('role') in ['user','assistant'] and p.get('channel') not in ['analysis','summary']:
    blocks=p.get('content',[]);body='\n'.join(x.get('text','') for x in blocks if isinstance(x,dict) and x.get('type') in ['text','input_text','output_text']);body=sanitize(body)
    if not body.strip():continue
    r=dict(timestamp=stamp,ordinal=d.get('ordinal'),role=p['role'],channel=p.get('channel'),text=body)
    messages.write(json.dumps(r,ensure_ascii=False)+'\n');human.write(f'## {stamp} | {p["role"]} | record {d.get("ordinal")}\n\n{body}\n\n');counts[p['role']]+=1
   elif typ and ('call' in typ or typ.endswith('_output')) and typ not in ['reasoning']:
    s=sanitize(json.dumps(dict(timestamp=stamp,ordinal=d.get('ordinal'),payload=p),ensure_ascii=False));actions.write(s+'\n');counts['execution_records']+=1
    if typ in ['function_call','custom_tool_call']:
     markers=[label for label,pattern in [('patch',r'apply_patch'),('file_write',r'write_text|write_bytes|copy2|copytree|cat >'),('git',r'git.{0,100}(?:commit|push|add)'),('scheduler',r'sbatch|scontrol|scancel')] if re.search(pattern,s)]
     if markers:revisions.append(dict(timestamp=stamp,ordinal=d.get('ordinal'),tool=p.get('name'),call_id=p.get('call_id'),actions=';'.join(markers),record_sha256=hashlib.sha256(s.encode()).hexdigest()))
 manifest=dict(exported_at=datetime.now(timezone.utc).isoformat(),thread_id='01a16fee-f301-7451-9440-7ddccd54b485',source=str(SOURCE),source_snapshot_bytes=read,source_snapshot_sha256=digest.hexdigest(),first_record=first,last_record=last,counts=dict(counts),response_types=dict(types),compactions=compactions,exclusions=['private reasoning','system and developer instructions','binary image data'],publication='Local archive only; do not push conversation/execution records to GitHub',scope='All matching records present in the source snapshot; no claim to recover messages absent from the local log')
 with (OUT/'REVISION_INDEX.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['timestamp','ordinal','tool','call_id','actions','record_sha256']);writer.writeheader();writer.writerows(revisions)
 manifest['mutation_candidates']=len(revisions)
 for name in ['visible_conversation.jsonl.gz','execution_records.jsonl.gz','CONVERSATION.md','REVISION_INDEX.csv']:
  p=OUT/name;manifest.setdefault('exports',[]).append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(counts=counts,compactions=len(compactions),first=first,last=last)))
if __name__=='__main__':main()
