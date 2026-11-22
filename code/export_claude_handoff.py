"""Preserve visible Claude handoff sessions locally, without reasoning or binaries."""
import gzip,json,hashlib,collections
from pathlib import Path
from datetime import datetime,timezone
from export_session_handoff import sanitize
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/session_handoff_20261122';BASE=Path.home()/'.claude/projects/-Users-jakubdusza-GTE-2026-ITR-CS8813'
counts=collections.Counter();sources=[]
with gzip.open(OUT/'claude_visible_conversation.jsonl.gz','wt',encoding='utf-8') as messages,gzip.open(OUT/'claude_execution_records.jsonl.gz','wt',encoding='utf-8') as actions,(OUT/'CLAUDE_CONVERSATION.md').open('w') as human:
 human.write('# Preserved visible Claude conversations\n\nPrivate reasoning and binary images are excluded. Execution records are separate.\n\n')
 for sid in ['abd21ea9-3bdc-40dc-8ba5-187343282f56','24afe0e5-359f-4563-bbd3-902dc74733ed']:
  p=BASE/f'{sid}.jsonl';digest=hashlib.sha256();ordinal=0
  with p.open('rb') as src:
   for line in src:
    digest.update(line);ordinal+=1
    try:d=json.loads(line)
    except json.JSONDecodeError:continue
    role=d.get('type')
    if role not in ['user','assistant']:continue
    blocks=d.get('message',{}).get('content',[]);stamp=d.get('timestamp')
    if isinstance(blocks,str):blocks=[dict(type='text',text=blocks)]
    text=sanitize('\n'.join(b.get('text','') for b in blocks if isinstance(b,dict) and b.get('type')=='text'))
    if text.strip():
     r=dict(session=sid,timestamp=stamp,source_line=ordinal,role=role,text=text);messages.write(json.dumps(r,ensure_ascii=False)+'\n');human.write(f'## {stamp} | {role} | {sid} line {ordinal}\n\n{text}\n\n');counts[role]+=1
    for block in blocks:
     if isinstance(block,dict) and block.get('type') in ['tool_use','tool_result']:
      b=dict(block)
      if b['type']=='tool_result' and isinstance(b.get('content'),list):b['content']=[x for x in b['content'] if x.get('type')=='text']
      actions.write(sanitize(json.dumps(dict(session=sid,timestamp=stamp,source_line=ordinal,record=b),ensure_ascii=False))+'\n');counts['execution_records']+=1
  sources.append(dict(session=sid,path=str(p),bytes=p.stat().st_size,sha256=digest.hexdigest()))
manifest=dict(exported_at=datetime.now(timezone.utc).isoformat(),sources=sources,counts=dict(counts),publication='Local only; do not push raw conversations/execution records',exclusions=['private reasoning','binary images'])
for name in ['claude_visible_conversation.jsonl.gz','claude_execution_records.jsonl.gz','CLAUDE_CONVERSATION.md']:
 p=OUT/name;manifest.setdefault('exports',[]).append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'claude_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(counts=counts)))
