from pathlib import Path
import hashlib,json
expected=json.loads(Path('reference_hashes.json').read_text())
for name,sha in expected.items():
    path=Path(name)
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
Path('reference_audit.json').write_text(json.dumps(dict(files=len(expected),match_published_frozen_references=True),indent=2)+'\n')
print('Verified',len(expected),'reference masks/ID maps against the published frozen hashes.')
