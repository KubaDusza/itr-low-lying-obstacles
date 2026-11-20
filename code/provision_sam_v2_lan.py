"""Stage authenticated TLS transfers entirely between the user's GTE hosts."""
from pathlib import Path
import json, secrets, shlex, subprocess, tempfile, sys

ROOT=Path(__file__).resolve().parents[1]
SSH=['ssh','-o','ControlPath=none','-o','ProxyCommand=ssh -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=6 -W %h:%p gtlpc119','-o','BatchMode=yes','-o','ConnectTimeout=9']
DEST='/mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121'

def remote(host,command,stdin=None):
    p=subprocess.run(SSH+[host,command],input=stdin,text=True,capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError(f'{host}: {p.stderr[-600:]} {p.stdout[-600:]}')
    return p.stdout

def put(host,path,text):
    parent=str(Path(path).parent)
    remote(host,f'umask 077; mkdir -p {shlex.quote(parent)}; cat > {shlex.quote(path)}',text)

SERVER=r'''
from pathlib import Path
import hmac,http.server,json,ssl,tarfile
cfg=json.loads(Path('config.json').read_text());archive=Path('payload.tar')
with tarfile.open(archive,'w',dereference=True) as tar:
    def accept(info):
        return None if any(p in ('__pycache__','.git','results','runs') for p in Path(info.name).parts) else info
    if cfg['kind']=='data':
        src=Path(cfg['data_source']);paired=src/'paired_v1'
        splits=json.loads((paired/'splits.json').read_text())
        for sites in splits.values():
            for site in sites:
                for kind in ('rgb','depth'):
                    for frame in range(100):
                        path=paired/'ISOD/data'/site/kind/f'{frame}.png'
                        assert path.is_file(),path
                        tar.add(path,arcname=f'ISOD/data/{site}/{kind}/{frame}.png',filter=accept)
        for name in ('ISOD_reference_val_v1','ISOD_reference_test_v1','OWN_manual','esanet'):
            tar.add(paired/name,arcname=name,filter=accept)
        tar.add(src/'OWN',arcname='OWN',filter=accept)
        tar.add(paired/'esanet_data_full_reference/depth_stats.json',arcname='depth_stats.json')
    else:
        src=Path('/opt/venv/jdusza_venv/lib/python3.12/site-packages')
        tar.add(src,arcname='lib/python3.12/site-packages',filter=accept)
        tar.add('/data/cs_courses/jdusza_itr/code/yolov8n-seg.pt',arcname='itr_assets/yolov8n-seg.pt')
        for name in ('Arial.ttf','Arial.Unicode.ttf'):
            path=Path.home()/'.config/Ultralytics'/name
            if path.exists():tar.add(path,arcname='itr_assets/'+name)
print('Archive ready',archive.stat().st_size,flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.client_address[0] not in cfg['allowed_ips'] or self.path!='/payload' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+cfg['token']):
            self.send_error(403);return
        self.send_response(200);self.send_header('Content-Length',str(archive.stat().st_size));self.end_headers()
        with archive.open('rb') as f:
            while data:=f.read(1024*1024):self.wfile.write(data)
        self.server.served.add(self.client_address[0])
    def log_message(self,*args):pass
server=http.server.HTTPServer((cfg['ip'],cfg['port']),Handler)
ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain('cert.pem','key.pem');server.socket=ctx.wrap_socket(server.socket,server_side=True)
server.served=set();server.timeout=60
while len(server.served)<len(cfg['allowed_ips']):server.handle_request()
server.server_close()
for file in ('payload.tar','config.json','key.pem'):
    Path(file).unlink(missing_ok=True)
print('Transfer served and temporary archive removed.',flush=True)
'''

def main():
    with tempfile.TemporaryDirectory(prefix='itr-lan-') as temporary:
        temp=Path(temporary)
        for host,ip,port,kind,source,dest in [
            ('shire','10.17.6.228',41818,'data','/data/cs_courses/jdusza_itr/sam_v2_transfer_data',DEST),
            ('shire','10.17.6.228',41817,'runtime','/data/cs_courses/jdusza_itr/sam_v2_transfer_runtime','/opt/venv/jdusza_itr_train214')]:
            if len(sys.argv)>1 and kind!=sys.argv[1]:continue
            token=secrets.token_hex(32)
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=ITR local transfer','-addext',f'subjectAltName=IP:{ip}','-keyout',str(temp/'key.pem'),'-out',str(temp/'cert.pem')],check=True,capture_output=True)
            put(host,source+'/server.py',SERVER)
            clients=[('morgoth','10.17.6.226',DEST)] if kind=='runtime' else [('morgoth','10.17.6.226',DEST),('glaurung','10.17.6.225','/data/dream/cs_courses/jdusza_itr/sam_v2_20261121'),('rivendell','10.17.6.227','/mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121')]
            put(host,source+'/config.json',json.dumps(dict(ip=ip,port=port,kind=kind,token=token,allowed_ips=[ip for _,ip,_ in clients],data_source='/data/cs_courses/jdusza_itr')))
            for name in ('cert.pem','key.pem'):put(host,source+'/'+name,(temp/name).read_text())
            remote(host,f'cd {shlex.quote(source)} && tmux new-session -d -s itr-v2-transfer-{kind} '+shlex.quote('timeout 7200s python3 -u server.py > transfer.log 2>&1'))
            for client,client_ip,client_work in clients:
                receive_dest=dest if kind=='runtime' else client_work
                transfer=client_work+'/.transfer-'+kind
                remote(client,'tmux kill-session -t itr-v2-receive-'+kind+' 2>/dev/null || true')
                put(client,transfer+'/cert.pem',(temp/'cert.pem').read_text())
                put(client,transfer+'/header','Authorization: Bearer '+token+'\n')
                script=f'''#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(transfer)}
trap 'printf "%s\\n" "$?" > transfer.exit; rm -f header' EXIT
mkdir -p {shlex.quote(receive_dest)}
'''
                if kind=='runtime':script+=f'python3 -m venv --without-pip {shlex.quote(receive_dest)}\n'
                script+=f'''curl --fail --silent --show-error --retry 120 --retry-connrefused --retry-delay 5 --max-time 3600 --cacert cert.pem --header @header https://{ip}:{port}/payload | tar -xf - -C {shlex.quote(receive_dest)}
touch {shlex.quote(client_work+'/'+kind+'.READY')}
printf 'Transfer complete: {kind}\\n'
'''
                put(client,transfer+'/receive.sh',script)
                remote(client,f'cd {shlex.quote(transfer)} && tmux new-session -d -s itr-v2-receive-{kind} '+shlex.quote('bash receive.sh > receive.log 2>&1'))
                print('Started detached',kind,'transfer',host,'to',client,flush=True)

if __name__=='__main__':main()
