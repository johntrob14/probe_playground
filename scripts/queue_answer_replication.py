"""Wait without occupying GPUs; then start services, sample, freeze, and train."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import answer_replication as prep
import run_judge_replay as coordinator

def idle_all():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus={int(i.strip()):(u.strip(),int(m.strip())) for i,u,m in (line.split(',') for line in result.splitlines())}
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    active={line.split(',')[0].strip() for line in result.splitlines()}
    return all(gpus[i][1]<=256 and gpus[i][0] not in active for i in (0,1,3,5))

def env(cuda):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=cuda,HF_HOME='/ssd1/john/.cache/huggingface',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONPATH='src:scripts',TESTBED_SYSTEM_SUFFIX='',TESTBED_STORE=str(prep.STORE),PYTHONDONTWRITEBYTECODE='1')

def child(script,args,log,cuda):
    with (prep.ROOT/log).open('x') as f:
        subprocess.run([sys.executable,'-B',script,*args],cwd=prep.REPO,env=env(cuda),stdout=f,stderr=subprocess.STDOUT,check=True)

def run():
    prep.validate_design(prep.read_json(prep.ROOT/'design_manifest.json'))
    with (prep.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Waiting for GPUs 0,1,3,5 and shared coordinator lease. No other jobs will be stopped.',flush=True)
        while True:
            if idle_all():
                try:
                    with coordinator.leases(prep.ROOT):
                        if not idle_all():continue
                        prep.validate_design(prep.read_json(prep.ROOT/'design_manifest.json'))
                        for port in (8003,8005):
                            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
                        prep.storage_guard(headroom=prep.WRITE_HEADROOM)
                        specs=[('rollout','5',['--model','Qwen/Qwen3-8B','--port','8005','--enable-lora','--max-model-len','4096','--gpu-mem','0.85']),
                            ('judge','1,3',['--model','meta-llama/Llama-3.3-70B-Instruct','--port','8003','--tp','2','--max-model-len','16384','--gpu-mem','0.9'])]
                        pids={}
                        for name,cuda,args in specs:
                            with (prep.ROOT/f'server_{name}.log').open('x') as f:
                                process=subprocess.Popen([sys.executable,'-B','-m','testbed.rollout_server',*args],cwd=prep.REPO,env=env(cuda),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                            pids[name]=process.pid
                        prep.write_new(prep.ROOT/'server_pids.json',pids)
                        print(json.dumps({'event':'servers_started','pids':pids}),flush=True)
                    break
                except BlockingIOError:pass
            time.sleep(30)
        deadline=time.monotonic()+1200
        while True:
            try:prep.pinned_servers();break
            except (OSError,ValueError):
                if time.monotonic()>deadline:raise
                time.sleep(10)
        print('Services ready; sampling shared first batches.',flush=True)
        child('scripts/answer_replication.py',['--sample'],'first_batches.log','')
        child('scripts/answer_replication.py',['--freeze'],'freeze.log','')
        child('scripts/run_answer_replication.py',[],'preflight.log','')
        print('Nine-run training/evaluation queue starting.',flush=True)
        child('scripts/run_answer_replication.py',['--run'],'run.log','0')
        print('All nine endpoints trained, audited, evaluated, and judge-scored.',flush=True)

if __name__=='__main__':run()
