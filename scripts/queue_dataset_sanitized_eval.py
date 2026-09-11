"""Bounded GPU queue for the dataset audit. Stops only its own child services."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import dataset_sanitized_eval as prep

def idle(devices):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    g={int(i):(u.strip(),int(m)) for i,u,m in (s.split(',') for s in raw.splitlines())}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    active={s.split(',')[0].strip() for s in raw.splitlines()}
    return all(g[d][1]<256 and g[d][0] not in active for d in devices)

def guard(m):
    assert shutil.disk_usage(prep.OUT).free>m['minimum_free_bytes']+1024**3
    assert sum(p.stat().st_size for p in prep.OUT.rglob('*') if p.is_file())<m['storage_cap_bytes']

def env(cuda):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=cuda,HF_HOME='/ssd1/john/.cache/huggingface',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONPATH='src:scripts',TESTBED_SYSTEM_SUFFIX='',PYTHONDONTWRITEBYTECODE='1')

def start(name,device,args):
    with (prep.OUT/f'{name}.log').open('x') as f:
        proc=subprocess.Popen([sys.executable,'-B','-m','testbed.rollout_server',*args],cwd=prep.REPO,
            env=env(device),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    prep.write(prep.OUT/f'{name}_pid.json',{'pid':proc.pid,'devices':device})
    return proc

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=50)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Own server did not exit; do not start conflicting work')

def run():
    m=prep.verify();prep.OUT.mkdir(parents=True,exist_ok=False)
    with (prep.OUT/'queue.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in (8016,8017):
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        guard(m)
        print('Waiting for free GPUs 3 and 5; GPUs 0,2,4 untouched.',flush=True)
        while not idle((3,5)):time.sleep(30)
        rollout=start('rollout','5',['--model','Qwen/Qwen3-8B','--port','8016','--enable-lora','--max-model-len','4096','--gpu-mem','0.85'])
        try:
            from testbed.rollout_client import RolloutClient
            RolloutClient([8016]).wait_ready(max_wait=1200)
            for i,cfg in enumerate(m['configs']):
                guard(m);assert idle((3,)),'Readout GPU taken by another job; stop safely'
                print(f'EVALUATING {i+1}/{len(m["configs"])} {cfg["name"]} {cfg["variant"]}',flush=True)
                with (prep.OUT/f'eval_{i:02d}.log').open('x') as f:
                    subprocess.run([sys.executable,'-B','scripts/dataset_sanitized_eval.py','--evaluate',str(i)],
                        cwd=prep.REPO,env=env('3'),stdout=f,stderr=subprocess.STDOUT,check=True)
        finally:stop(rollout)
        print('Generation complete. Waiting for GPUs 1 and 3 for CoT-only judge.',flush=True)
        while not idle((1,3)):time.sleep(30)
        judge=start('judge','1,3',['--model','meta-llama/Llama-3.3-70B-Instruct','--port','8017','--tp','2','--max-model-len','16384','--gpu-mem','0.9'])
        try:
            RolloutClient([8017]).wait_ready(max_wait=1200)
            for i,cfg in enumerate(m['configs']):
                guard(m)
                print(f'JUDGING {i+1}/{len(m["configs"])} {cfg["name"]} {cfg["variant"]}',flush=True)
                with (prep.OUT/f'judge_{i:02d}.log').open('x') as f:
                    subprocess.run([sys.executable,'-B','scripts/dataset_sanitized_eval.py','--judge',str(i)],
                        cwd=prep.REPO,env=env(''),stdout=f,stderr=subprocess.STDOUT,check=True)
        finally:stop(judge)
        prep.write(prep.OUT/'complete.json',{'status':'requested_evaluation_complete','conditions':len(m['configs'])})
        print('Paired evaluation and CoT judging complete; own GPU services stopped.',flush=True)

if __name__=='__main__':run()
