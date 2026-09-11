"""Wait for the dataset evaluation to release GPUs; run four frozen controls."""
import os
import subprocess
import sys
import time
import socket
import dataset_training_control as prep
import queue_dataset_sanitized_eval as resources

def command(flag,index=None,cuda=''):
    args=[sys.executable,'-B','scripts/dataset_training_control.py','--'+flag]
    if index is not None:args.append(str(index))
    path=prep.OUT/f'{flag}_{index if index is not None else "all"}.log'
    with path.open('x') as f:
        return subprocess.Popen(args,cwd=prep.REPO,env=resources.env(cuda),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)

def checked(flag,index=None,cuda=''):
    proc=command(flag,index,cuda)
    if proc.wait():raise RuntimeError(f'{flag} {index} failed; preserve outputs')

def run():
    prep.verify();prep.OUT.mkdir(parents=True,exist_ok=False);prep.guard()
    print('Waiting for paired dataset evaluation completion, then idle GPUs3,5. No other jobs stopped.',flush=True)
    dependency=prep.OUT.parent/'dataset_quality_audit_20260911/complete.json'
    while not dependency.exists() or not resources.idle((3,5)):time.sleep(30)
    resources.prep=prep
    for port in (8016,8017):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    rollout=resources.start('rollout','5',['--model','Qwen/Qwen3-8B','--port','8016','--enable-lora','--max-model-len','4096','--gpu-mem','0.85'])
    audits=[]
    try:
        from testbed.rollout_client import RolloutClient
        RolloutClient([8016]).wait_ready(max_wait=1200)
        checked('sample')
        for i in range(4):
            prep.guard();assert resources.idle((3,)),'Trainer GPU occupied by another job'
            print(f'TRAINING {i+1}/4',flush=True);checked('train',i,'3')
            audits.append(command('audit',i))
            for e in (2*i,2*i+1):
                assert resources.idle((3,))
                print(f'EVALUATING {e+1}/8',flush=True);checked('evaluate',e,'3')
    finally:resources.stop(rollout)
    for proc in audits:
        if proc.wait():raise RuntimeError('CPU execution audit failed')
    print('Waiting for idle GPUs1,3 for endpoint CoT judging.',flush=True)
    while not resources.idle((1,3)):time.sleep(30)
    judge=resources.start('judge','1,3',['--model','meta-llama/Llama-3.3-70B-Instruct','--port','8017','--tp','2','--max-model-len','16384','--gpu-mem','0.9'])
    try:
        RolloutClient([8017]).wait_ready(max_wait=1200)
        for i in range(8):
            print(f'JUDGING {i+1}/8',flush=True);checked('judge',i)
    finally:resources.stop(judge)
    prep.write(prep.OUT/'complete.json',{'status':'complete','training_runs':4,'evaluations':8})
    print('Four controls and both-view dataset evaluations complete; own services stopped.',flush=True)

if __name__=='__main__':run()
