"""Prospective 2x2 task-edit x length-penalty continuation, no replay."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace
import answer_pool_experiment as parent
import train_judge_replay as frozen
import dataset_sanitized_eval as evaluation

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'experiments/dataset_training_control_20260911'
OUT=Path('/ssd1/john/probe_playground/runs/dataset_training_control_20260911')
SEED=1423
MANIFEST=ROOT/'manifest_v3.json'
read,write,digest=evaluation.read,evaluation.write,evaluation.digest

def public_tasks(variant):
    from testbed.env import load_honeypot
    ids=read(ROOT/'train_task_ids.json');data=evaluation.official()
    selected=[]
    for t in load_honeypot('train'):
        if t.task_id not in ids:continue
        r=data[t.task_id]
        if variant=='sanitized':t=replace(t,text=r['prompt'],setup='\n'.join(r.get('test_imports',[])),shown_tests=r['test_list'][:1],original_tests=r['test_list'])
        selected.append(t)
    return sorted(selected,key=lambda t:t.task_id)

def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    inv=read(REPO/'exports/dataset_quality_audit_v1.json')
    ids=inv['splits']['train']['retained'];assert len(ids)==95
    write(ROOT/'train_task_ids.json',ids);write(ROOT/'task_ids.json',inv['splits']['eval']['retained'])
    for v in ('original','sanitized'):
        write(ROOT/f'public_{v}.json',[{k:getattr(t,k) for k in ('task_id','text','setup','shown_tests')} for t in public_tasks(v)])
    base_path=Path('/ssd1/john/probe_playground/runs/answer_replication_20260910/metadata/config_answer21_none_s941.json')
    template=read(base_path);configs=[];evals=[]
    # Alternating dataset/penalty order, fixed before new training generations.
    for i,(variant,clp) in enumerate((('original',.003),('sanitized',0.),('sanitized',.003),('original',0.))):
        name=f'{variant}_clp{str(clp).replace(".","p")}_s{SEED}'
        cfg=copy.deepcopy(template)
        cfg.update(seed=SEED,arm=name,trainer='dataset_training_control.py',replication_protocol='dataset_training_control_20260911',
            length_penalty=clp,lambda_ignored=None,run_dir=str(OUT/name),artifact_root=str(OUT),experiment_root=str(OUT),
            metadata_root=str(ROOT),output_manifest=str(ROOT/'manifest.json'),source_manifest=str(ROOT/'manifest.json'),
            train_task_ids=str(ROOT/'train_task_ids.json'),public_tasks_file=str(ROOT/f'public_{variant}.json'),
            first_batch_file=str(OUT/f'first_{variant}.json'),first_batch_sha256='SET_AFTER_SAMPLING',
            dataset_variant=variant,ports=[8016],lora_id_base=2060000000+i*1000000,
            storage_budget_bytes=20*1024**3,replay_label_source='none',replay_bank=None,replay_weight=0.)
        cfg.pop('lambda_ignored');cfg['lambda']=.5
        assert cfg['steps']==20 and cfg['batch_prompts']==16 and cfg['group_size']==8 and cfg['layer']==21
        configs.append(cfg)
        for view in ('original','sanitized'):
            evals.append(dict(name=name,variant=view,adapter=str(OUT/name/'serving_adapter'),
                probe_file=cfg['probe_file'],audit_bank=str(REPO/'experiments/replay_pilot_20260904/audit_bank.jsonl'),
                exclude_task_ids_file=template['train_task_ids'],out=str(OUT/f'eval_{name}_{view}'),
                lora_id=2070000000+len(evals)*100,seed=1234,n=4,max_tokens=2048,layer=21,pool='mean_answer'))
    paths=[Path(__file__),REPO/'scripts/queue_dataset_training_control.py',ROOT/'PLAN.md',base_path,
        Path(frozen.__file__),Path(parent.__file__),Path(evaluation.__file__),
        REPO/'src/testbed/env.py',REPO/'src/testbed/grpo.py',REPO/'src/testbed/replay.py',
        REPO/'scripts/train_matched_baselines.py',REPO/'scripts/eval_replay_replication.py',
        evaluation.SNAPSHOT,Path(template['probe_file']),Path(template['init_adapter'])/'adapter_model.safetensors']
    paths+=list(ROOT.glob('*.json'))
    write(ROOT/'manifest.json',dict(seed=SEED,configs=configs,eval_configs=evals,task_ids=inv['splits']['eval']['retained'],
        source_hashes={str(p):digest(p) for p in paths},storage_cap_bytes=20*1024**3,minimum_free_bytes=100*1024**3))

def verify():
    m=read(MANIFEST)
    assert all(digest(p)==h for p,h in m['source_hashes'].items()),'Frozen control source changed'
    return m

def guard(cfg=None):
    assert shutil.disk_usage(OUT).free>101*1024**3,'Free-space reserve reached'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<20*1024**3,'Control storage cap reached'

def sample():
    m=verify();guard()
    from transformers import AutoTokenizer
    from testbed.rollout_client import RolloutClient
    from testbed.sampling import build_prompt_text
    tok=AutoTokenizer.from_pretrained(m['configs'][0]['model_id'],local_files_only=True)
    client=RolloutClient([8016],timeout=1800);client.wait_ready(max_wait=120)
    for variant in ('original','sanitized'):
        tasks=[frozen.PublicTask(**r) for r in read(ROOT/f'public_{variant}.json')]
        selected=random.Random(SEED).sample(tasks,16)
        prompts=[tok(build_prompt_text(tok,t),add_special_tokens=False).input_ids for t in selected]
        assert max(map(len,prompts))+2048<=4096
        request=dict(prompt_token_ids=prompts,n=8,max_tokens=2048,temperature=1.,top_p=1.,seed=SEED*100000+1,
            lora_path=m['configs'][0]['init_adapter'],lora_id=2080000000+(variant=='sanitized'))
        write(OUT/f'first_{variant}.request.json',dict(request=request,task_ids=[t.task_id for t in selected]))
        rows=frozen.arithmetic.flatten_generated(client.generate(**request),[t.task_id for t in selected],prompts)
        write(OUT/f'first_{variant}.raw.json',rows)
        lookup={t.task_id:t for t in selected}
        scores=frozen.score_public_many([lookup[r['task_id']] for r in rows],[r['completion'] for r in rows],workers=8,timeout=6.,length_penalty=.003)
        write(OUT/f'first_{variant}.json',dict(request=request,task_ids=[t.task_id for t in selected],rows=rows,first_scores=scores,
            design_sha256=digest(MANIFEST)))

def config(index):
    cfg=copy.deepcopy(verify()['configs'][index]);cfg['first_batch_sha256']=digest(cfg['first_batch_file']);return cfg

def train(index):
    cfg=config(index);guard();parent.install_readout()
    def valid(c):
        assert c==config(index);return OUT,Path(c['run_dir'])
    def first(c):
        assert digest(c['first_batch_file'])==c['first_batch_sha256']
        r=copy.deepcopy(read(c['first_batch_file']))
        assert r['design_sha256']==digest(MANIFEST)
        for s in r['first_scores']:
            s['reward']=max(0.,1-c['length_penalty']*s['code_len']) if s['parsed'] and all(s['shown_pass']) else 0.
        return r
    def batchcheck(batch,c,ids,prompts):
        assert batch['task_ids']==ids and batch['request']['prompt_token_ids']==prompts
        rows=copy.deepcopy(batch['rows'])
        assert len(rows)==128 and {(r['task_id'],r['sample_idx']) for r in rows}=={(t,i) for t in ids for i in range(8)}
        for r in rows:assert r['prompt_token_ids']==prompts[ids.index(r['task_id'])]
        return rows
    def load_public(path,ids):
        assert digest(path)==cfg['public_tasks_sha256']
        rows=read(path);assert len(rows)==95 and [r['task_id'] for r in rows]==ids
        assert all(set(r)==frozen.PUBLIC_TASK_FIELDS and len(r['shown_tests'])==1 for r in rows)
        return [frozen.PublicTask(**r) for r in rows]
    def public_score(*args,**kwargs):
        kwargs['length_penalty']=cfg['length_penalty'];return frozen.score_public_many(*args,**kwargs)
    def validate_scores(scores,rows):return frozen.validate_public_scores(scores,rows,length_penalty=cfg['length_penalty'])
    arithmetic=SimpleNamespace(**vars(frozen.arithmetic));arithmetic.storage_guard=guard
    arithmetic.probe_scores=parent.clone_function(frozen.arithmetic.probe_scores,[('"mean_cot"','"mean_answer"',1)])
    scope=dict(vars(frozen),validate_config=valid,read_locked_first_batch=first,validate_first_batch=batchcheck,
        load_public_tasks=load_public,score_public_many=public_score,validate_public_scores=validate_scores,arithmetic=arithmetic)
    actor=parent.clone_function(frozen.train,[('os.environ.get("CUDA_VISIBLE_DEVICES") != "0"','os.environ.get("CUDA_VISIBLE_DEVICES") != "3"',1),
        ('Trainer must run with CUDA_VISIBLE_DEVICES=0','Trainer must run with CUDA_VISIBLE_DEVICES=3',1),('"mean_cot"','"mean_answer"',1)],scope)
    actor(cfg)

def evaluate(index,judge=False):
    m=verify()
    evaluation.ROOT=ROOT
    evaluation.verify=lambda:dict(m,configs=m['eval_configs'])
    original_digest=evaluation.digest
    evaluation.digest=lambda path:original_digest(MANIFEST if Path(path)==ROOT/'manifest.json' else path)
    (evaluation.judge if judge else evaluation.evaluate)(index)

def audit(index):
    cfg=config(index);run=Path(cfg['run_dir']);tasks={t.task_id:t for t in public_tasks(cfg['dataset_variant'])}
    from testbed.env import score_many
    rows=[json.loads(s) for s in (run/'rollouts.jsonl').read_text().splitlines()]
    with (run/'execution_audit.jsonl').open('x') as f:
        for start in range(0,len(rows),128):
            chunk=rows[start:start+128]
            scores=score_many([tasks[r['task_id']] for r in chunk],[r['completion'] for r in chunk],workers=8,timeout=6.,length_penalty=cfg['length_penalty'])
            for r,s in zip(chunk,scores):f.write(json.dumps({k:r[k] for k in ('step','task_id','sample_idx','finish_reason')}|s.to_dict())+'\n')
            f.flush()
    write(run/'audit_complete.json',{'rows':len(rows),'labels_read_by_trainer':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--sample',action='store_true')
    for flag in ('train','evaluate','judge','audit'):p.add_argument('--'+flag,type=int)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.sample:sample()
    elif a.train is not None:train(a.train)
    elif a.evaluate is not None:evaluate(a.evaluate)
    elif a.judge is not None:evaluate(a.judge,True)
    elif a.audit is not None:audit(a.audit)
    else:print('Verified',len(verify()['configs']),'training controls')
