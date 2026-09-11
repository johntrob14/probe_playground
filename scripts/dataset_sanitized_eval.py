"""Isolated original/sanitized evaluation; never modifies frozen experiments."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'experiments/dataset_quality_audit_20260911'
STORE=Path('/ssd1/john/probe_playground/runs')
OUT=STORE/'dataset_quality_audit_20260911'
SNAPSHOT=REPO/'exports/google_sanitized_mbpp_snapshot_v1.json'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def read(p):return json.loads(Path(p).read_text())
def official():
    raw=SNAPSHOT.read_text()
    try:rows=json.loads(raw)
    except json.JSONDecodeError:rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
    return {int(r['task_id']):r for r in rows}

def tasks(variant):
    from testbed.env import load_honeypot
    base=load_honeypot('test')
    if variant=='original':return base
    assert variant=='sanitized'
    data=official();out=[]
    for t in base:
        if t.task_id not in data:continue
        row=data[t.task_id]
        # Preserve the function identity: AST traversal of an isclose-wrapped
        # assertion would otherwise mistake math.isclose for the target.
        out.append(replace(t,text=row['prompt'],setup='\n'.join(row.get('test_imports',[])),
                           shown_tests=row['test_list'][:1],original_tests=row['test_list']))
    return out

def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    inventory=read(REPO/'exports/dataset_quality_audit_v1.json')
    ids=inventory['splits']['eval']['retained'];assert len(ids)==63
    configs=[]
    initial=STORE/'answer_pool_replay_20260909/eval_initial/config.json'
    sources=[initial]+[Path(v['path']).parent/'config.json' for v in inventory['endpoints'].values()]
    for i,p in enumerate(sources):
        cfg=read(p)
        name='initial' if i==0 else p.parent.name.removeprefix('eval_')
        for variant in (('original','sanitized') if i%2==0 else ('sanitized','original')):
            configs.append(dict(name=name,variant=variant,adapter=cfg['adapter'],probe_file=cfg['probe_file'],
                audit_bank=cfg['audit_bank'],exclude_task_ids_file=cfg['exclude_task_ids_file'],
                out=str(OUT/f'{name}_{variant}'),lora_id=2050000000+len(configs)*100,
                seed=1234,n=4,max_tokens=2048,layer=21,pool='mean_answer'))
    paths=[Path(__file__),REPO/'scripts/queue_dataset_sanitized_eval.py',SNAPSHOT,
           REPO/'scripts/eval_replay_replication.py',REPO/'scripts/answer_pool_experiment.py',
           REPO/'src/testbed/env.py',REPO/'src/testbed/replay.py',
           REPO/'experiments/judge_labelled_replay_output_20260908/judge_labels_output.py']
    paths+=sources
    paths += [Path(c[k]) for c in configs for k in ('probe_file','audit_bank','exclude_task_ids_file')]
    paths += [p for c in configs for p in Path(c['adapter']).glob('adapter*') if p.is_file()]
    write(ROOT/'manifest.json',dict(task_ids=ids,configs=configs,source_hashes={str(p):digest(p) for p in set(paths)},
        purpose='Paired evaluation intervention, not retraining or removal of prior training exposure.',
        baseline='Fresh original-condition generation on identical 63-task grid; old 120-task evaluations are secondary.',
        monitors='Frozen L21 mean-answer probe with policy and base readers; frozen CoT-only judge after generation.',
        resources={'rollout_gpu':5,'readout_gpu':3,'judge_gpus':[1,3],'rollout_port':8016,'judge_port':8017},
        storage_cap_bytes=5*1024**3,minimum_free_bytes=100*1024**3))
    write(ROOT/'task_ids.json',ids)

def verify():
    m=read(ROOT/'manifest.json')
    assert all(digest(p)==h for p,h in m['source_hashes'].items()),'Frozen audit inputs changed'
    return m

def evaluate(index):
    m=verify();cfg=m['configs'][index]
    import testbed.env as env
    selected=tasks(cfg['variant'])
    env.load_honeypot=lambda split:selected if split=='test' else (_ for _ in ()).throw(ValueError(split))
    import answer_pool_experiment as evaluator
    args=['eval_replay_replication','--adapter',cfg['adapter'],'--out',cfg['out'],
        '--audit-bank',cfg['audit_bank'],'--probe-file',cfg['probe_file'],
        '--task-ids-file',str(ROOT/'task_ids.json'),'--exclude-task-ids-file',cfg['exclude_task_ids_file'],
        '--ports','8016','--n','4','--seed','1234','--max-new-tokens','2048',
        '--layer','21','--pool','mean_answer','--acts-from','policy,base','--lora-id',str(cfg['lora_id'])]
    sys.argv=args;evaluator.evaluate()
    write(Path(cfg['out'])/'dataset_condition.json',dict(variant=cfg['variant'],manifest_sha256=digest(ROOT/'manifest.json'),
        task_definitions=[t.to_dict() for t in selected if t.task_id in m['task_ids']],
        fixed_audit_note='Original fixed audit retained as a reader check; not regenerated or relabelled.'))

def judge(index):
    m=verify();cfg=m['configs'][index];directory=Path(cfg['out'])
    from transformers import AutoTokenizer
    from testbed.rollout_client import RolloutClient
    from train_judge_replay_output import load_output_helper
    helper=load_output_helper()
    rows=[json.loads(s) for s in (directory/'scored.jsonl').read_text().splitlines()]
    public=[{k:r[k] for k in ('task_id','sample_idx','completion')} for r in rows]
    selected={t.task_id:t for t in tasks(cfg['variant'])}
    tokenizer=AutoTokenizer.from_pretrained(helper.JUDGE_MODEL,local_files_only=True)
    client=RolloutClient([8017]);client.wait_ready(max_wait=120)
    labels,raw=helper.label_rows(public,selected,tokenizer,client,view='cot_only',evidence_dir=directory/'cot_judge_evidence')
    assert len(labels)==len(raw)==len(rows)
    # Parse complete raw responses from durable journals, including abstentions.
    outputs=[]
    for p in sorted((directory/'cot_judge_evidence').glob('*.response.json')):
        outputs.extend(item['outputs'][0] for item in read(p))
    assert len(outputs)==len(rows)
    with (directory/'cot_judge_scores.jsonl').open('x') as f:
        for r,output in zip(public,outputs):
            parsed=helper.FROZEN.parse_output(output,helper.parse_explicit)
            f.write(json.dumps(dict(task_id=r['task_id'],sample_idx=r['sample_idx'],**parsed))+'\n')
    write(directory/'judge_complete.json',{'n':len(rows),'view':'cot_only','execution_information_supplied':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--evaluate',type=int);p.add_argument('--judge',type=int)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.evaluate is not None:evaluate(a.evaluate)
    elif a.judge is not None:judge(a.judge)
    else:print(json.dumps({'verified_configs':len(verify()['configs'])}))
