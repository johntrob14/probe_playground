"""Isolated layer21 answer-probe protocol; all generated artifacts live on SSD."""
from __future__ import annotations
import argparse
import ast
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import prepare_judge_replay as old
import train_judge_replay as frozen
import train_judge_replay_output as output

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = 'answer_pool_replay_20260909'
SOURCE_ROOT = REPO/'experiments'/PROTOCOL
ARTIFACT_ROOT = Path('/ssd1/john/probe_playground/runs')/PROTOCOL
ROOT = ARTIFACT_ROOT/'metadata'
PROBE = ROOT/'lin_answer_L21_mean_answer.npz'
RUN_SPECS = ((613,'none'),(613,'cot_answer'),(613,'cot_only'),
             (719,'cot_only'),(719,'cot_answer'),(719,'none'))
RUN_ORDER = tuple(f'answer21_{view}_s{seed}' for seed,view in RUN_SPECS)
GIB = 1024**3
STORAGE_CAP, WRITE_HEADROOM = 24*GIB, 3*GIB
require, digest, read_json, write_new = old.require, old.digest, old.read_json, old.write_new
pinned_servers = old.pinned_servers

def storage_guard(*args, headroom=0, **kwargs):
    require(type(headroom) is int and headroom>=0, 'Invalid headroom')
    used = 0
    for p in ARTIFACT_ROOT.rglob('*'):
        require(not p.is_symlink(), 'Unexpected artifact symlink')
        if p.is_file(): used += p.stat().st_size
    ssd, home = shutil.disk_usage(ARTIFACT_ROOT).free, shutil.disk_usage(REPO).free
    require(used+headroom<=STORAGE_CAP and ssd-headroom>=100*GIB and home>=10*GIB,
            'Storage reserve reached; preserve all files')
    return {'artifact_bytes':used,'ssd_free_bytes':ssd,'home_free_bytes':home,'write_headroom_bytes':headroom}

def previous_config(seed, view):
    return old.ROOT/f'config_prob_{"none" if view=="none" else "judge_replay"}_s{seed}.json'

def expected_config(seed, view):
    path = previous_config(seed,view)
    cfg = read_json(path)
    frozen.validate_config(cfg)
    name = f'answer21_{view}_s{seed}'
    bank = None if view=='none' else output.ROOT/f'replay_bank_{view}.jsonl'
    cfg.update(arm=f'answer21_{view}',replication_protocol=PROTOCOL,trainer='answer_pool_experiment.py',
               experiment_root=str(ARTIFACT_ROOT),artifact_root=str(ARTIFACT_ROOT),metadata_root=str(ROOT),
               run_dir=str(ARTIFACT_ROOT/name),layer=21,pool='mean_answer',probe_file=str(PROBE),
               replay_label_source='none' if view=='none' else 'judge_output',judge_view=view,
               replay_bank=str(bank) if bank else None,replay_bank_sha256=digest(bank) if bank else None,
               hack_min=.9,honest_max=0.,storage_budget_bytes=STORAGE_CAP,
               lora_id_base=1_910_000_000+RUN_ORDER.index(name)*1_000_000,
               old_config_path=str(path),old_config_sha256=digest(path),
               output_manifest=str(ROOT/'manifest.json'),answer_empty_logit=20.)
    return cfg

def sources():
    paths = {Path(__file__), SOURCE_ROOT/'PLAN.md', REPO/'scripts/fit_answer21_probe.py',
             REPO/'scripts/run_answer_pool.py', PROBE, ROOT/'probe_fit.json',
             output.ROOT/'judge_labels_output.py', output.ROOT/'manifest.json'}
    # Preserve the exact dependencies actually frozen by the previous label study.
    previous = read_json(output.ROOT/'manifest.json')
    paths.update(Path(p) for p in previous['source_sha256'])
    paths.update(Path(p) for p in previous['config_sha256'])
    for seed,view in RUN_SPECS:
        cfg = expected_config(seed,view)
        paths.update(Path(cfg[k]) for k in ('old_config_path','first_batch_file','public_tasks_file',
                                           'train_task_ids','fresh_test_task_ids','audit_bank','source_manifest'))
        if cfg['replay_bank']: paths.add(Path(cfg['replay_bank']))
    return {str(p.resolve()):digest(p) for p in sorted(paths)}

def freeze():
    require(not (ROOT/'manifest.json').exists(), 'Manifest already exists')
    fitted = read_json(ROOT/'probe_fit.json')
    require(fitted['status']=='complete' and fitted['probe_sha256']==digest(PROBE)
            and fitted['layer']==21 and fitted['pool']=='mean_answer', 'Probe fit incomplete')
    configs = {name:expected_config(*spec) for name,spec in zip(RUN_ORDER,RUN_SPECS)}
    for cfg in configs.values():
        frozen.read_locked_first_batch(read_json(cfg['old_config_path']))
        frozen.load_replay_bank(cfg,set(read_json(cfg['train_task_ids'])))
    servers = pinned_servers()
    storage_guard(headroom=WRITE_HEADROOM)
    source_hashes = sources()
    for name,cfg in configs.items(): write_new(ROOT/f'config_{name}.json',cfg)
    manifest = {'schema_version':1,'status':'frozen_before_training','replication_protocol':PROTOCOL,
                'run_order':list(RUN_ORDER),'configs':configs,'source_sha256':source_hashes,
                'config_sha256':{str(ROOT/f'config_{n}.json'):digest(ROOT/f'config_{n}.json') for n in RUN_ORDER},
                'rollout_server':servers['rollout'],'judge_server':servers['judge'],
                'evaluation_lora_ids':{n:1_920_000_000+i*100 for i,n in enumerate(RUN_ORDER)},
                'initial_lora_id':1_921_000_000,'initial_evaluation':str(ARTIFACT_ROOT/'eval_initial'),
                'probe':fitted,'storage_budget_bytes':STORAGE_CAP,'metadata_on_ssd':True}
    write_new(ROOT/'manifest.json',manifest)
    return manifest

def validate_manifest(manifest=None, root=ROOT, check_server=False):
    require(Path(root).resolve()==ROOT, 'Wrong metadata root')
    saved = read_json(ROOT/'manifest.json')
    require(manifest is None or manifest==saved,'Manifest differs')
    require(saved['replication_protocol']==PROTOCOL and saved['run_order']==list(RUN_ORDER), 'Protocol changed')
    require(saved['source_sha256']==sources(),'Frozen sources changed')
    configs = {name:expected_config(*spec) for name,spec in zip(RUN_ORDER,RUN_SPECS)}
    require(saved['configs']==configs, 'Frozen configs changed')
    for n,cfg in configs.items():
        p=ROOT/f'config_{n}.json'
        require(read_json(p)==cfg and saved['config_sha256'][str(p)]==digest(p),'Config file changed')
    if check_server: pinned_servers({'rollout':saved['rollout_server'],'judge':saved['judge_server']})
    return configs

def validate_config(cfg):
    name=Path(cfg['run_dir']).name
    require(validate_manifest().get(name)==cfg,'Undeclared trainer configuration')
    require(frozen.sha256_file(frozen.__file__)==output.FROZEN_TRAINER_SHA256,'Frozen trainer changed')
    require(not os.environ.get('TESTBED_SYSTEM_SUFFIX'),'Ambient prompt suffix forbidden')
    return ARTIFACT_ROOT,Path(cfg['run_dir'])

def clone_function(function, replacements, namespace=None):
    """Count-checked edits to an isolated function, never its frozen source file."""
    source=inspect.getsource(function)
    for before,after,count in replacements:
        require(source.count(before)==count,f'Unexpected source pattern: {before}')
        source=source.replace(before,after)
    scope=dict(function.__globals__) if namespace is None else dict(namespace)
    exec(compile(source,str(Path(__file__))+':'+function.__name__,'exec'),scope)
    return scope[function.__name__]

_original_logits = None

def answer_token_count(row,end_id):
    tokens=row['completion_token_ids']
    return len(tokens)-tokens.index(end_id)-1 if end_id in tokens else 0

def answer_replay_logits(policy, rows, probe, layer, pool, end_id):
    import torch
    require(layer==21 and pool=='mean_answer','Readout accidentally reverted')
    valid=[]
    for i,row in enumerate(rows):
        row['answer_token_count']=answer_token_count(row,end_id)
        row['answer_span_missing']=row['answer_token_count']==0
        if row['answer_token_count']: valid.append(i)
    result=torch.full((len(rows),),20.,dtype=torch.float32,device=policy.device)
    if valid:
        logits=_original_logits(policy,[rows[i] for i in valid],probe,layer,pool,end_id)
        result=result.index_copy(0,torch.tensor(valid,device=policy.device),logits)
    return result

def install_readout():
    global _original_logits
    import testbed.replay as replay
    if replay.replay_logits is not answer_replay_logits:
        _original_logits=replay.replay_logits
        replay.replay_logits=answer_replay_logits

def train(cfg):
    validate_config(cfg)
    install_readout()
    arithmetic=SimpleNamespace(**vars(frozen.arithmetic))
    arithmetic.probe_scores=clone_function(frozen.arithmetic.probe_scores,[('"mean_cot"','"mean_answer"',1)])
    def guard(config,additional_bytes=None):
        return storage_guard(headroom=config['write_headroom_bytes'] if additional_bytes is None else additional_bytes)
    arithmetic.storage_guard=guard
    def read_first(config):
        return frozen.read_locked_first_batch(read_json(config['old_config_path']))
    def refresh(*args,**kwargs):
        if kwargs['cfg']['replay_label_source']=='none': return frozen.refresh_candidates(*args,**kwargs)
        return output.refresh_candidates(*args,**kwargs)
    def bound_helper():
        helper=output.load_output_helper()
        calls=0
        def label_rows(*args,**kwargs):
            nonlocal calls
            calls+=1
            return helper.label_rows(*args,view=cfg['judge_view'],
                evidence_dir=Path(cfg['run_dir'])/'judge_evidence'/f'refresh_{calls:02d}',**kwargs)
        return SimpleNamespace(JUDGE_MODEL=helper.JUDGE_MODEL,label_rows=label_rows)
    scope=dict(frozen.__dict__)
    scope.update(validate_config=validate_config,read_locked_first_batch=read_first,arithmetic=arithmetic,
                 refresh_candidates=refresh,load_judge_helper=bound_helper)
    function=clone_function(frozen.train,[('cfg["replay_label_source"] == "judge"',
                                           'cfg["replay_label_source"] == "judge_output"',2),
                                          ('"mean_cot"','"mean_answer"',1)],scope)
    return function(cfg)

def evaluate():
    install_readout()
    import eval_replay_replication as evaluator
    original=evaluator.score_probe
    def score_probe(policy,probe,records,source,args):
        end_id=policy.tok.convert_tokens_to_ids('</think>')
        for row in records:
            row['answer_token_count']=answer_token_count(row,end_id)
            row['answer_span_missing']=row['answer_token_count']==0
        return original(policy,probe,records,source,args)
    evaluator.score_probe=score_probe
    evaluator.main()

def output_evaluator():
    import importlib.util
    source=REPO/'scripts/eval_judge_replay_output.py'
    spec=importlib.util.spec_from_file_location('_answer_pool_output_evaluator',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.prep=sys.modules[__name__]
    module.context=clone_function(module.context,[
        ('require(cfg["replay_label_source"] == LABEL_SOURCE, "Wrong replay label source")',
         'require(cfg["replay_label_source"] in ("none", LABEL_SOURCE), "Wrong replay label source")',1)],vars(module))
    module.configuration=clone_function(module.configuration,[
        ('str(prep.ROOT / "judge_labels_output.py")','str(output.ROOT / "judge_labels_output.py")',1),
        ('digest(prep.ROOT / "judge_labels_output.py")','digest(output.ROOT / "judge_labels_output.py")',1)],
        dict(vars(module),output=output))
    return module

if __name__=='__main__':
    if '--evaluate' in sys.argv:
        sys.argv.remove('--evaluate'); evaluate()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--freeze',action='store_true')
        parser.add_argument('--train',type=Path)
        parser.add_argument('--output-judge',type=Path)
        args=parser.parse_args()
        if args.freeze: print(json.dumps({'frozen':str(ROOT/'manifest.json'),'runs':list(freeze()['configs'])}))
        elif args.train: train(read_json(args.train))
        elif args.output_judge: print(json.dumps(output_evaluator().main(args.output_judge)))
        else: print(json.dumps({'configs':list(validate_manifest()),**storage_guard()}))
