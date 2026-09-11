"""Isolated new-seed extension of the frozen answer-pool recipe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import answer_pool_experiment as original

REPO=original.REPO
PROTOCOL='answer_replication_20260910'
SOURCE_ROOT=REPO/'experiments'/PROTOCOL
ARTIFACT_ROOT=Path('/ssd1/john/probe_playground/runs')/PROTOCOL
ROOT=ARTIFACT_ROOT/'metadata'
STORE=original.old.STORE
SEEDS=(827,941,1063)
ARMS=('task','none','cot_only')
RUN_SPECS=tuple((s,v) for s,views in zip(SEEDS,(ARMS,ARMS[1:]+ARMS[:1],ARMS[2:]+ARMS[:2])) for v in views)
RUN_ORDER=tuple(f'answer21_{v}_s{s}' for s,v in RUN_SPECS)
STORAGE_CAP=40*1024**3
WRITE_HEADROOM=3*1024**3
PROBE=original.PROBE
require,digest,read_json,write_new=original.require,original.digest,original.read_json,original.write_new
clone_function=original.clone_function
PUBLIC_SCORE_PROTOCOL=original.old.PUBLIC_SCORE_PROTOCOL
old=original.old

def base_config():return read_json(original.ROOT/'config_answer21_cot_only_s613.json')

def config_for(seed,view,batch_hash=None):
    prior=original.ROOT/f'config_answer21_{"cot_only" if view=="cot_only" else "none"}_s613.json'
    cfg=read_json(prior);name=f'answer21_{view}_s{seed}'
    batch=ROOT/f'first_batches/seed_{seed}.json'
    cfg.update(seed=seed,arm=f'answer21_{view}',replication_protocol=PROTOCOL,trainer='answer_replication.py',
        experiment_root=str(ARTIFACT_ROOT),artifact_root=str(ARTIFACT_ROOT),metadata_root=str(ROOT),
        run_dir=str(ARTIFACT_ROOT/name),output_manifest=str(ROOT/'manifest.json'),storage_budget_bytes=STORAGE_CAP,
        source_manifest=str(ROOT/'design_manifest.json'),first_batch_file=str(batch),
        first_batch_sha256=batch_hash if batch_hash is not None else digest(batch),
        lora_id_base=2_020_000_000+RUN_ORDER.index(name)*1_000_000,
        parent_config=str(prior),parent_config_sha256=digest(prior))
    cfg.pop('old_config_path');cfg.pop('old_config_sha256')
    if view=='task':cfg['lambda']=0.
    return cfg

def expected_config(arm,seed,batch_hash,design):return config_for(seed,arm,batch_hash)

def expected_request(seed,prompt_ids):
    return dict(prompt_token_ids=prompt_ids,n=8,max_tokens=2048,temperature=1.,top_p=1.,seed=seed*100000+1,
        lora_path=base_config()['init_adapter'],lora_id=2_010_000_000+SEEDS.index(seed)*100)

def sources():
    paths=set(map(Path,original.sources()))
    paths.update((Path(__file__),REPO/'scripts/run_answer_replication.py',REPO/'scripts/queue_answer_replication.py',
        SOURCE_ROOT/'PLAN.md',original.ROOT/'manifest.json',REPO/'scripts/sample_judge_initial.py',
        REPO/'scripts/run_answer_pool.py',REPO/'scripts/run_judge_replay_strict.py',
        original.frozen.METADATA_ROOT/'judge_labels.py'))
    return {str(p.resolve()):digest(p) for p in sorted(paths)}

def freeze_design():
    ROOT.mkdir(parents=True,exist_ok=True);storage_guard(headroom=WRITE_HEADROOM)
    write_new(ROOT/'design_manifest.json',dict(status='frozen_design_before_first_batches',protocol=PROTOCOL,
        seeds=list(SEEDS),run_order=list(RUN_ORDER),source_sha256=sources(),
        configs={n:config_for(*s,batch_hash='0'*64) for n,s in zip(RUN_ORDER,RUN_SPECS)},
        created_at=datetime.now(timezone.utc).isoformat()))

def validate_design(design,check_server=False):
    require(design==read_json(ROOT/'design_manifest.json'),'Design changed')
    require(design['source_sha256']==sources(),'Frozen sources changed')
    require(design['seeds']==list(SEEDS) and design['run_order']==list(RUN_ORDER),'Seed/order changed')
    require(design['configs']=={n:config_for(*s,batch_hash='0'*64) for n,s in zip(RUN_ORDER,RUN_SPECS)},'Recipe changed')
    if check_server:pinned_servers()
    return design

def pinned_servers(expected=None):
    from prepare_rank_replication import server_evidence
    pids=read_json(ROOT/'server_pids.json')
    judge=clone_function(old.judge_server_evidence,[('pid = 60011',f"pid = {int(pids['judge'])}",1)])
    observed={'rollout':server_evidence(pid=pids['rollout']),'judge':judge()}
    if expected:
        for name in observed:
            for key in ('pid','starttime_ticks','argv','cmdline_sha256','cuda_visible_devices','max_model_len'):
                require(observed[name][key]==expected[name][key],f'Pinned {name} changed: {key}')
    return observed

# Adapt only seed/protocol identity; retain strict first-batch schema and scoring checks.
batch_scope=dict(vars(original.frozen),SEEDS=SEEDS,PROTOCOL=PROTOCOL)
validate_first_batch=clone_function(original.frozen.validate_first_batch,
    [('1_800_000_000','2_010_000_000',1)],batch_scope)
original.frozen.validate_first_batch=validate_first_batch
read_locked_first_batch=clone_function(original.frozen.read_locked_first_batch,[],
    dict(vars(original.frozen),validate_first_batch=validate_first_batch))

def freeze():
    validate_design(read_json(ROOT/'design_manifest.json'))
    servers=pinned_servers();configs={n:config_for(*s) for n,s in zip(RUN_ORDER,RUN_SPECS)}
    for n,c in configs.items():
        read_locked_first_batch(c)
        original.frozen.load_replay_bank(c,set(read_json(c['train_task_ids'])))
        write_new(ROOT/f'config_{n}.json',c)
    write_new(ROOT/'manifest.json',dict(schema_version=1,replication_protocol=PROTOCOL,run_order=list(RUN_ORDER),
        configs=configs,source_sha256=sources(),config_sha256={str(ROOT/f'config_{n}.json'):digest(ROOT/f'config_{n}.json') for n in RUN_ORDER},
        rollout_server=servers['rollout'],judge_server=servers['judge'],
        evaluation_lora_ids={n:2_040_000_000+i*100 for i,n in enumerate(RUN_ORDER)},
        initial_evaluation=str(original.ARTIFACT_ROOT/'eval_initial'),design_sha256=digest(ROOT/'design_manifest.json')))

def validate_manifest(manifest=None,root=ROOT,check_server=False):
    require(Path(root).resolve()==ROOT,'Wrong metadata root')
    saved=read_json(ROOT/'manifest.json');require(manifest is None or manifest==saved,'Manifest changed')
    validate_design(read_json(ROOT/'design_manifest.json'))
    require(saved['design_sha256']==digest(ROOT/'design_manifest.json') and saved['source_sha256']==sources(),'Inputs changed')
    configs={n:config_for(*s) for n,s in zip(RUN_ORDER,RUN_SPECS)}
    require(saved['configs']==configs and saved['run_order']==list(RUN_ORDER),'Configs differ')
    for n,c in configs.items():
        p=ROOT/f'config_{n}.json';require(read_json(p)==c and digest(p)==saved['config_sha256'][str(p)],'Config file changed')
    if check_server:pinned_servers({'rollout':saved['rollout_server'],'judge':saved['judge_server']})
    return configs

def validate_config(cfg):
    require(validate_manifest().get(Path(cfg['run_dir']).name)==cfg,'Undeclared run')
    require(not os.environ.get('TESTBED_SYSTEM_SUFFIX'),'Prompt suffix forbidden')
    return ARTIFACT_ROOT,Path(cfg['run_dir'])

scope=dict(vars(original));scope.update(globals())
storage_guard=clone_function(original.storage_guard,[],scope)
scope['storage_guard']=storage_guard
train=clone_function(original.train,[
    ("return frozen.read_locked_first_batch(read_json(config['old_config_path']))","return read_locked_first_batch(config)",1)],scope)
# Use this module even when invoked as __main__; the original implementation's
# sys.modules[__name__] must refer to the extension, not the frozen parent.
scope['__name__']=__name__
_output_evaluator=clone_function(original.output_evaluator,[],scope)
def output_evaluator():
    module=_output_evaluator()
    module.main=clone_function(module.main,[
        ('prep.old.judge_server_evidence()',"prep.pinned_servers()['judge']",1)],vars(module))
    return module

def sample():
    import sample_judge_initial as sampler
    sampler.prep=sys.modules[__name__]
    sampler.main=clone_function(sampler.main,[
        ('prep.ROOT/"public_tasks.json"','Path(prep.base_config()["public_tasks_file"])',1),
        ('prep.pinned_servers({"rollout": design["rollout_server"], "judge": design["judge_server"]})','prep.pinned_servers()',1)],vars(sampler))
    sampler.main(generate=True)

if __name__=='__main__':
    if '--evaluate' in sys.argv:
        sys.argv.remove('--evaluate');original.evaluate()
    else:
        p=argparse.ArgumentParser();p.add_argument('--design',action='store_true');p.add_argument('--sample',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--train',type=Path);p.add_argument('--output-judge',type=Path);a=p.parse_args()
        if a.design:freeze_design()
        elif a.sample:sample()
        elif a.freeze:freeze()
        elif a.train:train(read_json(a.train))
        elif a.output_judge:print(json.dumps(output_evaluator().main(a.output_judge)))
        else:print(json.dumps({'runs':list(validate_manifest())}))
