"""Isolated zero-penalty controls; no edits to the frozen answer-probe study."""
import argparse
import json
from pathlib import Path
import sys
import answer_pool_experiment as original

REPO=original.REPO
PROTOCOL='answer_controls_20260909'
ARTIFACT_ROOT=Path('/ssd1/john/probe_playground/runs')/PROTOCOL
ROOT=ARTIFACT_ROOT/'metadata'
SOURCE_ROOT=REPO/'experiments'/PROTOCOL
RUN_SPECS=((613,'task'),(613,'replay'),(719,'replay'),(719,'task'))
RUN_ORDER=tuple(f'answer21_{view}_s{seed}' for seed,view in RUN_SPECS)
STORAGE_CAP=18*1024**3
WRITE_HEADROOM=3*1024**3
require,digest,read_json,write_new=original.require,original.digest,original.read_json,original.write_new
pinned_servers=original.pinned_servers
clone_function=original.clone_function
old=original.old


def expected_config(seed,view):
    prior=original.ROOT/f'config_answer21_{"none" if view=="task" else "cot_answer"}_s{seed}.json'
    cfg=read_json(prior)
    name=f'answer21_{view}_s{seed}'
    cfg.update(arm=f'answer21_{view}',replication_protocol=PROTOCOL,trainer='answer_pool_controls.py',
               experiment_root=str(ARTIFACT_ROOT),artifact_root=str(ARTIFACT_ROOT),metadata_root=str(ROOT),
               run_dir=str(ARTIFACT_ROOT/name),output_manifest=str(ROOT/'manifest.json'),
               storage_budget_bytes=STORAGE_CAP,lora_id_base=1_930_000_000+RUN_ORDER.index(name)*1_000_000,
               parent_config=str(prior),parent_config_sha256=digest(prior))
    cfg['lambda']=0.
    return cfg


def sources():
    previous=read_json(original.ROOT/'manifest.json')
    paths=set(map(Path,previous['source_sha256']))|set(map(Path,previous['config_sha256']))
    paths.update((Path(__file__),REPO/'scripts/run_answer_controls.py',SOURCE_ROOT/'PLAN.md',
                  original.ROOT/'manifest.json',original.ARTIFACT_ROOT/'eval_initial/audit_scored.jsonl'))
    return {str(p.resolve()):digest(p) for p in sorted(paths)}


def freeze():
    require(not (ROOT/'manifest.json').exists(),'Already frozen')
    original.validate_manifest(check_server=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    configs={n:expected_config(*s) for n,s in zip(RUN_ORDER,RUN_SPECS)}
    servers=pinned_servers(); storage_guard(headroom=WRITE_HEADROOM)
    for n,c in configs.items(): write_new(ROOT/f'config_{n}.json',c)
    manifest={'schema_version':1,'replication_protocol':PROTOCOL,'run_order':list(RUN_ORDER),
              'configs':configs,'source_sha256':sources(),
              'config_sha256':{str(ROOT/f'config_{n}.json'):digest(ROOT/f'config_{n}.json') for n in RUN_ORDER},
              'rollout_server':servers['rollout'],'judge_server':servers['judge'],
              'evaluation_lora_ids':{n:1_940_000_000+i*100 for i,n in enumerate(RUN_ORDER)},
              'initial_evaluation':str(original.ARTIFACT_ROOT/'eval_initial'),
              'parent_manifest_sha256':digest(original.ROOT/'manifest.json')}
    write_new(ROOT/'manifest.json',manifest)
    return manifest


def validate_manifest(manifest=None,root=ROOT,check_server=False):
    require(Path(root).resolve()==ROOT,'Wrong root')
    saved=read_json(ROOT/'manifest.json')
    require(manifest is None or saved==manifest,'Manifest differs')
    original.validate_manifest(check_server=False)
    require(saved['source_sha256']==sources(),'Frozen sources changed')
    configs={n:expected_config(*s) for n,s in zip(RUN_ORDER,RUN_SPECS)}
    require(saved['configs']==configs and saved['run_order']==list(RUN_ORDER),'Configs differ')
    for n,c in configs.items():
        path=ROOT/f'config_{n}.json'
        require(read_json(path)==c and digest(path)==saved['config_sha256'][str(path)],'Config changed')
    if check_server: pinned_servers({'rollout':saved['rollout_server'],'judge':saved['judge_server']})
    return configs


def validate_config(cfg):
    require(validate_manifest().get(Path(cfg['run_dir']).name)==cfg,'Undeclared control')
    require(cfg['lambda']==0. and cfg['replay_weight'] in (0.,.05),'Wrong factorial cell')
    require(not original.os.environ.get('TESTBED_SYSTEM_SUFFIX'),'Prompt suffix forbidden')
    return ARTIFACT_ROOT,Path(cfg['run_dir'])


scope=dict(vars(original));scope.update(globals())
storage_guard=original.clone_function(original.storage_guard,[],scope)
scope['storage_guard']=storage_guard
train=original.clone_function(original.train,[],scope)
output_evaluator=original.clone_function(original.output_evaluator,[],scope)

if __name__=='__main__':
    if '--evaluate' in sys.argv:
        sys.argv.remove('--evaluate');original.evaluate()
    else:
        p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--train',type=Path);p.add_argument('--output-judge',type=Path)
        a=p.parse_args()
        if a.freeze: print(json.dumps({'frozen':str(ROOT/'manifest.json'),'runs':list(freeze()['configs'])}))
        elif a.train: train(read_json(a.train))
        elif a.output_judge: print(json.dumps(output_evaluator().main(a.output_judge)))
        else: print(json.dumps({'configs':list(validate_manifest())}))
