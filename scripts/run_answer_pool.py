"""Six answer-pool runs and a fresh initialization readout, preserving old workflows."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import answer_pool_experiment as prep

REPO, ROOT, ARTIFACT_ROOT, RUN_ORDER = prep.REPO, prep.ROOT, prep.ARTIFACT_ROOT, prep.RUN_ORDER

def load_runner():
    spec=importlib.util.spec_from_file_location('_answer_pool_coordinator',REPO/'scripts/run_judge_replay_strict.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.ROOT,runner.ARTIFACT_ROOT,runner.RUN_ORDER=ROOT,ARTIFACT_ROOT,RUN_ORDER
    runner.prep_module=lambda:prep
    original_stages=runner.stage_list
    verify_eval=prep.clone_function(runner.OLD.COMMON.verify_evaluation,
                                   [('"layer": 15, "pool": "mean_cot"','"layer": 21, "pool": "mean_answer"',1)])
    # The inherited judge entrypoint also validates its input evaluation.
    runner.OLD.COMMON.verify_evaluation=verify_eval
    def stages(configs):
        directory=ARTIFACT_ROOT/'eval_initial'
        result=[{'name':RUN_ORDER[0],'kind':'initial_eval','marker':directory/'eval.json',
                 'targets':[directory],'log':ROOT/'initial_eval.log'}]
        for stage in original_stages(configs):
            result.append(stage)
            if stage['kind']=='judge':
                dest=ARTIFACT_ROOT/f"eval_{stage['name']}"/'output_monitors'
                result.append({'name':stage['name'],'kind':'output_judge','marker':dest/'complete.json',
                               'targets':[dest],'log':ROOT/f"output_judge_{stage['name']}.log"})
        return result
    def verify(stage,configs,manifest,preparer):
        name,kind=stage['name'],stage['kind']; cfg=configs[name]
        directory=ARTIFACT_ROOT/('eval_initial' if kind=='initial_eval' else f'eval_{name}')
        if kind=='train': return runner.OLD.verify_training(cfg)
        if kind=='audit': return runner.OLD.verify_execution_audit(cfg)
        if kind=='judge':
            with runner.adapted(preparer): return runner.JUDGE.verify(directory,ROOT/'manifest.json')
        if kind=='output_judge': return prep.output_evaluator().verify(directory,ROOT/'manifest.json')
        adapter=cfg['init_adapter'] if kind=='initial_eval' else str(Path(cfg['run_dir'])/'serving_adapter')
        identity=manifest['initial_lora_id'] if kind=='initial_eval' else manifest['evaluation_lora_ids'][name]
        result=verify_eval(directory,cfg,adapter,identity)
        for row in runner.OLD.COMMON.rows(directory/'scored.jsonl'):
            prep.require(type(row.get('answer_token_count')) is int and row['answer_span_missing']==(row['answer_token_count']==0),
                         'Missing answer-span evidence')
            if row['n_completion_tokens'] and row['answer_span_missing']:
                prep.require(row['probe_logit_policy']==20. and row['probe_logit_base']==20.,'Missing-span rule differs')
        if kind!='initial_eval':
            reference={r['key']:r for r in runner.OLD.COMMON.rows(ARTIFACT_ROOT/'eval_initial/audit_scored.jsonl')}
            for row in runner.OLD.COMMON.rows(directory/'audit_scored.jsonl'):
                for field,tolerance in [('probe_base',.0005),('probe_logit_base',.002)]:
                    prep.require(math.isclose(row[field],reference[row['key']][field],abs_tol=tolerance,rel_tol=.0001),
                                 'Answer-probe fixed-base invariance failed')
        return result
    def command(stage,configs,manifest):
        name,kind=stage['name'],stage['kind']; cfg=configs[name]; config=ROOT/f'config_{name}.json'
        if kind=='train': return [sys.executable,'-B',str(REPO/'scripts/answer_pool_experiment.py'),'--train',str(config)]
        if kind in ('audit','judge'): return [sys.executable,'-B',str(Path(__file__)),'--stage',kind,'--config',str(config)]
        if kind=='output_judge': return [sys.executable,'-B',str(REPO/'scripts/answer_pool_experiment.py'),'--output-judge',str(config)]
        adapter=cfg['init_adapter'] if kind=='initial_eval' else str(Path(cfg['run_dir'])/'serving_adapter')
        dest=ARTIFACT_ROOT/('eval_initial' if kind=='initial_eval' else f'eval_{name}')
        identity=manifest['initial_lora_id'] if kind=='initial_eval' else manifest['evaluation_lora_ids'][name]
        cmd=runner.OLD.COMMON.evaluation_command(adapter,dest,cfg,identity)
        cmd[1]=str(REPO/'scripts/answer_pool_experiment.py')
        cmd.insert(2,'--evaluate')
        cmd[cmd.index('--layer')+1]='21'; cmd[cmd.index('--pool')+1]='mean_answer'
        return cmd
    runner.stage_list,runner.verify_stage,runner.command=stages,verify,command
    # The inherited coordinator must classify initial evaluation as GPU0 work.
    runner.main=prep.clone_function(runner.main,[
        ('stage["kind"] in ("train", "eval")','stage["kind"] in ("train", "eval", "initial_eval")',1)],vars(runner))
    return runner

RUNNER=load_runner()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=('audit','judge')); parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.stage:
        prep.require(args.config is not None and not args.run and not args.resume,'Wrong internal stage arguments')
        RUNNER.internal_stage(args.stage,args.config)
    else:
        prep.require(args.config is None,'Unexpected config')
        RUNNER.main(execute=args.run,resume=args.resume)
