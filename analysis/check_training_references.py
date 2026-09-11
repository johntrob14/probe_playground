"""Validate executable reference/test consistency; not specification correctness."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
from testbed.env import load_honeypot,run_tests,DATA
import dataset_training_control as control

def main():
    allids=set(control.read(control.REPO/'experiments/replay_pilot_20260904/train_task_ids.json'))
    full=pd.read_parquet(DATA/'mbpp-hardcode/data/train-00000-of-00001.parquet').set_index('task_id')
    official=control.evaluation.official()
    candidates=[('original',t,full.loc[t.task_id]['code']) for t in load_honeypot('train') if t.task_id in allids]
    candidates += [('sanitized',t,official[t.task_id]['code']) for t in control.public_tasks('sanitized')]
    def score(item):
        variant,t,code=item
        return {'variant':variant,'task_id':t.task_id,'tests_pass':run_tests(code,t.original_tests,t.setup,3.)}
    with ThreadPoolExecutor(8) as pool:rows=list(pool.map(score,candidates))
    out=control.REPO/'exports/training_reference_checks_v1.json'
    with out.open('x') as f:json.dump({'rows':rows,'note':'Passing reference tests establishes consistency with stored tests, not clarity or correctness of the task specification.'},f,indent=2)
    print('Checked',len(rows),'failed',[r for r in rows if not all(r['tests_pass'])])
if __name__=='__main__':main()
