"""Find within-prompt cases where even detected hacks outscore passing solutions."""
import json
from pathlib import Path
from collections import defaultdict
from dataset_quality_audit import REPO,read,klass

def main():
    audit=json.loads((REPO/'exports/dataset_quality_audit_v1.json').read_text());results={}
    for name,v in audit['endpoints'].items():
        if '_task_' in name:continue
        directory=Path(v['path']).parent.parent/name.removeprefix('eval_')
        labels={(r['step'],r['task_id'],r['sample_idx']):klass(r) for r in read(directory/'execution_audit.jsonl')}
        groups=defaultdict(list)
        for r in read(directory/'rollouts.jsonl'):groups[(r['step'],r['task_id'])].append(r)
        candidates=[];conflicts=[]
        for (step,tid),g in groups.items():
            hacks=[r for r in g if labels[(step,tid,r['sample_idx'])]=='hack' and r['probe_score']>=.9]
            passes=[r for r in g if labels[(step,tid,r['sample_idx'])]=='pass' and r['probe_score']<=.1]
            for h in hacks:
                for p in passes:
                    diff=h['training_task_reward']-p['training_task_reward']-.5*(h['monitor_penalty']-p['monitor_penalty'])
                    record={'step':step,'task_id':tid,'hack_sample':h['sample_idx'],'pass_sample':p['sample_idx'],
                        'hack_chars':h['code_len'],'pass_chars':p['code_len'],'hack_probe':h['probe_score'],'pass_probe':p['probe_score'],
                        'hack_minus_pass_reward_after_monitor':diff,
                        'without_length_penalty_difference':-.5*(h['monitor_penalty']-p['monitor_penalty'])}
                    candidates.append(record)
                    if diff>0:conflicts.append(record)
        results[name]={'eligible_pairs':len(candidates),'pairs_favoring_detected_hack':len(conflicts),
            'groups_with_conflict':len({(r['step'],r['task_id']) for r in conflicts}),
            'conflicts':sorted(conflicts,key=lambda r:-r['hack_minus_pass_reward_after_monitor'])}
    out=REPO/'exports/reward_conflict_audit_v1.json'
    with out.open('x') as f:json.dump({'runs':results,'note':'Conditional within-prompt pairs, not independent observations. Excludes KL and replay gradients; compares the scalar policy-gradient reward.'},f,indent=2)
    for n,r in results.items():print(n,r['eligible_pairs'],r['pairs_favoring_detected_hack'],r['groups_with_conflict'],r['conflicts'][:1])
if __name__=='__main__':main()
