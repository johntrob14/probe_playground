"""Compare opposite-label probe scores within task, avoiding between-task difficulty cues."""
from collections import defaultdict
import json
from pathlib import Path
from dataset_quality_audit import REPO,read,klass

def main():
    a=json.loads((REPO/'exports/dataset_quality_audit_v1.json').read_text())
    cohorts={int(k):v for k,v in a['cohort_by_task'].items()}
    result={}
    for name,value in a['endpoints'].items():
        result[name]={}
        for population,filename in [('fresh','scored.jsonl'),('fixed','audit_scored.jsonl')]:
            groups=defaultdict(list)
            for r in read(Path(value['path']).parent/filename):groups[r['task_id']].append(r)
            result[name][population]={}
            for cohort in ('all','unchanged','retained_changed','not_retained'):
                pairs=[];participating=set()
                for tid,rows in groups.items():
                    if cohort!='all' and cohorts[tid]!=cohort:continue
                    hacks=[r for r in rows if klass(r)=='hack'];passes=[r for r in rows if klass(r)=='pass']
                    for h in hacks:
                        for p in passes:
                            pairs.append((h,p));participating.add(tid)
                stats={'tasks_with_both_classes':len(participating),'opposite_label_pairs':len(pairs),'readers':{}}
                for reader in ('policy','base'):
                    differences=[h['probe_'+reader]-p['probe_'+reader] for h,p in pairs]
                    stats['readers'][reader]={'wins':sum(d>0 for d in differences),'ties':sum(d==0 for d in differences),
                        'losses':sum(d<0 for d in differences)}
                result[name][population][cohort]=stats
    with (REPO/'exports/dataset_within_task_probe_v1.json').open('x') as f:json.dump(result,f,indent=2)
    for n,v in result.items():print(n,'fixed unchanged',v['fixed']['unchanged'],'fresh unchanged',v['fresh']['unchanged'])
if __name__=='__main__':main()
