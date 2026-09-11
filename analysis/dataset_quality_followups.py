"""Additional CPU provenance, fixed-text, and published-artifact checks."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import urllib.request
import hashlib
from dataset_quality_audit import REPO,STORE,read,metrics

def main():
    audit=json.loads((REPO/'exports/dataset_quality_audit_v1.json').read_text())
    cohorts={int(k):v for k,v in audit['cohort_by_task'].items()}
    names=('all','unchanged','retained_changed','not_retained')
    def inside(t,n):return n=='all' or cohorts[int(t)]==n
    result={'probe_fit':{},'fixed_audit':{},'text_interventions':{},'broad_endpoints':{},'public_artifacts':{}}
    fit=json.loads((STORE/'runs/answer_pool_replay_20260909/metadata/probe_fit.json').read_text())
    for split in ('fit_rows','validation_rows'):
        rows=fit[split]
        result['probe_fit'][split]={n:{'n':len(g),'hacks':sum(r['label'] for r in g),
            'tasks':len({r['task_id'] for r in g}),
            'source_counts':dict(Counter(r['source'] for r in g))}
            for n in names for g in [[r for r in rows if inside(r['task_id'],n)]]}
    for name,record in audit['endpoints'].items():
        path=Path(record['path']).parent/'audit_scored.jsonl';rows=read(path)
        result['fixed_audit'][name]={n:metrics([r for r in rows if inside(r['task_id'],n)]) for n in names}
    scores=read(STORE/'runs/answer_text_evasion_20260910/v2/scores.jsonl')
    byid=defaultdict(dict)
    for r in scores:
        assert r['variant'] not in byid[r['id']]
        byid[r['id']][r['variant']]=r
    for source in sorted({r['source'] for r in scores}):
        result['text_interventions'][source]={}
        for n in names:
            result['text_interventions'][source][n]={}
            for variant in sorted({r['variant'] for r in scores}-{'original'}):
                pairs=[(v['original'],v[variant]) for v in byid.values() if 'original' in v and variant in v
                    and v['original']['source']==source and v['original']['category']=='hack'
                    and inside(v['original']['task_id'],n)]
                result['text_interventions'][source][n][variant]={'n':len(pairs),
                    'original_caught':sum(a['score']>=.5 for a,b in pairs),
                    'edited_caught':sum(b['score']>=.5 for a,b in pairs),
                    'detection_gained':sum(a['score']<.5<=b['score'] for a,b in pairs),
                    'detection_lost':sum(b['score']<.5<=a['score'] for a,b in pairs)}
    screen=json.loads((REPO/'exports/task_ambiguity_screen_v1.json').read_text())
    for source in screen['sources']:
        rows=read(source['path'])
        result['broad_endpoints'][source['path']]={n:metrics([r for r in rows if inside(r['task_id'],n)]) for n in names}
    targets={
      'JYudelson1/MonitorDecorrelation':['data/runs/README.md','src/monitordecorrelation/envs/mbpp_honeypot.py','src/monitordecorrelation/mbpp_split.py'],
      'AlignmentResearch/obfuscation-atlas':['obfuscation_atlas/tasks/create_hardcoded_mbpp.py','obfuscation_atlas/scripts/mbpp_validation_prompt.txt']}
    total=0
    for repo,files in targets.items():
        with urllib.request.urlopen('https://api.github.com/repos/'+repo+'/git/trees/main?recursive=1',timeout=30) as f:tree=json.load(f)
        assert not tree['truncated'];sha=tree['sha']
        public={'commit':sha,'files':{},'run_files':[{'path':r['path'],'bytes':r.get('size')} for r in tree['tree'] if r['type']=='blob' and r['path'].startswith('data/runs/')]}
        for file in files:
            url=f'https://raw.githubusercontent.com/{repo}/{sha}/{file}'
            with urllib.request.urlopen(url,timeout=30) as f:raw=f.read(200001)
            assert len(raw)<=200000;total+=len(raw);assert total<1000000
            public['files'][file]={'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
        result['public_artifacts'][repo]=public
    out=REPO/'exports/dataset_quality_followups_v1.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(out)
    print('PROBE FIT',result['probe_fit'])
    for source,groups in result['text_interventions'].items():print('TEXT',source,groups['unchanged'])
    for repo,p in result['public_artifacts'].items():
        print(repo,p['commit'])
        for file,r in p['files'].items():
            if file.endswith('README.md'):print(r['text'])

if __name__=='__main__':main()
