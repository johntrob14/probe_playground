"""Reuse the verified answer coordinator with fresh paths and old init audit."""
from pathlib import Path
source=Path(__file__).with_name('run_answer_pool.py').read_text()
edits=[('import answer_pool_experiment as prep','import answer_replication as prep',1),
    ("REPO/'scripts/answer_pool_experiment.py'","REPO/'scripts/answer_replication.py'",3),
    ("result=[{'name':RUN_ORDER[0],'kind':'initial_eval','marker':directory/'eval.json',\n                 'targets':[directory],'log':ROOT/'initial_eval.log'}]","result=[]",1),
    ("ARTIFACT_ROOT/'eval_initial/audit_scored.jsonl'","prep.original.ARTIFACT_ROOT/'eval_initial/audit_scored.jsonl'",1)]
for before,after,count in edits:
    assert source.count(before)==count,(before,source.count(before))
    source=source.replace(before,after)
exec(compile(source,__file__+':adapted','exec'),globals())
