"""Explicit eligibility amendment; preserves the first prepared artifact set."""
from pathlib import Path
import sys
import answer_text_evasion as job

job.OUT = job.OUT / 'v2'
job.prepare = job.original.clone_function(job.prepare, [
    ("if len(matches)!=1 or answer.count('```')!=2:reason='not_single_complete_fence'",
     "if not matches:reason='no_complete_first_fence'", 1),
    ("code_sha256=hashlib.sha256(code.encode()).hexdigest(),reference_base_logit=r['probe_logit_base']",
     "code_sha256=hashlib.sha256(code.encode()).hexdigest(),original_fence_count=item['answer'].count('```'),reference_base_logit=r['probe_logit_base']", 1),
    ("paths=[Path(__file__),ROOT/'PLAN.md'",
     "paths=[Path(__file__),REPO/'scripts/answer_text_evasion_v2.py',ROOT/'ELIGIBILITY_V2.md',ROOT/'PLAN.md'", 1),
], vars(job))

if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:job.prepare()
    elif sys.argv[1:] == ['--run']:job.wait_and_score()
    else:raise SystemExit('Specify --prepare or --run')
