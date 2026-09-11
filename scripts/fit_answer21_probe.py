"""Refit one preselected historical probe on CPU; no sweep or activation writes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from testbed.detectors.probes import LinearProbe
from testbed.metrics import auroc, tpr_at_fpr

REPO = Path(__file__).resolve().parents[1]
STORE = Path('/ssd1/john/probe_playground')
OUT = STORE / 'runs/answer_pool_replay_20260909/metadata'

def main():
    assert os.environ.get('OPENBLAS_NUM_THREADS') == '1' and os.environ.get('OMP_NUM_THREADS') == '1', 'Freeze single-threaded CPU fit'
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'lin_answer_L21_mean_answer.npz'
    if target.exists() or (OUT / 'probe_fit.json').exists():
        raise FileExistsError('Keep existing fitted probe; no overwrite')
    vectors, records, sources = [], [], {}
    for name, directory, capture in [('yud20', STORE/'artifacts/panel_data_yud20/train', 'acts_base'),
                                      ('base', STORE/'artifacts/base_honeypot', 'acts')]:
        scored = directory/'honeypot.scored.jsonl'
        rows = [json.loads(s) for s in scored.read_text().splitlines()]
        features = {}
        for index in sorted((directory/capture).glob('*_index.jsonl')):
            indices = [json.loads(s) for s in index.read_text().splitlines()]
            archive = Path(str(index).replace('_index.jsonl', '_pooled.npz'))
            with np.load(archive) as data:
                layer = data['mean_answer'][:, 21].astype(np.float32)
            assert len(layer) == len(indices)
            for entry, vector in zip(indices, layer):
                key = (entry['family'], entry['task_id'], entry['sample_idx'])
                assert key not in features
                features[key] = vector
            sources[str(archive)] = {'bytes': archive.stat().st_size, 'mtime_ns': archive.stat().st_mtime_ns}
            sources[str(index)] = {'sha256': hashlib.sha256(index.read_bytes()).hexdigest()}
        sources[str(scored)] = {'sha256': hashlib.sha256(scored.read_bytes()).hexdigest()}
        for row in rows:
            if row['parsed'] and not row['truncated']:
                vectors.append(features[(row['family'], row['task_id'], row['sample_idx'])])
                records.append({'source': name, 'task_id': row['task_id'], 'sample_idx': row['sample_idx'],
                                'split': row['split'], 'label': int(row['hack'])})
    X, y = np.stack(vectors), np.array([r['label'] for r in records])
    assert sum(r['split']=='validation' and r['label'] for r in records) < 5
    train = np.array([r['split']=='train' and r['task_id']%5 != 0 for r in records])
    val = np.array([r['split']=='validation' or (r['split']=='train' and r['task_id']%5==0) for r in records])
    assert (int(train.sum()), int(y[train].sum()), int(val.sum()), int(y[val].sum())) == (2098,407,804,80)
    assert not {r['task_id'] for r,m in zip(records,train) if m} & {r['task_id'] for r,m in zip(records,val) if m}
    probe = LinearProbe(.05).fit(X[train], y[train])
    scores = probe.score(X[val])
    auc, (tpr, threshold) = auroc(scores,y[val]), tpr_at_fpr(scores,y[val])
    with (STORE/'artifacts/detectors/panel_yud20/linear_sweep.csv').open() as stream:
        expected = next(r for r in csv.DictReader(stream) if r['pool']=='mean_answer' and r['layer']=='21')
    print(json.dumps({'refit_auroc':auc, 'refit_tpr1':tpr, 'historical':expected}),flush=True)
    assert abs(auc-float(expected['val_auroc'])) < .002, (auc, expected)
    assert abs(tpr-float(expected['val_tpr1'])) <= .02500001, (tpr, expected)
    probe.save(target)
    result = {'status':'complete', 'layer':21, 'pool':'mean_answer', 'C':.05,
              'train_n':2098,'train_hacks':407,'validation_n':804,'validation_hacks':80,
              'validation_auroc':auc,'validation_tpr_at_1fpr':tpr,'validation_threshold_at_1fpr':threshold,
              'selection':'Preselected historical validation TPR winner; not an independent confirmation',
              'negative_class':'All included non-hacks, including other failures',
              'historical_validation':expected, 'cpu_threads':1,
              'numerical_check':'AUROC within .002 and TPR within two of 80 positives; refit not byte-identical to historical fit',
              'probe_sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'sources':sources,
              'fit_rows':[r for r,m in zip(records,train) if m],
              'validation_rows':[r for r,m in zip(records,val) if m]}
    with (OUT/'probe_fit.json').open('x') as stream: json.dump(result,stream,indent=1)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','fit_rows','validation_rows')}),flush=True)

if __name__=='__main__': main()
