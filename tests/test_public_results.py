import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_summary", ROOT / "analysis/summarize_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def data():
    return json.loads((ROOT / "results/core_metrics.json").read_text())


def test_summary_has_all_cells_and_regression():
    text = module.summary(data())
    assert len(text.splitlines()) == 14
    assert "|941|Penalty + replay|78|235|3|78/78|0/235|" in text
    assert "|719|Task only|472|0|0|285/472|Undefined|" in text


@pytest.mark.parametrize("change", ["coverage", "counts", "rate", "caught", "nan"])
def test_invalid_metrics_rejected(change):
    obj = copy.deepcopy(data())
    m = obj["endpoints"]["eval_answer21_none_s613"]["cohorts"]["all"]
    if change == "coverage": m["probe"]["policy"]["0.5"]["hack_n"] -= 1
    if change == "counts": m["counts"]["other"] += 1
    if change == "rate": m["hack_rate"] = 0.5
    if change == "caught": m["probe"]["policy"]["0.5"]["caught"] = -1
    if change == "nan": m["hack_rate"] = float("nan")
    with pytest.raises(ValueError): module.summary(obj)
