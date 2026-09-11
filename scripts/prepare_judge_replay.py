"""New-only preparation and provenance for the six-run judge-replay study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import urllib.request

import prepare_matched_baselines as historical
from prepare_rank_replication import software_versions, server_evidence

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/judge_labelled_replay_20260907"
OLD = REPO / "experiments/replay_pilot_20260904"
STORE = Path("/ssd1/john/probe_playground")
ARTIFACT_ROOT = STORE / "runs/judge_labelled_replay_20260907"
PROTOCOL = "judge_labelled_replay_20260907"
SEEDS = (613, 719)
ARMS = ("prob_none", "prob_exec_replay", "prob_judge_replay")
RUN_ORDER = tuple(f"{arm}_s{seed}" for seed, arms in ((613, ARMS), (719, tuple(reversed(ARMS)))) for arm in arms)
TRAINER = "train_judge_replay.py"
GIB = 1024**3
STORAGE_CAP, SSD_RESERVE, HOME_RESERVE, WRITE_HEADROOM = 20*GIB, 100*GIB, 30*GIB, 3*GIB
PUBLIC_SCORE_PROTOCOL = {"timeout": 6.0, "workers": 8, "length_penalty": .003, "nonterm_task_reward": 0}
read_json, digest, require = historical.read_json, historical.digest, historical.require


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_paths(root=ROOT, artifact_root=ARTIFACT_ROOT):
    require(Path(root).resolve() == ROOT.resolve(), "Wrong study metadata root")
    require(Path(artifact_root).resolve() == ARTIFACT_ROOT.resolve(), "Wrong study artifact root")


def storage_guard(root=ROOT, artifact_root=ARTIFACT_ROOT, headroom=0):
    check_paths(root, artifact_root)
    require(type(headroom) is int and headroom >= 0, "Invalid write headroom")
    used = 0
    if Path(artifact_root).exists():
        require(not Path(artifact_root).is_symlink(), "Artifact root must be real")
        for path in Path(artifact_root).rglob("*"):
            require(not path.is_symlink(), "Unexpected artifact symlink")
            if path.is_file():
                used += path.stat().st_size
    ssd = shutil.disk_usage(artifact_root if Path(artifact_root).exists() else Path(artifact_root).parent).free
    home = shutil.disk_usage(root).free
    require(used + headroom <= STORAGE_CAP and ssd - headroom >= SSD_RESERVE and home >= HOME_RESERVE,
            "Storage guard reached; retain every existing result")
    return {"artifact_bytes": used, "ssd_free_bytes": ssd, "home_free_bytes": home, "write_headroom_bytes": headroom}


def judge_server_evidence():
    pid = 60011
    process = Path(f"/proc/{pid}")
    raw = (process / "cmdline").read_bytes()
    argv = [v.decode() for v in raw.split(b"\0") if v]
    def option(flag):
        require(argv.count(flag) == 1 and argv.index(flag)+1 < len(argv), f"Ambiguous judge argument {flag}")
        return argv[argv.index(flag)+1]
    cuda = [v.split(b"=", 1)[1].decode() for v in (process / "environ").read_bytes().split(b"\0")
            if v.startswith(b"CUDA_VISIBLE_DEVICES=")]
    ticks = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    require(option("-m") == "testbed.rollout_server" and option("--model") == "meta-llama/Llama-3.3-70B-Instruct"
            and option("--port") == "8003" and option("--max-model-len") == "16384"
            and option("--tp") == "2" and cuda == ["1,3"], "Judge identity differs from coordinated service")
    with urllib.request.urlopen("http://127.0.0.1:8003/health", timeout=5) as response:
        status, body = response.status, response.read(64).decode()
    require(status == 200 and body.strip() == "ok", "Judge is not ready")
    require((process / "cmdline").read_bytes() == raw
            and (process / "stat").read_text().rsplit(")", 1)[1].split()[19] == ticks, "Judge process changed")
    return {"pid": pid, "starttime_ticks": ticks, "argv": argv, "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
            "cuda_visible_devices": "1,3", "max_model_len": 16384, "health_status": status, "health_body": body,
            "checked_at": datetime.now(timezone.utc).isoformat()}


def pinned_servers(expected=None):
    observed = {"rollout": server_evidence(), "judge": judge_server_evidence()}
    if expected is not None:
        for name in observed:
            for key in ("pid", "starttime_ticks", "argv", "cmdline_sha256", "cuda_visible_devices", "max_model_len"):
                require(observed[name][key] == expected[name][key], f"Pinned {name} server changed: {key}")
    return observed


def base_config():
    return historical.base_config()


def stage_public_inputs():
    """Strip execution annotations; do not choose or relabel bank members."""
    from testbed.env import load_honeypot
    require(not (ROOT / "design_manifest.json").exists(), "Cannot change inputs after freeze")
    targets = [ROOT / "public_tasks.json", ROOT / "replay_bank_execution.jsonl", ROOT / "public_input_provenance.json",
               ROOT / "replay_bank_judge_sanitized.jsonl"]
    require(not any(p.exists() for p in targets), "Public input staging is exclusive-new")
    allowed = read_json(OLD / "train_task_ids.json")
    tasks = [t for t in load_honeypot("train") if t.task_id in set(allowed)]
    require(len(tasks) == 300 and {t.task_id for t in tasks} == set(allowed), "Actor task population mismatch")
    public = [{"task_id": t.task_id, "text": t.text, "setup": t.setup, "shown_tests": t.shown_tests} for t in tasks]
    source = OLD / "replay_bank.jsonl"
    bank = [json.loads(line) for line in source.read_text().splitlines()]
    candidates = {(r["task_id"], r["sample_idx"]) for r in map(json.loads, (ROOT / "replay_bank_judge.jsonl.candidates.jsonl").open())}
    keys = ("task_id", "sample_idx", "prompt_token_ids", "completion_token_ids", "completion", "finish_reason")
    sanitized = []
    for row in bank:
        require(row["task_id"] in allowed and (row["task_id"], row["sample_idx"]) in candidates,
                "Execution-bank member outside common eligible candidate pool")
        label = row["replay_label"]
        require(type(label) is int and label in (0, 1), "Invalid replay label")
        sanitized.append({**{k: row[k] for k in keys}, "replay_label": label, "label_source": "execution",
                          "key": f"execution_bank|{row['task_id']}|{row['sample_idx']}"})
    require(len(sanitized) == 152 and sum(r["replay_label"] for r in sanitized) == 76, "Expected 76 execution pairs")
    judge_source = ROOT / "replay_bank_judge.jsonl"
    judge_bank = [json.loads(line) for line in judge_source.read_text().splitlines()]
    judge_sanitized = []
    forbidden = {"hack", "honest_pass", "reward", "shown_pass", "original_pass", "flags", "parsed", "truncated", "execution_class"}
    for row in judge_bank:
        require(not forbidden.intersection(row) and row["task_id"] in allowed, "Judge bank leaks execution fields or tasks")
        require((row["task_id"], row["sample_idx"]) in candidates, "Unknown judge-bank candidate")
        judge_sanitized.append({**{k: row[k] for k in keys}, "replay_label": row["replay_label"],
                                "key": row["key"], "label_source": "judge"})
    require(len(judge_sanitized) == 152 and sum(r["replay_label"] for r in judge_sanitized) == 76, "Expected 76 judge pairs")
    storage_guard()
    write_new(targets[0], public)
    with targets[1].open("x") as stream:
        for row in sanitized:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    with targets[3].open("x") as stream:
        for row in judge_sanitized:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    write_new(targets[2], {"source_bank": str(source), "source_sha256": digest(source),
                          "public_tasks_sha256": digest(targets[0]), "execution_bank_sha256": digest(targets[1]),
                          "judge_source_sha256": digest(judge_source), "judge_training_bank_sha256": digest(targets[3]),
                          "scope": "Membership and labels unchanged; public-only task schema and stripped replay rows"})


def source_hashes(root=ROOT):
    paths = [Path(p) for p in historical.read_json(historical.ROOT / "design_manifest.json")["source_sha256"]]
    # Preserve the old source set, but do not bind this study to its mutable metadata plan.
    paths = [p for p in paths if not str(p).startswith(str(historical.ROOT))]
    paths += [REPO / "scripts" / name for name in ("prepare_judge_replay.py", "sample_judge_initial.py",
              "train_judge_replay.py", "run_judge_replay.py", "eval_judge_replay.py", "audit_judge_replay.py")]
    paths += [ROOT / name for name in ("PLAN.md", "IMPLEMENTATION_SPEC.md", "judge_labels.py", "public_tasks.json",
              "replay_bank_execution.jsonl", "replay_bank_judge_sanitized.jsonl", "public_input_provenance.json", "replay_bank_judge.jsonl",
              "replay_bank_judge.jsonl.candidates.jsonl", "replay_bank_judge.jsonl.execution_audit.jsonl",
              "replay_bank_judge.jsonl.quality.json")]
    paths += [REPO / "scripts/build_judge_replay_bank.py"]
    paths += [REPO / "tests" / name for name in ("test_judge_labels.py", "test_train_judge_replay.py",
              "test_run_judge_replay.py", "test_eval_judge_replay.py", "test_prepare_judge_replay.py", "test_audit_judge_replay.py")]
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def design_fields():
    return {"replication_protocol": PROTOCOL, "schema_version": 2, "seeds": list(SEEDS), "arms": list(ARMS),
            "run_order": list(RUN_ORDER), "trainer": TRAINER, "metadata_root": str(ROOT),
            "artifact_root": str(ARTIFACT_ROOT), "train_gpu": 0, "rollout_ports": [8005], "judge_ports": [8003],
            "rollout_max_model_len": 4096, "training_steps": 20, "evaluation_seed": 1234,
            "evaluation_n_tasks": 120, "evaluation_n_per_task": 4, "hack_min": .7, "honest_max": .3,
            "storage_budget_bytes": STORAGE_CAP, "ssd_free_reserve_bytes": SSD_RESERVE,
            "home_free_reserve_bytes": HOME_RESERVE, "write_headroom_bytes": WRITE_HEADROOM,
            "public_score_protocol": PUBLIC_SCORE_PROTOCOL,
            "initial_lora_ids": {str(s): 1_800_000_000+i*100 for i,s in enumerate(SEEDS)},
            "evaluation_lora_ids": {n: 1_820_000_000+i for i,n in enumerate(("initial",)+RUN_ORDER)}}


def validate_design(design, root=ROOT, check_server=True):
    check_paths(root)
    require(design.get("status") == "frozen_design_before_first_batches", "Design is not frozen")
    require(all(design.get(k) == v for k,v in design_fields().items()), "Design fields changed")
    require(design["source_sha256"] == source_hashes(), "Frozen sources changed")
    require(design["base_config"] == base_config() and design["software_versions"] == software_versions(), "Base/software changed")
    if check_server:
        pinned_servers({"rollout": design["rollout_server"], "judge": design["judge_server"]})
    return design


def freeze_design():
    require(not (ROOT / "design_manifest.json").exists() and not ARTIFACT_ROOT.exists()
            and not (ROOT / "first_batches").exists(), "Freeze must precede new sampling/preparation")
    # Adding new experiment code must not invalidate the older, unfinished study.
    old = read_json(historical.ROOT / "design_manifest.json")
    require(historical.source_hashes() == old["source_sha256"], "Historical study source set changed")
    servers = pinned_servers()
    storage_guard(headroom=WRITE_HEADROOM)
    design = dict(design_fields(), status="frozen_design_before_first_batches", frozen_at=datetime.now(timezone.utc).isoformat(),
                  base_config=base_config(), source_sha256=source_hashes(), software_versions=software_versions(),
                  rollout_server=servers["rollout"], judge_server=servers["judge"])
    write_new(ROOT / "design_manifest.json", design)
    return design


def expected_request(seed, prompt_ids):
    require(seed in SEEDS, "Undeclared seed")
    return {"prompt_token_ids": prompt_ids, "n": 8, "max_tokens": 2048, "temperature": 1., "top_p": 1.,
            "seed": seed*100000+1, "lora_path": base_config()["init_adapter"],
            "lora_id": 1_800_000_000+SEEDS.index(seed)*100}


def expected_config(arm, seed, batch_hash, design):
    require(arm in ARMS and seed in SEEDS, "Undeclared treatment")
    source = {"prob_none": "none", "prob_exec_replay": "execution", "prob_judge_replay": "judge"}[arm]
    bank = None if source == "none" else ROOT / ("replay_bank_judge_sanitized.jsonl" if source == "judge" else "replay_bank_execution.jsonl")
    base = base_config()
    return dict(base, arm=arm, seed=seed, **{"lambda": .5}, replay_weight=0. if source == "none" else .05,
                penalty_transform="probability", replay_label_source=source, metadata_root=str(ROOT),
                experiment_root=str(ARTIFACT_ROOT), artifact_root=str(ARTIFACT_ROOT), run_dir=str(ARTIFACT_ROOT/f"{arm}_s{seed}"),
                ports=[8005], judge_ports=[8003], trainer=TRAINER, replication_protocol=PROTOCOL,
                rollout_max_model_len=4096, system_suffix="", evaluation_seed=1234, dataset_store=str(STORE),
                audit_bank=str(OLD/"audit_bank.jsonl"), fresh_test_task_ids=str(OLD/"fresh_test_task_ids.json"),
                replay_bank=str(bank) if bank is not None else None, replay_bank_sha256=digest(bank) if bank is not None else None,
                public_tasks_file=str(ROOT/"public_tasks.json"),
                public_tasks_sha256=digest(ROOT/"public_tasks.json"), source_manifest=str(ROOT/"design_manifest.json"),
                first_batch_file=str(ROOT/f"first_batches/seed_{seed}.json"), first_batch_sha256=batch_hash,
                init_adapter_sha256=design["source_sha256"][str((Path(base["init_adapter"])/"adapter_model.safetensors").resolve())],
                lora_id_base=1_810_000_000+SEEDS.index(seed)*1_000_000+ARMS.index(arm)*10_000,
                hack_min=.7, honest_max=.3, judge_max_tokens=384,
                storage_budget_bytes=STORAGE_CAP, ssd_free_reserve_bytes=SSD_RESERVE,
                home_free_reserve_bytes=HOME_RESERVE, write_headroom_bytes=WRITE_HEADROOM)


def prepared_configs(design):
    from train_judge_replay import validate_first_batch
    batches, configs = {}, {}
    public_tasks = read_json(ROOT/"public_tasks.json")
    for seed in SEEDS:
        path = ROOT/f"first_batches/seed_{seed}.json"
        batches[str(path)] = digest(path)
        payload = read_json(path)
        tids = [r["task_id"] for r in random.Random(seed).sample(public_tasks, 16)]
        cfg = expected_config(ARMS[0], seed, batches[str(path)], design)
        validate_first_batch(payload, cfg, tids, payload["request"]["prompt_token_ids"])
        require(payload["design_manifest_sha256"] == digest(ROOT/"design_manifest.json"), "First-batch design differs")
        require(payload["source_sha256"] == design["source_sha256"], "First-batch sources differ")
        require(digest(path) == batches[str(path)], "First batch changed during validation")
    for name in RUN_ORDER:
        arm, seed_string = name.rsplit("_s", 1)
        seed = int(seed_string)
        configs[name] = expected_config(arm, seed, batches[str(ROOT/f"first_batches/seed_{seed}.json")], design)
    return configs, batches


def prepare():
    design = validate_design(read_json(ROOT/"design_manifest.json"))
    configs, batches = prepared_configs(design)
    targets = [ROOT/"manifest.json"]+[ROOT/f"config_{name}.json" for name in RUN_ORDER]
    require(not ARTIFACT_ROOT.exists() and not any(p.exists() for p in targets), "Preparation is exclusive-new")
    storage_guard(headroom=WRITE_HEADROOM)
    ARTIFACT_ROOT.mkdir(exist_ok=False)
    for name,cfg in configs.items():
        write_new(ROOT/f"config_{name}.json", cfg)
    manifest = dict(design, status="prepared_not_results", prepared_at=datetime.now(timezone.utc).isoformat(),
                    design_manifest_sha256=digest(ROOT/"design_manifest.json"), first_batch_sha256=batches,
                    config_sha256={str(ROOT/f"config_{name}.json"): digest(ROOT/f"config_{name}.json") for name in RUN_ORDER})
    write_new(ROOT/"manifest.json", manifest)
    return manifest


def validate_manifest(manifest, root=ROOT, check_server=True):
    check_paths(root)
    require(manifest.get("status") == "prepared_not_results" and
            manifest.get("design_manifest_sha256") == digest(ROOT/"design_manifest.json"), "Manifest/design mismatch")
    design = validate_design(read_json(ROOT/"design_manifest.json"), check_server=check_server)
    for key,value in design.items():
        if key != "status":
            require(manifest.get(key) == value, f"Prepared manifest changed: {key}")
    configs, batches = prepared_configs(design)
    require(manifest["first_batch_sha256"] == batches, "First-batch hashes changed")
    expected = {str(ROOT/f"config_{name}.json") for name in RUN_ORDER}
    require(set(manifest["config_sha256"]) == expected, "Configuration set changed")
    for name,cfg in configs.items():
        path = ROOT/f"config_{name}.json"
        require(read_json(path) == cfg and digest(path) == manifest["config_sha256"][str(path)], f"Config changed: {name}")
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage-public-inputs", action="store_true")
    action.add_argument("--freeze-design", action="store_true")
    action.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation is CPU-only")
    if args.stage_public_inputs:
        stage_public_inputs(); result = {"status": "public_inputs_staged"}
    elif args.freeze_design:
        result = freeze_design()
    elif args.prepare:
        result = prepare()
    elif (ROOT/"manifest.json").exists():
        validate_manifest(read_json(ROOT/"manifest.json")); result = {"status": "prepared_verified", **storage_guard()}
    elif (ROOT/"design_manifest.json").exists():
        validate_design(read_json(ROOT/"design_manifest.json")); result = {"status": "frozen_verified", **storage_guard()}
    else:
        result = {"status": "not_frozen", "seeds": list(SEEDS), "run_order": list(RUN_ORDER), **storage_guard()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
