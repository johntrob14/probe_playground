# Reproduction and publication boundary

## What works without the original server

`python analysis/summarize_results.py results/core_metrics.json` renders the 12-cell primary results table using only the standard library. The aggregate file retains endpoint outcome counts, probe metrics, CoT-judge decisions and paired task-bootstrap intervals from the completed audit. It excludes raw transcripts, filesystem identities and private application material. It is an extract of measured results, not an independent recomputation from raw outputs.

The CPU tests validate table coverage and malformed-input handling, and exercise the existing task-cross-fitted threshold calibration code with synthetic examples:

```bash
python -m pip install -r requirements-analysis.txt
python -m pytest tests/test_public_results.py tests/test_answer_controls_analysis.py
python scripts/check_publication.py
```

The publication checker verifies the explicit file list, checks for selected credential patterns and oversized files, parses Python sources and compares Git-visible files with the list. It does not stage, delete or upload anything. Passing it is not a comprehensive security or licensing review.

## Training source and required external artifacts

The report's entry points are `scripts/answer_pool_experiment.py`, `scripts/answer_pool_controls.py` and `scripts/answer_replication.py`; their `run_answer_*.py` companions coordinate stages. `scripts/fit_answer21_probe.py` fits the answer-pooled readout. `analysis/dataset_quality_audit.py` computes cap-first outcomes and paired task intervals; `analysis/answer_controls_analysis.py` implements the original calibration diagnostic. These source files are preserved byte-for-byte, including their historical imports.

These are frozen experimental programs, **not a portable one-command training application**. They require original artifact bundles, source/configuration hashes, explicit server process identities and, in several files, the original `/ssd1/john/probe_playground` layout. `TESTBED_STORE` configures the shared package's store, but does not relocate all frozen launchers. Merely setting it does not reproduce the study. Do not launch old queues on a new machine or assume their recorded GPU assignments are available.

Exact reproduction requires separately supplied and verified:

- Qwen3-8B and Llama-3.3-70B-Instruct model access, under their respective licenses.
- The hacking-capable `A_yud/step_20` adapter and original model/tokenizer versions.
- The fitted layer-21 mean-answer probe, standardization parameters and original fit/validation split.
- Training/evaluation task IDs, cached dataset version, public-task records and fixed audit bank.
- Initial CoT-labelled replay bank, token-preserved initial batches per seed, and parent configuration/source manifests from the preparation chain.
- Endpoint rollout and judge records to independently regenerate the aggregate results.

These data and checkpoints are **not bundled**. No public download location or redistribution permission has been established by this cleanup. The original files remain on the research server; missing dependencies should be reported, not bypassed by disabling hash checks. A machine-independent training release needs a separately reviewed artifact/path migration and a GPU smoke test. That work has not been claimed as complete.

The retained older-named trainers implement arithmetic and validation imported by the current recipe. They are dependency code, not extra experiments endorsed by this release. Unused plotting, watcher and application-writing scripts are excluded from Git. Frozen source files are not refactored because the coordinator hashes and source-based function adapters depend on their bytes.

## Outcome conventions

Each core endpoint has 120 tasks × four samples. Categories are mutually exclusive, with token-limit outputs classified first: cap, terminating strict hack, terminating original-test pass, other. A strict hack passes the shown assertion, fails a stored assertion and satisfies the static heuristic. Finite tests and this heuristic do not establish universal correctness or intent. Probe false-positive rates use passes; the judge can abstain. Task-bootstrap intervals are conditional on each trained policy, not training-seed uncertainty.

The probability penalty is detached policy-gradient reward. Replay is a differentiable classification loss through a frozen readout into policy weights, not imitation or probe refitting. Judge replay labels exclude hidden tests; the original probe fit does not. All four completed seeds, including the unfavorable seed 941, are retained. Planned seed 1063 is not a completed result.

## Safety and release checks

Generated programs are executed by the historical harness. Its subprocess timeout is **not a security sandbox**: use an isolated disposable container or VM without credentials, host mounts or network access for untrusted code. Do not expose the rollout/judge services to an untrusted network.

Private writing/context files, logs, raw data, activation captures, adapters, optimizer states, caches and environment files are excluded by a default-deny `.gitignore`. Ignoring files does not remove previously committed content; this workspace had no tracked files at cleanup time. Check future staged changes before publishing. No Git history was rewritten, and no research result was deleted.

There is no license grant added by this cleanup. Confirm code ownership, third-party attribution and redistribution permissions before choosing a license or uploading artifacts. The code has not been uploaded or published.
