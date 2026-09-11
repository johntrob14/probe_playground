# Maintaining an activation monitor during policy optimization

Research code for comparing a probability penalty from an answer-activation probe with the same penalty plus chain-of-thought-labelled replay. The completed study uses Qwen3-8B, an MBPP public-test coding reward, and four matched training seeds. A frozen CoT-only text judge supplies replay labels; the frozen activation probe supplies the policy penalty.

Replay preserves detection at the original probe threshold more consistently than it improves behavior: hacking decreases in three seeds and increases in the fourth. Strong score discrimination and retrospective recalibration limit claims of information loss. This repository does not establish that replay prevents all obfuscation or outperforms online recalibration.

## Start with the results

Python 3.12 or newer is required. This command uses only the standard library:

```bash
python analysis/summarize_results.py results/core_metrics.json
```

It renders the complete four-seed comparison, including task-only baselines and the unfavorable seed 941. The small aggregate file contains measured endpoint counts and monitor metrics, not raw training examples. See [reproduction notes](REPRODUCING.md) for the outcome definitions, artifact requirements and limitations.

## CPU checks

Use an isolated environment; this does not install the GPU stack or download models:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python -m pytest tests/test_public_results.py tests/test_answer_controls_analysis.py
.venv/bin/python scripts/check_publication.py
```

The tests cover aggregate consistency and the existing task-cross-fitted threshold calibration diagnostic. They do not validate training on a new machine.

## Source layout

- `src/testbed/`: environment, activation readout, replay arithmetic and model services.
- `scripts/answer_pool_experiment.py`, `answer_pool_controls.py`, `answer_replication.py`: frozen experiment entry points and their historical dependencies.
- `analysis/`: retained calibration, outcome and robustness analyses, plus the portable summary command.
- `results/core_metrics.json`: path-free aggregate results for the report's 12 endpoints.
- `publication_files.json`: explicit list of public files, checked against Git's visible file set.

The probability penalty is a detached scalar reward. Replay instead differentiates a classification loss through the frozen probe into policy weights on saved transcripts; it does not imitate transcripts or refit the probe. The judge sees reasoning and public task information, not the final answer or hidden tests. The original probe fit uses execution-derived labels, so the pipeline is not entirely ground-truth-independent.

## Training reproduction status

The original scientific runtime is preserved, not rewritten. Its coordinators use exact source hashes, parent manifests, local artifacts and pinned service identities. Some paths are machine-specific. **A clean clone is not yet a self-contained GPU training reproduction.** The original adapters, probe, replay banks, batches and manifests must be supplied separately. See [required artifacts and launch limitations](REPRODUCING.md#training-source-and-required-external-artifacts) before using a historical launcher. Do not disable its hash checks or launch its recorded GPU assignments blindly.

`pyproject.toml` retains the historical GPU dependency declarations. No new GPU run or clean GPU installation was performed during publication cleanup.

## Safety and publication

The evaluator executes generated Python. Its subprocess timeout is not a security sandbox. Use a disposable isolated execution environment without credentials, host mounts or network access for untrusted programs. Keep model services off public networks.

Private documents, local data, captures, checkpoints, logs and unused workflows remain on the research server and are excluded from Git. Nothing has been deleted or uploaded. The earlier README is retained locally outside the public file set.

Code licensing, third-party attribution and artifact redistribution permissions still require owner review before publication. No license grant is implied by this repository preparation.
