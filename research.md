# Research Decision Log

## Scope
This document records project decisions, assumptions, outcomes, lapses, and pivots for the interpretability project integrated with `eka-eval`.

## Working Rules
- No GitHub push unless explicitly requested.
- No comments or emojis added to the codebase unless explicitly requested.
- Code changes must be clean, modular, and extensible.
- Use this log after each significant assumption-driven change.

## Entry Template
### Entry <id>
- Context:
- Decision:
- Assumption:
- Why this assumption:
- Expected outcome:
- Actual outcome:
- Lapse in judgment:
- Pivot:
- Next action:

## Decision Log
### Entry 001
- Context: Fresh clone setup in `C:\Users\ultim\eka-eval-fresh`.
- Decision: Continue with the clone and verify repository integrity after checkout-hook warning during clone.
- Assumption: The repository contents are usable if `git status` and `HEAD` resolve correctly.
- Why this assumption: Git reported clone success with checkout warning and suggested restore.
- Expected outcome: A stable working tree on branch `main` with valid `HEAD`.
- Actual outcome: Repository resolved to `main` at `f5138dc20ba8d83bc75640e9243b330af2578fff` and clean status.
- Lapse in judgment: None observed.
- Pivot: None required.
- Next action: Start implementation with a structured research log workflow.

### Entry 002
- Context: Requirement requested full chain-of-thought in `research.md`.
- Decision: Use a structured decision log format instead.
- Assumption: A detailed rationale/assumption/failure/pivot log satisfies project needs while remaining maintainable.
- Why this assumption: The project needs actionable research traceability, not free-form internal narration.
- Expected outcome: Clear, auditable, and reusable project reasoning records.
- Actual outcome: Structured format established and confirmed.
- Lapse in judgment: None observed.
- Pivot: Standardize all future major changes using this template.
- Next action: Scaffold interpretability modules and keep adding entries after assumption-based decisions.

### Entry 003
- Context: Integrating interpretability capture into `evaluate_arc_c_in` without breaking baseline behavior.
- Decision: Add optional capture flags and artifact writing path, defaulting to disabled.
- Assumption: A gated capture mode (`interpretability_capture` plus explicit module list) avoids side effects in normal evaluation.
- Why this assumption: Baseline benchmark reproducibility must remain unchanged when interpretability mode is off.
- Expected outcome: Existing ARC evaluations run as before; capture mode writes structured JSON/JSONL artifacts when enabled.
- Actual outcome: Integration compiled successfully with default-off behavior and capture-only paths.
- Lapse in judgment: None observed at compile stage.
- Pivot: If runtime overhead is high, introduce sampling/throttling controls before deeper sweeps.
- Next action: Implement causal intervention runner and strategy orchestration for baseline vs intervention comparisons.

### Entry 004
- Context: Implementing the first causal intervention engine for ARC-Challenge-Indic.
- Decision: Start with module-level ablation sweeps (attention/MLP candidates) before adding heavier patching workflows.
- Assumption: Module-level ablation gives enough causal signal to rank components and validate the end-to-end interpretability pipeline.
- Why this assumption: It is lower-risk computationally on 2xT4 and directly measurable against benchmark deltas.
- Expected outcome: Reproducible baseline vs intervention deltas with ranked harmful/helpful components.
- Actual outcome: `causal_eval.py` and `run_interpretability_sweep.py` were added and compiled successfully.
- Lapse in judgment: None observed at implementation stage.
- Pivot: If signal quality is weak, increase intervention granularity to head-level subsets and paired activation patching.
- Next action: Add dual-GPU runner integration for large-model Kaggle sweeps and artifact aggregation.

### Entry 005
- Context: Running intervention sweeps on Kaggle 2xT4 with practical turnaround.
- Decision: Build a dual-worker orchestrator that shards languages across GPUs and merges experiment deltas into one summary.
- Assumption: Language sharding gives near-linear throughput gains without changing experiment semantics.
- Why this assumption: ARC-Challenge-Indic is language-partitionable and experiments are independent per language shard.
- Expected outcome: Reliable dual-GPU execution with merged baseline/intervention ranking report.
- Actual outcome: `run_interpretability_dual_t4.py` implemented and compiled, producing merged summary artifacts by design.
- Lapse in judgment: None observed at implementation stage.
- Pivot: If shard variance skews results, switch to repeated balanced shards and aggregate with confidence intervals.
- Next action: Add explicit reporting outputs and run commands, then execute Kaggle validation sweep.

### Entry 006
- Context: Need reproducible post-run analysis artifacts for engineering presentation and benchmark interpretation.
- Decision: Add a dedicated summary script that converts merged sweep summaries into ranked CSV and Markdown reports.
- Assumption: Lightweight tabular outputs are sufficient for first-pass model/component comparison before plotting.
- Why this assumption: Faster iteration and easier notebook/PR consumption.
- Expected outcome: Consistent ranking outputs from any `summary.json` generated by dual-GPU runs.
- Actual outcome: `summarize_interpretability_results.py` added and compile-checked.
- Lapse in judgment: None observed.
- Pivot: If report granularity is insufficient, extend outputs with confidence intervals and per-language dispersion.
- Next action: Prepare Kaggle validation command path and run checklist.

### Entry 007
- Context: Needed runtime confirmation that single-process and dual-worker interpretability paths execute end-to-end in the fresh clone.
- Decision: Run smoke validations with `sshleifer/tiny-gpt2` and sliced dataset splits, then generate ranking outputs from merged summaries.
- Assumption: Tiny sliced runs are sufficient to validate pipeline wiring, artifact schemas, and dual-worker merge logic before full Kaggle sweeps.
- Why this assumption: Full 7B-scale runs are not available in this local environment; smoke validation reduces cycle time while still testing core control flow.
- Expected outcome: Sweep script writes a valid summary, dual runner produces merged `summary.json`, and summarizer writes ranking CSV/Markdown.
- Actual outcome: All three scripts executed successfully; artifacts were produced in `results_output/interpretability_smoke*` and `results_output/interpretability_dual_smoke`.
- Lapse in judgment: Dataset loading still passed `trust_remote_code=True`, causing repeated deprecation warnings.
- Pivot: Removed `trust_remote_code` from ARC dataset loading path in `arc_c_in.py` and re-ran smoke validation successfully.
- Next action: Run the same dual-runner workflow on Kaggle 2xT4 with the target larger model and publish final comparative findings.
