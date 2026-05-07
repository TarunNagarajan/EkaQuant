#!/usr/bin/env python3
"""
Dual-GPU ARC-Challenge-Indic comparison runner for Kaggle 2xT4.

Runs two modes on the same model:
1. Baseline (FWE disabled)
2. FWE enabled

Each mode shards target languages across two GPU workers to use both T4s.
The script writes a JSON report with overall and per-language deltas.

Example:
  python run_arc_fwe_2xt4.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --eka-eval-path ./eka-eval \
    --gpu-ids 0,1 \
    --languages bn,en,gu,hi,kn,ml,mr,or,pa,ta,te \
    --dataset-split validation \
    --max-new-tokens 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import torch


DEFAULT_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


@dataclass
class WorkerSpec:
    gpu_id: int
    languages: List[str]
    output_json: str
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC-Challenge-Indic baseline vs FWE on 2xT4.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    # Shared args
    parser.add_argument("--model", type=str, required=False, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--eka-eval-path", type=str, required=False, default="eka-eval")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--languages", type=str, default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--mode", choices=["both", "baseline", "fwe"], default="both")
    parser.add_argument("--results-dir", type=str, default="results_dual_t4")

    # FWE params
    parser.add_argument("--fwe-max-cache-tokens", type=int, default=1536)
    parser.add_argument("--fwe-preserve-prefix-tokens", type=int, default=96)
    parser.add_argument("--fwe-preserve-suffix-tokens", type=int, default=256)
    parser.add_argument("--fwe-fertility-weight", type=float, default=0.5)
    parser.add_argument("--fwe-recency-weight", type=float, default=0.35)
    parser.add_argument("--fwe-anchor-weight", type=float, default=0.15)

    # Worker-only args
    parser.add_argument("--languages-json", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--use-fwe", action="store_true")
    return parser.parse_args()


def add_eka_eval_to_path(eka_eval_path: str) -> str:
    project_root = os.path.abspath(eka_eval_path)
    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"eka-eval path not found: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def worker_main(args: argparse.Namespace) -> None:
    add_eka_eval_to_path(args.eka_eval_path)

    from eka_eval.benchmarks.tasks.multilingual.arc_c_in import evaluate_arc_c_in
    from eka_eval.core.model_loader import cleanup_model_resources, initialize_model_pipeline

    langs = json.loads(args.languages_json)
    start = time.time()

    pipe, param_count = initialize_model_pipeline(
        model_name_or_path=args.model,
        target_device_id=0,
        is_api_model=False,
    )
    if pipe is None:
        raise RuntimeError("Failed to initialize model pipeline in worker.")

    tokenizer = getattr(pipe, "tokenizer", None)
    device = getattr(pipe, "device", "cpu")

    try:
        scores = evaluate_arc_c_in(
            pipe=pipe,
            tokenizer=tokenizer,
            model_name_for_logging=args.model,
            device=device,
            dataset_name="sarvamai/arc-challenge-indic",
            target_languages=langs,
            dataset_split=args.dataset_split,
            max_new_tokens=args.max_new_tokens,
            save_detailed=False,
            use_checkpoints=False,
            prompt_template_name_zeroshot="arc_c_in_0shot",
            prompt_file_benchmark_key="arc_c_in",
            prompt_file_category="indic",
            use_fwe_kv_eviction=args.use_fwe,
            fwe_max_cache_tokens=args.fwe_max_cache_tokens,
            fwe_preserve_prefix_tokens=args.fwe_preserve_prefix_tokens,
            fwe_preserve_suffix_tokens=args.fwe_preserve_suffix_tokens,
            fwe_fertility_weight=args.fwe_fertility_weight,
            fwe_recency_weight=args.fwe_recency_weight,
            fwe_anchor_weight=args.fwe_anchor_weight,
        )
    finally:
        cleanup_model_resources(pipe)

    elapsed = time.time() - start
    payload = {
        "model": args.model,
        "param_count": param_count,
        "languages": langs,
        "use_fwe": bool(args.use_fwe),
        "scores": scores,
        "elapsed_sec": elapsed,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_csv_ints(value: str) -> List[int]:
    ints: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ints.append(int(part))
    if not ints:
        raise ValueError("No GPU IDs provided.")
    return ints


def parse_csv_strings(value: str) -> List[str]:
    vals = [item.strip() for item in value.split(",") if item.strip()]
    if not vals:
        raise ValueError("No languages provided.")
    return vals


def shard_languages(languages: List[str], shard_count: int) -> List[List[str]]:
    shards = [[] for _ in range(shard_count)]
    for idx, lang in enumerate(languages):
        shards[idx % shard_count].append(lang)
    return shards


def aggregate_mode_results(shard_payloads: List[Dict]) -> Dict:
    per_lang: Dict[str, float] = {}
    total_elapsed = 0.0

    for payload in shard_payloads:
        total_elapsed += float(payload.get("elapsed_sec", 0.0))
        scores = payload.get("scores", {})
        for key, value in scores.items():
            if key.startswith("ARC-Challenge-Indic_"):
                lang = key.replace("ARC-Challenge-Indic_", "", 1)
                per_lang[lang] = float(value)

    if not per_lang:
        overall = 0.0
    else:
        overall = sum(per_lang.values()) / len(per_lang)

    return {
        "overall": overall,
        "per_language": dict(sorted(per_lang.items())),
        "sum_worker_elapsed_sec": total_elapsed,
    }


def run_mode(
    mode_name: str,
    use_fwe: bool,
    args: argparse.Namespace,
    gpu_ids: List[int],
    languages: List[str],
    run_dir: str,
) -> Dict:
    mode_dir = os.path.join(run_dir, mode_name)
    os.makedirs(mode_dir, exist_ok=True)

    shards = shard_languages(languages, len(gpu_ids))
    worker_specs: List[WorkerSpec] = []
    for shard_idx, shard_langs in enumerate(shards):
        if not shard_langs:
            continue
        worker_specs.append(
            WorkerSpec(
                gpu_id=gpu_ids[shard_idx],
                languages=shard_langs,
                output_json=os.path.join(mode_dir, f"worker_{shard_idx}_output.json"),
                log_file=os.path.join(mode_dir, f"worker_{shard_idx}.log"),
            )
        )

    procs = []
    for spec in worker_specs:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--model",
            args.model,
            "--eka-eval-path",
            args.eka_eval_path,
            "--dataset-split",
            args.dataset_split,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--languages-json",
            json.dumps(spec.languages),
            "--output-json",
            spec.output_json,
            "--fwe-max-cache-tokens",
            str(args.fwe_max_cache_tokens),
            "--fwe-preserve-prefix-tokens",
            str(args.fwe_preserve_prefix_tokens),
            "--fwe-preserve-suffix-tokens",
            str(args.fwe_preserve_suffix_tokens),
            "--fwe-fertility-weight",
            str(args.fwe_fertility_weight),
            "--fwe-recency-weight",
            str(args.fwe_recency_weight),
            "--fwe-anchor-weight",
            str(args.fwe_anchor_weight),
        ]
        if use_fwe:
            cmd.append("--use-fwe")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
        log_fp = open(spec.log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log_fp, spec))

    for proc, log_fp, _ in procs:
        proc.wait()
        log_fp.close()

    for proc, _, spec in procs:
        if proc.returncode != 0:
            tail = ""
            try:
                with open(spec.log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                tail = "".join(lines[-40:])
            except Exception:
                pass
            raise RuntimeError(
                f"{mode_name} worker on GPU {spec.gpu_id} failed (rc={proc.returncode}).\n"
                f"Log: {spec.log_file}\n{tail}"
            )

    shard_payloads = []
    for spec in worker_specs:
        with open(spec.output_json, "r", encoding="utf-8") as f:
            shard_payloads.append(json.load(f))

    aggregate = aggregate_mode_results(shard_payloads)
    aggregate["worker_outputs"] = [spec.output_json for spec in worker_specs]
    aggregate["worker_logs"] = [spec.log_file for spec in worker_specs]
    return aggregate


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return

    gpu_ids = parse_csv_ints(args.gpu_ids)
    languages = parse_csv_strings(args.languages)

    if len(gpu_ids) < 2:
        raise ValueError("Use at least 2 GPU IDs for the dual-T4 runner (e.g., --gpu-ids 0,1).")

    visible_count = torch.cuda.device_count()
    if visible_count < 2:
        print(
            f"WARNING: Detected {visible_count} visible CUDA device(s) in current process. "
            "This script is designed for 2x T4 Kaggle workers."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"arc_fwe_dual_t4_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    selected_modes = []
    if args.mode in ("both", "baseline"):
        selected_modes.append(("baseline", False))
    if args.mode in ("both", "fwe"):
        selected_modes.append(("fwe", True))

    results: Dict[str, Dict] = {}
    for mode_name, use_fwe in selected_modes:
        results[mode_name] = run_mode(mode_name, use_fwe, args, gpu_ids[:2], languages, run_dir)

    comparison = {}
    if "baseline" in results and "fwe" in results:
        base = results["baseline"]
        fwe = results["fwe"]
        langs_union = sorted(set(base["per_language"].keys()).union(fwe["per_language"].keys()))
        per_lang_delta = {
            lang: fwe["per_language"].get(lang, 0.0) - base["per_language"].get(lang, 0.0)
            for lang in langs_union
        }
        comparison = {
            "overall_delta": fwe["overall"] - base["overall"],
            "per_language_delta": per_lang_delta,
        }

    report = {
        "model": args.model,
        "eka_eval_path": os.path.abspath(args.eka_eval_path),
        "gpu_ids": gpu_ids[:2],
        "languages": languages,
        "dataset_split": args.dataset_split,
        "max_new_tokens": args.max_new_tokens,
        "fwe_config": {
            "max_cache_tokens": args.fwe_max_cache_tokens,
            "preserve_prefix_tokens": args.fwe_preserve_prefix_tokens,
            "preserve_suffix_tokens": args.fwe_preserve_suffix_tokens,
            "fertility_weight": args.fwe_fertility_weight,
            "recency_weight": args.fwe_recency_weight,
            "anchor_weight": args.fwe_anchor_weight,
        },
        "results": results,
        "comparison": comparison,
        "run_dir": os.path.abspath(run_dir),
    }

    report_path = os.path.join(run_dir, "summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Dual-T4 ARC Comparison ===")
    for mode_name, payload in results.items():
        print(
            f"{mode_name:>8} | overall={payload['overall']:.6f} | "
            f"languages={len(payload['per_language'])} | sum_worker_elapsed_sec={payload['sum_worker_elapsed_sec']:.2f}"
        )
    if comparison:
        print(f"   delta | overall={comparison['overall_delta']:+.6f}")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
