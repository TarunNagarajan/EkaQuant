from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eka_eval.core.model_loader import (
    cleanup_model_resources,
    initialize_model_pipeline,
)
from eka_eval.interpretability import AblationExperiment, run_arc_ablation_comparison
from eka_eval.interpretability.artifacts import ensure_dir, write_json


DEFAULT_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


def _parse_csv(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected non-empty CSV value.")
    return values


def _parse_gpu_ids(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one GPU id.")
    return values


def _split_languages(languages: List[str], shard_count: int) -> List[List[str]]:
    shards = [[] for _ in range(shard_count)]
    for idx, language in enumerate(languages):
        shards[idx % shard_count].append(language)
    return [shard for shard in shards if shard]


def _load_experiments(path: str) -> List[AblationExperiment]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    experiments = []
    for item in raw:
        experiments.append(
            AblationExperiment(
                name=str(item["name"]),
                module_names=[str(name) for name in item["module_names"]],
                scale=float(item.get("scale", 0.0)),
            )
        )
    return experiments


def _worker_main(args: argparse.Namespace) -> None:
    languages = json.loads(args.languages_json)
    experiments = (
        _load_experiments(args.experiments_json) if args.experiments_json else None
    )

    pipe, _ = initialize_model_pipeline(
        model_name_or_path=args.model,
        target_device_id=0,
        is_api_model=False,
    )
    if pipe is None:
        raise RuntimeError("Worker failed to initialize model pipeline.")

    try:
        payload = run_arc_ablation_comparison(
            pipe=pipe,
            tokenizer=getattr(pipe, "tokenizer", None),
            model_name=args.model,
            target_languages=languages,
            dataset_split=args.dataset_split,
            max_new_tokens=args.max_new_tokens,
            experiments=experiments,
            artifact_dir=args.artifact_dir,
        )
    finally:
        cleanup_model_resources(pipe)

    write_json(args.worker_output_json, payload)


def _merge_payloads(payloads: List[Dict]) -> Dict:
    all_languages: List[str] = []
    baseline_per_language: Dict[str, float] = {}
    experiment_map: Dict[str, Dict] = {}

    for payload in payloads:
        languages = payload["target_languages"]
        all_languages.extend(languages)
        for language in languages:
            baseline_key = f"ARC-Challenge-Indic_{language}"
            baseline_per_language[language] = float(
                payload["baseline_scores"].get(baseline_key, 0.0)
            )

        for experiment in payload["experiments"]:
            name = experiment["name"]
            if name not in experiment_map:
                experiment_map[name] = {
                    "name": name,
                    "module_names": experiment["module_names"],
                    "scale": float(experiment["scale"]),
                    "per_language": {},
                }
            per_language = experiment["language_aggregate"]["per_language"]
            for language, row in per_language.items():
                experiment_map[name]["per_language"][language] = {
                    "baseline_score": float(row["baseline_score"]),
                    "intervention_score": float(row["intervention_score"]),
                    "delta": float(row["delta"]),
                    "relative_delta": float(row["relative_delta"]),
                }

    sorted_languages = sorted(set(all_languages))
    baseline_overall = sum(
        baseline_per_language.get(lang, 0.0) for lang in sorted_languages
    ) / max(len(sorted_languages), 1)

    merged_experiments = []
    for name, payload in experiment_map.items():
        per_language = payload["per_language"]
        if sorted_languages:
            intervention_overall = sum(
                per_language.get(lang, {}).get("intervention_score", 0.0)
                for lang in sorted_languages
            ) / len(sorted_languages)
        else:
            intervention_overall = 0.0
        merged_experiments.append(
            {
                "name": name,
                "module_names": payload["module_names"],
                "scale": payload["scale"],
                "overall_baseline": baseline_overall,
                "overall_intervention": intervention_overall,
                "overall_delta": intervention_overall - baseline_overall,
                "per_language": dict(sorted(per_language.items())),
            }
        )

    merged_experiments.sort(key=lambda row: row["overall_delta"])
    return {
        "languages": sorted_languages,
        "baseline_overall": baseline_overall,
        "baseline_per_language": dict(sorted(baseline_per_language.items())),
        "experiments": merged_experiments,
    }


def _main(args: argparse.Namespace) -> None:
    languages = _parse_csv(args.languages)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    shards = _split_languages(languages, len(gpu_ids))
    if len(shards) < 2:
        raise ValueError("At least two language shards are required for dual-GPU mode.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(args.artifact_dir, f"dual_t4_{run_stamp}"))
    worker_outputs = []
    processes = []

    for idx, shard_languages in enumerate(shards):
        worker_output = os.path.join(run_dir, f"worker_{idx}_summary.json")
        worker_log = os.path.join(run_dir, f"worker_{idx}.log")
        worker_outputs.append(worker_output)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--model",
            args.model,
            "--dataset-split",
            args.dataset_split,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--artifact-dir",
            args.artifact_dir,
            "--languages-json",
            json.dumps(shard_languages),
            "--worker-output-json",
            worker_output,
            "--experiments-json",
            args.experiments_json,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[idx])
        log_file = open(worker_log, "w", encoding="utf-8")
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env
        )
        processes.append((process, log_file, worker_log))

    for process, log_file, _ in processes:
        process.wait()
        log_file.close()

    for process, _, worker_log in processes:
        if process.returncode != 0:
            with open(worker_log, "r", encoding="utf-8") as file:
                log_tail = "".join(file.readlines()[-50:])
            raise RuntimeError(
                f"Worker failed with rc={process.returncode}. Log: {worker_log}\n{log_tail}"
            )

    payloads = []
    for worker_output in worker_outputs:
        with open(worker_output, "r", encoding="utf-8") as file:
            payloads.append(json.load(file))

    merged = _merge_payloads(payloads)
    summary_path = os.path.join(run_dir, "summary.json")
    write_json(
        summary_path,
        {
            "model": args.model,
            "gpu_ids": gpu_ids[: len(shards)],
            "dataset_split": args.dataset_split,
            "max_new_tokens": args.max_new_tokens,
            "worker_outputs": worker_outputs,
            "merged": merged,
        },
    )

    print("Dual-GPU interpretability sweep complete.")
    print(f"Summary: {summary_path}")
    print(f"Baseline overall: {merged['baseline_overall']:.6f}")
    for row in merged["experiments"][:5]:
        print(f"{row['name']} | delta={row['overall_delta']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual-T4 interpretability ablation sweep."
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--languages", type=str, default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument(
        "--artifact-dir", type=str, default="results_output/interpretability"
    )
    parser.add_argument("--experiments-json", type=str, default="")
    parser.add_argument("--languages-json", type=str, default="")
    parser.add_argument("--worker-output-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.worker:
        _worker_main(parsed_args)
    else:
        _main(parsed_args)
