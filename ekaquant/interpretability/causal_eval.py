from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from eka_eval.benchmarks.tasks.multilingual.arc_c_in import evaluate_arc_c_in

from .artifacts import make_run_dir, write_json
from .interventions import ablate_modules
from .metrics import aggregate_language_deltas


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    module_names: List[str]
    scale: float = 0.0


def _sample_names_evenly(candidates: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if len(candidates) <= count:
        return candidates
    step = max(len(candidates) // count, 1)
    sampled = candidates[::step][:count]
    return sampled


def build_default_ablation_experiments(model, attention_count: int = 6, mlp_count: int = 6) -> List[AblationExperiment]:
    named_modules = [name for name, _ in model.named_modules()]

    attention_candidates = sorted(
        {
            name
            for name in named_modules
            if ("self_attn" in name or name.endswith(".attn")) and "." in name
        }
    )
    mlp_candidates = sorted(
        {
            name
            for name in named_modules
            if ".mlp" in name or "feed_forward" in name
        }
    )

    sampled_attention = _sample_names_evenly(attention_candidates, attention_count)
    sampled_mlp = _sample_names_evenly(mlp_candidates, mlp_count)

    experiments: List[AblationExperiment] = []
    for module_name in sampled_attention:
        experiments.append(AblationExperiment(name=f"ablate::{module_name}", module_names=[module_name]))
    for module_name in sampled_mlp:
        experiments.append(AblationExperiment(name=f"ablate::{module_name}", module_names=[module_name]))
    return experiments


def _collect_language_rows(baseline_scores: Dict[str, float], intervention_scores: Dict[str, float], languages: Iterable[str]) -> List[Dict]:
    rows = []
    for language in languages:
        key = f"ARC-Challenge-Indic_{language}"
        rows.append(
            {
                "language": language,
                "baseline_score": float(baseline_scores.get(key, 0.0)),
                "intervention_score": float(intervention_scores.get(key, 0.0)),
            }
        )
    return rows


def run_arc_ablation_comparison(
    pipe,
    tokenizer,
    model_name: str,
    target_languages: List[str],
    dataset_split: str = "validation",
    max_new_tokens: int = 5,
    experiments: Optional[List[AblationExperiment]] = None,
    artifact_dir: str = "results_output/interpretability",
) -> Dict:
    base_eval_args = {
        "pipe": pipe,
        "tokenizer": tokenizer,
        "model_name_for_logging": model_name,
        "device": getattr(pipe, "device", "cpu"),
        "dataset_name": "sarvamai/arc-challenge-indic",
        "target_languages": target_languages,
        "dataset_split": dataset_split,
        "max_new_tokens": max_new_tokens,
        "save_detailed": False,
        "use_checkpoints": False,
        "prompt_template_name_zeroshot": "arc_c_in_0shot",
        "prompt_file_benchmark_key": "arc_c_in",
        "prompt_file_category": "indic",
    }

    baseline_scores = evaluate_arc_c_in(**base_eval_args)
    baseline_overall = float(baseline_scores.get("ARC-Challenge-Indic", 0.0))

    if experiments is None:
        experiments = build_default_ablation_experiments(pipe.model)

    experiment_reports = []
    for experiment in experiments:
        with ablate_modules(pipe.model, experiment.module_names, scale=experiment.scale):
            intervention_scores = evaluate_arc_c_in(**base_eval_args)
        rows = _collect_language_rows(baseline_scores, intervention_scores, target_languages)
        aggregate = aggregate_language_deltas(rows)
        experiment_reports.append(
            {
                "name": experiment.name,
                "module_names": experiment.module_names,
                "scale": experiment.scale,
                "overall_baseline": baseline_overall,
                "overall_intervention": float(intervention_scores.get("ARC-Challenge-Indic", 0.0)),
                "overall_delta": float(intervention_scores.get("ARC-Challenge-Indic", 0.0)) - baseline_overall,
                "language_aggregate": aggregate,
            }
        )

    experiment_reports.sort(key=lambda item: item["overall_delta"])

    run_dir = make_run_dir(artifact_dir, "arc_ablation")
    payload = {
        "model_name": model_name,
        "target_languages": target_languages,
        "dataset_split": dataset_split,
        "max_new_tokens": max_new_tokens,
        "baseline_scores": baseline_scores,
        "experiments": experiment_reports,
    }
    write_json(f"{run_dir}/summary.json", payload)
    payload["run_dir"] = run_dir
    return payload
