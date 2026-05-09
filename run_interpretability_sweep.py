import argparse
import json
import os
import sys
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eka_eval.core.model_loader import cleanup_model_resources, initialize_model_pipeline
from eka_eval.interpretability import AblationExperiment, run_arc_ablation_comparison


def _parse_languages(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one language code is required.")
    return values


def _load_experiments(path: str) -> List[AblationExperiment]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    experiments: List[AblationExperiment] = []
    for row in payload:
        experiments.append(
            AblationExperiment(
                name=str(row["name"]),
                module_names=[str(name) for name in row["module_names"]],
                scale=float(row.get("scale", 0.0)),
            )
        )
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal ablation sweep on ARC-Challenge-Indic.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--languages", type=str, default="bn,en,hi")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--artifact-dir", type=str, default="results_output/interpretability")
    parser.add_argument("--experiments-json", type=str, default="")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    languages = _parse_languages(args.languages)
    experiments = _load_experiments(args.experiments_json) if args.experiments_json else None

    pipe, _ = initialize_model_pipeline(
        model_name_or_path=args.model,
        target_device_id=args.device_id,
        is_api_model=False,
    )
    if pipe is None:
        raise RuntimeError("Model pipeline initialization failed.")

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

    print("Ablation sweep complete.")
    print(f"Run directory: {payload['run_dir']}")
    print(f"Baseline overall: {payload['baseline_scores'].get('ARC-Challenge-Indic', 0.0):.6f}")
    top = payload["experiments"][:5]
    for idx, row in enumerate(top, start=1):
        print(f"{idx}. {row['name']} | delta={row['overall_delta']:.6f}")


if __name__ == "__main__":
    main()
