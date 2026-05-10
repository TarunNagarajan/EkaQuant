from __future__ import annotations

from typing import Dict, Iterable, List


def compute_delta(baseline_score: float, intervention_score: float) -> float:
    return float(intervention_score) - float(baseline_score)


def compute_relative_delta(
    baseline_score: float, intervention_score: float, epsilon: float = 1e-9
) -> float:
    baseline = float(baseline_score)
    intervention = float(intervention_score)
    return (intervention - baseline) / (abs(baseline) + epsilon)


def aggregate_language_deltas(rows: Iterable[Dict]) -> Dict:
    normalized_rows: List[Dict] = list(rows)
    if not normalized_rows:
        return {
            "mean_delta": 0.0,
            "mean_relative_delta": 0.0,
            "count": 0,
            "per_language": {},
        }

    per_language: Dict[str, Dict[str, float]] = {}
    delta_sum = 0.0
    rel_sum = 0.0

    for row in normalized_rows:
        language = str(row["language"])
        baseline = float(row["baseline_score"])
        intervention = float(row["intervention_score"])
        delta = compute_delta(baseline, intervention)
        rel = compute_relative_delta(baseline, intervention)
        per_language[language] = {
            "baseline_score": baseline,
            "intervention_score": intervention,
            "delta": delta,
            "relative_delta": rel,
        }
        delta_sum += delta
        rel_sum += rel

    count = len(normalized_rows)
    return {
        "mean_delta": delta_sum / count,
        "mean_relative_delta": rel_sum / count,
        "count": count,
        "per_language": dict(sorted(per_language.items())),
    }
