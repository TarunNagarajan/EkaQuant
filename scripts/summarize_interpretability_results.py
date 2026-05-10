from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List


def _load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _rows_from_summary(summary: Dict) -> List[Dict]:
    experiments = summary["merged"]["experiments"]
    rows = []
    for experiment in experiments:
        rows.append(
            {
                "experiment": experiment["name"],
                "overall_baseline": float(experiment["overall_baseline"]),
                "overall_intervention": float(experiment["overall_intervention"]),
                "overall_delta": float(experiment["overall_delta"]),
                "module_count": len(experiment["module_names"]),
            }
        )
    rows.sort(key=lambda row: row["overall_delta"])
    return rows


def _write_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "overall_baseline",
                "overall_intervention",
                "overall_delta",
                "module_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("| experiment | baseline | intervention | delta | module_count |\n")
        file.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['experiment']} | "
                f"{row['overall_baseline']:.6f} | "
                f"{row['overall_intervention']:.6f} | "
                f"{row['overall_delta']:.6f} | "
                f"{row['module_count']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CSV/Markdown ranking from interpretability summary.json")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    summary = _load_summary(args.summary_json)
    rows = _rows_from_summary(summary)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.summary_json))
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ranking.csv")
    md_path = os.path.join(out_dir, "ranking.md")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if rows:
        print(f"Best delta: {rows[-1]['experiment']} ({rows[-1]['overall_delta']:.6f})")
        print(f"Worst delta: {rows[0]['experiment']} ({rows[0]['overall_delta']:.6f})")


if __name__ == "__main__":
    main()
