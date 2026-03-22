"""
Compute average metrics across datasets for a given sampling method.

Usage:
    python average_results.py --method margin_multilabel
    python average_results.py --method margin_multilabel --cycle 10
    python average_results.py --method random --cycle 5 --dir results/baselines
"""

import argparse
import sys
from pathlib import Path

import yaml


def find_method_files(baselines_dir: Path, method: str) -> dict[str, Path]:
    """Find files matching the method prefix across dataset subdirectories."""
    matches = {}
    for dataset_dir in sorted(baselines_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for f in dataset_dir.glob(f"{method}_*.yaml"):
            matches[dataset_dir.name] = f
            break  # one file per dataset
    return matches


def get_numeric_keys(cycles: list[dict]) -> set[str]:
    """Return keys whose values are numeric (int or float) in cycle dicts."""
    keys = set()
    for cycle in cycles:
        for k, v in cycle.items():
            if isinstance(v, (int, float)) and k != "cycle":
                keys.add(k)
    return keys


def average_cycles(all_cycles: list[list[dict]]) -> list[dict]:
    """Average per-cycle metrics across datasets."""
    if not all_cycles:
        return []

    n_cycles = min(len(c) for c in all_cycles)
    numeric_keys = get_numeric_keys(all_cycles[0])

    averaged = []
    for i in range(n_cycles):
        row = {"cycle": all_cycles[0][i]["cycle"]}
        for key in sorted(numeric_keys):
            values = [c[i][key] for c in all_cycles if key in c[i]]
            if values:
                row[key] = round(sum(values) / len(values), 6)
        averaged.append(row)
    return averaged


def average_supplementary(all_supps: list[dict]) -> dict:
    """Average top-level numeric supplementary fields across datasets."""
    if not all_supps:
        return {}
    numeric_keys = {k for s in all_supps for k, v in s.items() if isinstance(v, (int, float))}
    result = {}
    for key in sorted(numeric_keys):
        values = [s[key] for s in all_supps if key in s and isinstance(s[key], (int, float))]
        if values:
            result[key] = round(sum(values) / len(values), 6)
    return result


def print_cycle_row(row: dict, keys: list[str]) -> None:
    vals = "  ".join(f"{row.get(k, float('nan')):>12.5f}" for k in keys)
    print(f"  {row['cycle']:>5}  {vals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Average active learning results across datasets.")
    parser.add_argument("--method", required=True, help="Sampling method prefix, e.g. margin_multilabel")
    parser.add_argument("--cycle", type=int, default=None, help="Show only this cycle number (1-indexed)")
    parser.add_argument(
        "--dir",
        default=None,
        help="Path to baselines directory (default: <script_dir>/results)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    baselines_dir = Path(args.dir) if args.dir else script_dir / "results"

    if not baselines_dir.exists():
        print(f"Error: baselines directory not found: {baselines_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_method_files(baselines_dir, args.method)
    if not files:
        print(f"Error: no files found for method '{args.method}' in {baselines_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Method: {args.method}")
    print(f"Datasets found: {', '.join(files)}\n")

    all_data = {}
    for dataset, path in files.items():
        with open(path) as f:
            all_data[dataset] = yaml.safe_load(f)

    all_curves = [d["learning_curve"] for d in all_data.values() if "learning_curve" in d]
    all_supps = [d["supplementary"] for d in all_data.values() if "supplementary" in d]

    if args.cycle is not None:
        # --- Single cycle ---
        print(f"=== Cycle {args.cycle} results (averaged across {len(files)} datasets) ===\n")
        rows = []
        for dataset, data in all_data.items():
            curve = data.get("learning_curve", [])
            match = next((c for c in curve if c["cycle"] == args.cycle), None)
            if match is None:
                print(f"  Warning: cycle {args.cycle} not found in {dataset}")
            else:
                rows.append((dataset, match))

        if not rows:
            print("No data found for this cycle.")
            return

        # Print per-dataset values
        metric_keys = sorted(get_numeric_keys([r for _, r in rows]))
        col_w = max(len(k) for k in metric_keys) if metric_keys else 10
        header = f"{'Dataset':<12}" + "".join(f"  {k:>{max(col_w, 12)}}" for k in metric_keys)
        print(header)
        print("-" * len(header))
        for dataset, row in rows:
            vals = "".join(f"  {row.get(k, float('nan')):>{max(col_w, 12)}.5f}" for k in metric_keys)
            print(f"{dataset:<12}{vals}")

        # Averages
        avg_row: dict = {}
        for key in metric_keys:
            values = [r.get(key) for _, r in rows if isinstance(r.get(key), (int, float))]
            if values:
                avg_row[key] = round(sum(values) / len(values), 6)

        print("-" * len(header))
        vals = "".join(f"  {avg_row.get(k, float('nan')):>{max(col_w, 12)}.5f}" for k in metric_keys)
        print(f"{'Average':<12}{vals}")

    else:
        # --- All cycles averaged ---
        print(f"=== All cycles averaged across {len(files)} datasets ===\n")
        averaged = average_cycles(all_curves)
        if not averaged:
            print("No learning curve data found.")
        else:
            metric_keys = [k for k in averaged[0] if k != "cycle"]
            col_w = max(len(k) for k in metric_keys) if metric_keys else 10
            header = f"{'Cycle':>7}" + "".join(f"  {k:>{max(col_w, 12)}}" for k in metric_keys)
            print(header)
            print("-" * len(header))
            for row in averaged:
                vals = "".join(f"  {row.get(k, float('nan')):>{max(col_w, 12)}.5f}" for k in metric_keys)
                print(f"{row['cycle']:>7}{vals}")

        # Supplementary averages
        if all_supps:
            print("\n=== Supplementary averages ===\n")
            avg_supp = average_supplementary(all_supps)
            col_w = max(len(k) for k in avg_supp) if avg_supp else 10
            for k, v in avg_supp.items():
                print(f"  {k:<{col_w + 2}}: {v}")


if __name__ == "__main__":
    main()
