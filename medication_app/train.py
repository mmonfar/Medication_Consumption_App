"""Train and cache per-medication models: python -m medication_app.train"""

from __future__ import annotations

from .data import load_sample_data
from .pipeline import save, train_all


def main() -> None:
    print("loading data...")
    _, consumption, cohort = load_sample_data()
    print(f"  {len(consumption):,} administration records over {cohort['Date'].nunique()} days")

    print("cross-validating candidates per medication (this takes a minute)...")
    results = train_all(consumption, cohort)

    path = save(results, consumption)
    print(f"\n{'medication':22s} {'model':>16s} {'metric':>7s} {'score':>7s} {'bias':>8s}")
    for result in results.values():
        flag = "" if result.beats_benchmark else "  (benchmark)"
        print(
            f"{result.medication:22s} {result.model:>16s} {result.primary_metric:>7s} "
            f"{result.score:7.3f} {result.cumulative_bias:+8.1%}{flag}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
