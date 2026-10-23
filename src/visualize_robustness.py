
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "figures"

RESULTS_PATH = REPORT_DIR / "robustness_test_results.csv"
SUMMARY_PATH = REPORT_DIR / "robustness_summary.csv"


def main():
    if not RESULTS_PATH.exists() or not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Run src/robustness_testing.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    # Chart 1: Strategy return at the base transaction cost.
    base_cost = results[results["Cost"] == 0.001]

    returns = base_cost.pivot(
        index="Round",
        columns="Strategy",
        values="Strategy_Return",
    ) * 100

    ax = returns.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )
    ax.axhline(0, linewidth=1)
    ax.set_title("Walk-Forward Returns: Tuned vs Fixed Threshold")
    ax.set_xlabel("Test Round")
    ax.set_ylabel("Strategy Return (%)")
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "walk_forward_returns.png",
        dpi=160,
    )
    plt.close()

    # Chart 2: Average return as transaction costs increase.
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy in summary["Strategy"].unique():
        data = summary[summary["Strategy"] == strategy].sort_values("Cost")

        ax.plot(
            data["Cost"] * 100,
            data["Average_Strategy_Return"] * 100,
            marker="o",
            label=strategy,
        )

    ax.axhline(0, linewidth=1)
    ax.set_title("Transaction Cost Sensitivity")
    ax.set_xlabel("Assumed Round-Trip Cost (%)")
    ax.set_ylabel("Average Strategy Return per Round (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "transaction_cost_sensitivity.png",
        dpi=160,
    )
    plt.close()

    # Chart 3: Average maximum drawdown.
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy in summary["Strategy"].unique():
        data = summary[summary["Strategy"] == strategy].sort_values("Cost")

        ax.plot(
            data["Cost"] * 100,
            data["Average_Max_Drawdown"] * 100,
            marker="o",
            label=strategy,
        )

    ax.set_title("Average Maximum Drawdown vs Transaction Costs")
    ax.set_xlabel("Assumed Round-Trip Cost (%)")
    ax.set_ylabel("Average Maximum Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "drawdown_sensitivity.png",
        dpi=160,
    )
    plt.close()

    # Chart 4: Average excess return over the benchmark.
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy in summary["Strategy"].unique():
        data = summary[summary["Strategy"] == strategy].sort_values("Cost")

        ax.plot(
            data["Cost"] * 100,
            data["Average_Excess_Return"] * 100,
            marker="o",
            label=strategy,
        )

    ax.axhline(0, linewidth=1)
    ax.set_title("Excess Return vs Open-to-Close Benchmark")
    ax.set_xlabel("Assumed Round-Trip Cost (%)")
    ax.set_ylabel("Average Excess Return (percentage points)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "excess_return_sensitivity.png",
        dpi=160,
    )
    plt.close()

    print("Charts saved to:", OUTPUT_DIR)

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
