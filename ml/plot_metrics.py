"""Plot saved test metrics with Matplotlib; also called automatically by training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Save PNGs without opening a desktop window.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#8aab8d", "#21614f"]

def plot_comparison(report):
    folder = ROOT / "public/plots"
    folder.mkdir(exist_ok=True)
    models = report["models"]
    names = [model["name"] for model in models]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, metric, title, unit in zip(axes, ["mse", "r2"],
                                     ["MSE · lower is better", "R² · higher is better"],
                                     ["Mean squared error (kWh²)", "R² score"]):
        values = [model["test"][metric] for model in models]
        bars = ax.bar(names, values, color=COLORS, width=.5)
        ax.bar_label(bars, fmt="%.3f", padding=5)
        ax.set(title=title, ylabel=unit)
        low, high = min(0, min(values)), max(0, max(values))
        margin = max(.1, (high - low) * .2)
        ax.set_ylim(low - (margin if low < 0 else 0), high + margin)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Linear Regression vs Random Forest · held-out next-day test", fontsize=13)
    fig.savefig(folder / "regression-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    labels = ["accuracy", "precision", "recall"]
    x = np.arange(len(labels))
    width = .32
    for i, model in enumerate(models):
        values = [model["high_usage_test"][metric] for metric in labels]
        bars = ax.bar(x + (i - .5) * width, values, width, label=model["name"], color=COLORS[i])
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4, fontsize=10)
    threshold = report["high_usage_classification"]["threshold_kwh"]
    ax.set(title=f"High-usage detection · daily kWh ≥ {threshold:.2f} (training median)",
           ylabel="Classification score", ylim=(0, 1.16), xticks=x,
           xticklabels=[label.title() for label in labels])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", frameon=False)
    fig.supxlabel("Derived from regression predictions; these are not regression accuracy scores.", fontsize=10)
    fig.savefig(folder / "high-usage-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    plot_comparison(json.loads((ROOT / "models/report.json").read_text(encoding="utf-8")))
