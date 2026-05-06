"""Generate evaluation result charts from hard-coded report data.

This script does NOT re-run evaluation; it simply visualises the metrics
produced by a previous run of evaluate.py.

Usage:
    python generate_charts.py

Output:
    - output/charts/detection_confusion.png
    - output/charts/metrics_bar.png
    - output/charts/detection_pie.png
    - output/charts/ocr_gauge.png
    - output/charts/eval_dashboard.png
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Hard-coded results from evaluate.py run
# ---------------------------------------------------------------------------
RESULTS = {
    "images_evaluated": 1145,
    "time_seconds": 5325.7,
    "ground_truth_plates": 1294,
    "detections": 1105,
    "true_positives": 1042,
    "false_positives": 63,
    "false_negatives": 252,
    "precision": 0.9430,
    "recall": 0.8053,
    "f1_score": 0.8687,
    "plates_read": 1042,
    "avg_ocr_confidence": 0.9531,
}

OUTPUT_DIR = Path("output/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save_and_close(fig, filename):
    """Save figure to OUTPUT_DIR and close it."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close(fig)


# ===========================================================================
# Chart 1 – Detection Confusion-style Bar (TP / FP / FN)
# ===========================================================================
def chart_detection_confusion():
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["True Positives\n(Detected correctly)",
                  "False Positives\n(Wrong detections)",
                  "False Negatives\n(Missed plates)"]
    values = [RESULTS["true_positives"], RESULTS["false_positives"], RESULTS["false_negatives"]]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Count")
    ax.set_title("Detection Results Breakdown")
    ax.set_ylim(0, max(values) * 1.2)

    # Annotate bar tops
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_and_close(fig, "detection_confusion.png")


# ===========================================================================
# Chart 2 – Precision / Recall / F1 Bar Chart
# ===========================================================================
def chart_metrics_bar():
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Precision", "Recall", "F1-Score"]
    values = [RESULTS["precision"], RESULTS["recall"], RESULTS["f1_score"]]
    colors = ["#3498db", "#9b59b6", "#1abc9c"]

    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2, width=0.6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Detection Quality Metrics")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f"{val:.4f}" if val != RESULTS["f1_score"] else f"{val:.4f}"
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_and_close(fig, "metrics_bar.png")


# ===========================================================================
# Chart 3 – Detection Pie Chart (TP vs FP vs FN)
# ===========================================================================
def chart_detection_pie():
    fig, ax = plt.subplots(figsize=(7, 7))

    labels = ["True Positives", "False Positives", "False Negatives"]
    sizes = [RESULTS["true_positives"], RESULTS["false_positives"], RESULTS["false_negatives"]]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    explode = (0.03, 0.03, 0.03)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(sizes)))})",
        shadow=False, startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 1.2},
        textprops={"fontsize": 11}
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title("Detection Outcome Distribution")
    plt.tight_layout()
    save_and_close(fig, "detection_pie.png")


# ===========================================================================
# Chart 4 – OCR Confidence Gauge
# ===========================================================================
def chart_ocr_gauge():
    fig, ax = plt.subplots(figsize=(8, 4))

    confidence = RESULTS["avg_ocr_confidence"]
    # Horizontal bar from 0 to 1
    ax.barh([0], [1.0], color="#ecf0f1", height=0.4, edgecolor="black")
    ax.barh([0], [confidence], color="#2ecc71", height=0.4, edgecolor="black")

    ax.set_xlim(0, 1.15)
    ax.set_yticks([0])
    ax.set_yticklabels(["Avg OCR Confidence"])
    ax.set_xlabel("Confidence Score")
    ax.set_title(f"OCR Confidence on Matched Plates (n={RESULTS['plates_read']})")

    # Annotate score
    ax.text(confidence + 0.02, 0, f"{confidence:.2%}",
            va="center", ha="left", fontsize=14, fontweight="bold", color="#27ae60")

    # Add threshold markers
    for x, label, color in [(0.5, "0.5", "gray"), (0.75, "0.75", "gray"), (1.0, "1.0", "gray")]:
        ax.axvline(x=x, color=color, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_and_close(fig, "ocr_gauge.png")


# ===========================================================================
# Chart 5 – Ground-truth vs Detections Comparison
# ===========================================================================
def chart_gt_vs_det():
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Ground-truth Plates", "Total Detections"]
    values = [RESULTS["ground_truth_plates"], RESULTS["detections"]]
    colors = ["#34495e", "#e67e22"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2, width=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Ground-truth vs Predicted Detections")
    ax.set_ylim(0, max(values) * 1.2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_and_close(fig, "gt_vs_detections.png")


# ===========================================================================
# Chart 6 – Dashboard (2x2 grid summarising everything)
# ===========================================================================
def chart_dashboard():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ---------- Top-left: Detection outcomes ----------
    ax1 = fig.add_subplot(gs[0, 0])
    outcomes = ["TP", "FP", "FN"]
    outcome_vals = [RESULTS["true_positives"], RESULTS["false_positives"], RESULTS["false_negatives"]]
    outcome_colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    bars = ax1.bar(outcomes, outcome_vals, color=outcome_colors, edgecolor="black")
    ax1.set_title("Detection Outcomes", fontweight="bold")
    ax1.set_ylabel("Count")
    for bar, val in zip(bars, outcome_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(val), ha="center", va="bottom", fontweight="bold")

    # ---------- Top-right: Metrics ----------
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ["Precision", "Recall", "F1"]
    metric_vals = [RESULTS["precision"], RESULTS["recall"], RESULTS["f1_score"]]
    metric_colors = ["#3498db", "#9b59b6", "#1abc9c"]
    bars = ax2.bar(metrics, metric_vals, color=metric_colors, edgecolor="black", width=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Quality Metrics", fontweight="bold")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, metric_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2%}" if val >= 0.01 else f"{val:.4f}",
                 ha="center", va="bottom", fontweight="bold")

    # ---------- Bottom-left: Pie ----------
    ax3 = fig.add_subplot(gs[1, 0])
    pie_vals = outcome_vals
    wedges, texts, autotexts = ax3.pie(
        pie_vals, labels=["TP", "FP", "FN"], colors=outcome_colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "black", "linewidth": 1}
    )
    ax3.set_title("Outcome Proportion", fontweight="bold")

    # ---------- Bottom-right: OCR + summary text ----------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_text = (
        f"EVALUATION SUMMARY\n"
        f"{'='*30}\n\n"
        f"Images evaluated : {RESULTS['images_evaluated']}\n"
        f"Time elapsed     : {RESULTS['time_seconds']:.1f}s\n"
        f"Speed            : {RESULTS['time_seconds']/RESULTS['images_evaluated']:.2f}s/img\n\n"
        f"--- Detection ---\n"
        f"Ground-truth     : {RESULTS['ground_truth_plates']}\n"
        f"Detections       : {RESULTS['detections']}\n"
        f"Precision        : {RESULTS['precision']:.2%}\n"
        f"Recall           : {RESULTS['recall']:.2%}\n"
        f"F1-Score         : {RESULTS['f1_score']:.4f}\n\n"
        f"--- OCR ---\n"
        f"Plates read      : {RESULTS['plates_read']}\n"
        f"Avg confidence   : {RESULTS['avg_ocr_confidence']:.2%}"
    )
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=12, fontfamily="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7"))

    fig.suptitle("License Plate Vision – Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)
    save_and_close(fig, "eval_dashboard.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating charts from hard-coded evaluation results...\n")
    chart_detection_confusion()
    chart_metrics_bar()
    chart_detection_pie()
    chart_ocr_gauge()
    chart_gt_vs_det()
    chart_dashboard()
    print(f"\nAll charts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
