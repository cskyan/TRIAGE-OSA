#!/usr/bin/env python3
"""Plot matched Stage50 fusion controls without rerunning experiments.

Panel a displays every paired fold-seed utility difference in both candidate
universes. Panel b decomposes the mean 500-protein utility difference using
the fixed utility coefficients. No clinical accuracy or independent-sample
significance is inferred from these repeated fits.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CONTROLS = ("Concat_matched", "Concat_warm", "Concat_wide")
LABELS = ("Matched MLP\n2,823 parameters", "Same-start MLP\n2,823 parameters", "Wider MLP\n5,721 parameters")
SCOPES = ("all_500", "primary_250")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(source):
    raw = pd.read_csv(source, sep="\t")
    data = raw.loc[raw.method.isin(CONTROLS) & raw.scope.isin(SCOPES)].copy()
    if len(data) != 72 or not data.partition.eq("test").all():
        raise RuntimeError("Expected 72 test contrast rows: three controls, two scopes, four folds and three seeds")
    if data.duplicated(["method", "scope", "fold", "seed"]).any():
        raise RuntimeError("Duplicate paired observations")
    for method in CONTROLS:
        for scope in SCOPES:
            subset = data.loc[(data.method == method) & (data.scope == scope)]
            if len(subset) != 12 or set(subset.fold) != {1, 2, 3, 4} or subset.seed.nunique() != 3:
                raise RuntimeError("Incomplete fold-seed pairing")
    data["utility_gain"] = -data.difference_multi_axis_utility
    data["target_recovery_term"] = -0.75 * data.difference_Spearman
    data["backbone_retention_term"] = -0.25 * data.difference_structural_retention_Spearman
    data["BCE_term"] = 0.15 * data.difference_soft_BCE
    total = data[["target_recovery_term", "backbone_retention_term", "BCE_term"]].sum(axis=1)
    if not np.allclose(total, data.utility_gain, atol=1e-10, rtol=1e-8):
        raise RuntimeError("Utility decomposition does not reproduce the archived contrast")
    if not np.isfinite(data[["utility_gain", "target_recovery_term", "backbone_retention_term", "BCE_term"]]).all().all():
        raise RuntimeError("Nonfinite plotting values")
    return data


def make_figure(data, output, qa_tools):
    if qa_tools is None:
        raise RuntimeError("Supply --qa-tools or NATURE_FIGURE_QA_TOOLS with the panel-alignment tools directory")
    sys.path.insert(0, str(qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(170 / 25.4, 96 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.24, top=0.72, wspace=0.24)
    all_color, primary_color = "#718392", "#087F83"
    term_colors = ("#526C92", "#D49A4E", "#B9C0C7")
    y_positions = np.array([2.0, 1.0, 0.0])
    offset_by_scope = {"all_500": 0.13, "primary_250": -0.13}
    jitter = np.linspace(-0.060, 0.060, 12)

    for row, method in enumerate(CONTROLS):
        for scope, color, marker in (("all_500", all_color, "o"), ("primary_250", primary_color, "s")):
            subset = data.loc[(data.method == method) & (data.scope == scope)].sort_values(["fold", "seed"])
            center = y_positions[row] + offset_by_scope[scope]
            axes[0].scatter(subset.utility_gain, center + jitter, s=12, marker=marker,
                            color=color, alpha=0.50, edgecolors="none", zorder=3)
            axes[0].scatter([subset.utility_gain.mean()], [center], s=31, marker="D",
                            facecolor=color, edgecolor="white", linewidth=0.5, zorder=4)

    axes[0].set_yticks(y_positions, LABELS)
    axes[0].set_xlim(-0.002, 0.036)
    axes[0].set_xticks([0.00, 0.01, 0.02, 0.03])
    axes[0].set_xlabel("Utility difference\nStructured model minus MLP", labelpad=7)
    scope_handles = [Line2D([], [], color=all_color, marker="o", linestyle="none", markersize=4,
                            label="500-protein universe"),
                     Line2D([], [], color=primary_color, marker="s", linestyle="none", markersize=4,
                            label="250-protein subset")]
    axes[0].legend(handles=scope_handles, loc="lower left", bbox_to_anchor=(-0.01, 1.035),
                   frameon=False, handletextpad=0.5, borderaxespad=0, labelspacing=0.45)

    means = data.loc[data.scope == "all_500"].groupby("method")[[
        "target_recovery_term", "backbone_retention_term", "BCE_term", "utility_gain"]].mean().loc[list(CONTROLS)]
    for row, method in enumerate(CONTROLS):
        y = y_positions[row]
        positive, negative = 0.0, 0.0
        for field, color in zip(("target_recovery_term", "backbone_retention_term", "BCE_term"), term_colors):
            value = float(means.loc[method, field])
            start = positive if value >= 0 else negative
            axes[1].barh(y, value, left=start, height=0.34, color=color, edgecolor="none", zorder=2)
            if value >= 0:
                positive += value
            else:
                negative += value
        axes[1].scatter([means.loc[method, "utility_gain"]], [y], marker="D", s=26,
                        facecolor="#222222", edgecolor="white", linewidth=0.4, zorder=4)
    axes[1].set_xlim(-0.080, 0.105)
    axes[1].set_xticks([-0.05, 0.00, 0.05, 0.10])
    axes[1].set_xlabel("Contribution to mean utility difference\n500-protein universe", labelpad=7)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    term_handles = [Patch(facecolor=term_colors[0], label="Target recovery"),
                    Patch(facecolor=term_colors[1], label="Backbone retention"),
                    Patch(facecolor=term_colors[2], label="Soft-label BCE"),
                    Line2D([], [], color="#222222", marker="D", linestyle="none", markersize=4, label="Net difference")]
    axes[1].legend(handles=term_handles, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.035),
                   frameon=False, columnspacing=0.8, handlelength=1.1, handletextpad=0.4,
                   borderaxespad=0, labelspacing=0.45)

    for ax, label, title in zip(axes, ("a", "b"), ("Capacity and initialization controls", "Source of the utility difference")):
        ax.set_ylim(-0.5, 2.5)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=3)
        ax.axvline(0, color="#444444", lw=0.65, zorder=1)
        ax.grid(axis="x", color="#E9ECEF", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-14, 40), textcoords="offset points",
                    fontsize=10, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 40), textcoords="offset points",
                    fontsize=8.5, ha="left", va="bottom", annotation_clip=False)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(output) + ".alignment.json",
        overlay_svg=str(output) + ".alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(str(output) + ".pdf")
    fig.savefig(str(output) + ".svg")
    fig.savefig(str(output) + ".png", dpi=600)
    plt.close(fig)
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, default=Path("results/discovery_preprocess_v2/50_fusion_capacity_controls_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/fusion_capacity/fusion_capacity_controls"))
    parser.add_argument("--qa-tools", type=Path, default=os.environ.get("NATURE_FIGURE_QA_TOOLS", str(Path(__file__).resolve().parents[1] / "plotting/qa")))
    args = parser.parse_args()
    root = args.root.resolve()
    results = args.results if args.results.is_absolute() else root / args.results
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    source = results / "paired_differences_vs_common_Full.tsv"
    data = prepare_data(source)
    data.to_csv(str(output) + ".source.tsv", sep="\t", index=False)
    means = make_figure(data, output, args.qa_tools)
    means.to_csv(str(output) + ".decomposition.tsv", sep="\t")
    metadata = dict(input_sha256=sha(source), rows_before=len(pd.read_csv(source, sep="\t")),
        rows_plotted=len(data), folds=4, seeds=3, scopes=list(SCOPES), controls=list(CONTROLS),
        structured_model_parameters=2824, inference="descriptive paired differences; no iid clinical significance test",
        omitted_comparisons="Full self-comparison and Native_full original-objective reference are not common-objective controls",
        panel_b="Arithmetic decomposition of 12-run mean; variation shown in panel a, not a confidence interval",
        figure_archetype="quantitative grid", backend="Python/matplotlib")
    Path(str(output) + ".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Created " + str(output))


if __name__ == "__main__":
    main()
