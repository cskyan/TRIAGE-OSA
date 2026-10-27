#!/usr/bin/env python3
"""Create publication-grade TRIAGE-OSA V4 result figures from frozen outputs."""

from __future__ import print_function

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "discovery_preprocess_v2"
OUT = BASE / "26_v4_publication_figures"
SOURCE = OUT / "source_data"
COMBINED = OUT / "combined"
PANELS = OUT / "panels"

COLORS = {
    "v4": "#167C80",
    "v32": "#8B95A1",
    "gse133601": "#2F6B9A",
    "gse49800": "#8A5FA8",
    "accent": "#C65D3A",
    "dark": "#27323A",
    "mid": "#68747C",
    "light": "#D9E1E5",
    "pale_blue": "#EAF2F7",
    "pale_violet": "#F2ECF6",
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax, label):
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def clean_axis(ax):
    ax.grid(axis="x", color="#E7EBED", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    COMBINED.mkdir(parents=True, exist_ok=True)
    svg_path = COMBINED / (stem + ".svg")
    pdf_path = COMBINED / (stem + ".pdf")
    tiff_path = COMBINED / (stem + ".tiff")
    png_path = COMBINED / (stem + ".png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    return [svg_path, pdf_path, tiff_path, png_path]


def save_panel(fig, ax, figure_id, panel_stem, extra_axes=None):
    """Export one axes as an independently usable vector/raster panel."""
    panel_dir = PANELS / figure_id
    panel_dir.mkdir(parents=True, exist_ok=True)
    keep_axes = [ax] + list(extra_axes or [])
    visibility = {item: item.get_visible() for item in fig.axes}
    for item in fig.axes:
        item.set_visible(item in keep_axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [item.get_tightbbox(renderer) for item in keep_axes]
    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.04, 1.06)
    paths = []
    for suffix, dpi in (("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)):
        path = panel_dir / (panel_stem + "." + suffix)
        kwargs = {"bbox_inches": bbox, "facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    for item, was_visible in visibility.items():
        item.set_visible(was_visible)
    fig.canvas.draw()
    return paths


def rounded_box(ax, xy, width, height, title, lines, facecolor, edgecolor):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9, facecolor=facecolor, edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.025, xy[1] + height - 0.045, title,
            transform=ax.transAxes, fontsize=7.0, fontweight="bold",
            color=edgecolor, va="top")
    ax.text(xy[0] + 0.025, xy[1] + height - 0.095, "\n".join(lines),
            transform=ax.transAxes, fontsize=5.6, color=COLORS["dark"],
            va="top", linespacing=1.45)
    return patch


def connect_boxes(ax, start, end, color=None):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
        color=color or COLORS["mid"], transform=ax.transAxes,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def make_figure1():
    """Schematic of the frozen discovery-to-validation design."""
    role_rows = [
        ("TUnA", "generic structural pretraining", "training only"),
        ("GSE75097", "adult OSA, severity and patient context", "development"),
        ("GSE71356", "intermittent-hypoxia response", "development"),
        ("GSE133601 / GSE49800", "paired pre/post CPAP", "locked validation"),
        ("GSE135917", "adipose cross-tissue response", "locked validation"),
        ("GSE214865 / PXD032734", "pediatric cell/protein context", "locked validation"),
        ("GSE38792 / GSE293433", "independent adult replication", "locked exploratory"),
        ("IntAct / Complex Portal", "generic network plausibility", "sanity check"),
    ]
    pd.DataFrame(role_rows, columns=["dataset", "information", "role"]).to_csv(
        SOURCE / "Fig1_dataset_roles.tsv", sep="\t", index=False
    )

    fig = plt.figure(figsize=(7.2047, 5.35))
    gs = fig.add_gridspec(2, 1, height_ratios=(1.75, 1.0), hspace=0.22,
                          left=0.035, right=0.985, top=0.965, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    for ax in (ax_a, ax_b):
        ax.set_axis_off()

    rounded_box(ax_a, (0.015, 0.53), 0.205, 0.39, "Structural foundation",
                ["TUnA interaction supervision", "ESM2 protein representations",
                 "Frozen L1–L2–L3 backbone"], "#EEF2F5", COLORS["mid"])
    rounded_box(ax_a, (0.275, 0.53), 0.205, 0.39, "OSA development evidence",
                ["OSA status and severity", "AHI and intermittent hypoxia",
                 "Discovery cohorts only"], "#EAF2F7", COLORS["gse133601"])
    rounded_box(ax_a, (0.535, 0.53), 0.205, 0.39, "Patient-information conditioning",
                ["Age and sex auditing", "Hypertension and EDS", "BMI unavailable: not imputed"],
                "#F2ECF6", COLORS["gse49800"])
    rounded_box(ax_a, (0.795, 0.53), 0.19, 0.39, "TRIAGE-OSA V4",
                ["PPI expert", "L1–L2–L3 expert", "OSA-context expert",
                 "Patient-conditioned gating"], "#E6F3F1", COLORS["v4"])
    connect_boxes(ax_a, (0.22, 0.725), (0.275, 0.725))
    connect_boxes(ax_a, (0.48, 0.725), (0.535, 0.725))
    connect_boxes(ax_a, (0.74, 0.725), (0.795, 0.725))

    rounded_box(ax_a, (0.135, 0.06), 0.25, 0.29, "Frozen candidate universe",
                ["500-protein sensitivity universe", "250-protein primary universe",
                 "124,750 and 31,125 pairs"], "#F7F8F8", COLORS["mid"])
    rounded_box(ax_a, (0.455, 0.06), 0.41, 0.29, "Validation after ranking freeze",
                ["Paired CPAP • adipose • single-cell • pediatric proteome",
                 "Independent adult cohorts • known-interaction recovery",
                 "No validation-guided retraining or candidate selection"],
                "#FFF3EC", COLORS["accent"])
    connect_boxes(ax_a, (0.89, 0.53), (0.66, 0.35), COLORS["v4"])
    connect_boxes(ax_a, (0.455, 0.205), (0.385, 0.205), COLORS["accent"])
    panel_label(ax_a, "a")

    columns = [0.02, 0.28, 0.66]
    ax_b.text(columns[0], 0.92, "Dataset", transform=ax_b.transAxes,
              fontsize=6.2, fontweight="bold")
    ax_b.text(columns[1], 0.92, "Information contributed", transform=ax_b.transAxes,
              fontsize=6.2, fontweight="bold")
    ax_b.text(columns[2], 0.92, "Analysis role", transform=ax_b.transAxes,
              fontsize=6.2, fontweight="bold")
    for i, row in enumerate(role_rows):
        y = 0.82 - i * 0.105
        if i % 2 == 0:
            ax_b.add_patch(FancyBboxPatch((0.01, y - 0.052), 0.97, 0.092,
                           boxstyle="square,pad=0", transform=ax_b.transAxes,
                           facecolor="#F6F7F7", edgecolor="none"))
        ax_b.text(columns[0], y, row[0], transform=ax_b.transAxes, fontsize=5.35, va="center")
        ax_b.text(columns[1], y, row[1], transform=ax_b.transAxes, fontsize=5.35, va="center")
        role_color = COLORS["accent"] if "validation" in row[2] else COLORS["v4"]
        ax_b.text(columns[2], y, row[2], transform=ax_b.transAxes, fontsize=5.35,
                  va="center", color=role_color, fontweight="bold")
    panel_label(ax_b, "b")
    paths = save_figure(fig, "Fig1_TRIAGE_OSA_V4_framework_and_data_isolation")
    paths += save_panel(fig, ax_a, "Fig1", "Fig1a_model_framework")
    paths += save_panel(fig, ax_b, "Fig1", "Fig1b_dataset_roles")
    plt.close(fig)
    return paths


def make_figure2():
    """Protein-disjoint internal evaluation and patient-information ablation."""
    train = BASE / "06_model_training" / "triage_osa_patient_information_v4"
    cv = pd.read_csv(train / "cross_validation_metrics.tsv", sep="\t")
    availability = pd.read_csv(train / "patient_information_availability.tsv", sep="\t")
    summary = load_json(train / "run_summary.json")
    metrics = ["Spearman", "weak_top20_AUPRC", "top5_enrichment_ratio", "multi_axis_utility"]
    fold = cv.groupby(["outer_fold", "model"], as_index=False)[metrics].mean()
    selected_models = [
        "TRIAGE_OSA_specialized", "molecular_context_without_patient_information",
        "patient_information_only", "frozen_TRIAGE_p_PPI", "equal_L1_L2_L3",
    ]
    fold = fold[fold["model"].isin(selected_models)].copy()
    fold.to_csv(SOURCE / "Fig2_fourfold_internal_metrics.tsv", sep="\t", index=False)
    availability.to_csv(SOURCE / "Fig2_patient_information_availability.tsv", sep="\t", index=False)

    labels = {
        "TRIAGE_OSA_specialized": "Full V4",
        "molecular_context_without_patient_information": "No patient information",
        "patient_information_only": "Patient information only",
        "frozen_TRIAGE_p_PPI": "Frozen PPI",
        "equal_L1_L2_L3": "Equal L1–L2–L3",
    }
    order = selected_models
    fig = plt.figure(figsize=(7.2047, 5.55))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.15, 0.9),
                          height_ratios=(1.0, 1.0), left=0.09, right=0.98,
                          bottom=0.10, top=0.95, wspace=0.55, hspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    full = fold[fold.model == "TRIAGE_OSA_specialized"].set_index("outer_fold")
    nopat = fold[fold.model == "molecular_context_without_patient_information"].set_index("outer_fold")
    for f in sorted(set(full.index) & set(nopat.index)):
        ax_a.plot([0, 1], [nopat.loc[f, "multi_axis_utility"], full.loc[f, "multi_axis_utility"]],
                  color=COLORS["light"], lw=1.2)
        ax_a.scatter([0, 1], [nopat.loc[f, "multi_axis_utility"], full.loc[f, "multi_axis_utility"]],
                     c=[COLORS["v32"], COLORS["v4"]], s=20, edgecolor="white", linewidth=0.4)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["No patient\ninformation", "Full V4"])
    ax_a.set_ylabel("Multi-axis utility")
    ax_a.set_title("Patient-information gain", loc="left", fontweight="bold")
    ax_a.text(0.5, 0.02, "Mean Δ = +{:.3f}".format(
        summary["delta_multi_axis_utility_vs_molecular_context_without_patient_information"]),
        transform=ax_a.transAxes, ha="center", va="bottom", fontsize=5.5, color=COLORS["v4"])
    panel_label(ax_a, "a")

    ypos = np.arange(len(order))
    for i, model in enumerate(order):
        vals = fold.loc[fold.model == model, "multi_axis_utility"].to_numpy(float)
        ax_b.scatter(vals, np.full(len(vals), i), s=15, color=COLORS["v32"], alpha=0.75)
        ax_b.errorbar(vals.mean(), i, xerr=vals.std(ddof=1), fmt="o",
                      color=COLORS["v4"] if model == order[0] else COLORS["dark"],
                      ms=4, capsize=2, lw=0.9)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([labels[x] for x in order])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Multi-axis utility (mean ± fold SD)")
    ax_b.set_title("Internal baselines", loc="left", fontweight="bold")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    expert = summary["mean_test_expert_weights"]
    names = ["PPI", "L1–L2–L3", "OSA context"]
    vals = [expert["PPI"], expert["L1_L2_L3"], expert["OSA_context"]]
    colors = [COLORS["v32"], COLORS["gse49800"], COLORS["v4"]]
    bottom = 0
    for name, val, color in zip(names, vals, colors):
        ax_c.bar([0], [val], bottom=bottom, color=color, width=0.55, label=name)
        if val > 0.06:
            ax_c.text(0, bottom + val / 2, "{:.1f}%".format(100 * val), ha="center",
                      va="center", fontsize=5.2, color="white", fontweight="bold")
        bottom += val
    ax_c.set_ylim(0, 1)
    ax_c.set_xticks([0])
    ax_c.set_xticklabels(["Mean test gate"])
    ax_c.set_ylabel("Expert weight")
    ax_c.set_title("Adaptive fusion", loc="left", fontweight="bold")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=5.0)
    panel_label(ax_c, "c")

    full_metrics = full.reset_index()
    metric_specs = [("Spearman", "Spearman"), ("weak_top20_AUPRC", "AUPRC"),
                    ("top5_enrichment_ratio", "Top-5% enrichment")]
    x = np.arange(len(metric_specs))
    for j, (column, _) in enumerate(metric_specs):
        vals = full_metrics[column].to_numpy(float)
        ax_d.scatter(np.full(len(vals), j), vals, color=COLORS["v4"], s=18,
                     edgecolor="white", linewidth=0.4)
        ax_d.errorbar(j, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color=COLORS["dark"],
                      ms=4, capsize=2, lw=0.9)
    ax_d.axhline(1.0, color=COLORS["mid"], lw=0.6, ls=":")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([x[1] for x in metric_specs])
    ax_d.set_ylabel("Metric value (mean ± fold SD)")
    ax_d.set_title("Four-fold protein-disjoint evaluation", loc="left", fontweight="bold")
    panel_label(ax_d, "d")

    fields = ["age", "sex", "BMI", "AHI", "hypertension", "EDS"]
    lookup = availability.set_index("field")
    observed = [float(lookup.loc[f, "individual_values_observed"]) for f in fields]
    included = [str(lookup.loc[f, "included_in_v4"]).lower() == "yes" for f in fields]
    ax_e.barh(np.arange(len(fields)), observed,
              color=[COLORS["v4"] if x else COLORS["light"] for x in included])
    ax_e.set_yticks(np.arange(len(fields)))
    ax_e.set_yticklabels(fields)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Participants with observed values")
    ax_e.set_title("Patient-information coverage", loc="left", fontweight="bold")
    for i, (value, inc) in enumerate(zip(observed, included)):
        ax_e.text(value + 0.5, i, "included" if inc else "not available",
                  va="center", fontsize=5.0, color=COLORS["v4"] if inc else COLORS["mid"])
    ax_e.set_xlim(0, 40)
    clean_axis(ax_e)
    panel_label(ax_e, "e")

    paths = save_figure(fig, "Fig2_V4_internal_performance_and_ablation")
    paths += save_panel(fig, ax_a, "Fig2", "Fig2a_patient_information_gain")
    paths += save_panel(fig, ax_b, "Fig2", "Fig2b_internal_baselines")
    paths += save_panel(fig, ax_c, "Fig2", "Fig2c_adaptive_fusion_weights")
    paths += save_panel(fig, ax_d, "Fig2", "Fig2d_fourfold_internal_metrics")
    paths += save_panel(fig, ax_e, "Fig2", "Fig2e_patient_information_coverage")
    plt.close(fig)
    return paths


def make_figure4():
    """Independent adult replication and tissue-dependent pathway programs."""
    stage = BASE / "21_independent_validation_endpoint_triage_osa_v4"
    permutation = pd.read_csv(stage / "node_aware_permutation_summary.tsv", sep="\t")
    direction = pd.read_csv(stage / "candidate_directional_concordance.tsv", sep="\t")
    comparator = pd.read_csv(stage / "comparator_delta_permutation.tsv", sep="\t")
    pathway = pd.read_csv(stage / "reactome_pathway_axis_results.tsv", sep="\t")
    full = permutation[permutation.model == "full_TRIAGE_OSA"].copy()
    axes = pathway[pathway.level == "axis"].copy()
    full.to_csv(SOURCE / "Fig4_independent_permutation_endpoints.tsv", sep="\t", index=False)
    direction.to_csv(SOURCE / "Fig4_candidate_directional_concordance.tsv", sep="\t", index=False)
    comparator.to_csv(SOURCE / "Fig4_full_minus_comparator.tsv", sep="\t", index=False)
    axes.to_csv(SOURCE / "Fig4_pathway_axes.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2047, 5.65))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 0.92, 1.45),
                          left=0.09, right=0.98, bottom=0.10, top=0.95,
                          wspace=0.58, hspace=0.50)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1:3])

    top = full[full.statistic == "top_1pct_support_enrichment_ratio"].reset_index(drop=True)
    y = np.arange(len(top))
    ax_a.axvline(1, color=COLORS["dark"], lw=0.7, ls="--")
    ax_a.hlines(y, top.null_q025, top.null_q975, color=COLORS["light"], lw=5)
    ax_a.scatter(top.observed, y, color=COLORS["v4"], s=25, zorder=3)
    for yi, row in enumerate(top.itertuples(index=False)):
        offset = (5, -13) if str(row.endpoint) == "joint" else (5, 0)
        ax_a.annotate("{:.2f}; P={:.3f}".format(row.observed, row.one_sided_P),
                      (row.observed, yi), xytext=offset, textcoords="offset points",
                      va="center", fontsize=5.0)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(top.endpoint)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Top-1% support enrichment ratio")
    ax_a.set_title("Locked independent endpoints", loc="left", fontweight="bold")
    ax_a.text(1.0, 1.015, "Grey: permutation 95% interval  •  Teal: observed",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=5.0, color=COLORS["mid"])
    panel_label(ax_a, "a")

    y = np.arange(len(direction))
    ax_b.axvline(0.5, color=COLORS["dark"], lw=0.7, ls="--")
    ax_b.barh(y, direction.same_direction_fraction, color=[COLORS["gse133601"], COLORS["gse49800"]], height=0.55)
    for yi, row in enumerate(direction.itertuples(index=False)):
        ax_b.text(row.same_direction_fraction + 0.008, yi,
                  "{}/{}; P={:.3f}".format(row.same_direction_candidates,
                                            row.measurable_primary_candidates,
                                            row.one_sided_exact_sign_P),
                  va="center", fontsize=5.0)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(direction.dataset)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.45, 0.66)
    ax_b.set_xlabel("Same-direction candidate fraction")
    ax_b.set_title("Candidate-level direction", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    axis_order = ["hypoxia_HIF_ROS", "NFkB_innate_inflammation",
                  "antigen_lymphocyte_signalling", "PI3K_AKT_JAK_STAT_TGFb",
                  "proteostasis_ER_stress", "PPAR_adipose_lipid_metabolism"]
    datasets = ["GSE38792", "GSE293433"]
    eff = axes.pivot(index="dataset", columns="axis", values="competitive_effect").reindex(index=datasets, columns=axis_order)
    fdr = axes.pivot(index="dataset", columns="axis", values="BH_FDR_axis_family").reindex(index=datasets, columns=axis_order)
    short = ["Hypoxia/\nHIF/ROS", "NF-κB/\ninnate", "Antigen/\nlymphocyte",
             "PI3K–AKT/\nJAK–STAT", "Proteostasis/\nER stress", "PPAR/\nlipid"]
    im = ax_c.imshow(eff.to_numpy(float), aspect="auto", cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-0.85, vcenter=0, vmax=0.85))
    ax_c.set_xticks(np.arange(len(short)))
    ax_c.set_xticklabels(short)
    ax_c.set_yticks(np.arange(len(datasets)))
    ax_c.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(short)):
            value = eff.iloc[i, j]
            q = fdr.iloc[i, j]
            stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
            color = "white" if abs(value) > 0.43 else COLORS["dark"]
            ax_c.text(j, i, "{:+.2f}\n{}".format(value, stars), ha="center", va="center",
                      fontsize=5.5, color=color, fontweight="bold")
    cb = fig.colorbar(im, ax=ax_c, fraction=0.035, pad=0.025)
    cb.set_label("Competitive pathway effect", fontsize=5.5)
    cb.ax.tick_params(labelsize=5.0)
    ax_c.set_title("Pre-specified pathway programs are cohort dependent", loc="left", fontweight="bold")
    ax_c.text(0.0, -0.25, "* FDR<0.05; ** FDR<0.01; *** FDR<0.001; ns, not significant",
              transform=ax_c.transAxes, fontsize=5.0, color=COLORS["mid"])
    panel_label(ax_c, "c")

    paths = save_figure(fig, "Fig4_V4_independent_validation_and_pathway_heterogeneity")
    paths += save_panel(fig, ax_a, "Fig4", "Fig4a_locked_independent_endpoints")
    paths += save_panel(fig, ax_b, "Fig4", "Fig4b_candidate_directional_concordance")
    paths += save_panel(fig, ax_c, "Fig4", "Fig4c_pathway_heterogeneity", extra_axes=[cb.ax])
    plt.close(fig)
    return paths


def endpoint_rows():
    rows = []
    for version, suffix in (("V3.2", "v3_2"), ("V4", "v4")):
        stage11 = load_json(BASE / ("11_cpap_reversal_endpoint_triage_osa_" + suffix) / "run_summary.json")
        for dataset in ("GSE133601", "GSE49800"):
            item = stage11["cohorts"][dataset]
            rows.append({
                "version": version,
                "endpoint": "CPAP " + dataset,
                "ratio": item["primary_top_1pct_enrichment_ratio"],
                "P_value": item["top_1pct_permutation_P"],
                "AUEC": item["primary_AUEC"],
            })
        rows.append({
            "version": version,
            "endpoint": "CPAP pooled",
            "ratio": stage11["meta"]["pooled_enrichment_ratio"],
            "P_value": stage11["meta"]["one_sided_P"],
            "AUEC": np.nan,
        })
        stage14 = load_json(BASE / ("14_adipose_cross_tissue_endpoint_triage_osa_" + suffix) / "run_summary.json")
        rows.extend([
            {
                "version": version,
                "endpoint": "Adipose OSA state",
                "ratio": stage14["primary"]["state_top_1pct_ratio"],
                "P_value": stage14["primary"]["state_node_aware_P"],
                "AUEC": stage14["primary"]["state_AUEC"],
            },
            {
                "version": version,
                "endpoint": "Adipose CPAP",
                "ratio": stage14["primary"]["CPAP_top_1pct_ratio"],
                "P_value": stage14["primary"]["CPAP_node_aware_P"],
                "AUEC": stage14["primary"]["CPAP_AUEC"],
            },
        ])
    return pd.DataFrame(rows)


def make_figure3():
    subject_path = BASE / "25_cpap_subject_clinical_package_triage_osa_v4" / "subject_molecular_response_summary.tsv"
    subjects = pd.read_csv(subject_path, sep="\t")
    endpoints = endpoint_rows()
    signflip = load_json(BASE / "12_cpap_subject_signflip_triage_osa_v4" / "run_summary.json")

    subjects.to_csv(SOURCE / "Fig3_subject_molecular_response.tsv", sep="\t", index=False)
    endpoints.to_csv(SOURCE / "Fig3_V3_2_vs_V4_external_endpoints.tsv", sep="\t", index=False)
    sign_rows = []
    for dataset in ("GSE133601", "GSE49800"):
        item = signflip["cohorts"][dataset]
        sign_rows.append({
            "cohort": dataset,
            "subjects": item["subjects"],
            "top_1pct_ratio": item["observed_top_1pct_enrichment_ratio_unmoderated_t"],
            "P_value": item["top_1pct_one_sided_signflip_P"],
            "AUEC": item["observed_AUEC_unmoderated_t"],
            "AUEC_P_value": item["AUEC_one_sided_signflip_P"],
        })
    sign_rows.append({
        "cohort": "Pooled",
        "subjects": 33,
        "top_1pct_ratio": float(signflip["meta"]["pooled_enrichment_ratio"]),
        "P_value": float(signflip["meta"]["one_sided_P_normal_approximation"]),
        "AUEC": np.nan,
        "AUEC_P_value": np.nan,
    })
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(SOURCE / "Fig3_CPAP_subject_signflip.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2047, 5.9055))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.28, 1.0), height_ratios=(1.0, 1.0),
                          left=0.09, right=0.98, bottom=0.09, top=0.96,
                          wspace=0.56, hspace=0.46)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ordered = []
    ypos = []
    y = 0
    group_bounds = []
    for dataset in ("GSE133601", "GSE49800"):
        group = subjects[subjects["dataset"] == dataset].sort_values(
            "V4_priority_weighted_mean_oriented_post_minus_pre"
        )
        start = y
        for _, row in group.iterrows():
            ordered.append(row)
            ypos.append(y)
            y += 1
        group_bounds.append((dataset, start - 0.5, y - 0.5))
        y += 1.4
    ordered = pd.DataFrame(ordered).reset_index(drop=True)
    values = ordered["V4_priority_weighted_mean_oriented_post_minus_pre"].to_numpy(float)
    fractions = ordered["V4_top20_priority_reversal_fraction"].to_numpy(float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "reversal", ["#3C6E9B", "#F4F4F2", "#C65D3A"]
    )
    norm = TwoSlopeNorm(vmin=min(0.33, float(fractions.min())), vcenter=0.5,
                        vmax=max(0.69, float(fractions.max())))
    ax_a.axvline(0, color=COLORS["dark"], lw=0.7, ls="--", zorder=1)
    for dataset, lo, hi in group_bounds:
        color = COLORS["pale_blue"] if dataset == "GSE133601" else COLORS["pale_violet"]
        ax_a.axhspan(lo, hi, color=color, zorder=0)
        ax_a.text(0.98, lo + 0.35, dataset, fontsize=5.2, fontweight="bold",
                  color=COLORS["mid"], va="top", ha="right",
                  transform=ax_a.get_yaxis_transform())
    for yi, value in zip(ypos, values):
        ax_a.plot([0, value], [yi, yi], color="#AEB8BD", lw=0.7, zorder=2)
    scatter = ax_a.scatter(values, ypos, c=fractions, cmap=cmap, norm=norm,
                           s=18, edgecolor="white", linewidth=0.4, zorder=3)
    labels = [str(row["subject_id"]) for _, row in ordered.iterrows()]
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("V4-priority-weighted oriented CPAP effect")
    ax_a.set_ylabel("Participant")
    ax_a.set_title("Participant-level molecular response", loc="left", fontweight="bold")
    clean_axis(ax_a)
    cbar = fig.colorbar(scatter, ax=ax_a, orientation="horizontal", fraction=0.035,
                        pad=0.085, aspect=28)
    cbar.set_label("Top-20% priority reversal fraction", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5.0, length=2)
    panel_label(ax_a, "a")

    endpoint_order = ["CPAP GSE133601", "CPAP GSE49800", "CPAP pooled",
                      "Adipose OSA state", "Adipose CPAP"]
    pivot = endpoints.pivot(index="endpoint", columns="version", values="ratio").loc[endpoint_order]
    p_v4 = endpoints[endpoints["version"] == "V4"].set_index("endpoint")["P_value"].loc[endpoint_order]
    yy = np.arange(len(endpoint_order))
    x32 = pivot["V3.2"].to_numpy(float)
    x4 = pivot["V4"].to_numpy(float)
    ax_b.axvline(1, color=COLORS["dark"], lw=0.7, ls="--")
    for i in range(len(yy)):
        ax_b.plot([x32[i], x4[i]], [yy[i], yy[i]], color=COLORS["light"], lw=1.5)
    ax_b.scatter(x32, yy, color=COLORS["v32"], s=18, label="V3.2", zorder=3)
    ax_b.scatter(x4, yy, color=COLORS["v4"], s=22, label="V4", zorder=4)
    for i, (value, pval) in enumerate(zip(pivot["V4"], p_v4)):
        ax_b.text(x4[i] + 0.025, yy[i], "{:.2f}; P={:.3f}".format(value, pval),
                  va="center", ha="left", fontsize=5.0, color=COLORS["dark"])
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(endpoint_order)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Top-1% enrichment ratio")
    ax_b.set_title("External endpoint shift", loc="left", fontweight="bold")
    ax_b.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    for dataset, color, marker in (("GSE133601", COLORS["gse133601"], "o"),
                                   ("GSE49800", COLORS["gse49800"], "s")):
        group = subjects[subjects["dataset"] == dataset]
        xval = group["V4_priority_weighted_mean_oriented_post_minus_pre"].to_numpy(float)
        yval = group["V4_top20_priority_reversal_fraction"].to_numpy(float)
        ax_c.scatter(xval, yval, s=22, marker=marker, facecolor=color,
                     edgecolor="white", linewidth=0.45, alpha=0.9, label=dataset)
        if len(group) > 2:
            r = np.corrcoef(xval, yval)[0, 1]
            coef = np.polyfit(xval, yval, 1)
            xp = np.linspace(xval.min(), xval.max(), 50)
            ax_c.plot(xp, coef[0] * xp + coef[1], color=color, lw=1.0)
            anchor_x = xval.max()
            anchor_y = coef[0] * anchor_x + coef[1]
            ax_c.text(anchor_x, anchor_y, "  r={:.2f}".format(r), color=color,
                      fontsize=5.0, va="center", ha="left")
    ax_c.axvline(0, color=COLORS["dark"], lw=0.7, ls="--")
    ax_c.axhline(0.5, color=COLORS["mid"], lw=0.6, ls=":")
    ax_c.set_xlabel("V4-priority-weighted oriented CPAP effect")
    ax_c.set_ylabel("Top-20% priority reversal fraction")
    ax_c.set_title("Response heterogeneity", loc="left", fontweight="bold")
    ax_c.legend(loc="lower right")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    paths = save_figure(fig, "Fig3_V4_patient_response_heterogeneity")
    paths += save_panel(fig, ax_a, "Fig3", "Fig3a_participant_molecular_response", extra_axes=[cbar.ax])
    paths += save_panel(fig, ax_b, "Fig3", "Fig3b_external_endpoint_shift")
    paths += save_panel(fig, ax_c, "Fig3", "Fig3c_response_heterogeneity")
    plt.close(fig)
    return paths


def display_name(value):
    mapping = {
        "full_OSA_priority": "Full V4",
        "context_free_p_PPI": "PPI only",
        "context_only_r_OSA": "OSA context only",
        "L3_only": "L3 only",
        "L1_L2": "L1 + L2",
        "L1_L3": "L1 + L3",
        "L2_L3": "L2 + L3",
        "equal_L1_L2_L3": "Equal L1–L2–L3",
        "PPI25_context75": "PPI25 + context75",
        "PPI75_context25": "PPI75 + context25",
        "bidirectional_conservative": "Bidirectional conservative",
    }
    return mapping.get(value, value)


def make_legacy_network_sanity_figure():
    stage19 = BASE / "19_known_ppi_complex_recovery_triage_osa_v4"
    recovery = pd.read_csv(stage19 / "ranking_recovery_metrics.tsv", sep="\t")
    comparisons = pd.read_csv(stage19 / "full_vs_baseline_comparisons.tsv", sep="\t")
    coverage = pd.read_csv(stage19 / "reference_coverage.tsv", sep="\t")

    top1 = recovery[(recovery["reference"] == "IntAct_all_training_excluded") &
                    (np.isclose(recovery["fraction"], 0.01))].copy()
    top1["display_model"] = top1["model"].map(display_name)
    top1.to_csv(SOURCE / "Fig2_IntAct_all_top1_recovery.tsv", sep="\t", index=False)

    comp = comparisons[comparisons["reference"] == "IntAct_all_training_excluded"].copy()
    comp["display_comparator"] = comp["comparator"].map(display_name)
    comp.to_csv(SOURCE / "Fig2_full_minus_baseline_IntAct_all.tsv", sep="\t", index=False)
    coverage.to_csv(SOURCE / "Fig2_reference_coverage.tsv", sep="\t", index=False)

    version_rows = []
    for version, suffix in (("V3.2", "v3_2"), ("V4", "v4")):
        frame = pd.read_csv(
            BASE / ("19_known_ppi_complex_recovery_triage_osa_" + suffix) /
            "ranking_recovery_metrics.tsv", sep="\t"
        )
        keep = frame[(frame["model"] == "full_OSA_priority") &
                     (np.isclose(frame["fraction"], 0.01)) &
                     (frame["reference"].isin([
                         "IntAct_physical_training_excluded",
                         "IntAct_all_training_excluded",
                     ]))].copy()
        keep["version"] = version
        version_rows.append(keep)
    version_df = pd.concat(version_rows, ignore_index=True)
    version_df.to_csv(SOURCE / "Fig2_V3_2_vs_V4_IntAct_recovery.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2047, 5.0394))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0), height_ratios=(1.18, 0.82),
                          left=0.10, right=0.98, bottom=0.11, top=0.95,
                          wspace=0.58, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    plot = top1.sort_values("top_enrichment_ratio", ascending=True).reset_index(drop=True)
    yy = np.arange(len(plot))
    vals = plot["top_enrichment_ratio"].to_numpy(float)
    colors = [COLORS["v4"] if m == "full_OSA_priority" else
              COLORS["accent"] if m == "context_only_r_OSA" else COLORS["v32"]
              for m in plot["model"]]
    ax_a.axvline(1, color=COLORS["dark"], lw=0.7, ls="--")
    for y, val, color in zip(yy, vals, colors):
        ax_a.plot([1, val], [y, y], color=color, lw=1.4, alpha=0.72)
    ax_a.scatter(vals, yy, color=colors, s=24, edgecolor="white", linewidth=0.45, zorder=3)
    for y, val, pval in zip(yy, vals, plot["one_sided_hypergeometric_p"]):
        p_text = "P<0.001" if pval < 0.001 else "P={:.3f}".format(pval)
        ax_a.text(val + 0.10, y, "{:.2f}  {}".format(val, p_text),
                  va="center", ha="left", fontsize=5.0)
    ax_a.set_yticks(yy)
    ax_a.set_yticklabels(plot["display_model"])
    ax_a.set_xlabel("Top-1% IntAct-all enrichment ratio")
    ax_a.set_title("Known-interaction recovery", loc="left", fontweight="bold")
    ax_a.set_xlim(min(-0.2, vals.min() - 0.3), vals.max() + 1.2)
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    comp_plot = comp.sort_values("observed_reference_full_minus_comparator").reset_index(drop=True)
    yy = np.arange(len(comp_plot))
    diffs = comp_plot["observed_reference_full_minus_comparator"].to_numpy(float)
    sig = comp_plot["BH_FDR_within_reference"].to_numpy(float) < 0.05
    point_colors = np.where(sig, COLORS["v4"], COLORS["v32"])
    ax_b.axvline(0, color=COLORS["dark"], lw=0.7, ls="--")
    for y, val, color in zip(yy, diffs, point_colors):
        ax_b.plot([0, val], [y, y], color=color, lw=1.3)
    ax_b.scatter(diffs, yy, c=point_colors, s=20, edgecolor="white", linewidth=0.4, zorder=3)
    for y, fdr in zip(yy, comp_plot["BH_FDR_within_reference"]):
        ax_b.text(0.094, y, "{:.3f}".format(fdr), ha="left", va="center", fontsize=5.0)
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(comp_plot["display_comparator"])
    ax_b.set_xlabel("Full V4 − comparator reference percentile")
    ax_b.set_title("Fusion advantage versus baselines", loc="left", fontweight="bold")
    ax_b.text(
        0.94,
        1.025,
        "BH FDR",
        transform=ax_b.transAxes,
        fontsize=5.0,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    ax_b.set_xlim(-0.055, 0.125)
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    refs = ["IntAct_physical_training_excluded", "IntAct_all_training_excluded"]
    ref_labels = ["IntAct physical", "IntAct all"]
    pv = version_df.pivot(index="reference", columns="version", values="top_enrichment_ratio").loc[refs]
    yy = np.arange(2)
    x32 = pv["V3.2"].to_numpy(float)
    x4 = pv["V4"].to_numpy(float)
    y32 = yy - 0.07
    y4 = yy + 0.07
    for i in range(2):
        ax_c.plot([x32[i], x4[i]], [y32[i], y4[i]], color=COLORS["light"], lw=1.7)
    ax_c.scatter(x32, y32, color=COLORS["v32"], s=22, label="V3.2", zorder=3)
    ax_c.scatter(x4, y4, color=COLORS["v4"], s=24, label="V4", zorder=4)
    for i in range(2):
        ax_c.text(x32[i] - 0.08, y32[i], "{:.2f}".format(x32[i]),
                  va="center", ha="right", fontsize=5.0, color=COLORS["v32"])
        ax_c.text(x4[i] + 0.08, y4[i], "{:.2f}".format(x4[i]),
                  va="center", ha="left", fontsize=5.0, color=COLORS["v4"])
    ax_c.axvline(1, color=COLORS["dark"], lw=0.7, ls="--")
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels(ref_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Top-1% enrichment ratio")
    ax_c.set_title("V3.2-to-V4 change", loc="left", fontweight="bold")
    ax_c.legend(loc="upper center", ncol=2, handletextpad=0.4, columnspacing=0.8)
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    paths = save_figure(fig, "ExtendedData_V4_known_interaction_recovery")
    plt.close(fig)
    return paths


def make_figure5():
    """Known-interaction sanity, version stability and final mechanism modules."""
    stage19 = BASE / "19_known_ppi_complex_recovery_triage_osa_v4"
    stage22 = BASE / "22_consolidated_evidence_triage_osa_v4"
    recovery = pd.read_csv(stage19 / "ranking_recovery_metrics.tsv", sep="\t")
    stability = pd.read_csv(stage22 / "version_stability_metrics.tsv", sep="\t")
    pareto = pd.read_csv(stage22 / "pareto_front_candidates.tsv", sep="\t", low_memory=False)
    modules = pd.read_csv(stage22 / "mechanism_modules.tsv", sep="\t")
    protein = pd.read_csv(stage22 / "protein_evidence_matrix.tsv", sep="\t")

    top1 = recovery[(recovery.reference == "IntAct_all_training_excluded") &
                    np.isclose(recovery.fraction, 0.01)].copy()
    top1["display_model"] = top1.model.map(display_name)
    top1.to_csv(SOURCE / "Fig5_IntAct_top1_recovery.tsv", sep="\t", index=False)
    stability.to_csv(SOURCE / "Fig5_V3_2_V4_stability.tsv", sep="\t", index=False)

    axis_specs = [
        ("CPAP-1", "cpap_GSE133601_evaluable", "cpap_GSE133601_supportive"),
        ("CPAP-2", "cpap_GSE49800_evaluable", "cpap_GSE49800_supportive"),
        ("Adipose", "adipose_state_evaluable", "adipose_state_supportive"),
        ("Adipose CPAP", "adipose_CPAP_evaluable", "adipose_CPAP_supportive"),
        ("scRNA", "pediatric_scRNA_evaluable", "pediatric_scRNA_supportive"),
        ("Proteome", "pediatric_proteome_evaluable", "pediatric_proteome_supportive"),
        ("Known network", "known_network_evaluable", "known_network_supportive"),
        ("GSE38792", "independent_GSE38792_evaluable", "independent_GSE38792_supportive"),
        ("GSE293433", "independent_GSE293433_evaluable", "independent_GSE293433_supportive"),
    ]
    pareto = pareto.sort_values("triage_osa_rank_primary_250").reset_index(drop=True)
    matrix = np.full((len(pareto), len(axis_specs)), np.nan)
    for j, (_, eval_col, support_col) in enumerate(axis_specs):
        evaluable = pareto[eval_col].astype(bool).to_numpy()
        supportive = pareto[support_col].astype(bool).to_numpy()
        matrix[evaluable, j] = supportive[evaluable].astype(float)
    heat = pd.DataFrame(matrix, columns=[x[0] for x in axis_specs])
    heat.insert(0, "pair", pareto.gene_A.astype(str) + "–" + pareto.gene_B.astype(str))
    heat.to_csv(SOURCE / "Fig5_Pareto_pair_evidence.tsv", sep="\t", index=False)
    modules.to_csv(SOURCE / "Fig5_mechanism_modules.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2047, 7.15))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.25), height_ratios=(0.9, 1.35),
                          left=0.10, right=0.98, bottom=0.08, top=0.96,
                          wspace=0.50, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot = top1.sort_values("top_enrichment_ratio").reset_index(drop=True)
    y = np.arange(len(plot))
    vals = plot.top_enrichment_ratio.to_numpy(float)
    colors = [COLORS["v4"] if m == "full_OSA_priority" else
              COLORS["accent"] if m == "context_only_r_OSA" else COLORS["v32"]
              for m in plot.model]
    ax_a.axvline(1, color=COLORS["dark"], lw=0.7, ls="--")
    ax_a.hlines(y, 1, vals, color=colors, lw=1.3)
    ax_a.scatter(vals, y, c=colors, s=21, edgecolor="white", linewidth=0.4)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(plot.display_model)
    ax_a.set_xlabel("Top-1% IntAct-all enrichment ratio")
    ax_a.set_title("Generic network plausibility", loc="left", fontweight="bold")
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    metric_order = ["all_pair_rank_Spearman", "top_1pct_pairs", "top_5pct_pairs",
                    "pareto_pairs", "assay_shortlist_proteins", "mechanism_module_proteins"]
    labels = ["All-pair rank ρ", "Top 1% retention", "Top 5% retention",
              "Pareto pairs", "Assay shortlist", "Module proteins"]
    lookup = stability.set_index("comparison")
    vals = []
    for key in metric_order:
        row = lookup.loc[key]
        vals.append(float(row["value"]) if key == "all_pair_rank_Spearman" else
                    float(row["retention_fraction_of_v3_2"]))
    y = np.arange(len(vals))
    ax_b.barh(y, vals, color=[COLORS["v4"] if v >= 0.9 else COLORS["gse133601"] for v in vals])
    for yi, value in enumerate(vals):
        ax_b.text(min(value + 0.02, 1.04), yi, "{:.1f}%".format(100 * value), va="center", fontsize=5.2)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.12)
    ax_b.set_xlabel("V3.2 retained in V4")
    ax_b.set_title("Model-update stability", loc="left", fontweight="bold")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    cmap = mpl.colors.ListedColormap(["#E6A08D", "#5FA9A5"])
    cmap.set_bad("#E6E8E9")
    ax_c.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest")
    ax_c.set_xticks(np.arange(len(axis_specs)))
    ax_c.set_xticklabels([x[0] for x in axis_specs], rotation=48,
                         rotation_mode="anchor", ha="right")
    ax_c.set_yticks(np.arange(len(pareto)))
    ax_c.set_yticklabels(heat["pair"], fontsize=5.0)
    ax_c.set_title("Thirty-two Pareto pairs across external evidence", loc="left", fontweight="bold")
    ax_c.text(0.0, -0.23, "Teal, supportive; coral, opposing; grey, not evaluable",
              transform=ax_c.transAxes, fontsize=5.0, color=COLORS["mid"])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    panel_label(ax_c, "c")

    protein_gene = protein.set_index("UniProt")["gene_symbol"].astype(str).to_dict()
    module_colors = [COLORS["v4"], COLORS["gse49800"], COLORS["accent"]]
    centers = [(0.33, 0.68), (0.70, 0.66), (0.58, 0.23)]
    for idx, row in enumerate(modules.itertuples(index=False)):
        accessions = str(row.proteins).split("|")
        cx, cy = centers[idx]
        radius = 0.20 if len(accessions) > 5 else 0.12
        angles = np.linspace(0, 2 * np.pi, len(accessions), endpoint=False)
        pos = {a: (cx + radius * math.cos(t), cy + radius * math.sin(t))
               for a, t in zip(accessions, angles)}
        edges = pareto[pareto.protein_A.isin(accessions) & pareto.protein_B.isin(accessions)]
        for edge in edges.itertuples(index=False):
            x1, y1 = pos[edge.protein_A]
            x2, y2 = pos[edge.protein_B]
            edge_color = COLORS["accent"] if int(edge.evidence_conflict_count) > 0 else COLORS["mid"]
            ax_d.plot([x1, x2], [y1, y2], color=edge_color, lw=0.8, alpha=0.75, zorder=1)
        for accession in accessions:
            x, yv = pos[accession]
            ax_d.scatter(x, yv, s=25, color=module_colors[idx], edgecolor="white", linewidth=0.5, zorder=2)
            ax_d.text(x, yv - 0.035, protein_gene.get(accession, accession), fontsize=5.0,
                      ha="center", va="top")
        ax_d.text(cx, cy + radius + 0.08, "{} (n={})".format(row.module_id, len(accessions)),
                  ha="center", fontsize=5.8, fontweight="bold", color=module_colors[idx])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_title("Final mechanism modules", loc="left", fontweight="bold")
    ax_d.text(0.02, 0.02, "Coral edges indicate cross-axis evidence conflict",
              transform=ax_d.transAxes, fontsize=5.0, color=COLORS["mid"])
    panel_label(ax_d, "d")

    paths = save_figure(fig, "Fig5_V4_network_plausibility_and_mechanism_modules")
    paths += save_panel(fig, ax_a, "Fig5", "Fig5a_known_interaction_recovery")
    paths += save_panel(fig, ax_b, "Fig5", "Fig5b_model_update_stability")
    paths += save_panel(fig, ax_c, "Fig5", "Fig5c_Pareto_external_evidence")
    paths += save_panel(fig, ax_d, "Fig5", "Fig5d_mechanism_modules")
    plt.close(fig)
    return paths


def write_notes():
    contract = """# TRIAGE-OSA V4 main-figure contract and QA

## Figure 1
Core conclusion: the patient-informed V4 model preserves the L1–L2–L3 backbone and enforces discovery–validation isolation. Validation cohorts are never used for retraining or candidate selection.

## Figure 2
Core conclusion: patient information adds internal multi-axis utility across protein-disjoint folds while structural expert mass remains constrained. These are discovery weak-target recovery metrics, not clinical accuracy.

## Figure 3
Core conclusion: paired CPAP molecular responses vary substantially between participants and cohorts. Molecular response is not labelled as clinical response.

## Figure 4
Core conclusion: independent cohorts show modest candidate-level directional concordance but no significant global top-pair enrichment; pathway programs are strongly cohort dependent.

## Figure 5
Core conclusion: V4 retains generic network plausibility and a stable final Pareto/module set despite changes in the extreme top-ranked pairs. Conflicting external evidence is displayed rather than removed.

## Export and integrity contract
- Python/matplotlib only; 183-mm double-column layout.
- Editable SVG/PDF, 600-dpi TIFF and 300-dpi PNG.
- Complete composites are stored in `combined/`; 17 independently usable panels are grouped under `panels/Fig1` through `panels/Fig5`.
- All 33 CPAP participants and all 32 Pareto pairs are displayed.
- Fold points are shown with mean and fold SD where comparable.
- Grey evidence cells are not evaluable, not negative.
- No model refitting, reranking or validation-guided candidate selection occurs during plotting.
"""
    legends = """# Figure legends

## Fig. 1 | Patient-informed TRIAGE-OSA V4 and the frozen discovery–validation design
**a,** The pretrained structural backbone, OSA development evidence and available patient information condition three model experts before ranking a frozen 500-protein sensitivity universe and 250-protein primary universe. BMI was unavailable and was not imputed. **b,** Fixed roles of the public datasets and network references. CPAP, cross-tissue, single-cell, proteomic and independent adult outcomes were evaluated only after ranking was frozen.

## Fig. 2 | Protein-disjoint internal evaluation and patient-information contribution
**a,** Paired four-fold multi-axis utility with and without patient-information conditioning. **b,** Mean and fold SD of multi-axis utility for the full model and major internal baselines; points show outer folds. **c,** Mean learned expert weights. **d,** Four-fold Spearman correlation, weak-top-20 AUPRC and top-5% enrichment; points show folds and error bars show fold SD. **e,** Individual-level availability of patient-information fields in the development cohort. These internal endpoints recover development weak targets and are not estimates of clinical prediction accuracy.

## Fig. 3 | Patient-information-conditioned ranking exposes heterogeneous molecular responses to CPAP
**a,** Priority-weighted oriented post-minus-pre molecular effects for all 33 paired participants. Positive values indicate change opposite to the frozen discovery OSA direction; colour denotes the fraction of measured top-priority proteins showing directional reversal. **b,** V3.2-to-V4 changes in top-1% enrichment across CPAP and adipose endpoints. **c,** Participant-level relationship between continuous oriented effect and reversal fraction. Lines are descriptive cohort-specific least-squares summaries.

## Fig. 4 | Independent adult cohorts reveal candidate-level concordance and pathway heterogeneity
**a,** Observed top-1% support enrichment with node-aware permutation 95% null intervals and one-sided P values. **b,** Fractions of measurable primary candidates with concordant disease direction and nominal exact sign-test P values. **c,** Competitive effects for six prespecified Reactome pathway axes; symbols denote within-dataset axis-family BH FDR. The two cohorts do not support a universal direction of pathway change.

## Fig. 5 | Network plausibility and stable multi-evidence mechanism candidates
**a,** Top-1% enrichment of training-overlap-excluded IntAct-all pairs across frozen ranking models. **b,** V3.2-to-V4 all-pair rank correlation and set retention. **c,** External evidence states for all 32 V4 Pareto pairs; grey cells were not evaluable. **d,** Three Pareto-linked mechanism components; coral edges mark at least one conflicting evidence axis. Known-interaction recovery is a generic network sanity check and does not establish OSA-specific physical binding.
"""
    (OUT / "figure_contract_and_qa.md").write_text(contract, encoding="utf-8")
    (OUT / "figure_legends.md").write_text(legends, encoding="utf-8")


def main():
    if OUT.exists():
        shutil.rmtree(str(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    COMBINED.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    outputs = make_figure1() + make_figure2() + make_figure3() + make_figure4() + make_figure5()
    write_notes()
    outputs += sorted(SOURCE.glob("*.tsv"))
    outputs += [OUT / "figure_contract_and_qa.md", OUT / "figure_legends.md"]
    summary = {
        "status": "PASS_FIVE_MAIN_FIGURES_GENERATED",
        "backend": "Python_matplotlib",
        "figures": 5,
        "standalone_panels": 17,
        "formats_per_figure_or_panel": ["svg", "pdf", "tiff", "png"],
        "source_tables": len(list(SOURCE.glob("*.tsv"))),
        "patient_observations_plotted": 33,
        "pareto_pairs_plotted": 32,
        "external_validation_outcomes_omitted_for_convenience": False,
        "ranking_changed": False,
        "clinical_outcome_prediction_claimed": False,
    }
    with (OUT / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    outputs.append(OUT / "run_summary.json")
    manifest_rows = []
    for path in outputs:
        manifest_rows.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(manifest_rows).to_csv(OUT / "output_manifest.tsv", sep="\t", index=False)
    print("TRIAGE-OSA V4 five-figure package generated: {}".format(OUT))


def osa_case_study_closeout():
    """Aggregate completed OSA experiments only; never train or read benchmarks."""
    import os

    out = BASE / "32_osa_case_study_closeout_v1"
    for sub in ("main_figure_candidates", "supplementary_figures", "panels",
                "source_data", "qa"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    src = out / "source_data"

    stages = [
        "11_cpap_reversal_endpoint_triage_osa_v4",
        "12_cpap_subject_signflip_triage_osa_v4",
        "14_adipose_cross_tissue_endpoint_triage_osa_v4",
        "16_scrna_cell_context_endpoint_triage_osa_v4",
        "18_pediatric_proteome_endpoint_triage_osa_v4",
        "19_known_ppi_complex_recovery_triage_osa_v4",
        "21_independent_validation_endpoint_triage_osa_v4",
        "22_consolidated_evidence_triage_osa_v4",
        "25_cpap_subject_clinical_package_triage_osa_v4",
        "27_public_multimodal_experiments_v1", "29_publication_robustness_v1",
        "31_osa_controlled_experiments_v1/osa_ablation_suite",
        "31_osa_controlled_experiments_v1/supplement_completion_v1",
    ]
    inventory = []
    for stage in stages:
        p = BASE / stage / "run_summary.json"
        obj = load_json(p)
        status = obj["status"]
        if not (status.startswith("PASS") or status.startswith("AVAILABLE_SUPPLEMENTS_COMPLETE")):
            raise ValueError("Unfinished source stage: {} {}".format(stage, status))
        note = "Execution complete; status is not evidence of biological significance."
        if stage.startswith("11_"):
            note = "Subject robustness completed in Stage 12; original status preserved."
        elif stage.startswith("18_"):
            note = "Low coverage: pair endpoint not evaluable; do not claim validation success."
        elif stage.startswith("25_"):
            note = "33 molecular reports; individual clinical crosswalk unavailable."
        elif stage.endswith("supplement_completion_v1"):
            note = "OSA supplements complete; generic benchmark pending is OUT OF SCOPE."
        inventory.append(dict(stage=stage, source_status=status,
                              osa_execution_complete=True, interpretation=note))
    pd.DataFrame(inventory).to_csv(src / "osa_completion_inventory.tsv", sep="\t", index=False)

    sup = BASE / "31_osa_controlled_experiments_v1/supplement_completion_v1"
    metrics_path = sup / "02_fusion_convergence_and_context/all_comparisons_including_original.tsv"
    metrics = pd.read_csv(metrics_path, sep="\t")
    test = metrics.loc[metrics.partition.eq("test")].copy()
    assert len(test) == 168 and not test.duplicated(["method", "seed", "fold"]).any()
    assert test.groupby("method").size().eq(12).all()
    columns = ["Spearman", "weak_top20_AUPRC", "multi_axis_utility",
               "heldout_participant_target_rho"]
    assert np.isfinite(test[columns].values).all()
    fold = test.groupby(["method", "fold"])[columns].mean().reset_index()
    means = fold.groupby("method")[columns].mean()
    test.to_csv(src / "all_test_metrics_14_methods_168_runs.tsv", sep="\t", index=False)
    fold.to_csv(src / "fold_means_seeds_averaged.tsv", sep="\t", index=False)
    means.to_csv(src / "method_means.tsv", sep="\t")
    differences = []
    full = fold.loc[fold.method.eq("full")].set_index("fold")
    for method, block in fold.groupby("method"):
        for f, row in block.set_index("fold").iterrows():
            for metric in columns:
                differences.append(dict(comparator=method, fold=int(f), metric=metric,
                                        full_minus_comparator=float(full.loc[f, metric] - row[metric])))
    differences = pd.DataFrame(differences)
    differences.to_csv(src / "matched_fold_differences.tsv", sep="\t", index=False)

    raw_path = sup / "04_OSA_L2_sensitivity/pair_perturbations.tsv"
    raw = pd.read_csv(raw_path, sep="\t")
    assert len(raw) == 640 and raw.pair_id.nunique() == 128
    assert np.allclose(raw.absolute_PPI_change, raw.delta_PPI_logit.abs())
    assert raw.groupby(["pair_id", "mode"]).size().unstack().eq(
        pd.Series({"bottom": 1, "random": 3, "top": 1})).all().all()
    # Average absolute random changes AFTER abs; never abs(mean signed change).
    l2 = raw.groupby(["pair_id", "mode"])[
        ["absolute_PPI_change", "delta_PPI_logit", "delta_L2_logit"]].mean().reset_index()
    l2.to_csv(src / "L2_all_128_pairs_random_repeats_averaged.tsv", sep="\t", index=False)
    wide = l2.pivot(index="pair_id", columns="mode", values="absolute_PPI_change")
    wide["top_minus_random"] = wide.top - wide.random
    wide["top_minus_bottom"] = wide.top - wide.bottom
    wide.to_csv(src / "L2_paired_absolute_changes.tsv", sep="\t")
    shutil.copy2(raw_path, src / "L2_all_640_perturbations.tsv")
    l2_summary = {
        "pairs": 128, "random_repeats_per_pair": 3,
        "mean_absolute_change": {m: float(wide[m].mean()) for m in ("top", "random", "bottom")},
        "top_exceeds_random_pairs": int((wide.top > wide.random).sum()),
        "top_exceeds_bottom_pairs": int((wide.top > wide.bottom).sum()),
        "top_to_random_mean_ratio": float(wide.top.mean() / wide.random.mean()),
        "biological_independence_assumed": False,
        "binding_site_accuracy_measured": False,
    }
    (src / "L2_summary.json").write_text(json.dumps(l2_summary, indent=2), encoding="utf-8")

    # Complete OSA evidence tables, not a selection restricted to positive results.
    copies = {
        "22_consolidated_evidence_triage_osa_v4": ["model_axis_comparison.tsv", "mechanism_modules.tsv",
            "assay_shortlist.tsv", "version_stability_metrics.tsv", "evidence_conflict_matrix.tsv"],
        "25_cpap_subject_clinical_package_triage_osa_v4": ["clinical_subject_manifest.tsv",
            "subject_molecular_response_summary.tsv"],
        "29_publication_robustness_v1": ["participant_conditional_intervals.tsv",
            "paired_modality_differences.tsv", "stage27_reproduction.tsv"],
    }
    for stage, files in copies.items():
        for name in files:
            shutil.copy2(BASE / stage / name, src / (stage.split("_")[0] + "_" + name))

    labels = {
        "full": "Full TRIAGE-OSA", "no_patient": "No patient information",
        "no_context": "No OSA context", "no_gate_delta": "No gate delta",
        "no_residual": "No residual", "equal_fusion": "Equal fusion",
        "static_fusion": "Static (60 epochs)",
        "static_constrained_converged": "Static constrained (converged)",
        "static_simplex_converged": "Static simplex (converged)",
        "context_anchor_direct": "Context anchor (target copy)",
        "molecular_anchor_direct": "Molecular anchor",
        "patient_anchor_direct": "Patient anchor",
        "frozen_PPI_direct": "Frozen PPI readout",
        "frozen_tri_level_direct": "Frozen three-level readout",
    }
    order = list(labels)
    teal, orange, gray = "#167C80", "#C65D3A", "#788896"
    folds = [1, 2, 3, 4]
    fold_colors = ["#167C80", "#426F9A", "#957049", "#88669A"]

    def ap_panel(ax):
        for f, color in zip(folds, fold_colors):
            vals = [float(fold.loc[fold.method.eq(m) & fold.fold.eq(f), "weak_top20_AUPRC"].iloc[0])
                    for m in ("static_constrained_converged", "full")]
            ax.plot([0, 1], vals, "o-", color=color, lw=1, ms=3.5, label="Fold {}".format(f))
        ax.set(xticks=[0, 1], xticklabels=["Converged\nstatic", "Full\nTRIAGE-OSA"],
               xlim=(-0.3, 1.3), ylabel="Weak-target AUPRC", title="Matched fusion control")
        ax.legend(fontsize=5.5, loc="center right", bbox_to_anchor=(1, .37))

    contrast_order = ["static_constrained_converged", "no_patient", "no_gate_delta", "no_residual"]
    def contrast_panel(ax):
        ax.axvline(0, color=gray, lw=.7, ls="--")
        for j, method in enumerate(contrast_order):
            block = differences.loc[differences.comparator.eq(method) &
                                    differences.metric.eq("multi_axis_utility")].sort_values("fold")
            for k, value in enumerate(block.full_minus_comparator):
                ax.plot(value, j + (k - 1.5) * .10, "o", color=fold_colors[k], ms=3)
            ax.plot(block.full_minus_comparator.mean(), j, "|", color="#222222", ms=9, mew=1.4)
        ax.set(yticks=range(4), yticklabels=["Converged static", "No patient info", "No gate delta", "No residual"],
               xlabel="Full minus comparator: utility", title="Component contributions")
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)

    def l2_panel(ax):
        for mode, label, color in [("top", "High support", teal), ("random", "Random", gray),
                                    ("bottom", "Low support", orange)]:
            values = np.sort(wide[mode].values)
            ax.step(values, np.arange(1, 129) / 128., where="post", color=color, lw=1.3, label=label)
        ax.set(xlabel="Absolute change in PPI logit", ylabel="Cumulative fraction of pairs",
               title="OSA-candidate L2 sensitivity", ylim=(0, 1.03), xlim=(0, None))
        ax.legend(loc="lower right", fontsize=5.5)

    def all_metrics_panel(ax):
        arr = means.loc[order, columns[:3]].values
        ax.imshow(arr, vmin=0, vmax=1, cmap="Blues", aspect="auto", interpolation="nearest")
        for i in range(len(order)):
            for j in range(3):
                ax.text(j, i, "{:.3f}".format(arr[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if arr[i, j] > .65 else "#26333B")
        ax.set(yticks=range(len(order)), yticklabels=[labels[m] for m in order], xticks=range(3),
               xticklabels=["Spearman", "AUPRC", "Utility"], title="All OSA controls; mean test values")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    def generalization_panel(ax):
        ax.axvline(0, color=gray, lw=.8, ls="--")
        for j, method in enumerate(order):
            values = fold.loc[fold.method.eq(method)].sort_values("fold")["heldout_participant_target_rho"]
            ax.plot([values.min(), values.max()], [j, j], color=gray, lw=.7)
            for k, value in enumerate(values):
                ax.plot(value, j + (k - 1.5) * .09, "o", ms=2.6, color=fold_colors[k])
        ax.set(yticks=range(len(order)), yticklabels=[labels[m] for m in order],
               xlabel="Held-out participant-target Spearman", title="Target-transfer limitation",
               ylim=(len(order) - .5, -.5), xlim=(-.75, .10))
        ax.tick_params(axis="y", length=0)

    files = []
    def export(fig, path):
        fig.canvas.draw()
        for suffix in ("pdf", "svg", "png", "tiff"):
            kwargs = dict(dpi=600 if suffix == "tiff" else 220, facecolor="white")
            if suffix == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            target = path.with_suffix("." + suffix)
            fig.savefig(target, **kwargs)
            files.append(str(target.relative_to(out)))
        plt.close(fig)

    panels = [ap_panel, contrast_panel, l2_panel]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.60))
    fig.subplots_adjust(left=.075, right=.985, bottom=.24, top=.84, wspace=.78)
    for letter, ax, draw in zip("abc", axes, panels):
        draw(ax)
        ax.text(-.18, 1.20, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    export(fig, out / "main_figure_candidates/Fig_OSA_adaptation_and_L2")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3))
    fig.subplots_adjust(left=.245, right=.985, bottom=.14, top=.88, wspace=1.43)
    for letter, ax, draw in zip("ab", axes, [all_metrics_panel, generalization_panel]):
        draw(ax)
        ax.text(-.10, 1.10, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    export(fig, out / "supplementary_figures/SFig_OSA_all_controls_and_limits")
    for name, draw, size, left in [
        ("Main_a_matched_fusion", ap_panel, (3.4, 2.7), .20),
        ("Main_b_component_contributions", contrast_panel, (3.8, 2.7), .31),
        ("Main_c_L2_sensitivity", l2_panel, (3.4, 2.7), .20),
        ("Supp_a_all_controls", all_metrics_panel, (4.2, 4.3), .45),
        ("Supp_b_target_transfer", generalization_panel, (4.2, 4.3), .45),
    ]:
        fig, ax = plt.subplots(figsize=size)
        fig.subplots_adjust(left=left, right=.97, bottom=.19 if size[1] < 3 else .14, top=.87)
        draw(ax)
        export(fig, out / "panels" / name)

    summary = dict(status="COMPLETE_AVAILABLE_OSA_CASE_STUDY_ANALYSES",
        scope="OSA case study and OSA-specific ablations only",
        training_rerun=False, external_validation_used_for_tuning=False,
        generic_benchmarks="OUT_OF_SCOPE_USER_WILL_SUPPLY",
        test_methods=14, seeds=3, protein_disjoint_folds=4, test_metric_rows=168,
        ablation_runs_are_independent_clinical_replicates=False,
        full_vs_converged_static={c: float(means.loc["full", c] -
            means.loc["static_constrained_converged", c]) for c in columns[:3]},
        full_vs_no_patient={c: float(means.loc["full", c] - means.loc["no_patient", c]) for c in columns[:3]},
        L2_summary=l2_summary,
        limitations=["Weak-target metrics are not individual clinical accuracy.",
            "Full pipeline patient-level feature/candidate selection was not completely nested.",
            "Held-out participant-target transfer is negative.",
            "Previously inspected external cohorts are exploratory V4 reanalyses.",
            "Pediatric proteome pair endpoint is not evaluable at current coverage.",
            "Clinical crosswalk unavailable; hospital validation is future work."],
        previous_biological_figures="../30_manuscript_figure_package_v2/figure_manifest.json",
        outputs=files)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OSA case-study closeout: " + summary["status"])
    print("Output: " + str(out))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--osa-closeout"]:
        osa_case_study_closeout()
    elif sys.argv[1:]:
        raise SystemExit("Use no arguments for legacy figures, or --osa-closeout for OSA-only closure.")
    else:
        main()
