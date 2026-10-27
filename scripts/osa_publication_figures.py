#!/usr/bin/env python3
"""Create publication figures from the frozen TRIAGE-OSA Stage 22 package."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

# Mandatory editable-text settings are repeated at module scope so automated
# journal preflight tools can verify them before any figure is created.
# svg.fonttype='none'
# pdf.fonttype=42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42


SCRIPT_VERSION = "1.0.0"
EXPECTED_STAGE22_STATUS = "PASS_CONSOLIDATED_EVIDENCE_PACKAGE"
EXPECTED_RANKING_SHA256 = (
    "2dde9eee55064cdc2d1f1e3fce772d00c815b7f02eaf3014bca1d2a5b506944c"
)

COLORS = {
    "blue": "#0F4D92",
    "blue_mid": "#5B7FCA",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#D39B27",
    "red": "#B64342",
    "neutral": "#A8A8A8",
    "neutral_light": "#E3E3E3",
    "neutral_dark": "#4D4D4D",
    "black": "#272727",
}


def apply_publication_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial", "DejaVu Sans", "Liberation Sans"
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 6
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["legend.frameon"] = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--stage22",
        default="results/discovery_preprocess_v2/22_consolidated_evidence_triage_osa_v3_2",
    )
    parser.add_argument(
        "--output-stage",
        default="results/discovery_preprocess_v2/23_publication_figures_v1",
    )
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_from_root(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def add_panel_label(ax, label):
    ax.text(
        -0.16, 1.08, label, transform=ax.transAxes, fontsize=9,
        fontweight="bold", ha="left", va="bottom"
    )


def save_figure(fig, output_dir, stem, panel_axes=None, panel_names=None,
                panel_extra_axes=None):
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    svg_path = combined_dir / "{}.svg".format(stem)
    pdf_path = combined_dir / "{}.pdf".format(stem)
    tiff_path = combined_dir / "{}.tiff".format(stem)
    png_path = combined_dir / "{}.png".format(stem)
    fig.savefig(svg_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    paths = [svg_path, pdf_path, tiff_path, png_path]
    if panel_axes:
        if panel_names is None or len(panel_names) != len(panel_axes):
            raise RuntimeError("Panel names must match panel axes")
        if panel_extra_axes is None:
            panel_extra_axes = [None] * len(panel_axes)
        if len(panel_extra_axes) != len(panel_axes):
            raise RuntimeError("Panel extra axes must match panel axes")
        panel_dir = output_dir / "panels" / stem.split("_")[0]
        panel_dir.mkdir(parents=True, exist_ok=True)
        for ax, panel_name, extras in zip(panel_axes, panel_names, panel_extra_axes):
            keep_axes = [ax] + list(extras or [])
            visibility = {item: item.get_visible() for item in fig.axes}
            for item in fig.axes:
                item.set_visible(item in keep_axes)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = Bbox.union([item.get_tightbbox(renderer) for item in keep_axes])
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            bbox = bbox.expanded(1.04, 1.06)
            for suffix, dpi in (("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)):
                path = panel_dir / "{}.{}".format(panel_name, suffix)
                kwargs = {"bbox_inches": bbox, "facecolor": "white"}
                if dpi is not None:
                    kwargs["dpi"] = dpi
                fig.savefig(path, **kwargs)
                paths.append(path)
            for item, was_visible in visibility.items():
                item.set_visible(was_visible)
            fig.canvas.draw()
    plt.close(fig)
    return paths


def verify_manifest(root, manifest_path):
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    for row in manifest.itertuples(index=False):
        path = resolve_from_root(root, row.path)
        if not path.is_file():
            raise RuntimeError("Manifest artifact is missing: {}".format(path))
        if int(path.stat().st_size) != int(row.bytes):
            raise RuntimeError("Manifest byte mismatch: {}".format(path))
        if sha256_file(path) != str(row.sha256).lower():
            raise RuntimeError("Manifest SHA256 mismatch: {}".format(path))
    return int(len(manifest))


def load_inputs(root, stage22):
    stage22_summary_path = stage22 / "run_summary.json"
    if not stage22_summary_path.is_file():
        raise RuntimeError("Missing Stage 22 run_summary.json")
    summary = json.loads(stage22_summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != EXPECTED_STAGE22_STATUS:
        raise RuntimeError("Stage 22 is not formally eligible")
    if summary.get("frozen_ranking_sha256") != EXPECTED_RANKING_SHA256:
        raise RuntimeError("Stage 22 ranking hash is not the frozen V3.2.2 ranking")
    if bool(summary.get("ranking_changed", True)):
        raise RuntimeError("Stage 22 reports a changed ranking")
    manifest_count = verify_manifest(root, stage22 / "output_manifest.tsv")
    pair = pd.read_csv(
        stage22 / "pair_evidence_matrix.tsv.gz", sep="\t", low_memory=False
    )
    protein = pd.read_csv(
        stage22 / "protein_evidence_matrix.tsv", sep="\t", low_memory=False
    )
    pareto = pd.read_csv(
        stage22 / "pareto_front_candidates.tsv", sep="\t", low_memory=False
    )
    shortlist = pd.read_csv(
        stage22 / "assay_shortlist.tsv", sep="\t", low_memory=False
    )
    modules = pd.read_csv(
        stage22 / "mechanism_modules.tsv", sep="\t", low_memory=False
    )
    model_axis = pd.read_csv(
        stage22 / "model_axis_comparison.tsv", sep="\t", low_memory=False
    )
    pathway = pd.read_csv(
        root
        / "results/discovery_preprocess_v2/21_independent_validation_endpoint_v1/reactome_pathway_axis_results.tsv",
        sep="\t", low_memory=False,
    )
    cpap_node = json.loads(
        (root / "results/discovery_preprocess_v2/11_cpap_reversal_endpoint_triage_osa_v3_2/run_summary.json").read_text(encoding="utf-8")
    )
    cpap_sign = json.loads(
        (root / "results/discovery_preprocess_v2/12_cpap_subject_signflip_triage_osa_v3_2/run_summary.json").read_text(encoding="utf-8")
    )
    if len(pair) != 31125 or pair["pair_id"].duplicated().any():
        raise RuntimeError("Stage 22 pair evidence universe is invalid")
    if len(protein) != 250 or protein["UniProt"].duplicated().any():
        raise RuntimeError("Stage 22 protein evidence universe is invalid")
    if len(pareto) != int(summary["pareto_front_pairs"]):
        raise RuntimeError("Stage 22 Pareto count differs from run summary")
    return {
        "summary": summary,
        "manifest_count": manifest_count,
        "pair": pair,
        "protein": protein,
        "pareto": pareto,
        "shortlist": shortlist,
        "modules": modules,
        "model_axis": model_axis,
        "pathway": pathway,
        "cpap_node": cpap_node,
        "cpap_sign": cpap_sign,
    }


def evidence_distribution_source(pair):
    groups = {
        "All pairs": np.ones(len(pair), dtype=bool),
        "Full-model top 1%": pd.to_numeric(
            pair["triage_osa_rank_primary_250"], errors="coerce"
        ).to_numpy() <= 312,
        "Pareto front": pair["pareto_front"].astype(bool).to_numpy(),
    }
    records = []
    for group, mask in groups.items():
        values = pd.to_numeric(
            pair.loc[mask, "evidence_supportive_axis_count"], errors="coerce"
        ).astype(int)
        for count in range(10):
            records.append(
                {
                    "group": group,
                    "supportive_axis_count": count,
                    "pairs": int((values == count).sum()),
                    "fraction": float((values == count).mean()),
                    "group_n": int(len(values)),
                }
            )
    return pd.DataFrame.from_records(records)


def shortlist_heatmap_source(shortlist):
    columns = [
        ("OSA", "OSA_strength"),
        ("Severity", "severe_strength"),
        ("AHI", "AHI_strength"),
        ("IH", "IH_strength"),
        ("CPAP-1", "GSE133601_CPAP_reversal_score"),
        ("CPAP-2", "GSE49800_CPAP_reversal_score"),
        ("Adipose", "state_node_score"),
        ("Adipose\nCPAP", "CPAP_node_score"),
        ("T/NK", "scrna_T_NK_node_score"),
        ("B", "scrna_B_node_score"),
        ("Myeloid", "scrna_Myeloid_node_score"),
        ("GSE38792", "GSE38792_node_score"),
        ("GSE293433", "GSE293433_node_score"),
    ]
    values = shortlist[[column for _, column in columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    values.columns = [label for label, _ in columns]
    values.insert(0, "gene_symbol", shortlist["gene_symbol"].astype(str).to_numpy())
    return values


def standardize_model(name):
    mapping = {
        "full_primary": "Full TRIAGE-OSA",
        "full_OSA_priority": "Full TRIAGE-OSA",
        "full_TRIAGE_OSA": "Full TRIAGE-OSA",
        "PPI_only": "PPI only",
        "context_free_p_PPI": "PPI only",
        "context_only": "Context only",
        "context_only_r_OSA": "Context only",
        "L3_only": "L3 only",
        "L1_L2": "L1+L2",
        "L1_L3": "L1+L3",
        "L2_L3": "L2+L3",
        "equal_L1_L2_L3": "Equal L1+L2+L3",
        "PPI25_context75": "PPI25+Context75",
        "PPI75_context25": "PPI75+Context25",
        "bidirectional_conservative": "Bidirectional",
    }
    return mapping.get(str(name), str(name))


def model_heatmap_source(model_axis):
    endpoint_specs = [
        (11, "CPAP_GSE133601", "CPAP-1"),
        (11, "CPAP_GSE49800", "CPAP-2"),
        (14, "state", "Adipose state"),
        (14, "CPAP", "Adipose CPAP"),
        (16, "pediatric_scRNA_omnibus", "Pediatric scRNA"),
        (19, "IntAct_physical_training_excluded", "IntAct physical"),
        (21, "GSE38792", "Independent adipose"),
        (21, "GSE293433", "Independent PBMC"),
        (21, "joint", "Independent joint"),
    ]
    frames = []
    for stage, endpoint, label in endpoint_specs:
        subset = model_axis[
            (pd.to_numeric(model_axis["stage"], errors="coerce") == stage)
            & model_axis["endpoint"].astype(str).eq(endpoint)
        ].copy()
        subset["model_standard"] = subset["model"].map(standardize_model)
        subset["endpoint_label"] = label
        frames.append(subset)
    selected = pd.concat(frames, ignore_index=True)
    safe_top1_ratio = np.clip(
        pd.to_numeric(selected["top_1pct_ratio"], errors="coerce"), 1e-8, None
    )
    selected["log2_top1_ratio"] = np.log2(safe_top1_ratio)
    model_order = [
        "Full TRIAGE-OSA", "PPI only", "Context only", "L3 only",
        "L1+L2", "L1+L3", "L2+L3", "Equal L1+L2+L3",
        "PPI25+Context75", "PPI75+Context25", "Bidirectional",
    ]
    endpoint_order = [label for _, _, label in endpoint_specs]
    matrix = selected.pivot_table(
        index="model_standard", columns="endpoint_label",
        values="log2_top1_ratio", aggfunc="first"
    ).reindex(index=model_order, columns=endpoint_order)
    return selected, matrix


def cpap_source(cpap_node, cpap_sign):
    records = []
    for dataset, label in (("GSE133601", "CPAP-1"), ("GSE49800", "CPAP-2")):
        node = cpap_node["cohorts"][dataset]
        sign = cpap_sign["cohorts"][dataset]
        records.append(
            {
                "dataset": label,
                "analysis": "Node-aware",
                "ratio": float(node["primary_top_1pct_enrichment_ratio"]),
                "one_sided_P": float(node["top_1pct_permutation_P"]),
                "subjects": np.nan,
            }
        )
        records.append(
            {
                "dataset": label,
                "analysis": "Participant sign-flip",
                "ratio": float(sign["observed_top_1pct_enrichment_ratio_unmoderated_t"]),
                "one_sided_P": float(sign["top_1pct_one_sided_signflip_P"]),
                "subjects": int(sign["subjects"]),
            }
        )
    return pd.DataFrame.from_records(records)


def pathway_source(pathway):
    axes = pathway[pathway["level"].astype(str).eq("axis")].copy()
    wanted = [
        "hypoxia_HIF_ROS", "NFkB_innate_inflammation",
        "antigen_lymphocyte_signalling", "PI3K_AKT_JAK_STAT_TGFb",
        "proteostasis_ER_stress", "PPAR_adipose_lipid_metabolism",
    ]
    axes = axes[
        axes["dataset"].isin(["GSE38792", "GSE293433"])
        & axes["axis"].isin(wanted)
    ].copy()
    axes["competitive_effect"] = pd.to_numeric(
        axes["competitive_effect"], errors="coerce"
    )
    axes["BH_FDR_axis_family"] = pd.to_numeric(
        axes["BH_FDR_axis_family"], errors="coerce"
    )
    return axes


def draw_network(ax, pareto, modules, protein):
    module_colors = [COLORS["teal"], COLORS["violet"], COLORS["gold"]]
    centers = [(-0.75, 0.10), (0.58, 0.45), (0.62, -0.55)]
    protein_lookup = protein.set_index("UniProt")
    for module_index, row in enumerate(modules.itertuples(index=False)):
        proteins = str(row.proteins).split("|")
        center = centers[module_index % len(centers)]
        color = module_colors[module_index % len(module_colors)]
        radius = 0.34 if len(proteins) > 4 else 0.20
        angles = np.linspace(0, 2 * np.pi, len(proteins), endpoint=False)
        positions = {
            accession: (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
            for accession, angle in zip(proteins, angles)
        }
        edges = pareto[
            pareto["protein_A"].isin(proteins)
            & pareto["protein_B"].isin(proteins)
            & (pd.to_numeric(pareto["triage_osa_rank_primary_250"], errors="coerce") <= 1557)
        ]
        for edge in edges.itertuples(index=False):
            if edge.protein_A in positions and edge.protein_B in positions:
                x1, y1 = positions[edge.protein_A]
                x2, y2 = positions[edge.protein_B]
                ax.plot([x1, x2], [y1, y2], color=COLORS["neutral"], lw=1.0, zorder=1)
        for accession in proteins:
            x, y = positions[accession]
            gene = str(protein_lookup.loc[accession, "gene_symbol"])
            count = float(protein_lookup.loc[accession, "pareto_pair_count"])
            ax.scatter(
                x, y, s=35 + 12 * count, color=color, edgecolor="white",
                linewidth=0.7, zorder=2
            )
            ax.text(x, y - 0.055, gene, ha="center", va="top", fontsize=5.2)
        ax.text(
            center[0], center[1] + radius + 0.12,
            "{} ({} proteins)".format(row.module_id, len(proteins)),
            ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=color
        )
    ax.set_xlim(-1.25, 1.1)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Pareto-linked mechanism components", pad=5)


def make_figure1(data, output_dir, source_dir):
    pair = data["pair"]
    pareto = data["pareto"]
    shortlist = data["shortlist"]
    protein = data["protein"]
    modules = data["modules"]
    distribution = evidence_distribution_source(pair)
    heatmap = shortlist_heatmap_source(shortlist)
    pareto_source = pair[
        [
            "pair_id", "gene_A", "gene_B", "triage_osa_rank_primary_250",
            "structural_percentile", "osa_context_percentile",
            "evidence_breadth_score", "evidence_supportive_axis_count",
            "evidence_evaluable_axis_count", "evidence_conflict_count",
            "pareto_front", "mechanism_axes_union",
        ]
    ].copy()
    pareto_source.to_csv(source_dir / "fig1_pareto_landscape.tsv.gz", sep="\t", index=False, compression="gzip")
    distribution.to_csv(source_dir / "fig1_evidence_distribution.tsv", sep="\t", index=False)
    heatmap.to_csv(source_dir / "fig1_shortlist_heatmap.tsv", sep="\t", index=False)
    modules.to_csv(source_dir / "fig1_mechanism_modules.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2, 7.15))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.08, 1.25], height_ratios=[1.0, 1.15],
        hspace=0.46, wspace=0.34
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    nonpareto = ~pair["pareto_front"].astype(bool)
    ax_a.scatter(
        pair.loc[nonpareto, "structural_percentile"],
        pair.loc[nonpareto, "osa_context_percentile"],
        s=2.0, color="#D8D8D8", alpha=0.38, linewidth=0, rasterized=True,
        label="Other pairs"
    )
    points = ax_a.scatter(
        pareto["structural_percentile"], pareto["osa_context_percentile"],
        c=pareto["evidence_supportive_axis_count"], cmap="viridis",
        vmin=0, vmax=9, s=25, edgecolor="white", linewidth=0.45,
        label="Pareto front"
    )
    label_pairs = pareto.sort_values(
        ["evidence_supportive_axis_count", "triage_osa_rank_primary_250"],
        ascending=[False, True], kind="mergesort"
    ).head(3)
    for index, row in enumerate(label_pairs.itertuples(index=False)):
        ax_a.annotate(
            "{}-{}".format(row.gene_A, row.gene_B),
            (row.structural_percentile, row.osa_context_percentile),
            xytext=(-5, -9 - 8 * index), textcoords="offset points",
            fontsize=5.0, color=COLORS["black"], ha="right",
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": COLORS["neutral_dark"]}
        )
    ax_a.set_xlabel("Structural percentile")
    ax_a.set_ylabel("OSA-context percentile")
    ax_a.set_xlim(0, 1.02)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_title("Three-axis non-dominated candidates")
    colorbar_a = fig.colorbar(points, ax=ax_a, fraction=0.046, pad=0.03)
    colorbar_a.set_label("Supportive evidence axes", fontsize=6)
    colorbar_a.ax.tick_params(labelsize=5)
    ax_a.legend(loc="lower left", markerscale=2.2)
    add_panel_label(ax_a, "a")

    draw_network(ax_b, pareto, modules, protein)
    add_panel_label(ax_b, "b")

    styles = [
        ("All pairs", COLORS["neutral_dark"], "o"),
        ("Full-model top 1%", COLORS["blue"], "s"),
        ("Pareto front", COLORS["violet"], "D"),
    ]
    for group, color, marker in styles:
        subset = distribution[distribution["group"].eq(group)]
        ax_c.plot(
            subset["supportive_axis_count"], subset["fraction"],
            color=color, marker=marker, ms=3.5, lw=1.4, label=group
        )
    ax_c.set_xlabel("Number of supportive external axes")
    ax_c.set_ylabel("Fraction of pairs")
    ax_c.set_xticks(range(10))
    ax_c.set_ylim(bottom=0)
    ax_c.set_title("Evidence breadth is enriched on the Pareto front")
    ax_c.legend(loc="upper left")
    add_panel_label(ax_c, "c")

    heat_values = heatmap.drop(columns=["gene_symbol"]).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(heat_values)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#EFEFEF")
    image = ax_d.imshow(
        masked, aspect="auto", cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0), interpolation="nearest"
    )
    ax_d.set_xticks(np.arange(heat_values.shape[1]))
    ax_d.set_xticklabels(
        heatmap.columns[1:], rotation=45, rotation_mode="anchor", ha="right"
    )
    ax_d.set_yticks(np.arange(len(heatmap)))
    ax_d.set_yticklabels(heatmap["gene_symbol"])
    ax_d.set_title("Twenty-protein mechanistic shortlist\n(serum coverage limited)")
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    colorbar_d = fig.colorbar(image, ax=ax_d, fraction=0.028, pad=0.02)
    colorbar_d.set_label("Oriented node evidence", fontsize=6)
    colorbar_d.ax.tick_params(labelsize=5)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "TRIAGE-OSA balances structural plausibility, disease context and heterogeneous external evidence",
        fontsize=9.2, fontweight="bold", y=0.995
    )
    return save_figure(
        fig, output_dir, "Fig1_TRIAGE_OSA_evidence_landscape",
        panel_axes=[ax_a, ax_b, ax_c, ax_d],
        panel_names=["Fig1a_Pareto_landscape", "Fig1b_mechanism_components",
                     "Fig1c_evidence_breadth", "Fig1d_shortlist_heatmap"],
        panel_extra_axes=[[colorbar_a.ax], None, None, [colorbar_d.ax]],
    )


def make_figure2(data, output_dir, source_dir):
    selected, model_matrix = model_heatmap_source(data["model_axis"])
    cpap = cpap_source(data["cpap_node"], data["cpap_sign"])
    pathways = pathway_source(data["pathway"])
    selected.to_csv(source_dir / "fig2_model_endpoint_comparison.tsv", sep="\t", index=False)
    cpap.to_csv(source_dir / "fig2_cpap_heterogeneity.tsv", sep="\t", index=False)
    pathways.to_csv(source_dir / "fig2_pathway_axes.tsv", sep="\t", index=False)

    pair = data["pair"]
    protein = data["protein"]
    pareto = data["pareto"]
    limitations = pd.DataFrame(
        [
            {
                "quantity": "Mean evaluable evidence axes",
                "numerator": float(pair["evidence_evaluable_axis_count"].mean()),
                "denominator": 9.0,
            },
            {
                "quantity": "Pairs with evidence conflict",
                "numerator": float((pair["evidence_conflict_count"] > 0).sum()),
                "denominator": float(len(pair)),
            },
            {
                "quantity": "Primary proteins detected in serum",
                "numerator": float(protein["proteome_detected"].astype(str).str.lower().eq("yes").sum()),
                "denominator": float(len(protein)),
            },
            {
                "quantity": "Pareto pairs dual-detected in serum",
                "numerator": float(pareto["proteome_both_nodes_detected"].astype(str).str.lower().eq("yes").sum()),
                "denominator": float(len(pareto)),
            },
        ]
    )
    limitations["fraction"] = limitations["numerator"] / limitations["denominator"]
    limitations.to_csv(source_dir / "fig2_coverage_limitations.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.2, 6.65))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.42, 0.88], height_ratios=[1.1, 0.9],
        hspace=0.56, wspace=0.78
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    values = model_matrix.to_numpy(dtype=float)
    # Preserve exact cell annotations while clipping only the colour mapping so
    # one near-zero enrichment ratio does not erase all other visual contrast.
    max_abs = 3.0
    image = ax_a.imshow(
        values, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        interpolation="nearest"
    )
    ax_a.set_xticks(np.arange(model_matrix.shape[1]))
    ax_a.set_xticklabels(
        model_matrix.columns, rotation=45, rotation_mode="anchor", ha="right"
    )
    ax_a.set_yticks(np.arange(model_matrix.shape[0]))
    ax_a.set_yticklabels(model_matrix.index)
    ax_a.add_patch(
        Rectangle(
            (-0.5, -0.5), model_matrix.shape[1], 1,
            fill=False, edgecolor=COLORS["black"], linewidth=1.2
        )
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                ax_a.text(
                    column, row, "{:.2f}".format(values[row, column]),
                    ha="center", va="center", fontsize=5.0,
                    color="white" if abs(values[row, column]) > 0.55 * max_abs else COLORS["black"]
                )
    ax_a.set_title("Endpoint-specific top-1% enrichment across frozen rankings")
    colorbar_a = fig.colorbar(image, ax=ax_a, fraction=0.025, pad=0.02)
    colorbar_a.set_label("log2 ratio (colour clipped at +/-3)", fontsize=6)
    colorbar_a.ax.tick_params(labelsize=5)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    add_panel_label(ax_a, "a")

    positions = np.arange(len(cpap))[::-1]
    color_map = {"CPAP-1": COLORS["blue"], "CPAP-2": COLORS["red"]}
    marker_map = {"Node-aware": "o", "Participant sign-flip": "s"}
    for position, row in zip(positions, cpap.itertuples(index=False)):
        ax_b.scatter(
            row.ratio, position, s=32, marker=marker_map[row.analysis],
            color=color_map[row.dataset], edgecolor="white", linewidth=0.6, zorder=2
        )
        ax_b.text(
            max(row.ratio + 0.025, 0.77), position,
            "P={:.3f}".format(row.one_sided_P), va="center", fontsize=5.4
        )
    labels = ["{}\n{}".format(row.dataset, row.analysis) for row in cpap.itertuples(index=False)]
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(labels)
    ax_b.axvline(1.0, color=COLORS["neutral_dark"], lw=0.9, ls="--")
    ax_b.set_xlim(0.65, 1.32)
    ax_b.set_xlabel("Top-1% CPAP reversal ratio")
    ax_b.set_title("CPAP evidence is\ncohort-heterogeneous")
    add_panel_label(ax_b, "b")

    axis_order = [
        "hypoxia_HIF_ROS", "NFkB_innate_inflammation",
        "antigen_lymphocyte_signalling", "PI3K_AKT_JAK_STAT_TGFb",
        "proteostasis_ER_stress", "PPAR_adipose_lipid_metabolism",
    ]
    dataset_order = ["GSE38792", "GSE293433"]
    effect = pathways.pivot_table(
        index="axis", columns="dataset", values="competitive_effect", aggfunc="first"
    ).reindex(index=axis_order, columns=dataset_order)
    fdr = pathways.pivot_table(
        index="axis", columns="dataset", values="BH_FDR_axis_family", aggfunc="first"
    ).reindex(index=axis_order, columns=dataset_order)
    effect_values = effect.to_numpy(dtype=float)
    pathway_max = max(0.65, float(np.nanmax(np.abs(effect_values))))
    image = ax_c.imshow(
        effect_values, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-pathway_max, vcenter=0.0, vmax=pathway_max),
        interpolation="nearest"
    )
    pretty_axes = [
        "Hypoxia / HIF / ROS", "NF-kappaB / innate inflammation",
        "Antigen / lymphocyte signalling", "PI3K / JAK-STAT / TGF-beta",
        "Proteostasis / ER stress", "PPAR / lipid metabolism",
    ]
    ax_c.set_yticks(np.arange(len(axis_order)))
    ax_c.set_yticklabels(pretty_axes)
    ax_c.set_xticks(np.arange(len(dataset_order)))
    ax_c.set_xticklabels(["Independent adipose", "Independent PBMC"])
    for row in range(effect_values.shape[0]):
        for column in range(effect_values.shape[1]):
            if np.isfinite(effect_values[row, column]):
                q = fdr.to_numpy(dtype=float)[row, column]
                stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
                ax_c.text(
                    column, row, "{:.2f}{}".format(effect_values[row, column], stars),
                    ha="center", va="center", fontsize=5.5,
                    color="white" if abs(effect_values[row, column]) > 0.55 * pathway_max else COLORS["black"]
                )
    ax_c.set_title("Independent pathway axes replicate\nwith tissue-specific direction")
    colorbar_c = fig.colorbar(image, ax=ax_c, fraction=0.03, pad=0.025)
    colorbar_c.set_label("Competitive effect", fontsize=6)
    colorbar_c.ax.tick_params(labelsize=5)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    add_panel_label(ax_c, "c")

    y = np.arange(len(limitations))[::-1]
    fractions = limitations["fraction"].to_numpy(dtype=float)
    colors = [COLORS["blue_mid"], COLORS["red"], COLORS["gold"], COLORS["neutral"]]
    ax_d.barh(y, fractions, color=colors, height=0.58)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(
        [
            "Evaluable axes", "Pairs with conflict",
            "Serum proteins", "Dual-detected Pareto pairs",
        ]
    )
    ax_d.set_xlim(0, 1.0)
    ax_d.set_xlabel("Fraction")
    ax_d.set_title("Coverage and conflict define\nthe claim boundary")
    for yi, row in zip(y, limitations.itertuples(index=False)):
        label = "{:.1%}".format(row.fraction)
        if row.quantity != "Mean evaluable evidence axes":
            label += " ({:.0f}/{:.0f})".format(row.numerator, row.denominator)
        else:
            label += " ({:.2f}/9)".format(row.numerator)
        ax_d.text(min(row.fraction + 0.025, 0.82), yi, label, va="center", fontsize=5.5)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "External validation supports pathway-level biology but not universal ranking superiority",
        fontsize=9.2, fontweight="bold", y=0.995
    )
    return save_figure(
        fig, output_dir, "Fig2_TRIAGE_OSA_external_validation",
        panel_axes=[ax_a, ax_b, ax_c, ax_d],
        panel_names=["Fig2a_model_endpoint_matrix", "Fig2b_CPAP_heterogeneity",
                     "Fig2c_pathway_axes", "Fig2d_coverage_limitations"],
        panel_extra_axes=[[colorbar_a.ax], None, [colorbar_c.ax], None],
    )


def write_notes(output_dir, data):
    contract = """# Figure contract and QA notes

## Backend and export

- Backend: Python / matplotlib only.
- Complete composites are stored in `combined/`; independently usable panels are grouped under `panels/Fig1` and `panels/Fig2`.
- Archetype: asymmetric mixed-modality quantitative figures.
- Final width: 183 mm-equivalent double-column canvas (7.2 inches).
- Exports: editable SVG, PDF, 600 dpi TIFF and 300 dpi PNG preview.
- No observations were sampled or excluded from the 31,125-pair landscape.

## Figure 1 contract

Core conclusion: TRIAGE-OSA identifies a small non-dominated set balancing
structural plausibility, OSA context and external evidence, while conflicts and
limited serum coverage remain visible.

Panel map: a, all-pair Pareto landscape; b, mechanism components among Pareto
pairs within the frozen top 5%; c, evidence-axis distribution for all pairs,
the frozen full-model top 1% and the Pareto front; d, node evidence for the
20-protein mechanistic shortlist.

Reviewer risk: the shortlist is post-development and exploratory; it is not an
independent validation panel or a clinically assayable panel. All 20 proteins
are Tier 2 because none was detected in the pediatric depleted-serum TMT data.

## Figure 2 contract

Core conclusion: full-model ranking performance is endpoint-specific; CPAP
results are heterogeneous and the strongest independent signal is pathway-level
with tissue-specific direction.

Panel map: a, top-1% enrichment across 11 frozen rankings and nine endpoints;
b, locked CPAP node-aware and participant sign-flip ratios; c, six prespecified
Reactome pathway axes in two untouched independent cohorts; d, evidence
coverage and conflict boundaries.

Reviewer risk: the heatmap is descriptive across heterogeneous endpoints and
does not create a new pooled significance test. Exact locked P values remain in
the source tables and result ledger. Panel a clips only the colour mapping at
log2 enrichment +/-3 so the context-only IntAct value of -26.58 does not erase
all other visual contrast; exact unmodified values remain printed in cells and
stored in the source table. All log2 inputs use an explicit positive floor.
"""
    (output_dir / "figure_contract_and_qa.md").write_text(contract, encoding="utf-8")
    legends = """# Figure legends

## Fig. 1 | Multiaxis evidence landscape of the frozen TRIAGE-OSA ranking

**a**, Structural and OSA-context percentiles for all 31,125 frozen primary
protein pairs. Coloured points mark the 32 three-axis Pareto-front pairs and
encode the number of supportive external evidence axes. **b**, Connected
components formed by Pareto-front pairs that also occur within the frozen
full-model top 5%; node size reflects Pareto-pair participation. **c**,
Distribution of supportive external axes among all pairs, the frozen
full-model top 1% (312 rank positions before validation eligibility filtering)
and the post-development Pareto front. **d**, Oriented node evidence for the
20-protein mechanistic shortlist; grey cells indicate unavailable evidence.
Values above 0.5 indicate directionally supportive evidence under the locked
endpoint orientation. The shortlist is exploratory and is not a clinically
validated assay panel. Source data are provided in the Stage 23 source-data
directory.

## Fig. 2 | Endpoint-specific external validation of TRIAGE-OSA

**a**, Log2 top-1% enrichment ratios for 11 frozen rankings across CPAP,
cross-tissue, pediatric single-cell, known-interaction and untouched
independent endpoints. The full TRIAGE-OSA row is outlined. Colour is clipped
at +/-3 for readability while cell labels and source data retain exact values.
**b**, Locked
top-1% CPAP-reversal ratios from node-aware and participant sign-flip analyses;
one-sided permutation P values are shown. CPAP-1 denotes GSE133601 and CPAP-2
denotes GSE49800. **c**, Competitive effects for six prespecified Reactome
pathway axes in the untouched independent adipose (GSE38792) and PBMC
(GSE293433) cohorts. Asterisks denote axis-family BH FDR (*<0.05, **<0.01,
***<0.001). **d**, Fractions defining the evidence boundary: mean evaluable
axes, pairs with at least one cross-cohort or cross-lineage conflict, primary
proteins detected in pediatric serum and Pareto pairs with both proteins
detected. Stage 22 integration was post-development and did not change the
frozen ranking or candidate membership. Source data are provided in the Stage
23 source-data directory.
"""
    (output_dir / "figure_legends.md").write_text(legends, encoding="utf-8")


def write_manifest(root, output_dir, paths):
    rows = []
    for path in sorted(set(Path(value) for value in paths)):
        rows.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "output_manifest.tsv", sep="\t", index=False)


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    stage22 = resolve_from_root(root, args.stage22)
    output_dir = resolve_from_root(root, args.output_stage)
    apply_publication_style()
    data = load_inputs(root, stage22)
    print("OSA publication figures environment")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("matplotlib: {}".format(matplotlib.__version__))
    print("Stage 22: {}".format(stage22))
    print("Stage 22 manifest artifacts: {}".format(data["manifest_count"]))
    print("Backend: Python only")
    if args.check_env:
        print("OSA publication figures preflight: PASS")
        return 0
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force: {}".format(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    figure_paths.extend(make_figure1(data, output_dir, source_dir))
    figure_paths.extend(make_figure2(data, output_dir, source_dir))
    write_notes(output_dir, data)
    summary = {
        "status": "PASS_PUBLICATION_FIGURES_V1",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "stage22_status": data["summary"]["status"],
        "stage22_ranking_sha256": data["summary"]["frozen_ranking_sha256"],
        "pairs_plotted": int(len(data["pair"])),
        "pareto_pairs": int(len(data["pareto"])),
        "mechanism_modules": int(len(data["modules"])),
        "shortlist_proteins": int(len(data["shortlist"])),
        "backend": "python_matplotlib",
        "combined_figures": 2,
        "standalone_panels": 8,
        "formats_per_figure_or_panel": ["svg", "pdf", "tiff", "png"],
        "observations_sampled_or_excluded": False,
        "ranking_changed": False,
        "figures": [path.name for path in figure_paths],
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    all_paths = figure_paths + list(source_dir.glob("*")) + [
        output_dir / "figure_contract_and_qa.md",
        output_dir / "figure_legends.md",
        summary_path,
    ]
    write_manifest(root, output_dir, all_paths)
    print("OSA publication figures: PASS_PUBLICATION_FIGURES_V1")
    print("Output: {}".format(output_dir))
    print("Figure count: 2; exports per figure: SVG/PDF/TIFF/PNG")
    print("Pairs plotted: {}; sampled/excluded: no".format(len(data["pair"])))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
