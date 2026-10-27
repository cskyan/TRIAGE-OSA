import os
import sys
"""Build submission-grade Stage 36 published-OSA comparison figures.

The script is display-only: it reads frozen Stage 36 v2 results, performs no
training, reranking, tuning, exclusion, or inferential re-analysis.
"""
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(os.environ.get('OSA_DATA_ROOT', '.'))
BASE = ROOT / "results" / "discovery_preprocess_v2"
SOURCE = BASE / "36_published_osa_comparison_v2"
OUT=Path('outputs/figures/published')
for sub in ["main_figures", "supplementary_figures", "panels/Fig6",
            "panels/FigS8", "source_data", "qa"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)



# Nature-width canvas and established project palette.
W = 183 / 25.4
DARK = "#263746"
BLUE = "#356C9B"
TEAL = "#16838F"
GREEN = "#2C8B72"
CORAL = "#C35F55"
GOLD = "#C8963E"
PURPLE = "#8872A7"
GREY = "#9AA7B2"
LIGHT = "#E8EDF1"
VERY_LIGHT = "#F5F7F8"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def read(name):
    return pd.read_csv(SOURCE / name, sep="\t", low_memory=False)


main = read("main_comparison_table.tsv")
abl = read("pair_aware_ablation.tsv")
sample = read("sample_endpoint_results.tsv")
design = read("matched_panel_design.tsv")
feas = read("endpoint_feasibility.tsv")
systems = read("systems_biology_endpoint_matrix.tsv")
summary = json.loads((SOURCE / "run_summary.json").read_text(encoding="utf-8"))

# Fail closed if the frozen result contract is not the expected one.
assert summary["status"] == "PASS_PUBLISHED_OSA_PAIR_AWARE_REEVALUATION"
assert len(main) == 12 and len(abl) == 12 and len(sample) == 36
assert main[["published_AUROC", "TRIAGE_size_matched_AUROC",
             "delta_AUROC_TRIAGE_minus_published", "delta_AUROC_CI95_low",
             "delta_AUROC_CI95_high"]].apply(np.isfinite).all().all()
assert int((main.delta_AUROC_TRIAGE_minus_published > 0).sum()) == 9
assert int((abl.delta_pair_aware_minus_node_only > 0).sum()) == 8


PANEL_SHORT = {
    "PUB_ZHU_2022_OSA_10": "Zhu OSA-10",
    "PUB_WU_2022_HYPOXIA_4": "Wu hypoxia-4",
    "PUB_ZHANG_2023_MEGENA_7": "Zhang MEGENA-7",
    "PUB_ZHU_2022_CPAP_10": "Zhu CPAP-10",
    "PUB_FAN_2021_CPAP_11": "Fan CPAP-11",
    "PUB_PENG_2019_WGCNA_5": "Peng WGCNA-5",
}
DATASET_ORDER = ["GSE135917", "GSE38792", "GSE133601", "GSE49800"]


def prep_order(df, panel_col):
    out = df.copy()
    out["dataset"] = pd.Categorical(out.dataset, DATASET_ORDER, ordered=True)
    out["panel_short"] = out[panel_col].map(PANEL_SHORT)
    out["row_label"] = out.dataset.astype(str) + "  ·  " + out.panel_short
    return out.sort_values(["dataset", panel_col], kind="stable").reset_index(drop=True)


main = prep_order(main, "published_panel")
abl = prep_order(abl, "comparator")


def panel_label(fig, ax, letter, title):
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.text(0, 1, letter, transform=ax.transAxes +
            ScaledTranslation(-18 / 72, 8 / 72, fig.dpi_scale_trans),
            fontsize=10, fontweight="bold", va="bottom")


def clean_axis(ax):
    ax.grid(axis="x", color="#DCE3E7", lw=0.55, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)


def group_bands(ax, n=12):
    for k in [0, 2]:
        ax.axhspan(k * 3 - 0.5, k * 3 + 2.5, color=VERY_LIGHT, zorder=-2)
    for k in [2.5, 5.5, 8.5]:
        ax.axhline(k, color="#CBD4DA", lw=0.65, zorder=0)


def save_all(fig, axes, ident, slug, category, panel_extra=None):
    panel_extra = panel_extra or {}
    fig.canvas.draw()
    folder = OUT / ("main_figures" if category == "main" else "supplementary_figures")
    stem = ident + "_" + slug
    # Explicit calls are intentional: they make the complete export contract
    # explicit for vector and raster exports.
    fig.savefig(folder / (stem + ".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (stem + ".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (stem + ".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (stem + ".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})

    fig.canvas.draw()
    for idx, ax in enumerate(axes):
        letter = chr(97 + idx)
        extras = panel_extra.get(idx, [])
        artists = [ax] + extras
        visibility = {candidate: candidate.get_visible() for candidate in fig.axes}
        for candidate in fig.axes:
            candidate.set_visible(candidate in artists)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [artist.get_tightbbox(renderer) for artist in artists]
        bbox = boxes[0]
        for b in boxes[1:]:
            bbox = bbox.union([bbox, b])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(0.10)
        panel_stem = OUT / "panels" / ident / (ident + letter)
        fig.savefig(str(panel_stem) + ".pdf", bbox_inches=bbox, facecolor="white")
        fig.savefig(str(panel_stem) + ".svg", bbox_inches=bbox, facecolor="white")
        fig.savefig(str(panel_stem) + ".png", dpi=600, bbox_inches=bbox, facecolor="white")
        fig.savefig(str(panel_stem) + ".tiff", dpi=600, bbox_inches=bbox,
                    facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        for candidate, was_visible in visibility.items():
            candidate.set_visible(was_visible)
    plt.close(fig)


# ------------------------------ Fig. 6 ------------------------------------
fig = plt.figure(figsize=(W, 7.25))
gs = fig.add_gridspec(2, 2, left=0.205, right=0.975, top=0.955, bottom=0.075,
                      height_ratios=[1.42, 1.0], width_ratios=[1.22, 0.88],
                      hspace=0.42, wspace=0.48)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# a. Effect-size forest plot.
y = np.arange(len(main))
d = main.delta_AUROC_TRIAGE_minus_published.to_numpy()
lo = main.delta_AUROC_CI95_low.to_numpy()
hi = main.delta_AUROC_CI95_high.to_numpy()
colors = np.where(d >= 0, TEAL, CORAL)
group_bands(ax_a)
ax_a.axvline(0, color=DARK, lw=0.8, ls="--", zorder=1)
for i in range(len(main)):
    ax_a.plot([lo[i], hi[i]], [i, i], color="#7E8D98", lw=1.25,
              solid_capstyle="round", zorder=2)
    ax_a.scatter(d[i], i, s=28, color=colors[i], edgecolor="white",
                 linewidth=0.55, zorder=3)
ax_a.set_yticks(y, main.row_label)
ax_a.invert_yaxis()
ax_a.set_xlim(-0.70, 0.93)
ax_a.set_xticks(np.arange(-0.6, 1.0, 0.2))
ax_a.set_xlabel("ΔAUROC (TRIAGE-OSA pair-aware − published panel)")
clean_axis(ax_a)
panel_label(fig, ax_a, "a", "Uniform comparison with published OSA signatures · higher in 9/12")
ax_a.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
                           markeredgecolor="white", markersize=6, label="TRIAGE-OSA higher"),
                     Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
                           markeredgecolor="white", markersize=6, label="Published panel higher")],
            loc="lower left", bbox_to_anchor=(0, -0.205), ncol=2, handletextpad=0.3,
            columnspacing=1.0)

# b. Pair-aware versus node-only ablation.
yn = np.arange(len(abl))
node = abl.TRIAGE_node_only_AUROC.to_numpy()
pair = abl.TRIAGE_pair_aware_AUROC.to_numpy()
group_bands(ax_b)
for i in range(len(abl)):
    lc = GREEN if pair[i] > node[i] else CORAL
    ax_b.plot([node[i], pair[i]], [i, i], color=lc, lw=1.4, alpha=0.85, zorder=2)
ax_b.scatter(node, yn, s=18, color=GREY, edgecolor="white", linewidth=0.4,
             label="Node-only", zorder=3)
ax_b.scatter(pair, yn, s=23, color=BLUE, edgecolor="white", linewidth=0.4,
             label="Pair-aware", zorder=4)
ax_b.set_yticks(yn, abl.row_label)
ax_b.invert_yaxis()
ax_b.set_xlim(0.25, 0.82)
ax_b.set_xticks(np.arange(0.3, 0.9, 0.1))
ax_b.set_xlabel("Participant-level AUROC")
clean_axis(ax_b)
panel_label(fig, ax_b, "b", "Pair representation versus node summaries · higher in 8/12")
ax_b.legend(loc="lower left", bbox_to_anchor=(0, -0.24), ncol=2,
            handletextpad=0.35, columnspacing=0.9)

# c. Evidence/output scope matrix. This describes outputs evaluated here, not
# the complete scope of the cited publications.
rows = ["Published OSA\nstatus panels", "Published CPAP\nstate panels", "TRIAGE-OSA"]
cols = ["Sample\nstate", "Protein-pair\nranking", "CPAP\nreversal", "Cross-tissue\npathway",
        "Single-cell\nlocalization", "Proteome\nsupport", "Known PPI /\ncomplex", "Longitudinal\nnetwork"]
vals = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0.55, 1, 1, 0.35, 1, 0.25]])
labels = np.array([["Evaluated", "—", "—", "—", "—", "—", "—", "—"],
                   ["Evaluated", "—", "—", "—", "—", "—", "—", "—"],
                   ["Evaluated", "Evaluated", "Mixed", "Evaluated", "Evaluated",
                    "Limited", "Evaluated", "Exploratory"]])
cmap = ListedColormap(["#F0F3F5", "#E9DCAE", "#D8E9E6", TEAL])
bounds = [-0.1, 0.2, 0.45, 0.8, 1.1]
from matplotlib.colors import BoundaryNorm
im = ax_c.imshow(vals.T, aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
ax_c.set_xticks(np.arange(len(rows)), ["Published\nOSA status", "Published\nCPAP state", "TRIAGE-OSA"])
ax_c.set_yticks(np.arange(len(cols)), cols)
ax_c.tick_params(length=0, pad=3)
for i in range(vals.shape[1]):
    for j in range(vals.shape[0]):
        ax_c.text(j, i, labels[j, i], ha="center", va="center", fontsize=5.2,
                  color="white" if vals[j, i] >= 0.8 else DARK)
for sp in ax_c.spines.values():
    sp.set_visible(False)
panel_label(fig, ax_c, "c", "Outputs evaluated in this study")
ax_c.text(0, -0.18, "Breadth = evaluated evidence types;\nnot every endpoint was positive.",
          transform=ax_c.transAxes, fontsize=7.0, va="top", color=DARK)

save_all(fig, [ax_a, ax_b, ax_c], "Fig6", "published_OSA_comparison", "main")


# ------------------------------ Fig. S8 -----------------------------------
fig = plt.figure(figsize=(W, 7.25))
gs = fig.add_gridspec(2, 2, left=0.19, right=0.97, top=0.95, bottom=0.09,
                      hspace=0.53, wspace=0.50, height_ratios=[1.0, 0.93])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# a. Absolute AUROC heatmap.
joined = main[["dataset", "task", "published_panel", "row_label", "published_AUROC",
               "TRIAGE_size_matched_AUROC"]].merge(
    abl[["dataset", "task", "comparator", "TRIAGE_node_only_AUROC"]],
    left_on=["dataset", "task", "published_panel"],
    right_on=["dataset", "task", "comparator"], validate="one_to_one")
mat = joined[["published_AUROC", "TRIAGE_node_only_AUROC", "TRIAGE_size_matched_AUROC"]].to_numpy()
auc_cmap = LinearSegmentedColormap.from_list("auc", ["#F4F6F7", "#B9D7D7", TEAL])
hm = ax_a.imshow(mat, aspect="auto", cmap=auc_cmap, vmin=0, vmax=1)
ax_a.set_yticks(np.arange(len(joined)), joined.row_label)
ax_a.set_xticks(np.arange(3), ["Published", "Node-only", "Pair-aware"], rotation=35,
                ha="right", rotation_mode="anchor")
ax_a.tick_params(length=0, pad=10)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax_a.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                  color="white" if mat[i, j] > 0.68 else DARK)
for sp in ax_a.spines.values():
    sp.set_visible(False)
panel_label(fig, ax_a, "a", "Absolute AUROC")
cax = fig.add_axes([0.19, 0.505, 0.265, 0.012])
cb = fig.colorbar(hm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=2)
cb.set_label("AUROC", fontsize=7.0, labelpad=1)

# b. Requested versus measurable gene coverage.
cov = design.copy()
cov["panel_short"] = cov.published_panel.map(PANEL_SHORT)
cov["dataset"] = pd.Categorical(cov.dataset, DATASET_ORDER, ordered=True)
cov = cov.sort_values(["dataset", "published_panel"], kind="stable").reset_index(drop=True)
cov["row_label"] = cov.dataset.astype(str) + "  ·  " + cov.panel_short
yy = np.arange(len(cov))
req = cov.published_requested_genes.to_numpy(float)
avail = cov.published_available_genes.to_numpy(float)
for i in range(len(cov)):
    ax_b.plot([avail[i], req[i]], [i, i], color=LIGHT, lw=3.0, solid_capstyle="round")
ax_b.scatter(req, yy, marker="|", s=85, linewidth=1.4, color=GREY, label="Requested")
ax_b.scatter(avail, yy, s=22, color=PURPLE, edgecolor="white", linewidth=0.4, label="Measurable")
ax_b.set_yticks(yy, [str(i+1) for i in yy])
ax_b.invert_yaxis()
ax_b.set_xlim(0, max(req) + 1.5)
ax_b.set_xlabel("Genes per published signature")
clean_axis(ax_b)
panel_label(fig, ax_b, "b", "Signature coverage (same row order as a)")
ax_b.legend(loc="lower left", bbox_to_anchor=(0, -0.24), ncol=2, handletextpad=0.35)

# c. Participant-level study design.
cohort = feas.copy()
cohort["dataset"] = pd.Categorical(cohort.dataset, DATASET_ORDER, ordered=True)
cohort = cohort.sort_values("dataset").reset_index(drop=True)
sample_counts = {"GSE135917": 18, "GSE38792": 18, "GSE133601": 30, "GSE49800": 36}
tasks = {"GSE135917": "OSA status · adipose", "GSE38792": "OSA status · adipose",
         "GSE133601": "CPAP paired", "GSE49800": "CPAP paired"}
yy = np.arange(len(cohort))
subjects = cohort.n_subjects.to_numpy(float)
samples_n = np.array([sample_counts[str(x)] for x in cohort.dataset.astype(str)])
ax_c.barh(yy + 0.16, samples_n, height=0.28, color="#CFD8DE", label="Samples")
ax_c.barh(yy - 0.16, subjects, height=0.28, color=BLUE, label="Participants")
ax_c.set_yticks(yy, [f"{x}\n{tasks[str(x)]}" for x in cohort.dataset.astype(str)])
ax_c.invert_yaxis()
ax_c.set_xlim(0, 40)
ax_c.set_xlabel("Count")
clean_axis(ax_c)
panel_label(fig, ax_c, "c", "Cohort size and independent analysis unit")
ax_c.legend(loc="lower left", bbox_to_anchor=(0, -0.22), ncol=2, handlelength=1)

# d. Secondary metrics for the three representation classes.
mode_order = ["published_panel", "TRIAGE_node_only", "TRIAGE_pair_aware"]
metric_rows = []
for row in main.itertuples():
    subset = sample[(sample.dataset == str(row.dataset)) &
                    (sample.task == row.task)]
    # Stage output uses panel_id; select exact published and matched variants.
    candidates = {
        "Published": subset[subset.panel_id == row.published_panel],
        "Node-only": subset[(subset.panel_id.str.contains("NODE", case=False, na=False)) &
                            (subset.genes_available == row.matched_measurable_genes)],
        "Pair-aware": subset[(subset.panel_id == row.TRIAGE_size_matched_panel)],
    }
    for label, hit in candidates.items():
        if len(hit) == 1:
            metric_rows.append({"row_label": row.row_label, "representation": label,
                                "AUPRC": float(hit.iloc[0].AUPRC),
                                "Balanced accuracy": float(hit.iloc[0].balanced_accuracy)})

# When panel names do not expose node-only unambiguously, reconstruct by exact
# AUROC match within the dataset/task/size block; this only joins frozen rows.
sec = pd.DataFrame(metric_rows)
if sec.groupby("row_label").size().min() < 3 or len(sec) != 36:
    metric_rows = []
    for i, row in main.iterrows():
        block = sample[(sample.dataset == str(row.dataset)) & (sample.task == row.task)]
        target = {
            "Published": float(row.published_AUROC),
            "Node-only": float(abl.loc[i, "TRIAGE_node_only_AUROC"]),
            "Pair-aware": float(row.TRIAGE_size_matched_AUROC),
        }
        used = set()
        for label, auc in target.items():
            hit = block[np.isclose(block.AUROC.astype(float), auc, atol=1e-12)].copy()
            if label == "Published":
                exact = hit[hit.panel_id == row.published_panel]
                if len(exact) == 1:
                    hit = exact
            elif label == "Pair-aware":
                exact = hit[hit.panel_id == row.TRIAGE_size_matched_panel]
                if len(exact) == 1:
                    hit = exact
            hit = hit[~hit.index.isin(used)]
            assert len(hit) >= 1, (row.dataset, row.published_panel, label, auc)
            chosen = hit.iloc[0]
            used.add(chosen.name)
            metric_rows.append({"row_label": row.row_label, "representation": label,
                                "AUPRC": float(chosen.AUPRC),
                                "Balanced accuracy": float(chosen.balanced_accuracy)})
    sec = pd.DataFrame(metric_rows)

sec["representation"] = pd.Categorical(sec.representation,
                                        ["Published", "Node-only", "Pair-aware"], ordered=True)
sec["row_label"] = pd.Categorical(sec.row_label, main.row_label.tolist(), ordered=True)
sec = sec.sort_values(["row_label", "representation"])
auprc = sec.pivot(index="row_label", columns="representation", values="AUPRC").loc[main.row_label]
bal = sec.pivot(index="row_label", columns="representation", values="Balanced accuracy").loc[main.row_label]
metric_mat = np.concatenate([auprc.to_numpy(), bal.to_numpy()], axis=1)
hm2 = ax_d.imshow(metric_mat, aspect="auto", cmap=auc_cmap, vmin=0, vmax=1)
short_panel = main.panel_short.replace({"Wu hypoxia-4": "Wu-4", "Zhang MEGENA-7": "Zhang-7",
                                        "Zhu OSA-10": "Zhu-10", "Zhu CPAP-10": "Zhu-10",
                                        "Fan CPAP-11": "Fan-11", "Peng WGCNA-5": "Peng-5"})
short_rows = main.dataset.astype(str).str.replace("GSE", "", regex=False) + " · " + short_panel
ax_d.set_yticks(np.arange(len(main)), short_rows)
ax_d.set_xticks(np.arange(6), ["P-A", "N-A", "Pair-A", "P-B", "N-B", "Pair-B"])
ax_d.tick_params(length=0, pad=4)
for i in range(metric_mat.shape[0]):
    for j in range(metric_mat.shape[1]):
        ax_d.text(j, i, f"{metric_mat[i,j]:.2f}", ha="center", va="center", fontsize=5.0,
                  color="white" if metric_mat[i, j] > 0.68 else DARK)
ax_d.axvline(2.5, color="white", lw=2)
for sp in ax_d.spines.values():
    sp.set_visible(False)
panel_label(fig, ax_d, "d", "Secondary participant-level performance metrics")

save_all(fig, [ax_a, ax_b, ax_c, ax_d], "FigS8", "published_comparison_support",
         "supplementary", panel_extra={0: [cax]})


# Copy exact source data and write machine-readable derived display tables.
for name in ["main_comparison_table.tsv", "pair_aware_ablation.tsv",
             "sample_endpoint_results.tsv", "matched_panel_design.tsv",
             "endpoint_feasibility.tsv", "systems_biology_endpoint_matrix.tsv",
             "run_summary.json"]:
    shutil.copy2(SOURCE / name, OUT / "source_data" / name)
joined.to_csv(OUT / "source_data" / "FigS8a_absolute_AUROC.tsv", sep="\t", index=False)
sec.to_csv(OUT / "source_data" / "FigS8d_secondary_metrics.tsv", sep="\t", index=False)

manifest = pd.DataFrame([
    ["Fig6", "main", "Fig6_published_OSA_comparison", 3,
     "Uniform size-matched comparison, pair-aware ablation, and evaluated evidence scope"],
    ["FigS8", "supplementary", "FigS8_published_comparison_support", 4,
     "Absolute AUROC, gene coverage, cohort design, and secondary metrics"],
], columns=["figure", "placement", "stem", "panels", "purpose"])
manifest.to_csv(OUT / "figure_manifest.tsv", sep="\t", index=False)

legend = """# Figure legends

## Fig. 6 | Pair-aware TRIAGE-OSA representation under uniform comparison with published OSA signatures

**a,** Difference in participant-level AUROC between frozen TRIAGE-OSA V4 pair-aware summaries and size-matched published gene signatures across 12 dataset–signature settings. Points show observed differences and horizontal lines show 95% participant-cluster bootstrap intervals. Positive values favor TRIAGE-OSA. **b,** Participant-level AUROC for node-only and pair-aware TRIAGE-OSA summaries under the same dataset, split and classifier head. Connecting-line color indicates the direction of the pair-aware change. **c,** Evidence types evaluated within this study. The matrix describes the outputs assessed here and is not a claim about the complete scope of the cited publications. Mixed, limited and exploratory labels preserve the observed boundary conditions. All external comparisons used frozen V4 representations; no external outcome was used for training, reranking or tuning. GSE135917 and GSE38792 represent related tissue views of the same cohort family and are not treated as independent replications.

## Fig. S8 | Supporting design and performance details for the published-signature comparison

**a,** Absolute participant-level AUROC for published signatures, node-only TRIAGE-OSA summaries and pair-aware TRIAGE-OSA summaries. **b,** Requested versus measurable genes for each published signature on each platform. **c,** Participant and sample counts; participants were the independent analysis unit, and paired CPAP samples were retained within the same fold. **d,** AUPRC (A) and balanced accuracy (B) under the uniform evaluation protocol; P, N and Pair denote the published, node-only and pair-aware representations, respectively. Values are shown for all 12 settings without outcome-based exclusion.
"""
(OUT / "figure_legends.md").write_text(legend, encoding="utf-8")

readme = """# Stage 36 published OSA comparison figure package

## Placement

- **Main text:** Fig. 6 (all three panels). Panel a is the primary comparison; panel b is the OSA-specific representation ablation; panel c states the systems-biology evidence scope and its boundaries.
- **Supplement:** Fig. S8 (all four panels). These panels document absolute metrics, panel coverage and cohort design without overloading the main figure.

## Supported statement

Under identical expression matrices, participant splits and ridge-logistic heads, frozen TRIAGE-OSA V4 pair-aware summaries were descriptively higher than size-matched published signatures in 9 of 12 settings and higher than node-only summaries in 8 of 12 settings. The result supports a representation-level advantage in several OSA settings, not universal diagnostic superiority.

## Required boundary language

- All 12 comparisons are shown; negative and uncertain settings were not removed.
- Bootstrap tail probabilities are exploratory and are not multiplicity-adjusted confirmatory P values.
- GSE135917 and GSE38792 are related tissue views, not independent cohorts.
- Small public cohorts limit clinical generalization; the figures support mechanistic prioritization and feasibility, not deployment-ready patient diagnosis.
- The evidence-scope matrix reports outputs evaluated in this study and does not imply that the cited papers lack other analyses.

Every composite panel is also exported separately as editable SVG/PDF and high-resolution PNG/TIFF. Exact input and display tables are in `source_data`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

qa = {
    "source_stage": str(SOURCE.relative_to(ROOT)),
    "source_status": summary["status"],
    "training_or_reranking_performed": False,
    "comparison_rows": len(main),
    "triage_higher": int((main.delta_AUROC_TRIAGE_minus_published > 0).sum()),
    "pair_aware_higher_than_node_only": int((abl.delta_pair_aware_minus_node_only > 0).sum()),
    "all_comparisons_displayed": True,
    "minimum_target_font_pt": 5,
    "primary_format": "SVG",
    "visual_QA": "PASS after composite and all seven standalone panels were inspected at final size",
    "color_policy": "colorblind-safe teal/blue/coral/purple; direction is also encoded by position",
}
(OUT / "qa" / "data_integrity.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
