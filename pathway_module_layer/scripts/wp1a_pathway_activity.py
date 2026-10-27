#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WP1a: sample-level pathway activity via ssGSEA (Barbie 2009).

Python 3.7, standard library only.

For each discovery dataset this reads the frozen gene-level expression matrix
(HGNC_ID x samples) and produces a per-sample pathway-activity matrix.  These
activities feed the optional pathway-based OSA/severity prediction (plan P1) and
give a sample-resolved companion to the contrast-level GSEA in wp1b.

ssGSEA here follows the GSVA-package definition: per sample, rank genes by
expression, walk the ranked list accumulating a weighted hit CDF minus a uniform
miss CDF, and integrate the difference (area, not max deviation).  The score
matrix is range-normalised across all (sample, set) cells when --normalize is on.

Discovery-only: reads GSE75097 / GSE71356 expression.  No validation data.
"""

from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_STAGE = "21_pathway_activity_v1"
MAP_REL = "results/discovery_preprocess_v2/03_mapping"

# dataset -> (expression matrix, differential used only to map HGNC->GMT id space)
DISCOVERY_EXPR = [
    ("GSE75097",
     "results/discovery_preprocess_v2/03_mapping/GSE75097/gene_expression_primary_samples.tsv.gz",
     "results/discovery_preprocess_v2/04_discovery_v2/differential/GSE75097_OSA_state.tsv.gz"),
    ("GSE71356",
     "results/discovery_preprocess_v2/03_mapping/GSE71356/gene_expression_primary_samples.tsv.gz",
     "results/discovery_preprocess_v2/04_discovery_v2/differential/GSE71356_IH_x_time.tsv.gz"),
]


def build_hgnc_map(diff_path, id_space):
    """HGNC_ID -> GMT-space id (upper) from a differential table."""
    target = "Entrez" if id_space == "entrez" else "gene_symbol_current"
    mp = {}
    for row in C.iter_tsv(diff_path):
        h = (row.get("HGNC_ID") or "").strip()
        t = (row.get(target) or "").strip().upper()
        if h and t:
            mp[h] = t
    return mp


def read_expression(path):
    """Return (sample_ids, {gene_row_id: [values]})."""
    with C._opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r").split("\t")
        samples = header[1:]
        data = {}
        for line in handle:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            vals = []
            ok = True
            for x in parts[1:]:
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(float("nan"))
            if ok:
                data[gid] = vals
    return samples, data


def ssgsea_sample(expr_values, gene_order_idx, member_positions, alpha):
    """ssGSEA enrichment for one sample.

    expr_values      : list of expression values (index = gene index)
    gene_order_idx   : indices sorted by decreasing expression (precomputed)
    member_positions : set of ranks(1-based in the ordered list) that are members
    Returns the integrated ES (area between hit and miss CDFs).
    """
    n = len(gene_order_idx)
    k = len(member_positions)
    if k == 0 or k >= n:
        return 0.0
    ranks = C.average_ranks(expr_values)  # 1..n, higher value -> higher rank
    # walk in decreasing expression order; weight = rank_value^alpha at hits
    total_hit_w = 0.0
    for pos in member_positions:
        idx = gene_order_idx[pos - 1]
        total_hit_w += ranks[idx] ** alpha
    if total_hit_w <= 0.0:
        return 0.0
    miss_step = 1.0 / (n - k)
    cum_hit = 0.0
    cum_miss = 0.0
    area = 0.0
    memset = member_positions
    for j in range(n):
        idx = gene_order_idx[j]
        if (j + 1) in memset:
            cum_hit += (ranks[idx] ** alpha) / total_hit_w
        else:
            cum_miss += miss_step
        area += (cum_hit - cum_miss)
    return area


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--stage", default=DEFAULT_STAGE)
    ap.add_argument("--id-space", default="auto",
                    choices=["auto", "symbol", "entrez"])
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--max-size", type=int, default=500)
    ap.add_argument("--normalize", action="store_true",
                    help="range-normalise the score matrix across all cells")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, "results", "discovery_preprocess_v2", args.stage)
    os.makedirs(out_dir, exist_ok=True)

    sets = C.parse_gmt(args.gmt)
    if not sets:
        sys.exit("ERROR: no gene sets parsed from %s" % args.gmt)
    id_space = args.id_space
    if id_space == "auto":
        id_space = C.detect_gmt_id_space(sets)

    input_hashes = {"gmt": {"path": C.relative_path(args.gmt, root),
                            "sha256": C.sha256_file(args.gmt),
                            "n_sets": len(sets), "id_space": id_space}}
    manifest = []

    for ds, expr_rel, diff_rel in DISCOVERY_EXPR:
        expr_path = os.path.join(root, expr_rel)
        diff_path = os.path.join(root, diff_rel)
        if not (os.path.exists(expr_path) and os.path.exists(diff_path)):
            print("WARN: missing inputs for %s" % ds, file=sys.stderr)
            continue
        hmap = build_hgnc_map(diff_path, id_space)
        samples, data = read_expression(expr_path)
        # translate expression rows to GMT id space, collapse dup ids by mean
        translated = {}
        for gid, vals in data.items():
            tid = hmap.get(gid)
            if not tid:
                continue
            translated.setdefault(tid, []).append(vals)
        gene_ids = sorted(translated.keys())
        matrix = []
        for tid in gene_ids:
            grp = translated[tid]
            if len(grp) == 1:
                matrix.append(grp[0])
            else:
                merged = []
                for col in range(len(samples)):
                    merged.append(sum(g[col] for g in grp) / float(len(grp)))
                matrix.append(merged)
        gene_index = {g: i for i, g in enumerate(gene_ids)}

        # precompute, per sample, the gene order by decreasing expression
        per_sample_order = []
        for s in range(len(samples)):
            col = [matrix[i][s] for i in range(len(gene_ids))]
            order = sorted(range(len(gene_ids)), key=lambda i: (-col[i], gene_ids[i]))
            per_sample_order.append((col, order))

        # member positions per set per sample depend on ordering -> compute per sample
        kept_sets = []
        for gs in sets:
            members = [gene_index[g] for g in gs["genes"] if g in gene_index]
            if args.min_size <= len(members) <= args.max_size:
                kept_sets.append((gs["name"], set(members)))

        scores = {}  # set_name -> [per sample]
        for name, _ in kept_sets:
            scores[name] = [0.0] * len(samples)
        for s in range(len(samples)):
            col, order = per_sample_order[s]
            pos_of_gene = {order[r]: (r + 1) for r in range(len(order))}
            for name, memidx in kept_sets:
                member_positions = set(pos_of_gene[i] for i in memidx)
                scores[name][s] = ssgsea_sample(col, order, member_positions, args.alpha)

        if args.normalize and scores:
            allv = [v for row in scores.values() for v in row]
            lo, hi = min(allv), max(allv)
            rng = (hi - lo) or 1.0
            for name in scores:
                scores[name] = [(v - lo) / rng for v in scores[name]]

        fields = ["pathway"] + samples
        rows = []
        for name, _ in kept_sets:
            row = {"pathway": name}
            for i, s in enumerate(samples):
                row[s] = C.fmt(scores[name][i])
            rows.append(row)
        out_path = os.path.join(out_dir, "pathway_activity_%s.tsv.gz" % ds)
        C.write_gzip_tsv(out_path, rows, fields)
        manifest.append({"dataset": ds, "path": C.relative_path(out_path, root),
                         "sha256": C.sha256_file(out_path),
                         "n_sets": len(kept_sets), "n_samples": len(samples),
                         "n_genes_mapped": len(gene_ids)})
        input_hashes[ds] = {"expression": C.sha256_file(expr_path),
                            "idmap_differential": C.sha256_file(diff_path)}
        print("ssGSEA %-9s samples=%d sets=%d genes=%d"
              % (ds, len(samples), len(kept_sets), len(gene_ids)))

    run_summary = {
        "script": "wp1a_pathway_activity.py",
        "script_version": SCRIPT_VERSION, "shared_version": C.SHARED_VERSION,
        "created_at_utc": C.utc_now(), "stage": args.stage,
        "method": "ssGSEA", "discovery_only": True, "validation_data_read": False,
        "params": {"alpha": args.alpha, "id_space": id_space,
                   "min_size": args.min_size, "max_size": args.max_size,
                   "normalize": bool(args.normalize)},
        "input_hashes": input_hashes, "outputs": manifest,
    }
    C.atomic_json(os.path.join(out_dir, "run_summary.json"), run_summary)
    C.write_tsv(os.path.join(out_dir, "output_manifest.tsv"),
                [{"dataset": m["dataset"], "path": m["path"], "sha256": m["sha256"]}
                 for m in manifest], ["dataset", "path", "sha256"])
    print("OK  ssGSEA activity written for %d dataset(s)." % len(manifest))


if __name__ == "__main__":
    main()
