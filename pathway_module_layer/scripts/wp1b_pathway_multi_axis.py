#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WP1b: contrast-level pre-ranked GSEA -> discovery multi-axis pathway matrix.

Python 3.7, standard library only.

Reads the frozen discovery differential tables (GSE75097 OSA/severe/AHI/ordinal
and GSE71356 IH x time), ranks every gene by its signed moderated statistic, and
runs a seeded pre-ranked GSEA against a fixed gene-set database (Hallmark or
Reactome GMT).  Emits per-axis GSEA tables and one assembled multi-axis pathway
evidence matrix that WP2 consumes.

Blinding: only discovery contrasts are read here.  CPAP / adipose / scRNA /
proteome / known-PPI are never touched -- those axes are added post-freeze by a
separate external pass (see the plan doc, WP5).
"""

from __future__ import print_function

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_STAGE = "20_pathway_multi_axis_v1"
DIFF_REL = "results/discovery_preprocess_v2/04_discovery_v2/differential"

# axis -> discovery differential file (discovery-only; frozen upstream)
DISCOVERY_AXES = [
    ("OSA", "GSE75097_OSA_state.tsv.gz"),
    ("severe", "GSE75097_severe.tsv.gz"),
    ("AHI", "GSE75097_AHI.tsv.gz"),
    ("ordinal", "GSE75097_ordinal.tsv.gz"),
    ("IH", "GSE71356_IH_x_time.tsv.gz"),
]


def load_ranked_list(diff_path, rank_metric, id_space, weight_exponent):
    """Build a RankedList from a differential table.

    Collapses to one value per gene id (median of the signed metric across any
    duplicate rows) so multi-probe/isoform residue is deterministic.
    """
    id_col = "Entrez" if id_space == "entrez" else "gene_symbol_current"
    buckets = {}
    for row in C.iter_tsv(diff_path):
        gid = (row.get(id_col) or "").strip().upper()
        raw = (row.get(rank_metric) or "").strip()
        if not gid or gid in ("", "NA", "NAN"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        buckets.setdefault(gid, []).append(val)
    gene_ids = []
    scores = []
    for gid, vals in buckets.items():
        vals.sort()
        m = len(vals)
        med = vals[m // 2] if m % 2 == 1 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
        gene_ids.append(gid)
        scores.append(med)
    return C.RankedList(gene_ids, scores, weight_exponent=weight_exponent)


def run_axis_gsea(ranked, sets, min_size, max_size, n_perm, seed):
    """Run pre-ranked GSEA for one axis over all gene sets."""
    results = []
    for gs in sets:
        enr = ranked.enrichment(gs["genes"])
        if enr is None:
            continue
        k = enr["n_hits"]
        if k < min_size or k > max_size:
            continue
        # deterministic per-(axis,set) seed so results never depend on order
        rng = random.Random("%d|%s|%d" % (seed, gs["name"], k))
        null = ranked.permutation_null(k, n_perm, rng)
        nes, p = C.normalized_es_and_p(enr["es"], null)
        results.append({
            "pathway": gs["name"],
            "set_size_db": len(gs["genes"]),
            "n_hits": k,
            "ES": enr["es"],
            "NES": nes,
            "direction": "up" if enr["es"] >= 0 else "down",
            "P_value": p,
            "leading_edge_size": len(enr["leading_edge"]),
            "leading_edge": ";".join(enr["leading_edge"]),
        })
    # BH-FDR within this axis
    pvals = [r["P_value"] for r in results]
    qvals = C.benjamini_hochberg(pvals)
    for r, q in zip(results, qvals):
        r["FDR_BH"] = q
    results.sort(key=lambda r: (r["P_value"], -abs(r["NES"])))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--gmt", required=True,
                    help="Path to the fixed primary gene-set GMT (Hallmark or Reactome).")
    ap.add_argument("--stage", default=DEFAULT_STAGE)
    ap.add_argument("--rank-metric", default="signed_score",
                    choices=["signed_score", "moderated_t", "beta"])
    ap.add_argument("--id-space", default="auto",
                    choices=["auto", "symbol", "entrez"])
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--max-size", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--weight-exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260726)
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

    multi = {}   # pathway -> row of per-axis fields
    axis_order = []
    per_axis_manifest = []

    for axis, fname in DISCOVERY_AXES:
        diff_path = os.path.join(root, DIFF_REL, fname)
        if not os.path.exists(diff_path):
            print("WARN: missing differential for axis %s: %s" % (axis, diff_path),
                  file=sys.stderr)
            continue
        axis_order.append(axis)
        input_hashes[axis] = {"path": C.relative_path(diff_path, root),
                              "sha256": C.sha256_file(diff_path)}
        ranked = load_ranked_list(diff_path, args.rank_metric, id_space,
                                  args.weight_exponent)
        res = run_axis_gsea(ranked, sets, args.min_size, args.max_size,
                            args.n_perm, args.seed)
        # write per-axis table
        axis_fields = ["pathway", "set_size_db", "n_hits", "ES", "NES",
                       "direction", "P_value", "FDR_BH", "leading_edge_size",
                       "leading_edge"]
        axis_rows = [{k: (C.fmt(r[k]) if isinstance(r[k], float) else r[k])
                      for k in axis_fields} for r in res]
        axis_path = os.path.join(out_dir, "gsea_%s.tsv.gz" % axis)
        C.write_gzip_tsv(axis_path, axis_rows, axis_fields)
        per_axis_manifest.append((axis, C.relative_path(axis_path, root),
                                  C.sha256_file(axis_path), ranked.n, len(res)))
        # fold into multi-axis matrix
        for r in res:
            m = multi.setdefault(r["pathway"], {"pathway": r["pathway"]})
            m["%s_NES" % axis] = r["NES"]
            m["%s_P" % axis] = r["P_value"]
            m["%s_FDR" % axis] = r["FDR_BH"]
            m["%s_dir" % axis] = r["direction"]
            m["%s_hits" % axis] = r["n_hits"]
        print("axis %-8s genes=%d pathways_scored=%d" % (axis, ranked.n, len(res)))

    # assemble multi-axis matrix (rows = pathways seen in >=1 axis)
    fields = ["pathway"]
    for axis in axis_order:
        fields += ["%s_NES" % axis, "%s_P" % axis, "%s_FDR" % axis,
                   "%s_dir" % axis, "%s_hits" % axis]
    rows = []
    for pw in sorted(multi.keys()):
        m = multi[pw]
        row = {"pathway": pw}
        for f in fields[1:]:
            v = m.get(f, "")
            row[f] = C.fmt(v) if isinstance(v, float) else v
        rows.append(row)
    matrix_path = os.path.join(out_dir, "pathway_multi_axis_evidence.tsv")
    C.write_tsv(matrix_path, rows, fields)

    run_summary = {
        "script": "wp1b_pathway_multi_axis.py",
        "script_version": SCRIPT_VERSION,
        "shared_version": C.SHARED_VERSION,
        "created_at_utc": C.utc_now(),
        "stage": args.stage,
        "discovery_only": True,
        "validation_data_read": False,
        "params": {
            "rank_metric": args.rank_metric, "id_space": id_space,
            "min_size": args.min_size, "max_size": args.max_size,
            "n_perm": args.n_perm, "weight_exponent": args.weight_exponent,
            "seed": args.seed,
        },
        "axes": axis_order,
        "n_pathways_in_matrix": len(rows),
        "input_hashes": input_hashes,
        "outputs": {
            "multi_axis_matrix": {
                "path": C.relative_path(matrix_path, root),
                "sha256": C.sha256_file(matrix_path)},
            "per_axis": [
                {"axis": a, "path": p, "sha256": h, "genes_ranked": g,
                 "pathways_scored": n}
                for (a, p, h, g, n) in per_axis_manifest],
        },
    }
    C.atomic_json(os.path.join(out_dir, "run_summary.json"), run_summary)

    manifest_rows = [{"output": "pathway_multi_axis_evidence.tsv",
                      "path": C.relative_path(matrix_path, root),
                      "sha256": C.sha256_file(matrix_path)}]
    for (a, p, h, g, n) in per_axis_manifest:
        manifest_rows.append({"output": "gsea_%s.tsv.gz" % a, "path": p, "sha256": h})
    C.write_tsv(os.path.join(out_dir, "output_manifest.tsv"),
                manifest_rows, ["output", "path", "sha256"])

    print("OK  wrote %d-pathway multi-axis matrix over %d discovery axes -> %s"
          % (len(rows), len(axis_order), C.relative_path(matrix_path, root)))


if __name__ == "__main__":
    main()
