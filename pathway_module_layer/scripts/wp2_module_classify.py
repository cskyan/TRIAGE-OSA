#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WP2: classify pathways into the four OSA disease modules + evidence level.

Python 3.7, standard library only.

Consumes wp1b's discovery multi-axis matrix and assigns each pathway:
  - IH_triggered      : IH effect present + OSA direction consistent + a
                        severity/AHI axis consistent
  - OSA_sustained     : OSA effect stable + severity/AHI dose-response, IH not
                        required
  - CPAP_reversible   : OSA-abnormal + CPAP moves toward normal  (external axis)
  - CPAP_persistent   : OSA-abnormal + CPAP does not recover      (external axis)
and an evidence level A/B/C/D from how many independent axes agree in direction.

Blinding is enforced structurally: CPAP / adipose / scRNA are VALIDATION data,
so the CPAP-reversible / CPAP-persistent decision is only made when an external
axis table is supplied together with --discovery-frozen-sha256 matching the wp1b
matrix that was frozen first.  Without it those two classes stay 'pending_external'.
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
DEFAULT_STAGE = "22_pathway_modules_v1"
DISCOVERY_AXES = ["OSA", "severe", "AHI", "ordinal", "IH"]
SEVERITY_AXES = ["severe", "AHI", "ordinal"]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def axis_sig(row, axis, p_nom, fdr_sig):
    p = to_float(row.get("%s_P" % axis, ""))
    q = to_float(row.get("%s_FDR" % axis, ""))
    nom = (p is not None and p < p_nom)
    strong = (q is not None and q < fdr_sig)
    return nom, strong, row.get("%s_dir" % axis, "")


def classify_row(row, p_nom, fdr_sig):
    osa_nom, osa_strong, osa_dir = axis_sig(row, "OSA", p_nom, fdr_sig)
    discovery_sig = osa_nom or osa_strong

    # severity dose-response: any severity axis nominally significant, same dir as OSA
    severity_consistent = []
    for ax in SEVERITY_AXES:
        nom, strong, d = axis_sig(row, ax, p_nom, fdr_sig)
        if nom and d and osa_dir and d == osa_dir:
            severity_consistent.append(ax)
    severity_dose = len(severity_consistent) > 0

    ih_nom, ih_strong, ih_dir = axis_sig(row, "IH", p_nom, fdr_sig)
    ih_consistent = ih_nom and ih_dir and osa_dir and ih_dir == osa_dir

    ih_triggered = bool(discovery_sig and ih_consistent and severity_dose)
    osa_sustained = bool(discovery_sig and severity_dose and not ih_triggered)

    # count consistent discovery axes (same direction as OSA, nominally sig)
    consistent = 0
    conflict = False
    if discovery_sig:
        consistent += 1
    for ax in SEVERITY_AXES + ["IH"]:
        nom, strong, d = axis_sig(row, ax, p_nom, fdr_sig)
        if nom and d:
            if osa_dir and d == osa_dir:
                consistent += 1
            elif osa_dir and d != osa_dir:
                conflict = True

    if conflict:
        level = "D"
    elif consistent >= 4:
        level = "A"
    elif consistent == 3:
        level = "A"
    elif consistent == 2:
        level = "B"
    elif consistent == 1:
        level = "C"
    else:
        level = "D"

    return {
        "OSA_direction": osa_dir,
        "discovery_significant": "yes" if discovery_sig else "no",
        "severity_consistent_axes": ";".join(severity_consistent),
        "IH_triggered": "yes" if ih_triggered else "no",
        "OSA_sustained": "yes" if osa_sustained else "no",
        "n_axes_consistent": consistent,
        "direction_conflict": "yes" if conflict else "no",
        "evidence_level": level,
    }


def apply_external_cpap(cls, ext_row, osa_dir, p_nom):
    """Decide CPAP class from an external CPAP axis (post-freeze only)."""
    if ext_row is None or not osa_dir:
        return "pending_external"
    cp = ext_row.get("CPAP_dir", "")
    cp_p = to_float(ext_row.get("CPAP_P", ""))
    if not cp or cp_p is None:
        return "pending_external"
    # reversal = CPAP direction opposite to the OSA abnormality
    if cp != osa_dir and cp_p < p_nom:
        return "reversible"
    if cp == osa_dir:
        return "persistent"
    return "uncertain"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--matrix", default=None,
                    help="wp1b pathway_multi_axis_evidence.tsv (default: stage 20).")
    ap.add_argument("--stage", default=DEFAULT_STAGE)
    ap.add_argument("--p-nominal", type=float, default=0.05)
    ap.add_argument("--fdr-significant", type=float, default=0.25)
    ap.add_argument("--external-axes", default=None,
                    help="optional post-freeze CPAP/adipose/scRNA axis table.")
    ap.add_argument("--discovery-frozen-sha256", default=None,
                    help="required with --external-axes: sha256 of the frozen matrix.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    matrix_path = args.matrix or os.path.join(
        root, "results/discovery_preprocess_v2/20_pathway_multi_axis_v1",
        "pathway_multi_axis_evidence.tsv")
    if not os.path.exists(matrix_path):
        sys.exit("ERROR: multi-axis matrix not found: %s" % matrix_path)

    matrix_sha = C.sha256_file(matrix_path)
    external = None
    if args.external_axes:
        if not args.discovery_frozen_sha256:
            sys.exit("REFUSED: --external-axes requires --discovery-frozen-sha256 "
                     "(the frozen wp1b matrix hash) to preserve blinding.")
        if args.discovery_frozen_sha256 != matrix_sha:
            sys.exit("REFUSED: matrix sha256 mismatch; discovery set is not the "
                     "frozen one. external=%s current=%s"
                     % (args.discovery_frozen_sha256, matrix_sha))
        ext_rows, _ = C.read_tsv(args.external_axes)
        external = {r["pathway"]: r for r in ext_rows}

    rows, _ = C.read_tsv(matrix_path)
    out_fields = ["pathway", "OSA_direction", "discovery_significant",
                  "severity_consistent_axes", "IH_triggered", "OSA_sustained",
                  "CPAP_class", "n_axes_consistent", "direction_conflict",
                  "evidence_level", "modules"]
    out_rows = []
    counts = {"IH_triggered": 0, "OSA_sustained": 0, "CPAP_reversible": 0,
              "CPAP_persistent": 0, "A": 0, "B": 0, "C": 0, "D": 0}
    for r in rows:
        cls = classify_row(r, args.p_nominal, args.fdr_significant)
        ext_row = external.get(r["pathway"]) if external else None
        cpap = apply_external_cpap(cls, ext_row, cls["OSA_direction"], args.p_nominal)
        modules = []
        if cls["IH_triggered"] == "yes":
            modules.append("IH_triggered")
        if cls["OSA_sustained"] == "yes":
            modules.append("OSA_sustained")
        if cpap == "reversible":
            modules.append("CPAP_reversible")
            counts["CPAP_reversible"] += 1
        elif cpap == "persistent":
            modules.append("CPAP_persistent")
            counts["CPAP_persistent"] += 1
        out = {"pathway": r["pathway"], "CPAP_class": cpap,
               "modules": ";".join(modules) if modules else "none"}
        out.update(cls)
        out_rows.append({k: out.get(k, "") for k in out_fields})
        if cls["IH_triggered"] == "yes":
            counts["IH_triggered"] += 1
        if cls["OSA_sustained"] == "yes":
            counts["OSA_sustained"] += 1
        counts[cls["evidence_level"]] += 1

    out_rows.sort(key=lambda r: (r["evidence_level"], r["pathway"]))
    out_dir = os.path.join(root, "results", "discovery_preprocess_v2", args.stage)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pathway_classification.tsv")
    C.write_tsv(out_path, out_rows, out_fields)

    run_summary = {
        "script": "wp2_module_classify.py", "script_version": SCRIPT_VERSION,
        "shared_version": C.SHARED_VERSION, "created_at_utc": C.utc_now(),
        "stage": args.stage,
        "matrix": {"path": C.relative_path(matrix_path, root), "sha256": matrix_sha},
        "params": {"p_nominal": args.p_nominal, "fdr_significant": args.fdr_significant},
        "external_axes_used": bool(external),
        "external_axes_path": (C.relative_path(args.external_axes, root)
                               if external else None),
        "counts": counts, "n_pathways": len(out_rows),
        "note": ("CPAP classes pending_external until a post-freeze CPAP axis is "
                 "supplied; IH_triggered / OSA_sustained are discovery-derivable."),
    }
    C.atomic_json(os.path.join(out_dir, "run_summary.json"), run_summary)
    C.write_tsv(os.path.join(out_dir, "output_manifest.tsv"),
                [{"output": "pathway_classification.tsv",
                  "path": C.relative_path(out_path, root),
                  "sha256": C.sha256_file(out_path)}],
                ["output", "path", "sha256"])
    print("OK  classified %d pathways | IH_triggered=%d OSA_sustained=%d "
          "| levels A=%d B=%d C=%d D=%d"
          % (len(out_rows), counts["IH_triggered"], counts["OSA_sustained"],
             counts["A"], counts["B"], counts["C"], counts["D"]))
    if not external:
        print("    CPAP_reversible / CPAP_persistent = pending_external "
              "(supply --external-axes post-freeze).")


if __name__ == "__main__":
    main()
