#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WP3: map frozen candidate pairs onto pathway modules; find bridge pairs.

Python 3.7, standard library only.

Joins the frozen OSA interaction-priority ranking (pair_id, gene_A, gene_B,
S_OSA-PPI, p_PPI, r_OSA) to the WP2 pathway-module classification and the fixed
GMT, then labels every pair:
  - within_pathway : A and B share at least one disease module
  - cross_pathway_bridge : A and B each carry a module but the modules are
                           disjoint (candidate program-linking interaction)
  - unassigned : at least one protein has no assignable module

Emits the full mapping, the two focused subsets, and a compact GraphML of the
top bridge pairs for figure work.  Reads only frozen discovery-derived inputs.
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
DEFAULT_STAGE = "23_pair_to_pathway_v1"

# biological program tags inferred from pathway names (for bridge labelling)
PROGRAM_KEYWORDS = [
    ("hypoxia", ["HYPOXIA", "HIF", "OXYGEN"]),
    ("inflammation", ["INFLAMMAT", "TNF", "IL6", "INTERFERON", "NFKB",
                      "INNATE", "CYTOKINE"]),
    ("oxidative_stress", ["OXIDATIVE", "REACTIVE_OXYGEN", "ROS", "GLUTATHIONE"]),
    ("complement_coag", ["COMPLEMENT", "COAGULATION", "CLOTTING", "FIBRIN"]),
    ("metabolic", ["METABOL", "GLYCOLYSIS", "OXIDATIVE_PHOSPH", "FATTY_ACID",
                   "ADIPOGEN", "LIPID", "CHOLESTEROL", "INSULIN"]),
    ("vascular_endothelial", ["ANGIOGEN", "ENDOTHEL", "VASCUL", "VEGF"]),
    ("immune", ["IMMUNE", "TCELL", "T_CELL", "BCELL", "B_CELL", "LYMPHOCYTE",
                "ALLOGRAFT", "IL2_STAT5"]),
]

PRIORITY_BRIDGES = set([
    frozenset(["hypoxia", "inflammation"]),
    frozenset(["hypoxia", "oxidative_stress"]),
    frozenset(["inflammation", "complement_coag"]),
    frozenset(["metabolic", "vascular_endothelial"]),
    frozenset(["immune", "metabolic"]),
])


def program_of(pathway_name):
    up = pathway_name.upper()
    for prog, keys in PROGRAM_KEYWORDS:
        for k in keys:
            if k in up:
                return prog
    return "other"


def load_gene_to_pathways(gmt_sets, id_space):
    g2p = {}
    for gs in gmt_sets:
        for g in gs["genes"]:
            g2p.setdefault(g, set()).add(gs["name"])
    return g2p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--classification", default=None,
                    help="wp2 pathway_classification.tsv (default: stage 22).")
    ap.add_argument("--pairs", default=None,
                    help="frozen ranking with gene_A/gene_B/score "
                         "(default: stage 09 top_primary_5pct.tsv).")
    ap.add_argument("--stage", default=DEFAULT_STAGE)
    ap.add_argument("--assignable-levels", default="A,B,C",
                    help="evidence levels whose pathways count for mapping.")
    ap.add_argument("--top-bridges-graphml", type=int, default=100)
    ap.add_argument("--id-space", default="auto",
                    choices=["auto", "symbol", "entrez"])
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cls_path = args.classification or os.path.join(
        root, "results/discovery_preprocess_v2/22_pathway_modules_v1",
        "pathway_classification.tsv")
    pairs_path = args.pairs or os.path.join(
        root, "results/discovery_preprocess_v2/09_pair_context_ranking_v1",
        "top_primary_5pct.tsv")
    for p in (cls_path, pairs_path):
        if not os.path.exists(p):
            sys.exit("ERROR: required input missing: %s" % p)

    sets = C.parse_gmt(args.gmt)
    id_space = args.id_space
    if id_space == "auto":
        id_space = C.detect_gmt_id_space(sets)
    g2p = load_gene_to_pathways(sets, id_space)

    # pathway -> (modules set, program, level)
    cls_rows, _ = C.read_tsv(cls_path)
    levels_ok = set(x.strip() for x in args.assignable_levels.split(","))
    pw_modules = {}
    pw_program = {}
    for r in cls_rows:
        pw = r["pathway"]
        pw_program[pw] = program_of(pw)
        mods = set()
        if r.get("evidence_level", "") in levels_ok:
            m = r.get("modules", "")
            if m and m != "none":
                mods = set(m.split(";"))
        pw_modules[pw] = mods

    def gene_modules(gene):
        mods = set()
        progs = set()
        for pw in g2p.get(gene.upper(), ()):  # pathways containing this gene
            if pw in pw_modules and pw_modules[pw]:
                mods |= pw_modules[pw]
                progs.add(pw_program.get(pw, "other"))
        return mods, progs

    pair_rows, pfields = C.read_tsv(pairs_path)
    # tolerate different score column names
    score_col = None
    for c in ["OSA_interaction_priority_score", "S_OSA_PPI", "S_OSA-PPI"]:
        if c in pfields:
            score_col = c
            break

    out_fields = ["pair_id", "gene_A", "gene_B", "class",
                  "modules_A", "modules_B", "shared_modules",
                  "program_A", "program_B", "bridge_type", "priority_bridge",
                  "OSA_interaction_priority_score", "p_PPI", "r_OSA"]
    mapped = []
    n_within = n_bridge = n_unassigned = 0
    for r in pair_rows:
        gA = (r.get("gene_A") or "").strip()
        gB = (r.get("gene_B") or "").strip()
        modA, progA = gene_modules(gA)
        modB, progB = gene_modules(gB)
        if modA and modB:
            shared = modA & modB
            if shared:
                cls = "within_pathway"
                n_within += 1
            else:
                cls = "cross_pathway_bridge"
                n_bridge += 1
        else:
            cls = "unassigned"
            n_unassigned += 1
            shared = set()
        pa = program_of_first(progA)
        pb = program_of_first(progB)
        bridge_type = ""
        priority = "no"
        if cls == "cross_pathway_bridge":
            bridge_type = "%s<->%s" % tuple(sorted([pa, pb]))
            if frozenset([pa, pb]) in PRIORITY_BRIDGES:
                priority = "yes"
        mapped.append({
            "pair_id": r.get("pair_id", "%s__%s" % (gA, gB)),
            "gene_A": gA, "gene_B": gB, "class": cls,
            "modules_A": ";".join(sorted(modA)) or "none",
            "modules_B": ";".join(sorted(modB)) or "none",
            "shared_modules": ";".join(sorted(shared)) or "",
            "program_A": pa, "program_B": pb,
            "bridge_type": bridge_type, "priority_bridge": priority,
            "OSA_interaction_priority_score": r.get(score_col, "") if score_col else "",
            "p_PPI": r.get("p_PPI", ""), "r_OSA": r.get("r_OSA", ""),
        })

    def score_key(row):
        v = C._to_float_safe(row.get("OSA_interaction_priority_score", ""))
        return -(v if v is not None else -1.0)

    mapped.sort(key=score_key)
    out_dir = os.path.join(root, "results", "discovery_preprocess_v2", args.stage)
    os.makedirs(out_dir, exist_ok=True)
    C.write_tsv(os.path.join(out_dir, "pair_to_pathway_mapping.tsv"),
                mapped, out_fields)
    within = [m for m in mapped if m["class"] == "within_pathway"]
    bridges = [m for m in mapped if m["class"] == "cross_pathway_bridge"]
    C.write_tsv(os.path.join(out_dir, "within_pathway_pairs.tsv"),
                within, out_fields)
    C.write_tsv(os.path.join(out_dir, "cross_pathway_bridge_pairs.tsv"),
                bridges, out_fields)
    write_graphml(os.path.join(out_dir, "pathway_pair_network.graphml"),
                  bridges[:args.top_bridges_graphml])

    run_summary = {
        "script": "wp3_pair_to_pathway.py", "script_version": SCRIPT_VERSION,
        "shared_version": C.SHARED_VERSION, "created_at_utc": C.utc_now(),
        "stage": args.stage,
        "inputs": {
            "gmt": {"path": C.relative_path(args.gmt, root),
                    "sha256": C.sha256_file(args.gmt), "id_space": id_space},
            "classification": {"path": C.relative_path(cls_path, root),
                               "sha256": C.sha256_file(cls_path)},
            "pairs": {"path": C.relative_path(pairs_path, root),
                      "sha256": C.sha256_file(pairs_path),
                      "score_column": score_col}},
        "counts": {"within_pathway": n_within, "cross_pathway_bridge": n_bridge,
                   "unassigned": n_unassigned, "total": len(mapped)},
        "priority_bridges": sum(1 for m in bridges if m["priority_bridge"] == "yes"),
    }
    C.atomic_json(os.path.join(out_dir, "run_summary.json"), run_summary)
    print("OK  mapped %d pairs | within=%d bridge=%d unassigned=%d | priority_bridges=%d"
          % (len(mapped), n_within, n_bridge, n_unassigned,
             sum(1 for m in bridges if m["priority_bridge"] == "yes")))


def program_of_first(progs):
    if not progs:
        return "other"
    # deterministic: prefer a non-'other' program, else 'other'
    named = sorted(p for p in progs if p != "other")
    return named[0] if named else "other"


def write_graphml(path, bridge_rows):
    nodes = {}
    for r in bridge_rows:
        for g, prog, mod in [(r["gene_A"], r["program_A"], r["modules_A"]),
                             (r["gene_B"], r["program_B"], r["modules_B"])]:
            if g not in nodes:
                nodes[g] = (prog, mod)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
             '  <key id="program" for="node" attr.name="program" attr.type="string"/>',
             '  <key id="modules" for="node" attr.name="modules" attr.type="string"/>',
             '  <key id="score" for="edge" attr.name="score" attr.type="double"/>',
             '  <key id="bridge" for="edge" attr.name="bridge_type" attr.type="string"/>',
             '  <graph edgedefault="undirected">']
    for g, (prog, mod) in sorted(nodes.items()):
        lines.append('    <node id="%s"><data key="program">%s</data>'
                     '<data key="modules">%s</data></node>'
                     % (xml_escape(g), xml_escape(prog), xml_escape(mod)))
    for i, r in enumerate(bridge_rows):
        sc = r.get("OSA_interaction_priority_score", "") or "0"
        lines.append('    <edge id="e%d" source="%s" target="%s">'
                     '<data key="score">%s</data>'
                     '<data key="bridge_type">%s</data></edge>'
                     % (i, xml_escape(r["gene_A"]), xml_escape(r["gene_B"]),
                        xml_escape(str(sc)), xml_escape(r["bridge_type"])))
    lines += ['  </graph>', '</graphml>', '']
    with open(str(path), "w", encoding="utf-8") as h:
        h.write("\n".join(lines))


def xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


if __name__ == "__main__":
    main()
