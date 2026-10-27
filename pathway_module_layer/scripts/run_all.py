#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Run the discovery-only pathway-module layer end to end (WP1b -> WP2 -> WP3).\n\nPython 3.7, standard library only.\n\nThis is a thin, deterministic driver.  It runs the three P0 stages in order and\nstops on the first failure.  wp1a (sample-level ssGSEA) is optional and only run\nwith --with-activity.  External CPAP/adipose/scRNA axes are intentionally NOT\nwired here: they are added post-freeze via wp2's --external-axes, by hand, after\nthe discovery matrix hash is recorded.\n\nExample:\n  python3 run_all.py --root .       --gmt pathway_module_layer/genesets/PRIMARY.gmt --n-perm 1000\n"

from __future__ import print_function

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def run(cmd):
    print("\n$ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit("FAILED (rc=%d): %s" % (rc, " ".join(cmd)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--rank-metric", default="signed_score")
    ap.add_argument("--with-activity", action="store_true",
                    help="also run wp1a sample-level ssGSEA (slower).")
    args = ap.parse_args()

    py = sys.executable or "python3"

    run([py, os.path.join(HERE, "wp1b_pathway_multi_axis.py"),
         "--root", args.root, "--gmt", args.gmt,
         "--n-perm", str(args.n_perm), "--rank-metric", args.rank_metric])

    if args.with_activity:
        run([py, os.path.join(HERE, "wp1a_pathway_activity.py"),
             "--root", args.root, "--gmt", args.gmt, "--normalize"])

    run([py, os.path.join(HERE, "wp2_module_classify.py"), "--root", args.root])

    run([py, os.path.join(HERE, "wp3_pair_to_pathway.py"),
         "--root", args.root, "--gmt", args.gmt])

    print("\nAll discovery-only pathway-module stages completed.")
    print("Next (post-freeze, manual): record the wp1b matrix sha256, then add "
          "external CPAP/adipose/scRNA axes via wp2 --external-axes "
          "--discovery-frozen-sha256 <hash>.")


if __name__ == "__main__":
    main()
