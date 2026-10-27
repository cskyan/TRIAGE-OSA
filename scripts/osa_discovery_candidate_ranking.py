#!/usr/bin/env python3
"""Freeze discovery-only OSA protein candidates and preliminary controls.

Python 3.7+; numpy and pandas only.  This script consumes the audited Phase 4A
v2 discovery evidence and never reads validation, known-PPI, pathway, or model
outputs.  Candidate proteins are evidence-prioritized, not significant DE hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError("Object of type {} is not JSON serializable".format(
        type(value).__name__
    ))


def require_packages() -> Tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("This script requires numpy and pandas") from exc
    return np, pd


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read JSON {}: {}".format(path, exc))


def write_tsv(frame: Any, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "sep": "\t", "index": index,
        "compression": "gzip" if path.name.endswith(".gz") else None,
        "float_format": "%.10g", "na_rep": "NA",
    }
    try:
        frame.to_csv(str(path), lineterminator="\n", **kwargs)
    except TypeError:
        frame.to_csv(str(path), line_terminator="\n", **kwargs)


def midrank_strength(p_values: Any, pd: Any, np: Any) -> Any:
    p = pd.to_numeric(p_values, errors="coerce")
    if bool(p.isna().any()) or bool((p < 0.0).any()) or bool((p > 1.0).any()):
        raise RuntimeError("Invalid P values in a registered primary evidence source")
    evidence = -np.log10(np.maximum(p.to_numpy(dtype=float), 1e-300))
    ranks = pd.Series(evidence).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 0.5) / float(len(ranks))


def sign_label(values: Any, np: Any) -> Any:
    numeric = values.to_numpy(dtype=float)
    return np.where(numeric > 0.0, "positive", np.where(numeric < 0.0, "negative", "zero"))


def sign_agreement(a: Any, b: Any, np: Any) -> Any:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    return np.where((av == 0.0) | (bv == 0.0), "indeterminate",
                    np.where(av * bv > 0.0, "yes", "no"))


def load_result(stage: Path, contrast_id: str, pd: Any) -> Any:
    path = stage / "differential" / (contrast_id + ".tsv.gz")
    if not path.is_file():
        raise RuntimeError("Missing registered contrast result: " + str(path))
    frame = pd.read_csv(str(path), sep="\t", compression="gzip")
    if bool(frame["UniProt_accession"].duplicated().any()):
        raise RuntimeError("Duplicate UniProt in " + contrast_id)
    return frame


def add_stability_audit(ranking: Any, stage: Path, pd: Any, np: Any) -> Any:
    specifications = [
        ("OSA", "GSE75097_OSA_state", [
            ("adjusted", "GSE75097_OSA_state__adjusted"),
            ("all48", "GSE75097_OSA_state__all48"),
        ]),
        ("severe", "GSE75097_severe", [
            ("adjusted", "GSE75097_severe__adjusted"),
            ("all48", "GSE75097_severe__all48"),
        ]),
        ("AHI", "GSE75097_AHI", [
            ("adjusted", "GSE75097_AHI__adjusted"),
            ("all48", "GSE75097_AHI__all48"),
        ]),
    ]
    for prefix, primary_id, sensitivities in specifications:
        primary = load_result(stage, primary_id, pd)[["UniProt_accession", "beta"]]
        primary = primary.rename(columns={"beta": prefix + "_primary_beta_audit"})
        ranking = ranking.merge(primary, on="UniProt_accession", how="left", validate="one_to_one")
        for suffix, sensitivity_id in sensitivities:
            sensitivity = load_result(stage, sensitivity_id, pd)[["UniProt_accession", "beta"]]
            sensitivity = sensitivity.rename(columns={"beta": prefix + "_" + suffix + "_beta"})
            ranking = ranking.merge(
                sensitivity, on="UniProt_accession", how="left", validate="one_to_one"
            )
            ranking[prefix + "_sign_agreement_" + suffix] = sign_agreement(
                ranking[prefix + "_primary_beta_audit"],
                ranking[prefix + "_" + suffix + "_beta"], np,
            )
        ranking = ranking.drop(columns=[prefix + "_primary_beta_audit"])
    ih_sensitivity = load_result(
        stage, "GSE71356_IH_x_time__exclude_subject_403", pd
    )[["UniProt_accession", "beta"]].rename(columns={"beta": "IH_exclude403_beta"})
    ranking = ranking.merge(
        ih_sensitivity, on="UniProt_accession", how="left", validate="one_to_one"
    )
    ranking["IH_sign_agreement_exclude403"] = "not_available"
    mask = ranking["IH_beta"].notna() & ranking["IH_exclude403_beta"].notna()
    ranking.loc[mask, "IH_sign_agreement_exclude403"] = sign_agreement(
        ranking.loc[mask, "IH_beta"], ranking.loc[mask, "IH_exclude403_beta"], np
    )
    return ranking


def protein_covariates(root: Path, preprocess_run_id: str, pd: Any, np: Any) -> Any:
    feature_path = root / "results" / preprocess_run_id / "03_mapping" / "feature_gene_protein_map.tsv.gz"
    fmap = pd.read_csv(
        str(feature_path), sep="\t", compression="gzip", dtype=str,
        keep_default_na=False,
    )
    fmap = fmap[
        (fmap["dataset"] == "GSE75097")
        & (fmap["retained_primary_probe"] == "yes")
        & (fmap["UniProt_accession"] != "")
    ].copy()
    fmap["detected_fraction_numeric"] = pd.to_numeric(
        fmap["detected_fraction"], errors="coerce"
    )
    fmap["protein_length_numeric"] = pd.to_numeric(
        fmap["UniProt_length"], errors="coerce"
    )
    conflicting_length = fmap.groupby("UniProt_accession")["protein_length_numeric"].nunique()
    if bool((conflicting_length > 1).any()):
        raise RuntimeError("Conflicting reviewed canonical protein lengths in feature map")
    covariates = fmap.groupby("UniProt_accession", sort=True).agg({
        "detected_fraction_numeric": "median",
        "protein_length_numeric": "first",
        "original_feature": "nunique",
    }).reset_index().rename(columns={
        "detected_fraction_numeric": "protein_detection_fraction",
        "protein_length_numeric": "UniProt_length",
        "original_feature": "retained_probe_count_covariate",
    })
    covariates["log2_UniProt_length"] = np.log2(
        covariates["UniProt_length"].to_numpy(dtype=float)
    )
    return covariates


def standardized_mean_difference(a: Any, b: Any, np: Any) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    pooled = math.sqrt(0.5 * (float(np.var(av, ddof=1)) + float(np.var(bv, ddof=1))))
    return 0.0 if pooled == 0.0 else float((np.mean(av) - np.mean(bv)) / pooled)


def match_controls(ranking: Any, primary: Any, np: Any, pd: Any) -> Tuple[Any, Any]:
    covariates = [
        "GSE75097_mean_expression", "protein_detection_fraction",
        "log2_UniProt_length",
    ]
    candidates = primary.copy().sort_values("discovery_rank")
    pool = ranking[ranking["discovery_rank"] > 500].copy()
    if bool(candidates[covariates].isna().any().any()):
        raise RuntimeError("A primary candidate lacks a locked matching covariate")
    pool = pool.dropna(subset=covariates).copy()
    if pool.shape[0] < candidates.shape[0]:
        raise RuntimeError("Too few complete non-candidate proteins for matching")
    reference = pd.concat([candidates[covariates], pool[covariates]], ignore_index=True)
    means = reference.mean(axis=0)
    standard_deviations = reference.std(axis=0, ddof=1).replace(0.0, 1.0)
    candidate_z = ((candidates[covariates] - means) / standard_deviations).to_numpy(dtype=float)
    pool_z = ((pool[covariates] - means) / standard_deviations).to_numpy(dtype=float)
    available = np.ones(pool.shape[0], dtype=bool)
    rows = []  # type: List[Dict[str, Any]]
    pool_records = pool.reset_index(drop=True)
    for number in range(candidates.shape[0]):
        distances = np.sum((pool_z - candidate_z[number, :]) ** 2, axis=1)
        distances[~available] = np.inf
        minimum = float(np.min(distances))
        tied = np.where(np.isclose(distances, minimum, rtol=0.0, atol=1e-15))[0]
        if tied.size > 1:
            chosen = int(min(
                tied, key=lambda item: str(
                    pool_records.iloc[int(item)]["UniProt_accession"]
                )
            ))
        else:
            chosen = int(tied[0])
        available[chosen] = False
        candidate = candidates.iloc[number]
        control = pool_records.iloc[chosen]
        row = {
            "candidate_rank": int(candidate["discovery_rank"]),
            "candidate_UniProt": candidate["UniProt_accession"],
            "candidate_gene": candidate["gene_symbol_current"],
            "control_UniProt": control["UniProt_accession"],
            "control_gene": control["gene_symbol_current"],
            "control_discovery_rank": int(control["discovery_rank"]),
            "squared_standardized_distance": minimum,
            "matching_status": "preliminary_phase4_match",
        }
        for covariate in covariates:
            row["candidate_" + covariate] = float(candidate[covariate])
            row["control_" + covariate] = float(control[covariate])
        rows.append(row)
    matches = pd.DataFrame(rows)
    balance_rows = []
    for covariate in covariates:
        balance_rows.append({
            "covariate": covariate,
            "candidate_mean": float(matches["candidate_" + covariate].mean()),
            "control_mean": float(matches["control_" + covariate].mean()),
            "standardized_mean_difference": standardized_mean_difference(
                matches["candidate_" + covariate],
                matches["control_" + covariate], np,
            ),
        })
    return matches, pd.DataFrame(balance_rows)


def output_manifest(paths: Sequence[Path], root: Path, pd: Any) -> Any:
    rows = []
    for path in sorted(set(item.resolve() for item in paths if item.is_file())):
        rows.append({
            "path": relative_path(path, root), "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preprocess-run-id", default="discovery_preprocess_v2")
    parser.add_argument("--input-stage", default="04_discovery_v2")
    parser.add_argument("--output-stage", default="04_candidates_v1")
    parser.add_argument("--ranking-lock", type=Path, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    np, pd = require_packages()
    for value, label in [(args.input_stage, "input"), (args.output_stage, "output")]:
        if not re.match(r"^04_[A-Za-z0-9_.-]+$", value):
            raise RuntimeError("Invalid {} stage name: {}".format(label, value))
    stage = root / "results" / args.preprocess_run_id / args.input_stage
    lock_path = args.ranking_lock or (root / "config" / "discovery_candidate_ranking_lock.json")
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock_path = lock_path.resolve()
    required = [
        lock_path, stage / "run_summary.json", stage / "contrast_run_summary.tsv",
        stage / "protein_context_evidence.tsv.gz",
        root / "results" / args.preprocess_run_id / "03_mapping" / "feature_gene_protein_map.tsv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required inputs: " + "; ".join(missing))
    lock = load_json(lock_path)
    run = load_json(stage / "run_summary.json")
    contrasts = pd.read_csv(str(stage / "contrast_run_summary.tsv"), sep="\t")
    if set(contrasts["status"].astype(str)) != {"PASS"}:
        raise RuntimeError("Not all discovery v2 contrasts passed")
    if bool(contrasts["prior_df_boundary"].astype(str).str.lower().eq("true").any()):
        raise RuntimeError("A discovery v2 contrast has a prior-df boundary")
    if bool(run.get("validation_data_read")) or bool(run.get("candidate_ranking_performed")):
        raise RuntimeError("Discovery input violated validation/ranking isolation")
    if sha256_file(root / "config" / "discovery_contrast_lock.json") != run["contrast_lock"]["sha256"]:
        raise RuntimeError("Discovery contrast lock hash no longer matches v2 run")
    print("OSA discovery candidate ranking environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("pandas: {}".format(pd.__version__))
    print("Discovery input: {}".format(stage))
    print("Ranking lock: {}".format(lock_path))
    if args.check_env:
        return 0

    output_dir = root / "results" / args.preprocess_run_id / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace: " + str(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(
        str(stage / "protein_context_evidence.tsv.gz"), sep="\t",
        compression="gzip",
    )
    eligible = evidence[
        evidence["OSA_available"].eq("yes")
        & evidence["severe_available"].eq("yes")
        & evidence["AHI_available"].eq("yes")
    ].copy().reset_index(drop=True)
    if eligible.shape[0] != 11260:
        raise RuntimeError("Expected 11260 adult-eligible proteins, observed {}".format(
            eligible.shape[0]
        ))
    for prefix in ["OSA", "severe", "AHI"]:
        eligible[prefix + "_strength"] = midrank_strength(
            eligible[prefix + "_P_value"], pd, np
        )
        eligible[prefix + "_beta_sign"] = sign_label(eligible[prefix + "_beta"], np)
    ih_source = evidence[evidence["IH_available"].eq("yes")][
        ["UniProt_accession", "IH_P_value"]
    ].copy()
    ih_source["IH_strength"] = midrank_strength(
        ih_source["IH_P_value"], pd, np
    )
    eligible = eligible.drop(columns=["IH_strength"], errors="ignore").merge(
        ih_source[["UniProt_accession", "IH_strength"]],
        on="UniProt_accession", how="left", validate="one_to_one",
    )
    ih_available = eligible["IH_available"].eq("yes")
    eligible["IH_strength_or_neutral"] = eligible["IH_strength"].fillna(0.5)
    eligible["IH_beta_sign"] = "not_available"
    eligible.loc[ih_available, "IH_beta_sign"] = sign_label(
        eligible.loc[ih_available, "IH_beta"], np
    )
    eligible["adult_OSA_axis"] = (
        0.50 * eligible["OSA_strength"]
        + 0.25 * eligible["severe_strength"]
        + 0.25 * eligible["AHI_strength"]
    )
    eligible["discovery_score"] = (
        0.75 * eligible["adult_OSA_axis"]
        + 0.25 * eligible["IH_strength_or_neutral"]
    )
    adult_signs = np.column_stack([
        np.sign(eligible["OSA_beta"].to_numpy(dtype=float)),
        np.sign(eligible["severe_beta"].to_numpy(dtype=float)),
        np.sign(eligible["AHI_beta"].to_numpy(dtype=float)),
    ])
    eligible["adult_three_way_direction_agreement"] = np.where(
        np.all(adult_signs > 0.0, axis=1) | np.all(adult_signs < 0.0, axis=1),
        "yes", "no",
    )
    eligible["OSA_IH_direction_agreement_when_available"] = "not_available"
    eligible.loc[ih_available, "OSA_IH_direction_agreement_when_available"] = sign_agreement(
        eligible.loc[ih_available, "OSA_beta"], eligible.loc[ih_available, "IH_beta"], np
    )
    state_result = load_result(stage, "GSE75097_OSA_state", pd)[
        ["UniProt_accession", "mean_expression"]
    ].rename(columns={"mean_expression": "GSE75097_mean_expression"})
    eligible = eligible.merge(
        state_result, on="UniProt_accession", how="left", validate="one_to_one"
    )
    eligible = eligible.merge(
        protein_covariates(root, args.preprocess_run_id, pd, np),
        on="UniProt_accession", how="left", validate="one_to_one",
    )
    eligible = add_stability_audit(eligible, stage, pd, np)
    eligible = eligible.sort_values(
        ["discovery_score", "adult_OSA_axis", "OSA_strength", "UniProt_accession"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    eligible.insert(0, "discovery_rank", np.arange(1, eligible.shape[0] + 1))
    eligible.insert(1, "selection_label", "discovery_evidence_prioritized_not_FDR_significant")
    eligible["candidate_primary_250"] = np.where(eligible["discovery_rank"] <= 250, "yes", "no")
    eligible["candidate_sensitivity_500"] = np.where(eligible["discovery_rank"] <= 500, "yes", "no")
    eligible["model_asset_matching_status"] = "pending_L1_L2_L3_and_network_asset_join"

    primary = eligible.iloc[:250].copy()
    sensitivity = eligible.iloc[:500].copy()
    if not set(primary["UniProt_accession"]).issubset(set(sensitivity["UniProt_accession"])):
        raise RuntimeError("Primary candidate set is not nested in sensitivity set")
    matches, balance = match_controls(eligible, primary, np, pd)
    if matches.shape[0] != 250 or bool(matches["control_UniProt"].duplicated().any()):
        raise RuntimeError("Preliminary control matching did not produce 250 unique controls")

    ranking_path = output_dir / "protein_ranking_all.tsv.gz"
    primary_path = output_dir / "candidate_proteins_primary_250.tsv"
    sensitivity_path = output_dir / "candidate_proteins_sensitivity_500.tsv"
    controls_path = output_dir / "matched_control_proteins_preliminary.tsv"
    balance_path = output_dir / "preliminary_matching_balance.tsv"
    write_tsv(eligible, ranking_path)
    write_tsv(primary, primary_path)
    write_tsv(sensitivity, sensitivity_path)
    write_tsv(matches, controls_path)
    write_tsv(balance, balance_path)
    output_paths = [ranking_path, primary_path, sensitivity_path, controls_path, balance_path]

    summary = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preprocess_run_id": args.preprocess_run_id,
        "input_stage": {
            "path": relative_path(stage, root),
            "run_summary_sha256": sha256_file(stage / "run_summary.json"),
            "protein_context_evidence_sha256": sha256_file(
                stage / "protein_context_evidence.tsv.gz"
            ),
        },
        "ranking_lock": {
            "path": relative_path(lock_path, root), "sha256": sha256_file(lock_path),
        },
        "eligible_proteins": int(eligible.shape[0]),
        "eligible_with_IH": int(eligible["IH_available"].eq("yes").sum()),
        "primary_candidates": 250,
        "primary_with_IH": int(primary["IH_available"].eq("yes").sum()),
        "sensitivity_candidates": 500,
        "sensitivity_with_IH": int(sensitivity["IH_available"].eq("yes").sum()),
        "primary_score_range": [
            float(primary["discovery_score"].min()),
            float(primary["discovery_score"].max()),
        ],
        "candidate_interpretation": "discovery evidence-prioritized; not FDR-significant DE proteins",
        "primary_source_FDR_significant_counts": {
            "OSA": int((eligible["OSA_FDR_BH"] < 0.05).sum()),
            "severe": int((eligible["severe_FDR_BH"] < 0.05).sum()),
            "AHI": int((eligible["AHI_FDR_BH"] < 0.05).sum()),
            "IH_among_adult_eligible": int((eligible["IH_FDR_BH"] < 0.05).fillna(False).sum()),
        },
        "preliminary_controls": {
            "count": int(matches.shape[0]),
            "unique": int(matches["control_UniProt"].nunique()),
            "maximum_absolute_SMD": float(balance["standardized_mean_difference"].abs().max()),
            "status": "preliminary_until_L1_L2_L3_and_network_assets_are_joined",
        },
        "validation_data_read": False,
        "known_PPI_or_pathway_read": False,
        "model_output_read": False,
        "candidate_ranking_performed": True,
        "candidate_files_frozen": True,
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "next_gate": "Audit candidate composition and join frozen L1-L2-L3/network availability before final matched controls and pair construction",
    }
    summary_path = output_dir / "candidate_rank_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    output_paths.append(summary_path)
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(output_manifest(output_paths, root, pd), manifest_path)
    print("OSA discovery candidate ranking: PASS")
    print("Eligible adult OSA proteins: {}".format(eligible.shape[0]))
    print("Primary candidates: 250 (IH available: {})".format(
        int(primary["IH_available"].eq("yes").sum())
    ))
    print("Sensitivity candidates: 500 (IH available: {})".format(
        int(sensitivity["IH_available"].eq("yes").sum())
    ))
    print("Preliminary matched controls: 250; max |SMD|={:.4f}".format(
        float(balance["standardized_mean_difference"].abs().max())
    ))
    print("Output: {}".format(output_dir))
    print("No validation, known-PPI, pathway, or model output was read.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
