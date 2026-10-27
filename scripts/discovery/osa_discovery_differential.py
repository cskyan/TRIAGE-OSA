#!/usr/bin/env python3
"""Run locked OSA discovery differential-expression analyses.

Python 3.7+; numpy and pandas only.  The implementation uses ordinary least
squares followed by limma-style empirical-Bayes moderation of residual
variances.  It reads discovery datasets GSE75097 and GSE71356 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


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
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {} is not JSON serializable".format(
        type(value).__name__
    ))


def require_packages() -> Tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing numpy/pandas. For the current Python 3.7 environment use "
            "the already validated numpy 1.21.5 and pandas 1.1.5 environment."
        ) from exc
    return np, pd


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read JSON {}: {}".format(path, exc))


def write_tsv(frame: Any, path: Path, index: bool = False,
              index_label: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "sep": "\t", "index": index, "index_label": index_label,
        "compression": "gzip" if path.name.endswith(".gz") else None,
        "float_format": "%.10g", "na_rep": "NA",
    }
    try:
        frame.to_csv(str(path), lineterminator="\n", **kwargs)
    except TypeError:
        frame.to_csv(str(path), line_terminator="\n", **kwargs)


def digamma(x: float) -> float:
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)
    )


def trigamma(x: float) -> float:
    result = 0.0
    while x < 8.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + inv + 0.5 * inv2 + inv2 * inv * (
        1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0)
    )


def inverse_trigamma(y: float) -> float:
    if y <= 1e-8:
        return 1e8
    low, high = 1e-8, 1e8
    for _ in range(100):
        middle = math.sqrt(low * high)
        value = trigamma(middle)
        if value > y:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for m in range(1, maximum_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided(t_value: float, degrees_freedom: float) -> float:
    if not math.isfinite(t_value) or degrees_freedom <= 0.0:
        return float("nan")
    x = degrees_freedom / (degrees_freedom + t_value * t_value)
    value = regularized_beta(x, 0.5 * degrees_freedom, 0.5)
    return min(1.0, max(0.0, value))


def bh_adjust(p_values: Any, np: Any) -> Any:
    result = np.full(len(p_values), np.nan, dtype=float)
    finite = np.isfinite(p_values)
    values = np.asarray(p_values[finite], dtype=float)
    if values.size == 0:
        return result
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * float(values.size) / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty(values.size, dtype=float)
    restored[order] = adjusted
    result[finite] = restored
    return result


def estimate_prior_variance(s2: Any, residual_df: int, np: Any) -> Tuple[float, float, float, float, bool]:
    positive = np.asarray(s2[(np.isfinite(s2)) & (s2 > 0.0)], dtype=float)
    if positive.size < 10:
        raise RuntimeError("Too few positive residual variances for moderation")
    logs = np.log(positive)
    # Standard method-of-moments fit for a scaled-F prior.  Do not naively
    # winsorize log variances here: clipping can push their variance below the
    # theoretical sampling component and create an artificial infinite prior
    # df, especially in small participant-level sensitivity subsets.
    observed_variance = float(np.var(logs, ddof=1))
    residual_component = trigamma(0.5 * residual_df)
    excess_variance = observed_variance - residual_component
    boundary = bool(excess_variance <= 1e-8)
    remaining = max(excess_variance, 1e-8)
    prior_df = min(1e6, max(0.1, 2.0 * inverse_trigamma(remaining)))
    mean_log = float(np.mean(logs))
    log_prior = (
        mean_log - digamma(0.5 * residual_df) + math.log(0.5 * residual_df)
        + digamma(0.5 * prior_df) - math.log(0.5 * prior_df)
    )
    return (
        float(math.exp(log_prior)), float(prior_df), observed_variance,
        float(residual_component), boundary,
    )


def fit_model(expression: Any, sample_ids: Sequence[str], design: Any,
              contrast: Any, contrast_id: str, np: Any, pd: Any) -> Tuple[Any, Dict[str, Any]]:
    if list(design.index) != list(sample_ids):
        raise RuntimeError(contrast_id + " design/sample order mismatch")
    missing_columns = [sample for sample in sample_ids if sample not in expression.columns]
    if missing_columns:
        raise RuntimeError(contrast_id + " missing expression samples: " + ",".join(missing_columns))
    X = design.to_numpy(dtype=float)
    c = np.asarray(contrast, dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    residual_df = int(X.shape[0] - rank)
    if rank != X.shape[1]:
        raise RuntimeError(contrast_id + " design is rank deficient")
    if residual_df < 2:
        raise RuntimeError(contrast_id + " has fewer than two residual degrees of freedom")
    Y_all = expression.loc[:, list(sample_ids)].to_numpy(dtype=float)
    complete = np.all(np.isfinite(Y_all), axis=1)
    if int(np.sum(complete)) < 10:
        raise RuntimeError(contrast_id + " has too few complete proteins")
    Y = Y_all[complete, :]
    inverse_xtx = np.linalg.inv(np.dot(X.T, X))
    coefficients = np.dot(np.dot(inverse_xtx, X.T), Y.T)
    fitted = np.dot(X, coefficients)
    residuals = Y.T - fitted
    s2 = np.sum(residuals * residuals, axis=0) / float(residual_df)
    prior_s2, prior_df, log_s2_variance, theoretical_log_s2_variance, prior_boundary = (
        estimate_prior_variance(s2, residual_df, np)
    )
    posterior_s2 = (prior_df * prior_s2 + residual_df * s2) / (prior_df + residual_df)
    beta = np.dot(c, coefficients)
    unscaled_variance = float(np.dot(np.dot(c, inverse_xtx), c))
    unmoderated_standard_error = np.sqrt(s2 * unscaled_variance)
    unmoderated_t = beta / unmoderated_standard_error
    unmoderated_p = np.asarray([
        student_t_two_sided(float(value), float(residual_df))
        for value in unmoderated_t
    ], dtype=float)
    unmoderated_fdr = bh_adjust(unmoderated_p, np)
    standard_error = np.sqrt(posterior_s2 * unscaled_variance)
    t_values = beta / standard_error
    moderated_df = residual_df + prior_df
    p_values = np.asarray([
        student_t_two_sided(float(value), float(moderated_df)) for value in t_values
    ], dtype=float)
    fdr = bh_adjust(p_values, np)
    signed_score = np.sign(beta) * np.minimum(300.0, -np.log10(np.maximum(p_values, 1e-300)))
    valid_index = expression.index[complete]
    result = pd.DataFrame({
        "UniProt_accession": valid_index.astype(str),
        "mean_expression": np.mean(Y, axis=1),
        "beta": beta,
        "unmoderated_SE": unmoderated_standard_error,
        "unmoderated_t": unmoderated_t,
        "unmoderated_P_value": unmoderated_p,
        "unmoderated_FDR_BH": unmoderated_fdr,
        "moderated_SE": standard_error,
        "moderated_t": t_values,
        "P_value": p_values,
        "FDR_BH": fdr,
        "signed_score": signed_score,
        "residual_variance": s2,
        "posterior_variance": posterior_s2,
    })
    result = result.sort_values(
        ["P_value", "UniProt_accession"], ascending=[True, True]
    ).reset_index(drop=True)
    summary = {
        "contrast_id": contrast_id,
        "sample_count": int(X.shape[0]),
        "design_columns": list(design.columns),
        "design_rank": rank,
        "residual_df": residual_df,
        "prior_df": prior_df,
        "prior_variance": prior_s2,
        "log_residual_variance_observed": log_s2_variance,
        "log_residual_variance_sampling_component": theoretical_log_s2_variance,
        "prior_df_boundary": prior_boundary,
        "moderated_df": float(moderated_df),
        "proteins_tested": int(result.shape[0]),
        "proteins_incomplete_excluded": int(expression.shape[0] - result.shape[0]),
        "FDR_lt_0_05": int((result["FDR_BH"] < 0.05).sum()),
        "raw_P_lt_0_05": int((result["P_value"] < 0.05).sum()),
        "unmoderated_FDR_lt_0_05": int((result["unmoderated_FDR_BH"] < 0.05).sum()),
        "unmoderated_raw_P_lt_0_05": int((result["unmoderated_P_value"] < 0.05).sum()),
        "unmoderated_minimum_P": float(result["unmoderated_P_value"].min()),
        "minimum_P": float(result["P_value"].min()),
    }
    return result, summary


def batch_from_label(value: str) -> str:
    return re.sub(r"[A-Za-z]$", "", str(value).strip())


def add_batch_dummies(design: Any, metadata: Any, pd: Any) -> Any:
    batches = metadata["raw_array_label"].map(batch_from_label)
    dummies = pd.get_dummies(batches, prefix="batch", drop_first=True, dtype=float)
    dummies.index = metadata.index
    for column in dummies.columns:
        design[column] = dummies[column].astype(float)
    return design


def build_gse75097_design(metadata: Any, kind: str, adjusted: bool,
                          pd: Any) -> Tuple[Any, Any]:
    design = pd.DataFrame(index=metadata.index)
    design["intercept"] = 1.0
    if kind == "state":
        design["OSA_binary"] = metadata["study_group"].isin(["MSO", "VSO"]).astype(float)
        target = "OSA_binary"
    elif kind == "severe":
        design["VSO_binary"] = (metadata["study_group"] == "VSO").astype(float)
        target = "VSO_binary"
    elif kind == "AHI":
        design["AHI_per_10"] = metadata["AHI"].astype(float) / 10.0
        target = "AHI_per_10"
    elif kind == "ordinal":
        design["ordinal_group"] = metadata["study_group"].map(
            {"PS": 0.0, "MSO": 1.0, "VSO": 2.0}
        ).astype(float)
        target = "ordinal_group"
    elif kind == "VSOC":
        design["VSOC_binary"] = (metadata["study_group"] == "VSOC").astype(float)
        target = "VSOC_binary"
    else:
        raise RuntimeError("Unknown GSE75097 design kind: " + kind)
    if adjusted:
        design["age_per_10y"] = metadata["age"].astype(float) / 10.0
        design["female"] = (metadata["sex"].str.lower() == "female").astype(float)
        design = add_batch_dummies(design, metadata, pd)
    contrast = [1.0 if column == target else 0.0 for column in design.columns]
    return design, contrast


def build_gse71356_design(metadata: Any, pd: Any) -> Any:
    design = pd.DataFrame(index=metadata.index)
    design["intercept"] = 1.0
    subject_dummies = pd.get_dummies(
        metadata["subject_id"], prefix="subject", drop_first=True, dtype=float
    )
    subject_dummies.index = metadata.index
    for column in subject_dummies.columns:
        design[column] = subject_dummies[column].astype(float)
    design["IH"] = (metadata["exposure"] == "IH").astype(float)
    design["post_5h"] = (metadata["timepoint"] == "post_5h").astype(float)
    design["IH_x_post"] = design["IH"] * design["post_5h"]
    return design


def contrast_vector(columns: Sequence[str], weights: Dict[str, float]) -> List[float]:
    return [float(weights.get(column, 0.0)) for column in columns]


def load_expression(path: Path, pd: Any) -> Any:
    if not path.is_file():
        raise RuntimeError("Missing expression matrix: " + str(path))
    frame = pd.read_csv(
        str(path), sep="\t", compression="gzip", index_col=0
    )
    frame.index = frame.index.astype(str)
    if bool(frame.index.duplicated().any()):
        raise RuntimeError("Duplicate expression identifiers: " + str(path))
    return frame


def threshold_protein_expression(root: Path, preprocess_run_id: str,
                                 minimum_detected: int, np: Any, pd: Any) -> Any:
    qc = root / "results" / preprocess_run_id / "02_qc" / "discovery" / "GSE71356"
    mapping_dir = root / "results" / preprocess_run_id / "03_mapping"
    probes = load_expression(qc / "quantile_normalized_probe_matrix.tsv.gz", pd)
    fmap = pd.read_csv(
        str(mapping_dir / "feature_gene_protein_map.tsv.gz"), sep="\t",
        compression="gzip", dtype=str, keep_default_na=False,
    )
    fmap = fmap[fmap["dataset"] == "GSE71356"].copy()
    keep_column = "keep_detected_ge_{}".format(minimum_detected)
    if keep_column not in fmap.columns:
        raise RuntimeError("Missing threshold membership column: " + keep_column)
    mapped = fmap[(fmap[keep_column] == "yes") & (fmap["HGNC_ID"] != "")].copy()
    feature_to_gene = mapped.set_index("original_feature")["HGNC_ID"]
    features = [feature for feature in probes.index if feature in feature_to_gene.index]
    selected = probes.loc[features].copy()
    selected.insert(0, "HGNC_ID", feature_to_gene.loc[features].to_numpy())
    genes = selected.groupby("HGNC_ID", sort=True).median()
    gene_meta_rows = []
    for hgnc_id, group in mapped.groupby("HGNC_ID", sort=True):
        retained = group[group["original_feature"].isin(features)]
        if retained.empty:
            continue
        accessions = sorted(set(value for value in retained["UniProt_accession"] if value))
        if len(accessions) == 1:
            gene_meta_rows.append((hgnc_id, accessions[0]))
    gene_meta = pd.DataFrame(gene_meta_rows, columns=["HGNC_ID", "UniProt_accession"])
    counts = gene_meta["UniProt_accession"].value_counts()
    valid = gene_meta[~gene_meta["UniProt_accession"].isin(counts[counts > 1].index)].copy()
    valid = valid.sort_values("UniProt_accession")
    protein = genes.loc[valid["HGNC_ID"].tolist()].copy()
    protein.index = valid["UniProt_accession"].tolist()
    protein.index.name = "UniProt_accession"
    return protein


def attach_annotation(result: Any, annotation: Any, pd: Any) -> Any:
    return result.merge(annotation, on="UniProt_accession", how="left", validate="one_to_one")


def make_output_manifest(paths: Sequence[Path], root: Path, pd: Any) -> Any:
    rows = []
    for path in sorted(set(item.resolve() for item in paths if item.is_file())):
        rows.append({
            "path": relative_path(path, root),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preprocess-run-id", default="discovery_preprocess_v2")
    parser.add_argument(
        "--output-stage", default="04_discovery_v2",
        help="Versioned results subdirectory; default keeps the audited v1 output intact.",
    )
    parser.add_argument("--contrast-lock", type=Path, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    np, pd = require_packages()
    lock_path = args.contrast_lock or (root / "config" / "discovery_contrast_lock.json")
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock_path = lock_path.resolve()
    if not lock_path.is_file():
        raise RuntimeError("Missing discovery contrast lock: " + str(lock_path))
    load_json(lock_path)
    qc_dir = root / "results" / args.preprocess_run_id / "02_qc" / "discovery"
    mapping_dir = root / "results" / args.preprocess_run_id / "03_mapping"
    required = [
        qc_dir / "GSE75097" / "sample_mapping.tsv",
        qc_dir / "GSE71356" / "sample_mapping.tsv",
        mapping_dir / "GSE75097" / "protein_expression_all_samples.tsv.gz",
        mapping_dir / "GSE71356" / "protein_expression_all_samples.tsv.gz",
        mapping_dir / "feature_gene_protein_map.tsv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required inputs: " + "; ".join(missing))
    print("OSA discovery differential environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("pandas: {}".format(pd.__version__))
    print("Contrast lock: {}".format(lock_path))
    print("QC run: {}".format(qc_dir))
    print("Mapping run: {}".format(mapping_dir))
    if args.check_env:
        return 0

    if not re.match(r"^04_discovery[A-Za-z0-9_.-]*$", args.output_stage):
        raise RuntimeError(
            "--output-stage must be a simple versioned name beginning with 04_discovery"
        )
    output_dir = root / "results" / args.preprocess_run_id / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace: " + str(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to remove output outside project root")
        shutil.rmtree(str(output_dir))
    differential_dir = output_dir / "differential"
    design_dir = output_dir / "designs"
    differential_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    expressions = {}
    annotations = {}
    for dataset in ["GSE75097", "GSE71356"]:
        meta = pd.read_csv(
            str(qc_dir / dataset / "sample_mapping.tsv"), sep="\t",
            dtype=str, keep_default_na=False,
        ).set_index("sample_id", drop=False)
        freeze = pd.read_csv(
            str(qc_dir / dataset / "sample_freeze.tsv"), sep="\t",
            dtype=str, keep_default_na=False,
        ).set_index("sample_id", drop=False)
        if set(meta.index) != set(freeze.index):
            raise RuntimeError(dataset + " sample_mapping/sample_freeze membership mismatch")
        for column in freeze.columns:
            if column.startswith("include_"):
                meta["freeze_" + column] = freeze.loc[meta.index, column].to_numpy()
        metadata[dataset] = meta
        expressions[dataset] = load_expression(
            mapping_dir / dataset / "protein_expression_all_samples.tsv.gz", pd
        )
        annotation = pd.read_csv(
            str(mapping_dir / dataset / "protein_aggregation.tsv"), sep="\t",
            dtype=str, keep_default_na=False,
        )[["UniProt_accession", "HGNC_ID", "gene_symbol_current", "Entrez", "Ensembl_gene"]]
        annotations[dataset] = annotation

    summaries = []  # type: List[Dict[str, Any]]
    results = {}  # type: Dict[str, Any]
    output_paths = []  # type: List[Path]

    g750 = metadata["GSE75097"]
    specifications = [
        ("GSE75097_OSA_state", "state", ["PS", "MSO", "VSO"]),
        ("GSE75097_severe", "severe", ["PS", "VSO"]),
        ("GSE75097_AHI", "AHI", ["PS", "MSO", "VSO"]),
        ("GSE75097_ordinal", "ordinal", ["PS", "MSO", "VSO"]),
        ("GSE75097_VSOC", "VSOC", ["VSO", "VSOC"]),
    ]
    for contrast_id, kind, groups in specifications:
        sample_modes = [("primary", "yes")]
        if kind in ["state", "severe", "AHI", "ordinal"]:
            sample_modes.append(("all48", "yes"))
        for sample_mode, expected in sample_modes:
            include_column = (
                "freeze_include_primary" if sample_mode == "primary"
                else "freeze_include_sensitivity_all_48"
            )
            subset = g750[
                (g750[include_column] == expected) & g750["study_group"].isin(groups)
            ].copy()
            expected_counts = {
                ("state", "primary"): 32, ("state", "all48"): 34,
                ("severe", "primary"): 17, ("severe", "all48"): 18,
                ("AHI", "primary"): 32, ("AHI", "all48"): 34,
                ("ordinal", "primary"): 32, ("ordinal", "all48"): 34,
                ("VSOC", "primary"): 25,
            }
            expected_count = expected_counts[(kind, sample_mode)]
            if int(subset.shape[0]) != expected_count:
                raise RuntimeError(
                    "{} {} expected {} locked samples, observed {}".format(
                        contrast_id, sample_mode, expected_count, subset.shape[0]
                    )
                )
            if kind == "AHI" and bool((subset["AHI"] == "").any()):
                raise RuntimeError("AHI missing in locked analysis subset")
            for adjusted in [False, True]:
                suffix = ""
                if sample_mode == "all48":
                    suffix += "__all48"
                if adjusted:
                    suffix += "_adjusted" if suffix else "__adjusted"
                run_id = contrast_id + suffix
                design, contrast = build_gse75097_design(subset, kind, adjusted, pd)
                rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=float)))
                if rank != design.shape[1]:
                    summaries.append({
                        "contrast_id": run_id, "dataset": "GSE75097",
                        "status": "SKIPPED_RANK_DEFICIENT_SENSITIVITY",
                        "sample_count": int(design.shape[0]),
                        "design_columns": list(design.columns),
                    })
                    continue
                design_path = design_dir / (run_id + "__design.tsv")
                samples_path = design_dir / (run_id + "__samples.tsv")
                write_tsv(design.reset_index().rename(columns={"sample_id": "sample_id"}), design_path)
                write_tsv(subset.reset_index(drop=True), samples_path)
                result, summary = fit_model(
                    expressions["GSE75097"], subset.index.tolist(), design,
                    contrast, run_id, np, pd,
                )
                result = attach_annotation(result, annotations["GSE75097"], pd)
                result.insert(0, "dataset", "GSE75097")
                result.insert(1, "contrast_id", run_id)
                result_path = differential_dir / (run_id + ".tsv.gz")
                write_tsv(result, result_path)
                summary.update({
                    "dataset": "GSE75097", "status": "PASS",
                    "sample_mode": sample_mode, "adjusted": adjusted,
                    "result_path": relative_path(result_path, root),
                    "result_sha256": sha256_file(result_path),
                })
                summaries.append(summary)
                results[run_id] = result
                output_paths.extend([design_path, samples_path, result_path])
                print("[{}] PASS n={} proteins={} FDR<0.05={}".format(
                    run_id, summary["sample_count"], summary["proteins_tested"],
                    summary["FDR_lt_0_05"]
                ))

    g713 = metadata["GSE71356"]
    ih_contrasts = [
        ("GSE71356_IH_x_time", {"IH_x_post": 1.0}),
        ("GSE71356_IH_post_vs_pre", {"post_5h": 1.0, "IH_x_post": 1.0}),
        ("GSE71356_control_post_vs_pre", {"post_5h": 1.0}),
        ("GSE71356_IH_vs_control_post", {"IH": 1.0, "IH_x_post": 1.0}),
    ]
    for mode, excluded_subject in [("primary", ""), ("exclude_subject_403", "#403")]:
        subset = g713[g713["freeze_include_primary"] == "yes"].copy()
        if excluded_subject:
            subset = subset[subset["subject_id"] != excluded_subject].copy()
        expected_count = 16 if mode == "primary" else 12
        if int(subset.shape[0]) != expected_count:
            raise RuntimeError(
                "GSE71356 {} expected {} locked assays, observed {}".format(
                    mode, expected_count, subset.shape[0]
                )
            )
        design = build_gse71356_design(subset, pd)
        for base_id, weights in ih_contrasts:
            run_id = base_id if mode == "primary" else base_id + "__exclude_subject_403"
            contrast = contrast_vector(design.columns, weights)
            design_path = design_dir / (run_id + "__design.tsv")
            samples_path = design_dir / (run_id + "__samples.tsv")
            write_tsv(design.reset_index(), design_path)
            write_tsv(subset.reset_index(drop=True), samples_path)
            result, summary = fit_model(
                expressions["GSE71356"], subset.index.tolist(), design,
                contrast, run_id, np, pd,
            )
            result = attach_annotation(result, annotations["GSE71356"], pd)
            result.insert(0, "dataset", "GSE71356")
            result.insert(1, "contrast_id", run_id)
            result_path = differential_dir / (run_id + ".tsv.gz")
            write_tsv(result, result_path)
            summary.update({
                "dataset": "GSE71356", "status": "PASS", "sample_mode": mode,
                "adjusted": False, "result_path": relative_path(result_path, root),
                "result_sha256": sha256_file(result_path),
            })
            summaries.append(summary)
            results[run_id] = result
            output_paths.extend([design_path, samples_path, result_path])
            print("[{}] PASS n={} proteins={} FDR<0.05={}".format(
                run_id, summary["sample_count"], summary["proteins_tested"],
                summary["FDR_lt_0_05"]
            ))

    g713_primary = g713[g713["freeze_include_primary"] == "yes"].copy()
    primary_ih_design = build_gse71356_design(g713_primary, pd)
    primary_ih_contrast = contrast_vector(primary_ih_design.columns, {"IH_x_post": 1.0})
    for threshold in [3, 4, 6, 8]:
        run_id = "GSE71356_IH_x_time__detected_ge_{}".format(threshold)
        threshold_expression = threshold_protein_expression(
            root, args.preprocess_run_id, threshold, np, pd
        )
        result, summary = fit_model(
            threshold_expression, g713_primary.index.tolist(), primary_ih_design,
            primary_ih_contrast, run_id, np, pd,
        )
        result = attach_annotation(result, annotations["GSE71356"], pd)
        result.insert(0, "dataset", "GSE71356")
        result.insert(1, "contrast_id", run_id)
        result_path = differential_dir / (run_id + ".tsv.gz")
        write_tsv(result, result_path)
        summary.update({
            "dataset": "GSE71356", "status": "PASS",
            "sample_mode": "detection_threshold_sensitivity",
            "minimum_detected_samples": threshold, "adjusted": False,
            "result_path": relative_path(result_path, root),
            "result_sha256": sha256_file(result_path),
        })
        summaries.append(summary)
        results[run_id] = result
        output_paths.append(result_path)
        print("[{}] PASS n={} proteins={} FDR<0.05={}".format(
            run_id, summary["sample_count"], summary["proteins_tested"],
            summary["FDR_lt_0_05"]
        ))

    evidence_specs = [
        ("OSA", "GSE75097_OSA_state"),
        ("severe", "GSE75097_severe"),
        ("AHI", "GSE75097_AHI"),
        ("ordinal", "GSE75097_ordinal"),
        ("IH", "GSE71356_IH_x_time"),
    ]
    universe = sorted(set().union(*[
        set(results[run_id]["UniProt_accession"].astype(str))
        for _, run_id in evidence_specs
    ]))
    evidence = pd.DataFrame({"UniProt_accession": universe})
    annotation_union = pd.concat(
        [annotations["GSE75097"], annotations["GSE71356"]],
        ignore_index=True, sort=False,
    ).drop_duplicates("UniProt_accession", keep="first")
    evidence = evidence.merge(annotation_union, on="UniProt_accession", how="left")
    metrics = [
        "beta", "unmoderated_SE", "unmoderated_t", "unmoderated_P_value",
        "unmoderated_FDR_BH", "moderated_SE", "moderated_t", "P_value",
        "FDR_BH", "signed_score",
    ]
    for prefix, run_id in evidence_specs:
        source = results[run_id][["UniProt_accession"] + metrics].copy()
        source = source.rename(columns={metric: prefix + "_" + metric for metric in metrics})
        source[prefix + "_available"] = "yes"
        evidence = evidence.merge(source, on="UniProt_accession", how="left")
        evidence[prefix + "_available"] = evidence[prefix + "_available"].fillna("no")
    evidence_path = output_dir / "protein_context_evidence.tsv.gz"
    write_tsv(evidence, evidence_path)
    output_paths.append(evidence_path)

    summary_table = pd.DataFrame(summaries)
    summary_table_path = output_dir / "contrast_run_summary.tsv"
    write_tsv(summary_table, summary_table_path)
    output_paths.append(summary_table_path)
    run_summary = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preprocess_run_id": args.preprocess_run_id,
        "output_stage": args.output_stage,
        "contrast_lock": {
            "path": relative_path(lock_path, root), "sha256": sha256_file(lock_path),
        },
        "mapping_run_summary_sha256": sha256_file(mapping_dir / "mapping_run_summary.json"),
        "contrast_runs": summaries,
        "protein_context_evidence": {
            "path": relative_path(evidence_path, root),
            "sha256": sha256_file(evidence_path),
            "protein_rows": int(evidence.shape[0]),
        },
        "inference": {
            "method": "OLS plus limma-style empirical-Bayes variance moderation",
            "variance_prior_estimation": "standard un-winsorized log residual variance method of moments",
            "P_value": "two-sided moderated Student t",
            "FDR": "Benjamini-Hochberg within contrast",
            "unmoderated_diagnostics": "two-sided ordinary OLS t and Benjamini-Hochberg FDR are retained alongside moderated inference",
        },
        "method_amendment": {
            "previous_script_version": "1.0.0",
            "reason": "Audit found that naive 5/95 log-variance clipping caused prior_df=1000000 in several GSE71356 sensitivity analyses even though raw log-variance moments supported finite priors.",
            "selection_or_validation_outcomes_used": False,
        },
        "validation_data_read": False,
        "candidate_selection_performed": False,
        "candidate_ranking_performed": False,
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "next_gate": "Audit discovery contrasts and lock protein evidence integration before pair construction",
    }
    run_summary_path = output_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=True, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    output_paths.append(run_summary_path)
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(make_output_manifest(output_paths, root, pd), manifest_path)
    print("OSA discovery differential analysis: PASS")
    print("Protein context evidence: {}".format(evidence_path))
    print("Output: {}".format(output_dir))
    print("No validation dataset was read and no candidate ranking was performed.")
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
