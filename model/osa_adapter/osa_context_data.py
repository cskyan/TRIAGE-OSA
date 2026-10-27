#!/usr/bin/env python3
"""Discovery-only data construction for the specialized TRIAGE-OSA model."""

from __future__ import print_function

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from triage_osa_model import (
    BASE_FEATURE_COLUMNS,
    CONTEXT_FEATURE_COLUMNS,
    INPUT_FEATURE_COLUMNS,
)


DATA_SCHEMA_VERSION = "4.0.0"


def stable_bucket(value, buckets, seed=20260716):
    payload = "{}|{}".format(int(seed), value).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % int(buckets)


def stratified_sample_folds(metadata, folds=5, seed=20260716):
    rows = []
    for group, part in metadata.groupby("study_group", sort=True):
        identifiers = sorted(
            part["sample_id"].astype(str).tolist(),
            key=lambda value: hashlib.sha256(
                "{}|{}|{}".format(seed, group, value).encode("utf-8")
            ).hexdigest(),
        )
        for index, sample_id in enumerate(identifiers):
            rows.append(
                {
                    "dataset": "GSE75097",
                    "unit_id": sample_id,
                    "stratum": str(group),
                    "fold": int(index % int(folds)),
                }
            )
    return pd.DataFrame(rows)


def subject_folds(metadata, seed=20260716):
    subjects = sorted(
        metadata["subject_id"].astype(str).unique().tolist(),
        key=lambda value: hashlib.sha256(
            "{}|GSE71356|{}".format(seed, value).encode("utf-8")
        ).hexdigest(),
    )
    rows = []
    for index, subject_id in enumerate(subjects):
        rows.append(
            {
                "dataset": "GSE71356",
                "unit_id": subject_id,
                "stratum": "crossover_subject",
                "fold": int(index),
            }
        )
    return pd.DataFrame(rows)


def load_discovery_inputs(root, proteins):
    root = Path(root).resolve()
    mapping = root / "results" / "discovery_preprocess_v2" / "03_mapping"
    paths = {
        "GSE75097_expression": mapping / "GSE75097" / "protein_expression_primary_samples.tsv.gz",
        "GSE71356_expression": mapping / "GSE71356" / "protein_expression_primary_samples.tsv.gz",
        "discovery_patient_information": root / "metadata" / "discovery_patient_information.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing discovery-only input files: {}".format(missing))

    expression = {}
    for dataset in ("GSE75097", "GSE71356"):
        frame = pd.read_csv(paths[dataset + "_expression"], sep="\t")
        frame["UniProt_accession"] = frame["UniProt_accession"].astype(str)
        frame = frame.drop_duplicates("UniProt_accession", keep="first")
        frame = frame.set_index("UniProt_accession")
        expression[dataset] = frame.reindex(list(proteins)).apply(
            pd.to_numeric, errors="coerce"
        )

    information = pd.read_csv(paths["discovery_patient_information"], sep="\t")
    required_information_columns = {
        "dataset", "sample_id", "subject_id", "training_allowed",
        "clinical_information_training_allowed", "model_role", "sex",
        "age_years", "BMI_kg_m2", "AHI_events_h", "hypertension", "EDS",
        "study_group", "exposure", "timepoint",
        "cohort_summary_used_as_individual_value",
        "clinical_outcome_label_available",
    }
    missing_information_columns = sorted(
        required_information_columns - set(information.columns)
    )
    if missing_information_columns:
        raise RuntimeError(
            "Discovery patient-information manifest lacks columns: {}".format(
                missing_information_columns
            )
        )
    if not information["training_allowed"].astype(str).str.lower().eq("yes").all():
        raise RuntimeError("Discovery patient-information manifest contains forbidden rows")
    if information["cohort_summary_used_as_individual_value"].astype(str).str.lower().ne("no").any():
        raise RuntimeError("Cohort summaries cannot be used as individual training values")
    if information["clinical_outcome_label_available"].astype(str).str.lower().ne("no").any():
        raise RuntimeError("External or clinical outcomes are forbidden in V4 discovery training")
    information = information.rename(
        columns={"age_years": "age", "BMI_kg_m2": "BMI", "AHI_events_h": "AHI"}
    )
    metadata = {
        dataset: information.loc[information["dataset"].astype(str) == dataset].copy()
        for dataset in ("GSE75097", "GSE71356")
    }
    if not metadata["GSE75097"]["clinical_information_training_allowed"].astype(str).str.lower().eq("yes").all():
        raise RuntimeError("GSE75097 patient-information rows are not authorized for training")
    if metadata["GSE71356"]["clinical_information_training_allowed"].astype(str).str.lower().ne("no").any():
        raise RuntimeError("GSE71356 has no authorized individual clinical-information fields")
    expected_counts = {"GSE75097": 32, "GSE71356": 16}
    for dataset in metadata:
        metadata[dataset]["sample_id"] = metadata[dataset]["sample_id"].astype(str)
        metadata[dataset]["subject_id"] = metadata[dataset]["subject_id"].astype(str)
        available = set(expression[dataset].columns.astype(str))
        metadata[dataset] = metadata[dataset].loc[
            metadata[dataset]["sample_id"].isin(available)
        ].copy()
        if metadata[dataset]["sample_id"].duplicated().any():
            raise RuntimeError("{} metadata contains duplicate samples".format(dataset))
        if len(metadata[dataset]) != expected_counts[dataset]:
            raise RuntimeError(
                "{} patient-information row count drift: expected={} observed={}".format(
                    dataset, expected_counts[dataset], len(metadata[dataset])
                )
            )
        selected_columns = metadata[dataset]["sample_id"].astype(str).tolist()
        if not selected_columns:
            raise RuntimeError("{} has no frozen discovery samples".format(dataset))
        # GSE75097's mapped matrix also contains the VSOC cross-sectional arm.
        # The locked OSA-state design intentionally selects only untreated
        # PS/MSO/VSO samples, so extra matrix columns are not training data.
        expression[dataset] = expression[dataset].loc[:, selected_columns]
    return expression, metadata, paths


def _safe_standardized_difference(values, positive, negative):
    positive = np.asarray(positive, dtype=bool)
    negative = np.asarray(negative, dtype=bool)
    if positive.sum() < 1 or negative.sum() < 1:
        return np.full(values.shape[0], np.nan, dtype=float)
    pos = values[:, positive]
    neg = values[:, negative]
    difference = np.nanmean(pos, axis=1) - np.nanmean(neg, axis=1)
    joined = np.concatenate([pos, neg], axis=1)
    scale = np.nanstd(joined, axis=1, ddof=1)
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > 1e-8)])
    if not np.isfinite(fallback):
        fallback = 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    return difference / scale


def _safe_row_correlation(values, covariate):
    covariate = np.asarray(covariate, dtype=float)
    valid_covariate = np.isfinite(covariate)
    result = np.full(values.shape[0], np.nan, dtype=float)
    if valid_covariate.sum() < 3:
        return result
    x = covariate[valid_covariate]
    x = x - x.mean()
    x_norm = np.sqrt(np.sum(x * x))
    if x_norm <= 1e-12:
        return result
    y = values[:, valid_covariate]
    y_mean = np.nanmean(y, axis=1, keepdims=True)
    y_centered = y - y_mean
    numerator = np.nansum(y_centered * x.reshape(1, -1), axis=1)
    denominator = np.sqrt(np.nansum(y_centered * y_centered, axis=1)) * x_norm
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _binary_effect(values, metadata, column, positive_value=None):
    if column not in metadata:
        return np.full(values.shape[0], np.nan, dtype=float), 0.0
    series = metadata[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if positive_value is None and numeric.notna().sum() >= 4:
        positive = numeric.to_numpy(dtype=float) > 0.5
        valid = np.isfinite(numeric.to_numpy(dtype=float))
    else:
        text = series.astype(str).str.strip().str.lower()
        positive = text.eq(str(positive_value).lower()).to_numpy()
        valid = ~text.isin(("", "nan", "na", "none")).to_numpy()
    negative = valid & ~positive
    if positive.sum() < 2 or negative.sum() < 2:
        return np.full(values.shape[0], np.nan, dtype=float), 0.0
    return _safe_standardized_difference(values, positive, negative), 1.0


def _continuous_effect(values, metadata, column):
    if column not in metadata:
        return np.full(values.shape[0], np.nan, dtype=float), 0.0
    covariate = pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(covariate).sum() < 6 or np.nanstd(covariate) <= 1e-8:
        return np.full(values.shape[0], np.nan, dtype=float), 0.0
    return _safe_row_correlation(values, covariate), 1.0


def _signed_percentile(raw):
    raw = np.asarray(raw, dtype=float)
    finite = np.isfinite(raw)
    strength = np.zeros(raw.shape[0], dtype=float)
    signed = np.zeros(raw.shape[0], dtype=float)
    if finite.sum() > 0:
        absolute = np.abs(raw[finite])
        order = np.argsort(np.argsort(absolute, kind="mergesort"), kind="mergesort")
        denominator = max(1, int(finite.sum()) - 1)
        percentile = order.astype(float) / float(denominator)
        strength[finite] = percentile
        signed[finite] = np.sign(raw[finite]) * percentile
    return signed, strength, finite.astype(float)


def _ih_difference_in_difference(expression, metadata, subjects):
    values = expression.to_numpy(dtype=float)
    subject_effects = []
    for subject in subjects:
        rows = metadata.loc[metadata["subject_id"].astype(str) == str(subject)]
        lookup = {}
        for _, row in rows.iterrows():
            exposure = str(row.get("exposure", ""))
            timepoint = str(row.get("timepoint", ""))
            lookup[(exposure, timepoint)] = str(row["sample_id"])
        required = [
            ("IH", "baseline_0h"),
            ("IH", "post_5h"),
            ("control_exposure", "baseline_0h"),
            ("control_exposure", "post_5h"),
        ]
        if not all(key in lookup for key in required):
            continue
        index = dict((str(column), i) for i, column in enumerate(expression.columns))
        if not all(lookup[key] in index for key in required):
            continue
        ih_pre = values[:, index[lookup[("IH", "baseline_0h")]]]
        ih_post = values[:, index[lookup[("IH", "post_5h")]]]
        control_pre = values[:, index[lookup[("control_exposure", "baseline_0h")]]]
        control_post = values[:, index[lookup[("control_exposure", "post_5h")]]]
        subject_effects.append((ih_post - ih_pre) - (control_post - control_pre))
    if not subject_effects:
        return np.full(values.shape[0], np.nan, dtype=float)
    stacked = np.stack(subject_effects, axis=1)
    count = np.sum(np.isfinite(stacked), axis=1)
    total = np.nansum(stacked, axis=1)
    return np.divide(
        total,
        count,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=count > 0,
    )


def compute_node_evidence(expression, metadata, proteins, sample_ids, ih_subjects):
    gse75097_meta = metadata["GSE75097"].loc[
        metadata["GSE75097"]["sample_id"].isin(set(sample_ids))
    ].copy()
    columns = gse75097_meta["sample_id"].astype(str).tolist()
    gse75097 = expression["GSE75097"].loc[list(proteins), columns]
    values = gse75097.to_numpy(dtype=float)
    groups = gse75097_meta["study_group"].astype(str).to_numpy()

    osa_raw = _safe_standardized_difference(values, groups != "PS", groups == "PS")
    ordinal_map = {"PS": 0.0, "MSO": 1.0, "VSO": 2.0}
    ordinal = np.asarray([ordinal_map.get(value, np.nan) for value in groups], dtype=float)
    severity_raw = _safe_row_correlation(values, ordinal)
    ahi = pd.to_numeric(gse75097_meta["AHI"], errors="coerce").to_numpy(dtype=float)
    ahi_raw = _safe_row_correlation(values, ahi)
    ih_raw = _ih_difference_in_difference(
        expression["GSE71356"], metadata["GSE71356"], ih_subjects
    )

    age_raw, age_available = _continuous_effect(values, gse75097_meta, "age")
    bmi_raw, bmi_available = _continuous_effect(values, gse75097_meta, "BMI")
    sex_raw, sex_available = _binary_effect(
        values, gse75097_meta, "sex", positive_value="male"
    )
    hypertension_raw, hypertension_available = _binary_effect(
        values, gse75097_meta, "hypertension"
    )
    eds_raw, eds_available = _binary_effect(values, gse75097_meta, "EDS")

    osa_signed, osa_strength, osa_available = _signed_percentile(osa_raw)
    sev_signed, sev_strength, sev_available = _signed_percentile(severity_raw)
    ahi_signed, ahi_strength, ahi_available = _signed_percentile(ahi_raw)
    ih_signed, ih_strength, ih_available = _signed_percentile(ih_raw)
    age_signed, age_strength, _ = _signed_percentile(age_raw)
    bmi_signed, bmi_strength, _ = _signed_percentile(bmi_raw)
    sex_signed, sex_strength, _ = _signed_percentile(sex_raw)
    htn_signed, htn_strength, _ = _signed_percentile(hypertension_raw)
    eds_signed, eds_strength, _ = _signed_percentile(eds_raw)

    clinical_available_count = hypertension_available + eds_available
    if clinical_available_count > 0:
        clinical_signed = (
            hypertension_available * htn_signed + eds_available * eds_signed
        ) / clinical_available_count
        clinical_strength = (
            hypertension_available * htn_strength + eds_available * eds_strength
        ) / clinical_available_count
    else:
        clinical_signed = np.zeros(values.shape[0], dtype=float)
        clinical_strength = np.full(values.shape[0], 0.5, dtype=float)
    demographic_available_count = age_available + sex_available + bmi_available
    if demographic_available_count > 0:
        demographic_strength = (
            age_available * age_strength
            + sex_available * sex_strength
            + bmi_available * bmi_strength
        ) / demographic_available_count
    else:
        demographic_strength = np.zeros(values.shape[0], dtype=float)
    patient_information_coverage = np.full(
        values.shape[0],
        (age_available + sex_available + bmi_available
         + hypertension_available + eds_available) / 5.0,
        dtype=float,
    )

    adult_signed = 0.50 * osa_signed + 0.25 * sev_signed + 0.25 * ahi_signed
    adult_strength = np.abs(adult_signed)
    coverage = (
        osa_available + sev_available + ahi_available + ih_available
    ) / 4.0
    frame = pd.DataFrame(
        {
            "UniProt_accession": list(proteins),
            "OSA_raw": osa_raw,
            "OSA_signed": osa_signed,
            "OSA_strength": osa_strength,
            "severity_raw": severity_raw,
            "severity_signed": sev_signed,
            "severity_strength": sev_strength,
            "AHI_raw": ahi_raw,
            "AHI_signed": ahi_signed,
            "AHI_strength": ahi_strength,
            "IH_raw": ih_raw,
            "IH_signed": ih_signed,
            "IH_strength": ih_strength,
            "IH_available": ih_available,
            "age_signed": age_signed,
            "age_strength": age_strength,
            "sex_signed": sex_signed,
            "sex_strength": sex_strength,
            "BMI_signed": bmi_signed,
            "BMI_strength": bmi_strength,
            "hypertension_signed": htn_signed,
            "hypertension_strength": htn_strength,
            "EDS_signed": eds_signed,
            "EDS_strength": eds_strength,
            "clinical_signed": clinical_signed,
            "clinical_strength": clinical_strength,
            "demographic_strength": demographic_strength,
            "patient_information_coverage": patient_information_coverage,
            "adult_signed": adult_signed,
            "adult_strength": adult_strength,
            # Direction-agnostic OSA evidence anchor.  This restores the
            # robust discovery-score semantics used by the locked r_OSA
            # baseline while keeping every CV fold participant-isolated.
            "discovery_anchor": (
                0.75
                * (0.50 * osa_strength + 0.25 * sev_strength + 0.25 * ahi_strength)
                + 0.25 * np.where(ih_available > 0.5, ih_strength, 0.50)
            ),
            "context_coverage": coverage,
        }
    )
    frame.attrs["patient_information_availability"] = {
        "age": int(age_available), "sex": int(sex_available),
        "BMI": int(bmi_available),
        "hypertension": int(hypertension_available), "EDS": int(eds_available),
    }
    return frame


def _pair_values(node, pairs, column):
    lookup = node.set_index("UniProt_accession")[column]
    first = pairs["protein_A"].map(lookup).fillna(0.0).to_numpy(dtype=float)
    second = pairs["protein_B"].map(lookup).fillna(0.0).to_numpy(dtype=float)
    return first, second


def build_pair_context(pairs, node, information_fusion=False):
    result = pd.DataFrame(index=pairs.index)
    for prefix, column in (
        ("OSA", "OSA_strength"),
        ("severity", "severity_strength"),
        ("AHI", "AHI_strength"),
        ("IH", "IH_strength"),
    ):
        first, second = _pair_values(node, pairs, column)
        result[prefix + "_strength_min"] = np.minimum(first, second)
        result[prefix + "_strength_geomean"] = np.sqrt(
            np.clip(first * second, 0.0, None)
        )
        if prefix == "OSA":
            result["OSA_strength_mean"] = 0.5 * (first + second)
            result["OSA_strength_absdiff"] = np.abs(first - second)

    adult_a, adult_b = _pair_values(node, pairs, "adult_signed")
    ih_a, ih_b = _pair_values(node, pairs, "IH_signed")
    coverage_a, coverage_b = _pair_values(node, pairs, "context_coverage")
    result["adult_signed_product"] = adult_a * adult_b
    result["adult_direction_agreement"] = (adult_a * adult_b >= 0.0).astype(float)
    result["IH_signed_product"] = ih_a * ih_b
    result["IH_direction_agreement"] = (ih_a * ih_b >= 0.0).astype(float)
    result["cross_axis_direction_agreement"] = (
        (adult_a * ih_a >= 0.0).astype(float)
        + (adult_b * ih_b >= 0.0).astype(float)
    ) / 2.0
    result["context_coverage_min"] = np.minimum(coverage_a, coverage_b)
    result["context_coverage_mean"] = 0.5 * (coverage_a + coverage_b)

    anchor_a, anchor_b = _pair_values(node, pairs, "discovery_anchor")
    molecular_anchor = np.clip(
        0.05 + 0.90 * np.sqrt(np.clip(anchor_a * anchor_b, 0.0, None)),
        0.02,
        0.98,
    )
    result["context_anchor"] = molecular_anchor
    result["molecular_context_anchor"] = molecular_anchor

    adult_strength_a, adult_strength_b = _pair_values(node, pairs, "adult_strength")
    adult_pair = np.sqrt(np.clip(adult_strength_a * adult_strength_b, 0.0, None))
    adult_pair *= 0.65 + 0.35 * result["adult_direction_agreement"].to_numpy()
    ih_strength_a, ih_strength_b = _pair_values(node, pairs, "IH_strength")
    ih_pair = np.sqrt(np.clip(ih_strength_a * ih_strength_b, 0.0, None))
    ih_pair *= 0.65 + 0.35 * result["IH_direction_agreement"].to_numpy()
    ih_available_a, ih_available_b = _pair_values(node, pairs, "IH_available")
    ih_pair_available = (ih_available_a * ih_available_b) > 0.5
    combined = np.where(
        ih_pair_available,
        0.80 * adult_pair + 0.20 * ih_pair,
        adult_pair,
    )
    result["weak_target"] = np.clip(0.05 + 0.90 * combined, 0.02, 0.98)

    for prefix, column in (
        ("hypertension", "hypertension_strength"),
        ("EDS", "EDS_strength"),
        ("clinical_manifestation", "clinical_strength"),
    ):
        first, second = _pair_values(node, pairs, column)
        result[prefix + "_strength_min"] = np.minimum(first, second)
        result[prefix + "_strength_geomean"] = np.sqrt(
            np.clip(first * second, 0.0, None)
        )
    clinical_a, clinical_b = _pair_values(node, pairs, "clinical_signed")
    result["clinical_direction_agreement"] = (
        clinical_a * clinical_b >= 0.0
    ).astype(float)
    demographic_a, demographic_b = _pair_values(
        node, pairs, "demographic_strength"
    )
    result["demographic_confounding_strength_max"] = np.maximum(
        demographic_a, demographic_b
    )
    result["demographic_confounding_strength_geomean"] = np.sqrt(
        np.clip(demographic_a * demographic_b, 0.0, None)
    )
    info_a, info_b = _pair_values(node, pairs, "patient_information_coverage")
    result["patient_information_coverage_min"] = np.minimum(info_a, info_b)
    result["patient_information_coverage_mean"] = 0.5 * (info_a + info_b)
    # pandas <=2 exposed a writable column view here; the multiplication below
    # historically updated BOTH this feature and the clinical anchor. Preserve
    # those frozen V4 values explicitly, including under pandas 3 copy-on-write.
    clinical_pair = result["clinical_manifestation_strength_geomean"].to_numpy(copy=True)
    clinical_pair *= (
        0.65 + 0.35 * result["clinical_direction_agreement"].to_numpy()
    )
    result["clinical_manifestation_strength_geomean"] = clinical_pair
    result["clinical_information_anchor"] = np.clip(
        0.05 + 0.90 * clinical_pair, 0.02, 0.98
    )
    if information_fusion:
        coverage_weight = np.clip(
            result["patient_information_coverage_min"].to_numpy(), 0.0, 1.0
        )
        information_weight = 0.20 * coverage_weight
        result["context_anchor"] = np.clip(
            (1.0 - information_weight) * molecular_anchor
            + information_weight * result["clinical_information_anchor"].to_numpy(),
            0.02, 0.98,
        )
        result["weak_target"] = np.clip(
            0.80 * result["weak_target"].to_numpy()
            + 0.20 * result["clinical_information_anchor"].to_numpy(),
            0.02, 0.98,
        )
    if list(result[CONTEXT_FEATURE_COLUMNS].columns) != CONTEXT_FEATURE_COLUMNS:
        raise RuntimeError("Pair-context feature schema drift")
    return result


def assemble_input_matrix(base_pairs, pair_context, input_feature_columns=None):
    columns = list(input_feature_columns or INPUT_FEATURE_COLUMNS)
    missing = [column for column in BASE_FEATURE_COLUMNS if column not in base_pairs]
    if missing:
        raise RuntimeError("Frozen TRIAGE pair output lacks columns: {}".format(missing))
    frame = pd.concat(
        [
            base_pairs[BASE_FEATURE_COLUMNS].reset_index(drop=True),
            pair_context[
                [column for column in columns if column not in BASE_FEATURE_COLUMNS]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if list(frame.columns) != columns:
        raise RuntimeError("OSA adapter input feature order drift")
    return frame.to_numpy(dtype=np.float32)


def role_for_fold(unit_fold, outer_fold, number_of_folds):
    if int(unit_fold) == int(outer_fold) % int(number_of_folds):
        return "test"
    if int(unit_fold) == (int(outer_fold) + 1) % int(number_of_folds):
        return "validation"
    return "train"
