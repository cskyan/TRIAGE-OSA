#!/usr/bin/env python3
"""Uniform, subject-aware evaluation of published OSA gene panels.

This module is called only through train_triage_osa.py.  It does not retrain or
rerank TRIAGE-OSA.  Reported paper AUCs are metadata, never pooled with the
uniform local evaluation.  Python 3.7 compatible; numpy and pandas only.
"""
from __future__ import print_function

import gzip
import io
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "1.2.1"  # Tie-correct AUROC/AP; existing archived outputs are not rewritten.
DEFAULT_OUTPUT = "results/discovery_preprocess_v2/36_published_osa_comparison_v2"
RANKING = "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/pair_context_scores_triage_osa_primary_250.tsv.gz"
NODE_EVIDENCE = "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/full_discovery_node_evidence.tsv"


PUBLISHED = [
    dict(panel_id="PUB_ZHU_2022_OSA_10", task="OSA_status", year=2022,
         citation="Zhu et al., Frontiers in Genetics 2022", doi="10.3389/fgene.2022.927545",
         genes="PTPN3,TXLNA,GLT1D1,SMAD4,REV3L,MOAP1,GRPEL1,MGAT5,TBX3,CRYBB1",
         reported_metric="mean CV AUROC >0.99; external ROC shown",
         reproduction_level="published_panel_protocol_matched_ridge"),
    dict(panel_id="PUB_ZHU_2022_CPAP_10", task="CPAP_state", year=2022,
         citation="Zhu et al., Frontiers in Genetics 2022", doi="10.3389/fgene.2022.927545",
         genes="PPL,TBX3,TMEM121,EYA2,TFF2,FGD4,CAND1,TXLNA,TCF12,ABCF1",
         reported_metric="mean CV AUROC >0.99; external ROC shown",
         reproduction_level="published_panel_protocol_matched_ridge"),
    dict(panel_id="PUB_WU_2022_HYPOXIA_4", task="OSA_status", year=2022,
         citation="Wu et al., Frontiers in Medicine 2022", doi="10.3389/fmed.2022.813459",
         genes="AREG,ATF3,ZFP36,DUSP1", reported_metric="development AUROC 0.842; reported validation AUROC 0.787",
         reproduction_level="published_panel_protocol_matched_ridge"),
    dict(panel_id="PUB_ZHANG_2023_MEGENA_7", task="OSA_status", year=2023,
         citation="Zhang et al., PeerJ 2023", doi="10.7717/peerj.16608",
         genes="C12orf54,FOS,GPR1,OR9A4,MYO5B,RAB39B,KLHL4",
         reported_metric="development AUROC 1.00; reported validation AUROC 0.74",
         reproduction_level="published_panel_protocol_matched_ridge"),
    dict(panel_id="PUB_FAN_2021_CPAP_11", task="CPAP_state", year=2021,
         citation="Fan et al., PLOS ONE 2021", doi="10.1371/journal.pone.0255708",
         genes="TRAV10,SNORA36A,RPL10,OBP2B,IGLV1-40,H2BC8,ESAM,DNASE1L3,CD22,ANK3,ACP3",
         reported_metric="random 2:1 sample split AUROC 0.92",
         reproduction_level="published_panel_protocol_matched_ridge"),
    dict(panel_id="PUB_PENG_2019_WGCNA_5", task="CPAP_state", year=2019,
         citation="Peng et al., Journal of Cellular Biochemistry 2019", doi="10.1002/jcb.28203",
         genes="SOD2,SH3BP5L,WSB2,MTHFD2L,YPEL4", reported_metric="module and hub-gene discovery; no predictive AUROC",
         reproduction_level="published_panel_protocol_matched_ridge"),
]


DATASETS = [
    dict(dataset="GSE135917", task="OSA_status", matrix="geo/GSE135917_series_matrix.txt.gz",
         platform="GPL6244", subset="study_group_1_case_control", label_column="disease_group",
         positive="OSA", grouping="subject", interpretation="within_subcutaneous_adipose_LOSO"),
    dict(dataset="GSE38792", task="OSA_status", matrix="geo/GSE38792_series_matrix.txt.gz",
         platform="GPL6244", subset="all", label_column="disease_group", positive="OSA",
         grouping="subject", interpretation="within_visceral_adipose_LOSO_same_cohort_family_as_GSE135917"),
    dict(dataset="GSE133601", task="CPAP_state", matrix="geo/GSE133601_series_matrix.txt.gz",
         platform="ENTREZ_CUSTOM", subset="all", label_column="timepoint", positive="post",
         grouping="paired_subject", interpretation="paired_PBMC_leave_one_subject_out"),
    dict(dataset="GSE49800", task="CPAP_state", matrix="geo/GSE49800_series_matrix.txt.gz",
         platform="GPL6244", subset="all", label_column="timepoint", positive="post",
         grouping="paired_subject", interpretation="paired_PBL_leave_one_subject_out"),
]


def _json_safe(value):
    if isinstance(value, dict):
        return dict((str(k), _json_safe(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _series_matrix(path):
    header = None
    rows = []
    with gzip.open(str(path), "rt", encoding="utf-8", errors="replace") as handle:
        in_table = False
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            fields = line.rstrip("\n\r").split("\t")
            if header is None:
                header = [item.strip('"') for item in fields]
            else:
                rows.append([fields[0].strip('"')] + [float(x) if x not in ("", "null", "NA") else np.nan for x in fields[1:]])
    if header is None or not rows:
        raise RuntimeError("Series matrix table is empty: {}".format(path))
    frame = pd.DataFrame(rows, columns=header).set_index("ID_REF")
    return frame


def _gpl6244_map(path):
    with gzip.open(str(path), "rt", encoding="utf-8", errors="replace") as handle:
        # GEO annotation files begin with ^Annotation/!Annotation metadata and
        # # field descriptions.  pandas would otherwise treat ^Annotation as
        # the header.  Start only at the real tabular header.
        table = []
        started = False
        for line in handle:
            if line.startswith("ID\t"):
                started = True
            if started:
                table.append(line)
    if not table:
        raise RuntimeError("GPL6244 annotation table header was not found")
    frame = pd.read_csv(io.StringIO("".join(table)), sep="\t", dtype=str, low_memory=False)
    required = ["ID", "Gene symbol"]
    if any(c not in frame.columns for c in required):
        raise RuntimeError("GPL6244 annotation lacks ID/Gene symbol")
    frame = frame[required].dropna()
    frame = frame[~frame["Gene symbol"].str.contains("///", regex=False)]
    frame["ID"] = frame["ID"].astype(str)
    frame["Gene symbol"] = frame["Gene symbol"].astype(str).str.strip()
    frame = frame[(frame["Gene symbol"] != "") & (frame["Gene symbol"] != "---")]
    return dict(zip(frame["ID"], frame["Gene symbol"]))


def _entrez_map(path):
    frame = pd.read_csv(str(path), sep="\t", dtype=str, usecols=["symbol", "entrez_id"], low_memory=False)
    frame = frame.dropna().drop_duplicates("entrez_id")
    return dict(zip(frame["entrez_id"].astype(str), frame["symbol"].astype(str)))


def _gene_matrix(probe_by_sample, mapping, platform):
    genes = []
    keep = []
    for feature in probe_by_sample.index.astype(str):
        key = feature
        if platform == "ENTREZ_CUSTOM":
            key = feature[:-3] if feature.endswith("_at") else feature
        gene = mapping.get(key)
        if gene:
            keep.append(feature)
            genes.append(gene)
    if not keep:
        raise RuntimeError("No expression features mapped for {}".format(platform))
    x = probe_by_sample.loc[keep].copy()
    x["gene"] = genes
    x = x.groupby("gene", sort=True).mean(numeric_only=True)
    # Some deposited series matrices are unlogged intensities. This is a fixed,
    # label-free scale check, applied before any fold is constructed.
    finite = x.to_numpy(dtype=float)
    if np.nanpercentile(finite, 99.0) > 100.0:
        x = np.log2(x.clip(lower=0.0) + 1.0)
    return x.transpose()


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    # Midranks make AUROC invariant to sample order when predictions tie.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1.0) / 2.0
    return ranks


def _roc_auc(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = _rank(score) + 1.0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1.0) / 2.0) / (pos.sum() * neg.sum()))


def _average_precision(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy = y[order]
    # Evaluate precision/recall at distinct score thresholds, not within ties.
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(yy) - 1]
    tp = np.cumsum(yy)[ends]
    recall = tp / float(yy.sum())
    precision = tp / (ends + 1.0)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_ridge_logistic(x, y, penalty=1.0, steps=1500, learning_rate=0.05):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mean = np.nanmean(x, axis=0); sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    xx = (np.where(np.isfinite(x), x, mean) - mean) / sd
    design = np.column_stack([np.ones(len(xx)), xx])
    beta = np.zeros(design.shape[1], dtype=float)
    n0 = max(1.0, float((y == 0).sum())); n1 = max(1.0, float((y == 1).sum()))
    weights = np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
    for _ in range(steps):
        pred = _sigmoid(np.dot(design, beta))
        gradient = np.dot(design.T, weights * (pred - y)) / len(y)
        gradient[1:] += penalty * beta[1:] / len(y)
        beta -= learning_rate * gradient
    return beta, mean, sd


def _predict(model, x):
    beta, mean, sd = model
    x = np.asarray(x, dtype=float)
    xx = (np.where(np.isfinite(x), x, mean) - mean) / sd
    return _sigmoid(beta[0] + np.dot(xx, beta[1:]))


def _loso_predictions(x, y, groups):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=int); groups = np.asarray(groups).astype(str)
    pred = np.full(len(y), np.nan, dtype=float)
    for group in sorted(set(groups)):
        test = groups == group; train = ~test
        if len(set(y[train].tolist())) < 2:
            continue
        pred[test] = _predict(_fit_ridge_logistic(x[train], y[train]), x[test])
    return pred


def _pair_summary_features(z, genes, directions, edges, task):
    """Fixed three-output representation from the frozen V4 graph."""
    z = np.asarray(z, dtype=float)
    direction = np.asarray([directions.get(g, 0.0) for g in genes], dtype=float)
    direction[~np.isfinite(direction)] = 0.0
    endpoint_multiplier = -1.0 if task == "CPAP_state" else 1.0
    signed = z * (endpoint_multiplier * direction.reshape(1, -1))
    node_activity = np.sum(signed, axis=1) / max(float(np.sum(np.abs(direction))), 1e-8)
    index = dict((g, i) for i, g in enumerate(genes))
    selected_edges = [(index[a], index[b], w) for a, b, w in edges if a in index and b in index]
    if not selected_edges:
        return np.column_stack([node_activity, node_activity, np.zeros(len(z), dtype=float)])
    edge_weight = np.asarray([max(float(item[2]), 1e-8) for item in selected_edges], dtype=float)
    edge_weight = edge_weight / edge_weight.sum()
    a = np.column_stack([signed[:, item[0]] for item in selected_edges])
    b = np.column_stack([signed[:, item[1]] for item in selected_edges])
    joint_activity = np.dot(0.5 * (a + b), edge_weight)
    pair_conflict = np.dot(np.abs(a - b), edge_weight)
    return np.column_stack([node_activity, joint_activity, pair_conflict])


def _pair_summary_loso_predictions(x, y, groups, genes, directions, edges, task):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=int); groups = np.asarray(groups).astype(str)
    pred = np.full(len(y), np.nan, dtype=float)
    for group in sorted(set(groups)):
        test = groups == group; train = ~test
        if len(set(y[train].tolist())) < 2:
            continue
        mean = np.nanmean(x[train], axis=0); sd = np.nanstd(x[train], axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        train_z = (np.where(np.isfinite(x[train]), x[train], mean) - mean) / sd
        test_z = (np.where(np.isfinite(x[test]), x[test], mean) - mean) / sd
        train_features = _pair_summary_features(train_z, genes, directions, edges, task)
        test_features = _pair_summary_features(test_z, genes, directions, edges, task)
        pred[test] = _predict(_fit_ridge_logistic(train_features, y[train]), test_features)
    return pred


def _metrics(y, pred):
    valid = np.isfinite(pred); y = np.asarray(y, dtype=int)[valid]; pred = np.asarray(pred, dtype=float)[valid]
    if len(y) == 0:
        return dict(n_samples=0, AUROC=float("nan"), AUPRC=float("nan"), balanced_accuracy=float("nan"), Brier=float("nan"))
    call = pred >= 0.5
    sensitivity = float(np.mean(call[y == 1])) if (y == 1).any() else float("nan")
    specificity = float(np.mean(~call[y == 0])) if (y == 0).any() else float("nan")
    return dict(n_samples=int(len(y)), AUROC=_roc_auc(y, pred), AUPRC=_average_precision(y, pred),
                balanced_accuracy=0.5 * (sensitivity + specificity), Brier=float(np.mean((pred - y) ** 2)))


def _bootstrap_ci(y, pred, groups, iterations=2000, seed=20260904):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=float); groups = np.asarray(groups).astype(str)
    unique = np.asarray(sorted(set(groups))); rng = np.random.RandomState(seed); values = []
    for _ in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == group)[0] for group in chosen])
        value = _roc_auc(y[idx], pred[idx])
        if np.isfinite(value): values.append(value)
    if len(values) < max(100, iterations // 4):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_auc_delta(y, triage_pred, published_pred, groups, iterations=5000, seed=20260905):
    """Participant-cluster bootstrap for paired AUROC differences."""
    y = np.asarray(y, dtype=int)
    triage_pred = np.asarray(triage_pred, dtype=float)
    published_pred = np.asarray(published_pred, dtype=float)
    groups = np.asarray(groups).astype(str)
    valid = np.isfinite(triage_pred) & np.isfinite(published_pred)
    y = y[valid]; triage_pred = triage_pred[valid]; published_pred = published_pred[valid]; groups = groups[valid]
    unique = np.asarray(sorted(set(groups))); rng = np.random.RandomState(seed); values = []
    for _ in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == group)[0] for group in chosen])
        a = _roc_auc(y[idx], triage_pred[idx]); b = _roc_auc(y[idx], published_pred[idx])
        if np.isfinite(a) and np.isfinite(b):
            values.append(a - b)
    if len(values) < max(100, iterations // 4):
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(values, dtype=float)
    low = float(np.quantile(values, 0.025)); high = float(np.quantile(values, 0.975))
    p = float(min(1.0, 2.0 * min((np.sum(values <= 0.0) + 1.0) / (len(values) + 1.0),
                                 (np.sum(values >= 0.0) + 1.0) / (len(values) + 1.0))))
    return low, high, p


def _triage_panels(ranking_path, model_id="TRIAGE_OSA_V4"):
    rank = pd.read_csv(str(ranking_path), sep="\t", low_memory=False)
    rank = rank.sort_values("triage_osa_rank_primary_250", kind="mergesort")
    output = []
    for size in (4, 5, 7, 10, 11):
        genes = []
        for row in rank.itertuples(index=False):
            for gene in (str(row.gene_A), str(row.gene_B)):
                if gene and gene not in genes:
                    genes.append(gene)
                if len(genes) >= size:
                    break
            if len(genes) >= size:
                break
        output.append(dict(panel_id=model_id+"_TOP{}_NODES".format(size), task="both", year=2026,
                           citation="TRIAGE-OSA discovery-derived ranking", doi="", genes=",".join(genes),
                           reported_metric="not applicable; uniformly evaluated here",
                           reproduction_level="frozen_candidate_panel"))
    return output


def _triage_gene_order(ranking_path):
    rank = pd.read_csv(str(ranking_path), sep="\t", low_memory=False)
    rank = rank.sort_values("triage_osa_rank_primary_250", kind="mergesort")
    genes = []
    for row in rank.itertuples(index=False):
        for gene in (str(row.gene_A), str(row.gene_B)):
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def _triage_graph(ranking_path, node_path):
    rank = pd.read_csv(str(ranking_path), sep="\t", low_memory=False)
    rank = rank.sort_values("triage_osa_rank_primary_250", kind="mergesort")
    node = pd.read_csv(str(node_path), sep="\t", dtype={"UniProt_accession": str}, low_memory=False)
    protein_direction = dict(zip(node["UniProt_accession"].astype(str), node["adult_signed"].astype(float)))
    directions = {}
    edges = []
    for row in rank.itertuples(index=False):
        gene_a = str(row.gene_A); gene_b = str(row.gene_B)
        if gene_a not in directions:
            directions[gene_a] = protein_direction.get(str(row.protein_A), 0.0)
        if gene_b not in directions:
            directions[gene_b] = protein_direction.get(str(row.protein_B), 0.0)
        edges.append((gene_a, gene_b, float(row.triage_osa_score)))
    return directions, edges


def _metadata(root, dataset, subset):
    if dataset == "GSE38792":
        path = root / "results/discovery_preprocess_v2/20_independent_validation_preprocess_v1/independent_sample_manifest.tsv"
        meta = pd.read_csv(str(path), sep="\t", dtype=str)
    else:
        meta = pd.read_csv(str(root / "metadata/sample_manifest.tsv"), sep="\t", dtype=str)
    meta = meta[meta["dataset"] == dataset].copy()
    if subset != "all":
        meta = meta[meta["study_group"] == subset].copy()
    return meta


def _comparability_rows():
    return [
        dict(comparator="PUB_FAN_2021_CPAP_11", shared_dataset="GSE133601", current_stage="Stage11/12",
             same_endpoint="no", same_independent_unit="no", same_split="no", direct_comparison="no",
             reason="Published random 2:1 split classifies 30 samples; current endpoint tests ranked-pair enrichment with 15 participants."),
        dict(comparator="PUB_ZHU_2022_OSA_10/CPAP_10", shared_dataset="GSE71356,GSE75097,GSE133601,GSE49800,GSE135917,GSE38792",
             current_stage="V4+Stage11/12/21", same_endpoint="no", same_independent_unit="no", same_split="no", direct_comparison="no",
             reason="Published ANN is sample classification after cross-platform merging; V4 is frozen pair prioritization without a clinical probability output."),
        dict(comparator="PUB_WU_2022_HYPOXIA_4", shared_dataset="GSE135917,GSE38792", current_stage="Stage14/21",
             same_endpoint="no", same_independent_unit="partly", same_split="no", direct_comparison="no",
             reason="Published diagnostic ROC differs from pair/pathway support; the two adipose datasets share a cohort family."),
        dict(comparator="PUB_ZHANG_2023_MEGENA_7", shared_dataset="GSE135917,GSE38792", current_stage="Stage14/21",
             same_endpoint="no", same_independent_unit="partly", same_split="no", direct_comparison="no",
             reason="Published diagnostic ROC differs from pair/pathway support; reported validation is cross-tissue within a shared cohort family."),
        dict(comparator="PUB_PENG_2019_WGCNA_5", shared_dataset="GSE49800", current_stage="Stage11/12",
             same_endpoint="no", same_independent_unit="yes", same_split="no", direct_comparison="no",
             reason="Published result is module/hub discovery without predictive AUROC; current endpoint is frozen ranking enrichment."),
    ]


def run(args):
    root = Path(args.root).expanduser().resolve()
    out = root / DEFAULT_OUTPUT
    ranking_path, node_path = root/RANKING, root/NODE_EVIDENCE
    comparison = getattr(args,'comparison_ranking',None)
    model_id = 'TRIAGE_OSA_CROSSPARTICIPANT_'+args.comparison_method if comparison else 'TRIAGE_OSA_V4'
    if comparison:
        source = (root/comparison).resolve()
        with (source/'run_summary.json').open() as f:
            summary = json.load(f)
        expected = 'PASS_SMOKE' if args.smoke else 'PASS_CROSS_PARTICIPANT_EXECUTION'
        if summary['status'] != expected or summary.get('external_data_read') is not False:
            raise RuntimeError('Stage39 must complete without external fitting before comparison')
        from osa_external_strong_controls import sha
        if sha(source/'rankings_primary_250.tsv.gz') != summary['ranking_sha256']:
            raise RuntimeError('Stage39 ranking provenance mismatch')
        ranking_path = source/('comparison_'+args.comparison_method+'.tsv.gz')
        node_path = source/'full_discovery_node_evidence.tsv'
        out = root/'results/discovery_preprocess_v2'/('36_published_osa_comparison_crossparticipant_v1'+('_smoke' if args.smoke else ''))/args.comparison_method
    required = [ranking_path, node_path, root / "metadata/sample_manifest.tsv",
                root / "annotations/GPL6244.annot.gz", root / "annotations/hgnc_complete_set_20260714.tsv"]
    required += [root / item["matrix"] for item in DATASETS]
    missing = [str(p) for p in required if not p.is_file()]
    print("Published OSA comparison environment")
    print("Python/numpy/pandas compatible: PASS")
    print("Frozen V4 changed: no")
    print("Published paper AUC directly comparable to existing stages: no")
    if missing:
        raise RuntimeError("Missing comparison inputs: {}".format(missing))

    panels = PUBLISHED + _triage_panels(ranking_path, model_id)
    triage_gene_order = _triage_gene_order(ranking_path)
    triage_directions, triage_edges = _triage_graph(ranking_path, node_path)
    registry = pd.DataFrame(panels)
    registry["gene_count"] = registry["genes"].str.split(",").map(len)
    primary_rank = pd.read_csv(str(ranking_path), sep="\t", usecols=["gene_A", "gene_B"], low_memory=False)
    v4_genes = set(primary_rank["gene_A"].astype(str)) | set(primary_rank["gene_B"].astype(str))
    registry["overlap_with_V4_primary250"] = registry["genes"].map(lambda s: len(set(s.split(",")) & v4_genes))

    if args.check_env:
        print("Datasets: {}".format(", ".join(item["dataset"] for item in DATASETS)))
        print("Published panels: {}".format(len(PUBLISHED)))
        print("TRIAGE representations: size-matched node-only plus fixed pair-aware three-output")
        print("Preflight: PASS")
        return 0

    if out.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force: {}".format(out))
        resolved = out.resolve()
        allowed = (root/'results/discovery_preprocess_v2').resolve()
        if allowed not in resolved.parents or not resolved.name:
            raise RuntimeError('Refusing to remove an output outside the result directory')
        shutil.rmtree(str(out))
    out.mkdir(parents=True)
    if comparison:
        from osa_external_strong_controls import write_json, sha
        write_json(out/'comparison_provenance.json',dict(source=str(source),method=args.comparison_method,
            ranking_sha256=sha(ranking_path),node_evidence_sha256=sha(node_path),
            protocol='unchanged Stage36 participant splits, classifiers and signatures',
            inference='exploratory; published panels may overlap development cohorts; not original-paper independent validation'))
    registry.to_csv(str(out / "published_model_registry.tsv"), sep="\t", index=False)
    pd.DataFrame(_comparability_rows()).to_csv(str(out / "direct_comparability_audit.tsv"), sep="\t", index=False)

    gpl = _gpl6244_map(root / "annotations/GPL6244.annot.gz")
    entrez = _entrez_map(root / "annotations/hgnc_complete_set_20260714.tsv")
    results = []
    coverage = []
    predictions = []
    prediction_lookup = {}
    endpoint_lookup = {}
    matched_design = []
    for spec in DATASETS:
        print("[dataset] {}".format(spec["dataset"]))
        probes = _series_matrix(root / spec["matrix"])
        genes = _gene_matrix(probes, entrez if spec["platform"] == "ENTREZ_CUSTOM" else gpl, spec["platform"])
        meta = _metadata(root, spec["dataset"], spec["subset"])
        common_samples = [s for s in meta["sample_id"].astype(str) if s in genes.index]
        meta = meta.set_index("sample_id").loc[common_samples].reset_index()
        genes = genes.loc[common_samples]
        y = (meta[spec["label_column"]].astype(str).str.lower() == spec["positive"].lower()).astype(int).to_numpy()
        groups = meta["subject_id"].fillna(meta["sample_id"]).astype(str).to_numpy()
        if len(set(y.tolist())) != 2:
            raise RuntimeError("{} lacks two endpoint classes".format(spec["dataset"]))
        dataset_panels = [panel for panel in PUBLISHED if panel["task"] == spec["task"]]
        for published_panel in list(dataset_panels):
            published_available = [g for g in published_panel["genes"].split(",") if g in genes.columns]
            matched_genes = [g for g in triage_gene_order if g in genes.columns][:len(published_available)]
            matched_id = (model_id+"_MATCHED_{}").format(published_panel["panel_id"].replace("PUB_", ""))
            pair_id = (model_id+"_PAIR_AWARE_MATCHED_{}").format(published_panel["panel_id"].replace("PUB_", ""))
            dataset_panels.append(dict(panel_id=matched_id, task=spec["task"], year=2026,
                                       citation="TRIAGE-OSA discovery-derived ranking", doi="",
                                       genes=",".join(matched_genes), reported_metric="uniform evaluation",
                                       reproduction_level="dataset-measurability-matched_frozen_candidate_panel",
                                       evaluation_mode="gene_ridge"))
            dataset_panels.append(dict(panel_id=pair_id, task=spec["task"], year=2026,
                                       citation="TRIAGE-OSA discovery-derived pair ranking", doi="",
                                       genes=",".join(matched_genes), reported_metric="uniform evaluation",
                                       reproduction_level="fixed_directional_pair_summary",
                                       evaluation_mode="triage_pair_summary"))
            matched_design.append(dict(dataset=spec["dataset"], task=spec["task"],
                                       published_panel=published_panel["panel_id"],
                                       published_requested_genes=len(published_panel["genes"].split(",")),
                                       published_available_genes=len(published_available),
                                       TRIAGE_matched_panel=matched_id,
                                       TRIAGE_pair_aware_panel=pair_id,
                                       TRIAGE_available_genes=len(matched_genes),
                                       matching_rule="same_number_of_dataset-measurable_genes_label_free"))
        for panel in dataset_panels:
            evaluation_mode = panel.get("evaluation_mode", "gene_ridge")
            requested = panel["genes"].split(",")
            available = [g for g in requested if g in genes.columns]
            coverage.append(dict(dataset=spec["dataset"], task=spec["task"], panel_id=panel["panel_id"],
                                 requested_genes=len(requested), available_genes=len(available),
                                 coverage_fraction=float(len(available)) / len(requested),
                                 available_gene_symbols=",".join(available)))
            if len(available) < 2:
                results.append(dict(dataset=spec["dataset"], task=spec["task"], panel_id=panel["panel_id"],
                                    status="NOT_EVALUABLE_LT2_GENES", n_subjects=len(set(groups)), n_samples=len(y),
                                    genes_available=len(available), AUROC=float("nan"), AUROC_CI95_low=float("nan"),
                                    AUROC_CI95_high=float("nan"), AUPRC=float("nan"), balanced_accuracy=float("nan"),
                                    Brier=float("nan"), interpretation=spec["interpretation"]))
                continue
            if evaluation_mode == "triage_pair_summary":
                pred = _pair_summary_loso_predictions(genes[available].to_numpy(dtype=float), y, groups,
                                                      available, triage_directions, triage_edges, spec["task"])
            else:
                pred = _loso_predictions(genes[available].to_numpy(dtype=float), y, groups)
            met = _metrics(y, pred); low, high = _bootstrap_ci(y, pred, groups)
            row = dict(dataset=spec["dataset"], task=spec["task"], panel_id=panel["panel_id"],
                       evaluation_mode=evaluation_mode, status="EVALUATED",
                       n_subjects=len(set(groups)), genes_available=len(available), AUROC_CI95_low=low,
                       AUROC_CI95_high=high, interpretation=spec["interpretation"])
            row.update(met); results.append(row)
            prediction_lookup[(spec["dataset"], panel["panel_id"])] = pred
            endpoint_lookup[spec["dataset"]] = (y, groups)
            for i in range(len(y)):
                predictions.append(dict(dataset=spec["dataset"], task=spec["task"], panel_id=panel["panel_id"],
                                        sample_id=meta.iloc[i]["sample_id"], subject_id=groups[i], label=int(y[i]),
                                        probability=float(pred[i]) if np.isfinite(pred[i]) else np.nan))

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(str(out / "uniform_panel_results.tsv"), sep="\t", index=False)
    result_frame.to_csv(str(out / "sample_endpoint_results.tsv"), sep="\t", index=False)
    pd.DataFrame(coverage).to_csv(str(out / "panel_gene_coverage.tsv"), sep="\t", index=False)
    pd.DataFrame(matched_design).to_csv(str(out / "matched_panel_design.tsv"), sep="\t", index=False)
    pd.DataFrame(predictions).to_csv(str(out / "out_of_subject_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    result_frame[result_frame["task"] == "CPAP_state"].to_csv(
        str(out / "paired_cpap_results.tsv"), sep="\t", index=False)

    evaluated = result_frame[result_frame["status"] == "EVALUATED"].copy()
    main = []
    for design in matched_design:
        dataset = design["dataset"]
        block = evaluated[evaluated["dataset"] == dataset]
        published = block[block["panel_id"] == design["published_panel"]]
        triage_id = design["TRIAGE_pair_aware_panel"]
        triage = block[block["panel_id"] == triage_id]
        if len(published) != 1 or len(triage) != 1:
            continue
        pub = published.iloc[0]
        triage_row = triage.iloc[0]
        y, groups = endpoint_lookup[dataset]
        low, high, p = _bootstrap_auc_delta(y, prediction_lookup[(dataset, triage_id)],
                                             prediction_lookup[(dataset, design["published_panel"])], groups)
        main.append(dict(dataset=dataset, task=design["task"],
                         published_panel=design["published_panel"],
                         published_requested_genes=design["published_requested_genes"],
                         matched_measurable_genes=design["published_available_genes"],
                         published_AUROC=float(pub["AUROC"]),
                         TRIAGE_size_matched_panel=triage_id,
                         TRIAGE_size_matched_AUROC=float(triage_row["AUROC"]),
                         delta_AUROC_TRIAGE_minus_published=float(triage_row["AUROC"] - pub["AUROC"]),
                         delta_AUROC_CI95_low=low, delta_AUROC_CI95_high=high,
                         cluster_bootstrap_P=p, same_expression_matrix="yes",
                         same_participant_split="yes", same_classifier_head="ridge_logistic",
                         representation="published_raw_genes_vs_TRIAGE_fixed_three-output_pair_summary",
                         superiority_claim_allowed="only_if_prespecified_and_CI_excludes_zero"))
    main_frame = pd.DataFrame(main)
    if len(main_frame):
        main_frame["bootstrap_tail_probability_exploratory"] = main_frame.pop("cluster_bootstrap_P")
    main_frame.to_csv(str(out / "main_comparison_table.tsv"), sep="\t", index=False)

    ablation = []
    for design in matched_design:
        dataset = design["dataset"]
        block = evaluated[evaluated["dataset"] == dataset]
        node = block[block["panel_id"] == design["TRIAGE_matched_panel"]]
        pair = block[block["panel_id"] == design["TRIAGE_pair_aware_panel"]]
        if len(node) == 1 and len(pair) == 1:
            ablation.append(dict(dataset=dataset, task=design["task"], comparator=design["published_panel"],
                                 measurable_genes=design["published_available_genes"],
                                 TRIAGE_node_only_AUROC=float(node.iloc[0]["AUROC"]),
                                 TRIAGE_pair_aware_AUROC=float(pair.iloc[0]["AUROC"]),
                                 delta_pair_aware_minus_node_only=float(pair.iloc[0]["AUROC"] - node.iloc[0]["AUROC"])))
    pd.DataFrame(ablation).to_csv(str(out / "pair_aware_ablation.tsv"), sep="\t", index=False)

    feasibility = []
    for spec in DATASETS:
        block = result_frame[result_frame["dataset"] == spec["dataset"]]
        n_subjects = int(block["n_subjects"].max()) if len(block) else 0
        feasibility.append(dict(dataset=spec["dataset"], task=spec["task"], n_subjects=n_subjects,
                                independent_unit="participant", paired_samples_kept_together=("yes" if spec["task"] == "CPAP_state" else "not_applicable"),
                                uniform_panel_comparison="feasible", clinical_prediction_claim="not_allowed",
                                limitation="small public cohort; wide participant-cluster bootstrap intervals"))
    pd.DataFrame(feasibility).to_csv(str(out / "endpoint_feasibility.tsv"), sep="\t", index=False)

    missingness = []
    for spec in DATASETS:
        missingness.append(dict(dataset=spec["dataset"], task=spec["task"],
                                individual_age_available="no", individual_sex_available="no",
                                individual_BMI_available="no", individual_AHI_available="no",
                                individual_symptom_outcome_available="no", ablation_status="NOT_EVALUABLE",
                                reason="No harmonized individual clinical covariates are publicly linked to these expression rows."))
    pd.DataFrame(missingness).to_csv(str(out / "information_missingness_ablation.tsv"), sep="\t", index=False)

    learning = []
    for spec in DATASETS:
        learning.append(dict(dataset=spec["dataset"], task=spec["task"], status="NOT_FORMALLY_ESTIMABLE",
                             n_subjects=int(result_frame[result_frame["dataset"] == spec["dataset"]]["n_subjects"].max()),
                             reason="With 15-18 participants, repeated down-sampling would mainly measure split noise and is not presented as a learning curve."))
    pd.DataFrame(learning).to_csv(str(out / "learning_curve.tsv"), sep="\t", index=False)

    systems_rows = []
    for panel in PUBLISHED:
        systems_rows.append(dict(method=panel["panel_id"], sample_state_panel="yes",
                                 frozen_protein_pair_ranking="no", CPAP_reversal_endpoint="no",
                                 cross_tissue_pathway_endpoint="no", single_cell_localization="no",
                                 pediatric_proteome_support="no", known_PPI_complex_recovery="no",
                                 longitudinal_network_endpoint="no",
                                 interpretation="Published gene signature re-evaluated only for its matching sample-state task."))
    systems_rows.append(dict(method=model_id, sample_state_panel="yes_stage36",
                             frozen_protein_pair_ranking="yes", CPAP_reversal_endpoint="yes_mixed_stage11_12",
                             cross_tissue_pathway_endpoint="yes_stage14", single_cell_localization="yes_stage16",
                             pediatric_proteome_support="limited_stage18", known_PPI_complex_recovery="yes_stage19",
                             longitudinal_network_endpoint="exploratory_negative_stage35",
                             interpretation="Broader systems-biology scope; breadth does not imply every endpoint is positive."))
    pd.DataFrame(systems_rows).to_csv(str(out / "systems_biology_endpoint_matrix.tsv"), sep="\t", index=False)

    summary = dict(status="PASS_PUBLISHED_OSA_PAIR_AWARE_REEVALUATION", script_version=VERSION,
                   datasets=len(DATASETS), published_panels=len(PUBLISHED),
                   triage_representation="fixed directional node activity, pair joint activity, and pair conflict",
                   evaluated_rows=int((result_frame["status"] == "EVALUATED").sum()),
                   matched_comparisons=int(len(main_frame)),
                   TRIAGE_higher_descriptive=int((main_frame["delta_AUROC_TRIAGE_minus_published"] > 0).sum()),
                   TRIAGE_lower_descriptive=int((main_frame["delta_AUROC_TRIAGE_minus_published"] < 0).sum()),
                   TRIAGE_higher_CI_excludes_zero=int((main_frame["delta_AUROC_CI95_low"] > 0).sum()),
                   TRIAGE_lower_CI_excludes_zero=int((main_frame["delta_AUROC_CI95_high"] < 0).sum()),
                   direct_reuse_of_existing_AUC_allowed=False, V4_retrained=False, V4_reranked=False,
                   validation_driven_weight_tuning=False,
                   independent_unit="participant; paired pre/post samples remain in the same test fold",
                   claim_boundary="Candidate-panel information comparison, not a validated clinical predictor or proof of physical interaction",
                   gse135917_gse38792_independent=False,
                   next_step="Audit all rows, then add systems-biology breadth without selecting only favourable endpoints.")
    _write_json(out / "run_summary.json", summary)
    print("Published OSA comparison: {}".format(summary["status"]))
    print("Output: {}".format(out))
    print("V4 retrained/reranked: no/no")
    return 0
