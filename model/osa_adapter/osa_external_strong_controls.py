"""Post hoc OSA external strong controls. Python 3.7+, CPU, no Full retraining.

Use the existing convex static objective and archived discovery-only inputs.
Randomization intervals are explicitly not confidence intervals. Bootstrap
intervals resample complete participant delta vectors, never protein pairs.
"""
from __future__ import print_function

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

METHODS = ("Full", "Static_constrained", "Direct_anchor")
BASE = "results/discovery_preprocess_v2"
VERSION = "1.0.0"


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.minimum(q, 1.)
    return out


def save_table(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def fit_rankings(root, out):
    from osa_controlled_experiments import convex_static_fit
    base = root / BASE
    stage = base / "09_pair_context_ranking_triage_osa_v4"
    manifest = read_json(stage / "run_summary.json")
    source = stage / "pair_context_scores_triage_osa_500.tsv.gz"
    if sha(source) != manifest["output_all_sha256"] or manifest["external_validation_read"]:
        raise RuntimeError("Full discovery ranking provenance mismatch")
    x = pd.read_csv(source, sep="\t")
    if len(x) != 124750 or x.pair_id.duplicated().any():
        raise RuntimeError("Expected the complete 500-protein discovery universe")
    target = x.context_anchor.to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any((target <= 0) | (target >= 1)):
        raise RuntimeError("Invalid archived context anchor")
    # Match run_fusion_and_context_supplements, including float32 backbone features.
    gates = x[["g_L1", "g_L2", "g_L3"]].to_numpy(dtype=np.float32).clip(1e-6)
    gates = .1 + .7 * gates / gates.sum(axis=1, keepdims=True)
    tri = np.sum(gates * x[["logit_L1", "logit_L2", "logit_L3"]].to_numpy(dtype=np.float32), axis=1)
    experts = np.column_stack([x.logit_PPI.to_numpy(dtype=np.float32), tri,
                               np.log(target / (1.-target))])
    contract = read_json(base / "31_osa_controlled_experiments_v1/osa_ablation_suite/experiment_contract.json")
    seed = int(contract["optimization"]["seed"])
    fit = convex_static_fit(experts, target, seed, constrained=True)
    if not fit["converged"]:
        raise RuntimeError("Static solver did not meet the existing convergence criterion")
    write_json(out / "static_final_fit.json", dict(fit, seed=seed,
        final_refit="all discovery pairs; no external selection", full_retrained=False,
        expert_definition="original PPI; floor-constrained original tri-level gates; logit(context_anchor)"))
    x = x.copy()
    x["Full"] = x.triage_osa_score
    x["Static_constrained"] = expit(experts.dot(np.asarray(fit["weights"])))
    x["Direct_anchor"] = target
    primary = x.primary_pair.astype(str).str.lower().eq("yes")
    columns = ["pair_id", "protein_A", "protein_B", "primary_pair"] + list(METHODS)
    all_ranks = x[columns].copy()
    for method in METHODS:
        all_ranks["rank_" + method] = all_ranks[method].rank(method="first", ascending=False).astype(int)
    all_ranks.to_csv(out / "rankings_500.tsv.gz", sep="\t", index=False)
    p = x.loc[primary, columns + ["rank_r_OSA", "OSA_interaction_priority_rank"]].copy()
    for method in METHODS:
        p["rank_" + method] = p[method].rank(method="first", ascending=False).astype(int)
    if len(p) != 31125 or not np.array_equal(p.rank_Full, p.OSA_interaction_priority_rank):
        raise RuntimeError("Full primary ranking did not reproduce archived order")
    p.to_csv(out / "rankings_primary_250.tsv.gz", sep="\t", index=False)
    top = lambda c: set(p.loc[p[c] <= 312, "pair_id"])
    audit = dict(pairs=len(p), ranks_identical=bool(np.array_equal(p.rank_Direct_anchor, p.rank_r_OSA)),
                 rank_Spearman=float(p.rank_Direct_anchor.corr(p.rank_r_OSA, method="spearman")),
                 top312_overlap=len(top("rank_Direct_anchor") & top("rank_r_OSA")),
                 external_context_only="inherited rank_r_OSA from baseline ranking",
                 direct_anchor="current information-conditioned context_anchor",
                 equivalent=False)
    audit["equivalent"] = audit["ranks_identical"]
    write_json(out / "context_only_identity_audit.json", audit)
    write_json(out / "ranking_provenance.json", dict(source=str(source), source_sha256=sha(source),
        ranking_sha256=sha(out / "rankings_primary_250.tsv.gz"),
        external_data_read_during_fit=False, tie_rule="first in archived discovery row order",
        static_seed=seed, static_seed_selected_using_external_data=False))
    return p.reset_index(drop=True)


def _validated_external_methods(ranks, methods):
    """Validate comparison ranks without changing their row or tie order."""
    methods = tuple(METHODS if methods is None else methods)
    if (len(methods) < 2 or methods[0] != "Full"
            or any(not isinstance(m, str) or not m.strip() for m in methods)
            or len(set(methods)) != len(methods)):
        raise RuntimeError("External methods must be unique names with Full first and at least one control")
    fields = ["pair_id", "protein_A", "protein_B"]
    fields += list(methods) + ["rank_" + m for m in methods]
    missing = [name for name in fields if name not in ranks.columns]
    if missing:
        raise RuntimeError("Missing external ranking fields: " + str(missing))
    if ranks.empty or ranks["pair_id"].isna().any() or ranks["pair_id"].duplicated().any():
        raise RuntimeError("External ranking requires nonempty, unique pair identities")
    if ranks[["protein_A", "protein_B"]].isna().any().any():
        raise RuntimeError("External ranking contains missing protein identities")
    expected_ranks = np.arange(1, len(ranks) + 1, dtype=np.int64)
    for method in methods:
        try:
            values = ranks[method].to_numpy(dtype=float)
            provided = ranks["rank_" + method].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise RuntimeError("Nonnumeric external scores or ranks: " + method)
        if not np.isfinite(values).all() or not np.isfinite(provided).all():
            raise RuntimeError("Nonfinite external scores or ranks: " + method)
        if not np.array_equal(np.sort(provided), expected_ranks):
            raise RuntimeError("External ranks must be a permutation of 1..N: " + method)
        from_scores = ranks[method].rank(method="first", ascending=False).to_numpy(dtype=float)
        if not np.array_equal(provided, from_scores):
            raise RuntimeError("External ranks disagree with stable descending scores: " + method)
    return methods


def cpap(root, out, ranks, iterations, bootstraps, methods=None):
    methods = _validated_external_methods(ranks, methods)
    legacy_methods = methods == METHODS
    import osa_cpap_reversal_endpoint as endpoint
    import osa_cpap_subject_signflip as subject
    base = root / BASE
    candidates = pd.read_csv(base / endpoint.CANDIDATE_REL.split(BASE + "/")[-1], sep="\t").to_dict("records")
    ids = [r["UniProt_accession"] for r in candidates]
    index = {p:i for i,p in enumerate(ids)}
    left = np.asarray([index[p] for p in ranks.protein_A])
    right = np.asarray([index[p] for p in ranks.protein_B])
    pair_rows = ranks.to_dict("records")
    comparisons, observed_rows, boot_rows = [], [], []
    seed = int(read_json(root / "config/osa_cpap_validation_lock.json")["permutation"]["seed"])
    for ci, cohort in enumerate(endpoint.COHORTS):
        pre = base / "10_cpap_validation_preprocess_v1" / cohort
        evrows, _ = endpoint.read_tsv(pre / "candidate_validation_evidence.tsv")
        evidence, available = endpoint.build_node_evidence(np, candidates, evrows)
        available_set = set(available)
        valid_nodes = np.asarray([p in available_set for p in ids])
        eligible = valid_nodes[left] & valid_nodes[right]
        nodes = np.asarray([evidence[p]["score"] if p in available_set else 0. for p in ids])
        support = np.sqrt(nodes[left] * nodes[right])
        reversed_nodes = np.asarray([bool(evidence[p]["direction_reversed"]) for p in ids])
        binary = reversed_nodes[left] & reversed_nodes[right]
        masks, observed, null = {}, {}, {}
        for method in methods:
            r = ranks["rank_" + method].to_numpy()
            masks[method] = [eligible & (r <= int(math.ceil(len(ranks)*f))) for f in endpoint.FRACTIONS]
            if any(not m.any() for m in masks[method]):
                raise RuntimeError("No measurable selected pairs: " + cohort + "/" + method)
            rows, area, ratio = endpoint.endpoint_metrics(np, pair_rows, support, binary, eligible, "rank_"+method)
            observed[method] = np.asarray([ratio, area])
            for row in rows:
                observed_rows.append(dict(row, cohort=cohort, method=method,
                                          evidence="archived moderated_t"))
            # Reset to the same seed: exact same node permutations for all methods.
            nr, na, blocks = endpoint.node_aware_permutation(np, np.random.RandomState(seed+ci),
                iterations, 50, candidates, evidence, pair_rows, eligible, left, right, r)
            null[method] = np.column_stack([nr, na])
            save_table(out / (cohort + "_matched_blocks.tsv"), blocks)
        null_export = {method+"_"+metric:null[method][:,j] for method in methods
                       for j,metric in enumerate(("ratio", "AUEC"))}
        pd.DataFrame(null_export).to_csv(out / (cohort+"_shared_node_null.tsv.gz"), sep="\t", index=False)
        for control in methods[1:]:
            for j, metric in enumerate(("top1_enrichment_ratio", "normalized_AUEC")):
                delta = observed["Full"][j] - observed[control][j]
                nd = null["Full"][:,j] - null[control][:,j]
                comparisons.append(dict(cohort=cohort, comparison="Full_minus_"+control, metric=metric,
                    Full=observed["Full"][j], control=observed[control][j], difference=delta,
                    null_q025=float(np.quantile(nd,.025)), null_q975=float(np.quantile(nd,.975)),
                    two_sided_randomization_P=(1.+np.sum(np.abs(nd)>=abs(delta)))/(iterations+1.),
                    null_unit="matched node evidence permutation; not a patient CI"))
        # Participant bootstrap uses the existing unmoderated paired-t sensitivity,
        # not the moderated-t point estimate above. No mixed-estimator confidence intervals.
        d = pd.read_csv(pre / "candidate_participant_deltas.tsv.gz", sep="\t").set_index("UniProt_accession")
        fields = [c for c in d if c.startswith("subject_")]
        ordered = [p for p in ids if p in available_set]
        values = d.loc[ordered, fields].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite participant deltas")
        emap = {r["UniProt_accession"]:r for r in evrows}
        direction = np.asarray([1. if emap[p]["discovery_OSA_beta_sign"]=="positive" else -1. for p in ordered])
        node_idx = np.asarray([index[p] for p in ordered])
        def evaluate(v):
            t = subject.paired_t(np, v, np.ones((1,v.shape[1])))[:,0]
            ns = np.zeros((1,len(ids)))
            ns[0,node_idx] = endpoint.midrank_percentiles(np, -direction*t)
            return np.asarray([np.asarray(subject.pair_statistics(np,ns,left,right,eligible,masks[m]))[:,0] for m in methods])
        actual = evaluate(values)
        rng = np.random.RandomState(seed+200+ci)
        draws = np.empty((bootstraps,len(methods),2))
        for b in range(bootstraps):
            draws[b] = evaluate(values[:,rng.randint(len(fields),size=len(fields))])
            if (b+1)%1000==0:
                print("[{} participant bootstrap] {}/{}".format(cohort,b+1,bootstraps), flush=True)
        for c,control in enumerate(methods[1:],1):
            for j,metric in enumerate(("top1_enrichment_ratio", "normalized_AUEC")):
                diffs = draws[:,0,j]-draws[:,c,j]
                boot_rows.append(dict(cohort=cohort,comparison="Full_minus_"+control,metric=metric,
                    n_subjects=len(fields),Full=actual[0,j],control=actual[c,j],difference=actual[0,j]-actual[c,j],
                    CI95_low=float(np.quantile(diffs,.025)),CI95_high=float(np.quantile(diffs,.975)),
                    estimator="unmoderated paired_t sensitivity",CI_method="paired participant percentile bootstrap"))
        pd.DataFrame(draws.reshape(bootstraps,2*len(methods)),columns=[m+"_"+v for m in methods for v in ("ratio","AUEC")]).to_csv(
            out / (cohort+"_participant_bootstrap.tsv.gz"),sep="\t",index=False)
    for row,q in zip(comparisons,bh([r["two_sided_randomization_P"] for r in comparisons])):
        if legacy_methods:
            row["BH_FDR_8_CPAP_comparisons"] = q
        else:
            row["BH_FDR_CPAP_comparisons"] = q
            row["BH_family_size"] = len(comparisons)
            row["common_reference"] = "Full"
    if not legacy_methods:
        for row in boot_rows:
            row["common_reference"] = "Full"
    save_table(out / "cpap_moderated_endpoint_curves.tsv", observed_rows)
    save_table(out / "cpap_paired_node_randomization.tsv", comparisons)
    save_table(out / "cpap_paired_participant_bootstrap.tsv", boot_rows)


def longitudinal(root, out, ranks, bootstraps):
    import osa_independent_validation as endpoint
    base = root / BASE
    data = pd.read_csv(base / "34_longitudinal_validation_endpoint_v1/subject_node_responses.tsv",sep="\t")
    contract = read_json(base / "33_longitudinal_validation_design_v2/analysis_contract.json")
    graphs = {m:list(ranks.loc[ranks["rank_"+m]<=312,["protein_A","protein_B"]].itertuples(index=False,name=None)) for m in METHODS}
    scores, coverage = [], []
    for (contrast,participant), group in data.groupby(["contrast","subject_id"],sort=True):
        values = dict(zip(group.protein,group.signed_reversal))
        for m in METHODS:
            edges = [(a,b) for a,b in graphs[m] if a in values and b in values]
            nodes = set(p for e in edges for p in e)
            all_nodes = set(p for e in graphs[m] for p in e)
            valid = (len(edges)>=contract["minimum_edges"] and len(nodes)>=contract["minimum_nodes"]
                and len(edges)/float(len(graphs[m]))>=contract["minimum_edge_coverage"]
                and len(nodes)/float(len(all_nodes))>=contract["minimum_node_coverage"])
            coverage.append(dict(contrast=contrast,subject_id=participant,method=m,mapped_edges=len(edges),
                                 selected_edges=len(graphs[m]),evaluable=valid))
            if valid:
                scores.append(dict(contrast=contrast,subject_id=participant,method=m,
                                   score=endpoint.longitudinal_graph_score(np,edges,values)))
    save_table(out / "longitudinal_coverage.tsv",coverage)
    save_table(out / "longitudinal_subject_scores.tsv",scores)
    frame = pd.DataFrame(scores)
    tests = []; rng = np.random.RandomState(contract["seed"])
    for contrast,group in frame.groupby("contrast",sort=True):
        wide = group.pivot(index="subject_id",columns="method",values="score").reindex(columns=METHODS).dropna()
        for control in METHODS[1:]:
            row = dict(contrast=contrast,comparison="Full_minus_"+control,n_subjects=len(wide),status="NOT_EVALUABLE")
            if len(wide)>=contract["minimum_subjects"]:
                v = (wide.Full-wide[control]).to_numpy()
                effect,p = endpoint.longitudinal_signflip(np,v)
                indices = rng.randint(len(v),size=(bootstraps,len(v)))
                ci = np.quantile(v[indices].mean(axis=1),[.025,.975])
                row.update(status="EVALUATED",difference=effect,CI95_low=ci[0],CI95_high=ci[1],exact_signflip_P=p)
            tests.append(row)
    good = [r for r in tests if r["status"]=="EVALUATED"]
    for row,q in zip(good,bh([r["exact_signflip_P"] for r in good])):
        row["BH_FDR_longitudinal_family"] = q
    save_table(out / "longitudinal_paired_comparisons.tsv",tests)


def tissue_controls(root, out, ranks, iterations, methods=None):
    """Reuse the exact archived node evidence and matched-node endpoint family.

    These null quantiles measure randomization variability, not patient CIs.
    GSE38792 and the GSE135917 state view share participants; no pooled claim.
    """
    methods = _validated_external_methods(ranks, methods)
    legacy_methods = methods == METHODS
    import osa_cpap_reversal_endpoint as cp
    import osa_independent_validation as adult
    import osa_adipose_cross_tissue_endpoint as adipose
    base = root / BASE
    candidates = pd.read_csv(root/cp.CANDIDATE_REL,sep="\t").to_dict("records")
    source = pd.read_csv(base/"20_independent_validation_preprocess_v1/candidate_independent_evidence_500.tsv",sep="\t").fillna("").to_dict("records")
    adip = pd.read_csv(base/"13_adipose_validation_preprocess_v1/candidate_adipose_evidence.tsv",sep="\t").fillna("").to_dict("records")
    datasets = []
    for name in ("GSE38792","GSE293433"):
        ev,_ = adult.build_independent_node_evidence(np,candidates,source,name)
        datasets.append((name,ev))
    for name in ("state","state_adjusted","CPAP"):
        ev,_ = adipose.evidence_from_candidate(np,candidates,adip,name)
        datasets.append(("GSE135917_"+name,ev))
    ids = [r["UniProt_accession"] for r in candidates]
    ix = {p:i for i,p in enumerate(ids)}
    left = np.asarray([ix[p] for p in ranks.protein_A]); right = np.asarray([ix[p] for p in ranks.protein_B])
    rows = ranks.to_dict("records"); tests=[]; curves=[]
    for di,(name,ev) in enumerate(datasets):
        available = np.asarray([ev[p]["available"] for p in ids])
        score = np.asarray([ev[p]["score"] if ev[p]["available"] else 0. for p in ids])
        positive = np.asarray([bool(ev[p].get("same_direction",ev[p].get("positive",False))) for p in ids])
        eligible = available[left] & available[right]
        support = np.sqrt(score[left]*score[right])
        observed={}; null={}
        for method in methods:
            rr = ranks["rank_"+method].to_numpy()
            if any(not (eligible & (rr<=math.ceil(len(ranks)*f))).any() for f in cp.FRACTIONS):
                raise RuntimeError("No eligible top pairs: "+name+"/"+method)
            values,area,ratio = adult.observed_pair_metrics(np,rows,support,positive[left]&positive[right],eligible,"rank_"+method)
            curves.extend(dict(v,endpoint=name,method=method) for v in values)
            observed[method]=np.asarray([ratio,area])
            nr,na,blocks = cp.node_aware_permutation(np,np.random.RandomState(20260715+500+di),iterations,
                50,candidates,ev,rows,eligible,left,right,rr)
            null[method]=np.column_stack([nr,na])
            save_table(out/(name+"_matching_blocks.tsv"),blocks)
        pd.DataFrame({m+"_"+v:null[m][:,j] for m in methods for j,v in enumerate(("ratio","AUEC"))}).to_csv(
            out/(name+"_shared_node_null.tsv.gz"),sep="\t",index=False)
        for control in methods[1:]:
            for j,metric in enumerate(("top1_enrichment_ratio","normalized_AUEC")):
                delta=observed["Full"][j]-observed[control][j]
                nd=null["Full"][:,j]-null[control][:,j]
                tests.append(dict(endpoint=name,comparison="Full_minus_"+control,metric=metric,
                    Full=observed["Full"][j],control=observed[control][j],difference=delta,
                    two_sided_randomization_P=(1.+np.sum(np.abs(nd)>=abs(delta)))/(iterations+1.),
                    null_q025=np.quantile(nd,.025),null_q975=np.quantile(nd,.975),
                    interval_kind="matched-node null quantiles, not patient confidence interval"))
    for row,q in zip(tests,bh([r["two_sided_randomization_P"] for r in tests])):
        if legacy_methods:
            row["BH_FDR_tissue_family"]=q
        else:
            row["BH_FDR_tissue_comparisons"]=q
            row["BH_family_size"]=len(tests)
            row["common_reference"]="Full"
    save_table(out/"tissue_endpoint_curves.tsv",curves)
    save_table(out/"tissue_paired_node_randomization.tsv",tests)


def regression_check(root, out):
    """Fail closed if the retained Full no longer reproduces archived endpoints."""
    base = root / BASE
    checks = []
    fresh = pd.read_csv(out/"cpap_moderated_endpoint_curves.tsv",sep="\t")
    for cohort in ("GSE133601","GSE49800"):
        old = pd.read_csv(base/"11_cpap_reversal_endpoint_triage_osa_v4"/cohort/"ranking_endpoint_metrics.tsv",sep="\t")
        old = old[(old.model=="full_primary") & (old.metric_scope=="fraction")].sort_values("fraction")
        new = fresh[(fresh.cohort==cohort)&(fresh.method=="Full")].sort_values("fraction")
        ok = len(old)==len(new)==4 and np.allclose(old.support_enrichment_ratio,new.support_enrichment_ratio,atol=1e-8,rtol=0)
        checks.append(dict(endpoint=cohort,rows=len(new),pass_check=bool(ok)))
    fresh = pd.read_csv(out/"tissue_endpoint_curves.tsv",sep="\t")
    for stage, file, model, prefix in (
            ("14_adipose_cross_tissue_endpoint_triage_osa_v4","ranking_endpoint_metrics.tsv","full_primary","GSE135917_"),
            ("21_independent_validation_endpoint_triage_osa_v4","pair_support_metrics.tsv","full_TRIAGE_OSA","")):
        old = pd.read_csv(base/stage/file,sep="\t")
        old = old[(old.model==model)&(old.metric_scope=="fraction")]
        for endpoint, g in old.groupby("endpoint"):
            if endpoint in ("joint","joint_secondary"):
                continue
            name = "state_adjusted" if endpoint=="state_covariate_adjusted_sensitivity" else endpoint
            new = fresh[(fresh.endpoint==prefix+name)&(fresh.method=="Full")].sort_values("fraction")
            g=g.sort_values("fraction")
            ok=len(g)==len(new)==4 and np.allclose(g.support_enrichment_ratio,new.support_enrichment_ratio,atol=1e-8,rtol=0)
            checks.append(dict(endpoint=prefix+endpoint,rows=len(new),pass_check=bool(ok)))
    old = pd.read_csv(base/"34_longitudinal_validation_endpoint_v1/subject_graph_responses.tsv",sep="\t")
    old = old[(old.panel=="ALL")&(old.method=="full_TRIAGE_OSA")]
    new = pd.read_csv(out/"longitudinal_subject_scores.tsv",sep="\t")
    joined = new[new.method=="Full"].merge(old,on=["contrast","subject_id"],validate="one_to_one")
    ok = len(joined)==len(old)>0 and np.allclose(joined.score,joined.edge_response,atol=1e-12,rtol=0)
    checks.append(dict(endpoint="longitudinal_ALL",rows=len(joined),pass_check=bool(ok)))
    save_table(out/"full_endpoint_regression_checks.tsv",checks)
    if not all(row["pass_check"] for row in checks):
        raise RuntimeError("Full endpoint regression mismatch; do not interpret comparator results")


def run(args):
    root = Path(args.root).resolve()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/validation"))
    base = root / BASE
    comparison = getattr(args, 'comparison_ranking', None)
    ranking_source = None
    if comparison:
        ranking_source = (root / comparison).resolve()
        summary = read_json(ranking_source/'run_summary.json')
        expected = 'PASS_SMOKE' if args.smoke else 'PASS_CROSS_PARTICIPANT_EXECUTION'
        if summary['status'] != expected or summary.get('external_data_read') is not False:
            raise RuntimeError('Stage39 must finish in matching smoke/formal mode without external fitting')
        if sha(ranking_source/'rankings_primary_250.tsv.gz') != summary['ranking_sha256']:
            raise RuntimeError('Stage39 ranking provenance mismatch')
    required = [base/"09_pair_context_ranking_triage_osa_v4/pair_context_scores_triage_osa_500.tsv.gz",
                base/"09_pair_context_ranking_triage_osa_v4/run_summary.json",
                base/"31_osa_controlled_experiments_v1/osa_ablation_suite/experiment_contract.json",
                base/"04_candidates_v1/candidate_proteins_primary_250.tsv",
                base/"20_independent_validation_preprocess_v1/candidate_independent_evidence_500.tsv",
                base/"13_adipose_validation_preprocess_v1/candidate_adipose_evidence.tsv",
                base/"33_longitudinal_validation_design_v2/analysis_contract.json",
                base/"34_longitudinal_validation_endpoint_v1/subject_node_responses.tsv",
                base/"34_longitudinal_validation_endpoint_v1/subject_graph_responses.tsv",
                base/"14_adipose_cross_tissue_endpoint_triage_osa_v4/ranking_endpoint_metrics.tsv",
                base/"21_independent_validation_endpoint_triage_osa_v4/pair_support_metrics.tsv",
                root/"config/osa_cpap_validation_lock.json"]
    for cohort in ("GSE133601","GSE49800"):
        required += [base/"10_cpap_validation_preprocess_v1"/cohort/n for n in
                     ("candidate_validation_evidence.tsv","candidate_participant_deltas.tsv.gz")]
        required.append(base/"11_cpap_reversal_endpoint_triage_osa_v4"/cohort/"ranking_endpoint_metrics.tsv")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Missing existing inputs: "+str(missing))
    if args.check_env:
        import scipy
        import osa_cpap_reversal_endpoint
        import osa_cpap_subject_signflip
        import osa_independent_validation
        import osa_adipose_cross_tissue_endpoint
        print("OSA external strong controls environment: PASS; numpy={} pandas={} scipy={}".format(np.__version__,pd.__version__,scipy.__version__))
        print("CPU only; Full is not retrained; no new clinical cohort required")
        return 0
    iterations = 100 if args.smoke else int(args.external_control_permutations)
    bootstraps = 100 if args.smoke else int(args.external_control_bootstrap)
    if iterations<100 or bootstraps<100:
        raise RuntimeError("At least 100 repetitions required; formal defaults are 10000/5000")
    name = '38_external_strong_controls_crossparticipant_v1' if comparison else '38_external_strong_controls_v1'
    out = base / (name + ('_smoke' if args.smoke else ''))
    if out.exists() and any(out.iterdir()) and not args.force:
        raise RuntimeError("Output exists. --force reruns ONLY this Stage38 directory; original stages are never overwritten.")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"analysis_contract.json",dict(version=VERSION,methods=list(METHODS),posthoc=True,
        primary_focus="two adult CPAP cohorts, Full minus converged constrained static",
        secondary="Direct anchor comparisons; unmoderated participant bootstrap; six longitudinal contrasts",
        fractions=[.005,.01,.02,.05],permutations=iterations,bootstrap=bootstraps,
        no_external_model_selection=True,no_full_retraining=True,
        scope="adult CPAP, independent adult state, adipose state/adjusted/CPAP, longitudinal ALL top1pct",
        other_assays="scRNA/proteomics/known interaction recovery unchanged; not new clinical evidence"))
    write_json(out/"run_summary.json",dict(status="RUNNING",version=VERSION))
    if ranking_source is None:
        ranks = fit_rankings(root,out)
    else:
        ranks = pd.read_csv(ranking_source/'rankings_primary_250.tsv.gz',sep='\t')
        old_order = pd.read_csv(base/'38_external_strong_controls_v1/rankings_primary_250.tsv.gz',sep='\t')
        if ranks.pair_id.duplicated().any() or set(ranks.pair_id) != set(old_order.pair_id):
            raise RuntimeError('New and original primary universes differ')
        ranks = ranks.set_index('pair_id').loc[old_order.pair_id].reset_index()
        for field in ('protein_A','protein_B'):
            if not np.array_equal(ranks[field], old_order[field]):
                raise RuntimeError('Pair identity mismatch')
        for method in METHODS:
            if not np.isfinite(ranks[method]).all():
                raise RuntimeError('Nonfinite comparison scores')
            ranks['rank_'+method] = ranks[method].rank(method='first',ascending=False).astype(int)
        ranks.to_csv(out/'rankings_primary_250.tsv.gz',sep='\t',index=False)
        write_json(out/'ranking_provenance.json',dict(source=str(ranking_source),source_summary=summary,
            methods_meaning='Full=cross-participant ensemble; static and input anchor matched to same units',
            external_evaluation='previously inspected cohorts; exploratory re-evaluation',
            no_external_model_selection=True))
    print("[rankings] complete; now opening external outcomes",flush=True)
    cpap(root,out,ranks,iterations,bootstraps)
    tissue_controls(root,out,ranks,iterations)
    longitudinal(root,out,ranks,bootstraps)
    if ranking_source is None:
        regression_check(root,out)
    else:
        write_json(out/'legacy_regression_scope.json',dict(
            status='NOT_APPLICABLE_TO_NEW_RANKING',
            reason='New ranking cannot reproduce old Full endpoints; original Stage38 regression records retained'))
    save_table(out/"input_manifest.tsv",[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in required])
    write_json(out/"run_summary.json",dict(status="PASS_SMOKE" if args.smoke else "PASS_EXTERNAL_STRONG_CONTROLS",
        full_retrained=False,original_ranking_changed=False,static_fit_converged=True,
        pairs=31125,methods=list(METHODS),adult_tissue_endpoints_complete=True,
        interpretation="post hoc matched external controls; execution does not imply superiority"))
    print("External strong controls complete: "+str(out),flush=True)
    return 0
