# Manuscript source data

Supporting data for **TRIAGE-OSA prioritizes protein pairs in obstructive sleep apnoea through multiscale evidence integration**.

## Downloads

| File | Main contents |
|---|---|
| [Candidate_evidence.zip](Candidate_evidence.zip) | Complete 31,125-pair evidence matrix, method rankings, 15 original candidate edges, three modules, rank stability, selection-rule comparisons, assay annotations and source records. |
| [Analysis_outputs.zip](Analysis_outputs.zip) | External matched-control comparisons, fusion-capacity comparisons, robustness analyses and structural-source records. |

Use Git clone or the file's **Download raw file** button to download an archive.
The GitHub page preview does not display the archive contents.

## Unpack

Run from the repository root:

```bash
python -m zipfile -e source_data/Candidate_evidence.zip outputs/source_data/candidate_evidence
python -m zipfile -e source_data/Analysis_outputs.zip outputs/source_data/analysis_outputs
```

Keep the archives in separate directories as shown above. Both contain a
`fusion_controls/` group with different analysis outputs.

## Files to start with

### Candidate evidence

- `candidate_network/complete_pair_evidence.tsv.gz`: complete pair-level evidence matrix.
- `candidate_network/original_pareto_pairs.tsv` and
  `candidate_network/original_modules.tsv`: original Pareto candidates and displayed modules.
- `candidate_network/nine_axis_data_dictionary.tsv`: evidence-axis definitions.
- `fusion_controls/candidate_method_comparison.tsv` and
  `fusion_controls/complete_method_rankings.tsv.gz`: candidate and full-universe
  method comparisons.
- `ranking_stability/`: adapter, seed and partition comparisons and complete
  rank-stability results.
- `support_rule_sensitivity/`: alternative selection rules, pair membership,
  endpoint-direction scores and the selection protocol.
- `tissue_and_treatment/`: candidate expression states, gene statistics,
  context-specific pair evidence and processing records.
- `protein_localization/`: candidate localization annotations and source
  UniProt records.
- `molecular_backbone/`: molecular-model training and feature-adaptation records.
- `data_description.md`: archive overview and interpretation.

### Analysis outputs

- `external_assay_controls/`: Full, converged constrained static fusion and
  direct-anchor comparisons under matched external endpoints.
- `fusion_controls/`: matched-capacity and wider-MLP controls, parameter counts,
  tuning choices, fold/seed metrics and external comparisons.
- `candidate_evidence_sensitivity/`: sensitivity analyses of candidate evidence.
- `ranking_stability/`: complete candidate-ranking stability analyses.
- `published_signature_checks/`: published-signature evaluation and metric checks.
- `candidate_pathway_analysis/`: candidate-to-pathway evidence.
- `structure_mapping_validation/`: sequence, score-index and coordinate checks.
- `structure_case_studies/`: complete residue profiles, sparse L2 support,
  native arrays and source records for the two structural illustrations.

TSV files are tab-delimited UTF-8. Read compressed tables directly with
`pandas.read_csv(path, sep="\t")`. Interpret missingness, signs, coverage and
support using the included data dictionaries. Sparse L2 support is not a
complete experimental contact map.

## Regenerate the fusion-control figure without training

After extracting `Analysis_outputs.zip` and installing `requirements.txt`:

```bash
python run.py plot fusion-capacity --results outputs/source_data/analysis_outputs/fusion_controls
```

This reads archived paired fold/seed contrasts and writes PDF, SVG, 600-dpi
PNG and source tables under `outputs/figures/fusion_capacity/`. It does not
retrain models or rerun external statistics. The small candidate example also
runs without a checkpoint:

```bash
python run.py example
```

These are derived research outputs, not new patient records or clinical
accuracy estimates. Reused source studies and access routes are listed in
the repository README. Original third-party terms continue to apply.
