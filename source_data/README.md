# Manuscript source data

Supporting data for **TRIAGE-OSA: structured multiscale evidence fusion for
protein-pair prioritization in obstructive sleep apnoea**.

## Downloads

| File | Approximate size | Main contents |
|---|---|---|
| [Supplementary_Data_1_Candidate_Evidence.zip](Supplementary_Data_1_Candidate_Evidence.zip) | 31.1 MiB | 31,125-pair evidence matrix, method rankings, 15 original candidate edges, three modules, rank stability, selection-rule comparisons, assay annotations and provenance. |
| [Supplementary_Data_2_Analysis_Outputs.zip](Supplementary_Data_2_Analysis_Outputs.zip) | 7.4 MiB | External strong-control comparisons, fusion-capacity comparisons, robustness analyses and structural-source records. |

Use Git clone or the file's **Download raw file** button to download an archive.
The GitHub page preview does not display the archive contents. SHA-256 digests
are listed in [SHA256SUMS.txt](SHA256SUMS.txt); [manifest.json](manifest.json)
also records archive sizes, original hashes and public-copy changes.

## Unpack

Run from the repository root:

```bash
python -m zipfile -e source_data/Supplementary_Data_1_Candidate_Evidence.zip outputs/source_data/candidate_evidence
python -m zipfile -e source_data/Supplementary_Data_2_Analysis_Outputs.zip outputs/source_data/analysis_outputs
```

The archives use different top-level layouts. Keep them in separate directories
as shown above; do not merge their files into a single flat folder.

## Files to start with

### Supplementary Data 1

- `01_complete_pair_evidence.tsv.gz`: complete pair-level evidence matrix.
- `02_original_pareto_pairs.tsv` and `03_original_modules.tsv`: original
  Pareto candidates and displayed modules.
- `14_nine_axis_data_dictionary.tsv`: evidence-axis definitions.
- `15_candidate_method_comparison.tsv` and `17_complete_method_rankings.tsv.gz`:
  candidate and full-universe method comparisons.
- `24_candidate_rule_review.tsv` and `25_complete_rule_membership.tsv.gz`:
  selection-rule sensitivity.
- `38_candidate_evidence_status.tsv` through
  `41_candidate_context_evidence_status.tsv`: evidence-assessment rules,
  node statistics and candidate-context annotations.
- `README.md`, `README_rule_sensitivity.md` and `rule_sensitivity_protocol.md`:
  archived interpretation and selection protocol.

### Supplementary Data 2

- `external_strong_controls/`: Full, converged constrained static fusion and
  direct-anchor comparisons under matched external endpoints.
- `Fusion_Control_Source_Data/`: matched-capacity and wider-MLP controls,
  parameter counts, tuning choices, fold/seed metrics and external comparisons.
- `public_evidence_closeout/`: candidate stability and evidence-sensitivity
  outputs. Stage identifiers identify historical analyses, not figure numbers.
- `structure_provenance/`: sequence, score-index and coordinate checks.
- The remaining structural records include full residue profiles and archived
  native arrays; follow their included README and provenance files. Sparse L2
  support is not a complete experimental contact map.

TSV files are tab-delimited UTF-8. Read compressed tables directly with
`pandas.read_csv(path, sep="\t")`; do not manually edit large matrices.
Interpret missingness, signs, coverage and support using the archived data
dictionaries rather than treating every absent value as zero.

## Regenerate the fusion-control figure without training

After extracting Supplementary Data 2 and installing `requirements.txt`:

```bash
python run.py plot fusion-capacity --results outputs/source_data/analysis_outputs/Fusion_Control_Source_Data
```

This reads archived paired fold/seed contrasts and writes PDF, SVG, 600-dpi
PNG and source tables under `outputs/figures/fusion_capacity/`. It does not
retrain models or rerun external statistics. Historical `FigS5` filenames in
the archive are preserved source identifiers, not the current manuscript
figure numbering. The small candidate example also runs without a checkpoint:

```bash
python run.py example
```

## Public-copy changes and provenance

The original submission-source archives were retained by the authors.
Public copies replace local disk and server prefixes with logical source
aliases and translate the structural-source README into English. No numerical
results, rankings, compressed matrices or native arrays were modified.
`PUBLIC_RELEASE_NOTES.md` explains the aliases. `PUBLIC_FILE_MANIFEST.tsv`
inside each archive records every original/public member hash and any change.
Historical provenance hashes continue to describe the original files; use
the public manifest to verify distributed metadata files.

These are derived research outputs, not new patient records or clinical
accuracy estimates. Reused source studies and access routes are listed in
the repository README. Original third-party terms continue to apply.
