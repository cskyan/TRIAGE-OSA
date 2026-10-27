# Inputs not bundled with the code

## Data workspace

Maintain the original relative stage names inside a chosen workspace:

```text
config/                         Analysis contracts from this repository
metadata/                       Sample, assay and participant manifests
geo/                            GEO matrices, raw archives and study metadata
annotations/                    Platform and identifier mapping files
raw/                            Deposited study files, as expected by each stage
resources/                      ESM, reference and backbone assets
results/discovery_preprocess_v2/ Discovery and external analysis outputs
pathway_module_layer/genesets/   Locked GMT files
```

The scripts and their contracts define the exact filenames; no personal machine prefix is required. CLI `--root` locates a workspace; it does not generate missing metadata or copy configurations. Check each preprocessing script's help for its raw-file layout.

## Model resources

- The ESM model file and protein embeddings required by sequence preparation and molecular inference.
- A checkpoint compatible with `model/triage_server_candidate/`.
- The OSA starting checkpoint at the relative path stated in `config/triage_osa_training_lock_v4.json`. Its archived SHA-256 identity check is retained.
- Complete pair inference tables for the intended candidate universe, followed by the OSA checkpoint for prediction.

The default resources directories are portable placeholders, not distributed weights. A file with the same name but a different training run is not interchangeable with the archived checkpoint. Serialized checkpoint configuration may itself contain legacy paths; supply the current resource paths using the native supported overrides and audit loading before training. Do not edit the identity checks merely to make an unrelated checkpoint load.

## Clinical and public-data metadata

The workflow requires sample-to-participant mappings, phenotype fields, pairing/timepoint labels and availability indicators. These are not included in the Git-ready copy. Public accession links, cohort roles and molecular resource links are listed in the [README data-source inventory](../README.md#data-sources-and-download-links). File-level GEO URLs are recorded in the source download scripts and analysis locks. Dataset availability does not by itself reproduce the exact analysis sample manifest.

RBP400 and RBP400-Expanded share the authors' Zenodo deposit at https://doi.org/10.5281/zenodo.21320101. They are related molecular resources, not clinical OSA cohorts. Select the appropriate resource and split records from that deposit; the presence of an inherited dataset loader does not demonstrate its use in the study's checkpoint.

For treatment and tissue analyses, keep the participant pairing, shared-cohort identifiers and assay restrictions used in the archived protocol. Do not replace paired designs with independent sample comparisons.

## Stage-specific derived inputs

Stages 36 and 38 require prepared external matrices and ranking tables, not only the OSA checkpoint. Stage 38 also checks agreement with existing Full endpoint outputs. Ranking stability needs all relevant per-run predictions. Pathway analyses need the locked reference gene sets. Figure builders require their result-stage TSV/CSV files.

## Before a public release

Supply stable access locations for weights and permitted derived data, the checkpoint's training run and split manifests, and an exact tested environment. These items should be verified and linked, not inferred from a filename. No raw clinical record or large checkpoint is copied into this staging repository.
