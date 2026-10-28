# TRIAGE-OSA

**TRIAGE-OSA prioritizes protein pairs in obstructive sleep apnoea through multiscale evidence integration**

TRIAGE-OSA integrates multiscale protein-pair outputs with OSA-associated expression evidence to prioritize candidate pairs. This repository contains the molecular model, disease-specialization code, controlled experiments, external evidence analyses, and plotting workflows.

The study concerns disease-conditioned prioritization and traceable candidate evidence. Its outputs are not diagnostic predictions, established physical interactions, or validated treatment-response biomarkers.

## Repository layout

```text
run.py                         Unified command-line entry point
model/
  triage_server_candidate/      Checkpoint-compatible molecular backbone
  osa_adapter/                 OSA context, specialization, inference and controls
  osa_public_multimodal/       Additional participant-profile analyses
scripts/
  data/                        Public-data downloads and manifest preparation
  discovery/                   Discovery QC, statistics, mapping and rankings
  validation/                  Treatment, tissue, cell and proteomic evidence
  candidates/                  Candidate stability, sensitivity and pathway analyses
  checks/                      Structural-source checks
  experiments/                 Linux launcher for the fusion-capacity controls
config/                        Versioned analysis and training contracts
pathway_module_layer/          Discovery pathway and module analyses
plotting/
  figures/                     All statistical figure builders
  data/                        Small archived aggregate figure tables
examples/candidate_evidence/   Complete 15-edge candidate example
tests/                         Regression tests using explicit test fixtures
tools/                         Repository integrity checks
source_data/                   Candidate evidence, analysis outputs and file guide
```

## Quick start: one complete example

Use a Python environment with NumPy, pandas, SciPy and Matplotlib. Python 3.10 or later is recommended for this portable snapshot; this is not a claim that historical experiments used the same environment.

```bash
python -m pip install -r requirements.txt
python run.py example
```

This reads the archived ranks of **all 15 original candidate edges (17 proteins)** under Full, converged constrained static fusion, direct anchor, and PPI-only prioritization. It joins evidence-rule membership and exports:

- `outputs/candidate_evidence/candidate_summary.tsv`
- `outputs/candidate_evidence/candidate_ranks.pdf`, `.svg` and `.png`
- `outputs/candidate_evidence/run_summary.json`

The example does not require a GPU, checkpoint, raw expression data, or patient records. It does not retrain a model or reconstruct the complete candidate-selection pipeline. See [the example description](examples/candidate_evidence/README.md) for interpretation and data provenance.

## One entry point

```bash
python run.py list
python run.py train --help
python run.py predict --help
python run.py experiment external-controls --help
python run.py stage cpap-endpoint --help
python run.py --dry-run experiment cross-participant --device cuda:0
python run.py plot benchmark
python run.py plot candidate
python run.py audit
```

The launcher always uses the active Python interpreter. It does not activate another environment, select an unspecified physical GPU, or download dependencies automatically. Global launcher options go **before** the command. Arguments after a workflow name are passed to its original script.

Model training and inference additionally require PyTorch and ESM:

```bash
python -m pip install -r requirements-model.txt
python run.py test
```

Three longitudinal integration checks need the archived cohort metadata/design and report an explicit skip when those inputs are absent. To include them, use `python run.py --data-root /path/to/data_workspace test`. The remaining tests are self-contained.

Choose the PyTorch build appropriate for your platform. CUDA inference requires a compatible driver and CUDA-enabled PyTorch build. Dependency ranges above are installation requirements, not an exact historical environment lock.

## Data and weights

The two manuscript source-data archives are included under [`source_data/`](source_data/README.md). They contain derived evidence tables, analysis outputs and structural-source records. Raw clinical records, raw cohort expression matrices, trained weights and ESM weights are not bundled. Public accession identifiers and relative resource names remain in the scripts and contracts because they are needed for reproducibility.

| Archive | Contents |
|---|---|
| [Candidate evidence](source_data/Candidate_evidence.zip) | Complete 31,125-pair evidence matrix; rankings and stability; original candidates and modules; alternative selection rules; per-candidate assay evidence and source records. |
| [Analysis outputs](source_data/Analysis_outputs.zip) | Matched external controls; fusion-capacity controls; candidate-stability and public-evidence analyses; complete archived L1 profiles, sparse L2 support and structural mappings. |

The [source-data guide](source_data/README.md) describes the experiment groups, extraction and how to regenerate the fusion-control plot directly from archived results. The archives organize derived tables and arrays by analysis and complement the Supplementary Information. They do not contain model checkpoints.

The default data workspace is the repository root, with ignored `geo/`, `annotations/`, `metadata/`, `raw/`, `resources/`, and `results/` directories. An existing data workspace can also be used:

```bash
python run.py --data-root /path/to/data_workspace train --device cuda:0
python run.py --data-root /path/to/data_workspace experiment external-controls
python run.py --data-root /path/to/data_workspace plot published
```

Replace the example path with your own directory. It must include the relevant `config/`, `metadata/` and result-stage inputs; `--data-root` does not copy them. For individual preprocessing stages, pass their native `--root` option after the stage name where supported. Inspect `--help` first. Legacy plot builders use their original repository-relative layouts.

The backbone's portable default assets live below `resources/backbone/`, benchmark resources below `resources/benchmark_data/`, and ESM weights below `resources/esm/`. Explicit native CLI options override supported defaults. See [required inputs](#required-inputs).

## Data sources and download links

Choose the resources for the workflow you intend to run. The bundled candidate example and aggregate benchmark plots need no additional downloads. Reproducing the OSA analyses requires the relevant public cohorts, sample manifests and derived model outputs. Reconstructing molecular training additionally requires the training resources, split manifests and checkpoint provenance; downloading the public source alone does not reconstruct those processed inputs.

### Molecular resources

| Resource | Public access | Role and preparation |
|---|---|---|
| RBP400 and RBP400-Expanded | [Shared Zenodo archive, DOI: 10.5281/zenodo.21320101](https://doi.org/10.5281/zenodo.21320101) | Both resources are referenced under this DOI in the authors' manuscript records. They are related molecular-model resources, not OSA cohorts. Use the resource-specific files and split/provenance records from the deposit. Their availability does not establish that both were used to train the OSA backbone checkpoint. |
| Protein-Protein Docking Benchmark 5.5 (DB5.5) | [Official benchmark and structural files](https://zlab.wenglab.org/benchmark/benchmark5.5.html) | Structural training-source resource. Reproducing the study also requires the actual complex/chain selection, residue mapping, processed labels and training splits associated with the checkpoint. |
| Bernett sequence-based PPI gold-standard dataset | [Figshare, DOI: 10.6084/m9.figshare.21591618](https://figshare.com/articles/dataset/PPI_prediction_from_sequence_gold_standard_dataset/21591618) | Generic protein-pair benchmark. Preserve the released Intra1/Intra0/Intra2 train/validation/test assignments and record the version actually used. The landing page's latest version must not be assumed to be the historical experiment version. |
| ESM-2 sequence model | [Official ESM repository and model downloads](https://github.com/facebookresearch/esm) | Sequence preparation uses `esm2_t33_650M_UR50D` (`esm2_t33_650M_UR50D.pt`). These are sequence-model weights, not the molecular interaction checkpoint or the OSA specialization checkpoint. |

The shared RBP400 archive identifier above is taken from the authors' existing data-availability records. Its online file inventory was not independently verified for this repository snapshot; archive filenames and unverified licensing terms are therefore not specified here. Other dataset modes supported by the inherited molecular toolkit are not automatically part of the OSA study's training or evaluation protocol.

### OSA discovery and external evidence

Download study files from the linked accession pages. Use the scripts' expected filenames and the analysis contracts to select the correct processed matrix, raw assay files, annotations and metadata; these file types are not interchangeable.

| Accession | Material/context | Role in this repository |
|---|---|---|
| [GSE75097](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75097) | Adult peripheral blood mononuclear cells (PBMCs) | Discovery associations with OSA status, severity and AHI, using the study's selected untreated participants and phenotype manifest. |
| [GSE71356](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71356) | PBMC intermittent-hypoxia crossover experiment | Discovery intermittent-hypoxia context; preserve participant identity and exposure pairing. |
| [GSE133601](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133601) | Adult PBMCs before and after CPAP | External paired treatment-associated molecular reversal and matched method comparisons. |
| [GSE49800](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49800) | Adult peripheral blood leukocytes before and after CPAP | Second external paired treatment-associated molecular reversal cohort. |
| [GSE135917](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135917) | Subcutaneous adipose tissue | External disease-state and treatment views; retain the distinction between its case-control and CPAP study groups. |
| [GSE38792](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE38792) | Visceral adipose tissue | External tissue context. This view shares participants with Study Group 1 of GSE135917; the two tissue views are not two independent population replications. |
| [GSE293433](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293433) | Adult PBMC transcriptomics | External adult disease-state/pathway evidence; the preparation code uses the deposited processed mRNA workbook. |
| [GSE214865](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214865) | Pediatric PBMC single-cell transcriptomics | External cell-type context. Preserve participant-level replication rather than treating cells as independent participants. |
| [GSE283703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283703) | Longitudinal blood transcriptomics during CPAP follow-up | Longitudinal molecular/module analyses; retain participant, visit and sampling-time assignments from the archived design. |
| [PXD032734](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD032734) | Pediatric serum proteomics | External proteomic coverage/support. Use the deposited study and associated supplementary tables; this is not an adult paired CPAP cohort. |

These accessions supply different evidence types, not one pooled training cohort. External evidence is used for assessment and exploratory shortlisting under the documented protocol; it is not a replacement for discovery training data. Tissue, age group, participant overlap and pairing remain part of the interpretation.

The GEO downloader contains a file-level catalog with public URLs and relative destinations. Inspect it without downloading:

```bash
python run.py stage downloads --root . --list
python run.py stage downloads --help
```

The catalog is in `scripts/data/osa_geo_server_pipeline.py`; longitudinal GSE283703 downloads are implemented separately in `scripts/validation/osa_independent_validation.py`. Proteomic evidence follows its own preparation workflow. The catalog is not a single command that prepares all study inputs. Most GEO files are placed under `geo/` and platform annotations under `annotations/`.

### Identifier, pathway and interaction references

| Resource | Access | Use |
|---|---|---|
| UniProtKB | [UniProt](https://www.uniprot.org/) | Reviewed human protein identifiers and sequences. Preserve the release and checksums in `config/identifier_mapping_lock.json`. |
| HGNC | [HGNC download archives](https://www.genenames.org/download/archive/) | Gene-symbol and identifier harmonization using the recorded snapshot. |
| Reactome | [Reactome downloads](https://reactome.org/download-data) | Human pathway gene sets for pathway/module analyses; retain the archived GMT version and checksum. |
| IntAct | [IntAct interaction database](https://www.ebi.ac.uk/intact/) | Curated interaction-reference evidence, using the saved queries and response records. |
| Complex Portal | [Complex Portal](https://www.ebi.ac.uk/complexportal/) | Curated human complex-membership evidence. The access records are specified with IntAct in `config/osa_known_ppi_complex_lock.json`. |
| PDB and SIFTS | [RCSB PDB](https://www.rcsb.org/), [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) | Structural examples and structure-to-sequence residue correspondence. Mapping requires the actual inference sequence and saved score indices, not only a matching protein name. |

Public databases may change. Keep the downloaded release, retrieval date and checksum together with each analysis manifest rather than silently replacing archived references with the current release. Original data remain subject to their providers' terms. The GEO/ProteomeXchange accessions above identify study sources; exact sample selections, pairing tables, processed inputs and model weights are also required for reproduction.

## Required inputs

Keep these relative paths within the selected data workspace:

```text
config/                          Versioned contracts from this repository
metadata/                        Sample, assay, participant and pairing manifests
geo/                             GEO matrices, raw archives and study metadata
annotations/                     Platform and identifier mapping files
raw/                             Deposited files expected by each analysis
resources/                       ESM, reference and backbone assets
results/discovery_preprocess_v2/  Discovery and external analysis outputs
pathway_module_layer/genesets/    Versioned pathway GMT files
```

`--root` locates the workspace; it does not create missing metadata or copy configurations. Use each stage's `--help` and its analysis contract for the required files. Sample-to-participant mappings, phenotype fields, availability indicators and treatment timepoints must match the archived cohort selection. Preserve paired designs and the shared-participant identifiers of the two adipose tissue views.

Molecular inference needs ESM weights/embeddings and a checkpoint compatible with `model/triage_server_candidate/`. OSA training additionally needs the starting checkpoint identified in `config/triage_osa_training_lock_v4.json` and complete molecular pair outputs. The resource paths are placeholders, not distributed weights; checkpoint identity checks remain enabled. Use supported resource overrides when loading checkpoints whose serialized configuration contains old paths.

The two source-data archives contain derived outputs, not all training inputs. External comparisons require prepared cohort matrices and complete rankings; stability analyses require the per-run predictions; pathway analyses require the recorded gene-set snapshot. Weight download links are not included in this release.

## Main experiments

| Command | Purpose |
|---|---|
| `experiment ablations` | Retrained component variants and fusion controls |
| `experiment strong-controls` | Supplementary control analyses, including converged fusion and L2 support perturbations |
| `experiment external-controls` | Matched external comparison of Full, constrained static fusion and direct anchor (Stage 38) |
| `experiment published-comparison` | Common-protocol comparison with published OSA signatures (Stage 36) |
| `experiment cross-participant` | Cross-participant adaptation and associated ranking outputs (Stage 39) |
| `stage fusion-capacity` | Equal-input, parameter-matched, same-start and wider-MLP fusion controls (Stage 50) |
| `stage ranking-stability` | Candidate ranking and shortlist stability |
| `stage shortlist-sensitivity` | Sensitivity to evidence-selection rules |
| `stage cpap-endpoint` | Paired treatment-associated molecular reversal |

Run discovery processing in this order: downloads and manifests, QC, differential analysis, identifier mapping, candidate-universe selection, sequence/model-input preparation, both-order pair inference, and OSA training/prediction. Generate complete discovery rankings before external assessment. Stage names are available with `python run.py list`; native arguments are available with `python run.py stage NAME --help`.

The common external protocol compares Full, constrained static fusion and direct anchor separately from the published-signature comparison. Cross-participant outputs can be evaluated with `--comparison-ranking results/discovery_preprocess_v2/39_cross_participant_adaptation_v1`; the published-signature comparison also accepts `--comparison-method Full`. These commands require the completed run and its manifests, not merely a directory with that name.

Some archived ablation values are aggregate result summaries. The supported trainer variants define executable recipes; a table alone does not establish the availability of the original training implementation. No launcher command silently starts every experiment.

### Fusion-capacity controls

Stage 50 compares equal-input, parameter-matched, same-start and wider MLP controls. The common-objective `Full` is distinct from the original-objective `Native_full` reference. Model selection uses discovery validation; external evaluation is separate.

```bash
python run.py stage fusion-capacity --check-env --device cpu
python run.py stage fusion-capacity --smoke --device cpu
python run.py stage fusion-capacity --device cuda:0 --resume
python run.py stage fusion-capacity --external-only --resume
```

These commands require the archived Stage 31 fold inputs/checkpoints, Stage 38 ranking background and expression inputs specified in `config/osa_fusion_capacity_control_v1.json`. Smoke mode checks execution and is not a formal experiment. `--resume` verifies existing contracts and output fingerprints. Place a stage's `--root` after its name when using a separate data workspace.

On Linux, use `bash scripts/experiments/run_osa_fusion_capacity_controls.sh check`, `smoke` or `formal`. It defaults to CPU; set `OSA_DEVICE=cuda:0` and `CUDA_VISIBLE_DEVICES` explicitly for GPU use. `OSA_PYTHON` selects the interpreter. The launcher records progress in `logs/`.

## Figures

`python run.py plot benchmark` regenerates the benchmark and ablation figures from bundled aggregate tables. It retains the current **Combined molecular weight** label. `python run.py example` produces the independent candidate-rank example.

All plotting code is collected under `plotting/figures/` and selected through `python run.py plot NAME`:

| Name | Required inputs |
|---|---|
| `candidate` | Bundled 15-edge candidate example |
| `benchmark` | Bundled benchmark/ablation JSON and control TSV |
| `fusion-capacity` | `fusion_controls/` from `Analysis_outputs.zip` |
| `published` | Archived Stage 36 v2 comparisons |
| `bio` | Biological-analysis results and metadata |
| `coverage` | Cohort metadata and coverage outputs |
| `supp` | Supplementary control/evidence results |
| `v4`, `legacy` | Earlier versioned result trees |

For extraction and plotting commands, see the [source-data guide](source_data/README.md). Plotting reads completed results without retraining. Original figure IDs can precede final manuscript renumbering or manual layout revisions. Framework slides and PyMOL sessions are not reconstructed by these statistical builders. The candidate example exports PDF, SVG and 600-dpi PNG.

## Reproducibility and release status

This repository packages the study's code and archived supporting outputs. Model calculations, parameter keys, archived numerical values and checkpoint identity checks are retained. Long-running training and cohort analyses require the external inputs listed above.

`config/source_checksums.json` records source checksums. `model/triage_server_candidate/source_manifest.json` distinguishes the original runtime hashes from the portable copy. Unit tests use test fixtures; their passing does not establish clinical validity or reproduce the full training runs.

The repository uses the [Apache-2.0 license](LICENSE). External datasets and resources remain subject to their providers' terms. The included CPU example and source-data plotting workflow do not require checkpoints.
