# TRIAGE-OSA

**TRIAGE-OSA: structured multiscale evidence fusion for protein-pair prioritization in obstructive sleep apnoea**

TRIAGE-OSA integrates multiscale protein-pair outputs with OSA-associated expression evidence to prioritize candidate pairs. This repository contains the molecular model, disease-specialization code, controlled experiments, external evidence analyses, and plotting workflows.

The study concerns disease-conditioned prioritization and traceable candidate evidence. Its outputs are not diagnostic predictions, established physical interactions, or validated treatment-response biomarkers.

## Repository layout

```text
run.py                         Unified command-line entry point
model/
  triage_server_candidate/      Checkpoint-compatible molecular backbone
  osa_adapter/                 OSA context, specialization, inference and controls
  osa_public_multimodal/       Additional participant-profile analyses
scripts/                       Preprocessing and external evidence analyses
config/                        Versioned analysis and training contracts
pathway_module_layer/          Discovery pathway and module analyses
plotting/
  manuscript/                  Benchmark, biological and supplementary plot builders
  data/                        Small archived aggregate figure tables
  qa/                          Figure alignment checker
examples/candidate_evidence/   Complete 15-edge candidate example
tests/                         Regression tests using explicit test fixtures
tools/                         Repository integrity checks
docs/                          Workflow, input and provenance documentation
source_data/                   Supplementary Data 1 and 2, checksums and file guide
third_party/                   Notices for the bundled QA helper
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

Choose the PyTorch build appropriate for your platform. CUDA inference requires a compatible driver and CUDA-enabled PyTorch build. Dependency ranges above are installation requirements, not an exact historical environment lock. The local verification environment is recorded in `docs/verification.json` after testing.

## Data and weights

The two manuscript source-data archives are included under [`source_data/`](source_data/README.md). They contain derived evidence tables, analysis outputs and structural-source records. Raw clinical records, raw cohort expression matrices, trained weights and ESM weights are not bundled. Public accession identifiers and relative resource names remain in the scripts and contracts because they are needed for reproducibility.

| Archive | Contents |
|---|---|
| [Supplementary Data 1: Candidate Evidence](source_data/Supplementary_Data_1_Candidate_Evidence.zip) | Complete 31,125-pair evidence matrix; rankings and stability; original candidates and modules; alternative selection rules; per-candidate assay evidence and source records. |
| [Supplementary Data 2: Analysis Outputs](source_data/Supplementary_Data_2_Analysis_Outputs.zip) | Matched external controls; fusion-capacity controls; candidate-stability and public-evidence analyses; complete archived L1 profiles, sparse L2 support and structural mappings. |

The [source-data guide](source_data/README.md) describes extraction, checksums and how to regenerate the fusion-control plot directly from archived results. Public copies normalize machine-specific provenance paths and translate one README; numerical tables and arrays are unchanged. The in-archive manifests distinguish original and distributed hashes. These archives complement the Supplementary Information; they do not contain model checkpoints.

The default data workspace is the repository root, with ignored `geo/`, `annotations/`, `metadata/`, `raw/`, `resources/`, and `results/` directories. An existing data workspace can also be used:

```bash
python run.py --data-root /path/to/data_workspace train --device cuda:0
python run.py --data-root /path/to/data_workspace experiment external-controls
python run.py --data-root /path/to/data_workspace plot published
```

Replace the example path with your own directory. It must include the relevant `config/`, `metadata/` and result-stage inputs; `--data-root` does not copy them. For individual preprocessing stages, pass their native `--root` option after the stage name where supported. Inspect `--help` first. Legacy plot builders use their original repository-relative layouts.

The backbone's portable default assets live below `resources/backbone/`, benchmark resources below `resources/benchmark_data/`, and ESM weights below `resources/esm/`. Explicit native CLI options override supported defaults. See [required inputs](docs/INPUTS.md).

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

The catalog is in `scripts/osa_geo_server_pipeline.py`; longitudinal GSE283703 downloads are implemented separately in `scripts/osa_independent_validation.py`. Proteomic evidence follows its own preparation workflow. The catalog is not a single command that prepares all study inputs. Most GEO files are placed under `geo/` and platform annotations under `annotations/`.

### Identifier, pathway and interaction references

| Resource | Access | Use |
|---|---|---|
| UniProtKB | [UniProt](https://www.uniprot.org/) | Reviewed human protein identifiers and sequences. Preserve the release and checksums in `config/identifier_mapping_lock.json`. |
| HGNC | [HGNC download archives](https://www.genenames.org/download/archive/) | Gene-symbol and identifier harmonization using the recorded snapshot. |
| Reactome | [Reactome downloads](https://reactome.org/download-data) | Human pathway gene sets for pathway/module analyses; retain the archived GMT version and checksum. |
| IntAct | [IntAct interaction database](https://www.ebi.ac.uk/intact/) | Curated interaction-reference evidence, using the saved queries and response records. |
| Complex Portal | [Complex Portal](https://www.ebi.ac.uk/complexportal/) | Curated human complex-membership evidence. The access records are specified with IntAct in `config/osa_known_ppi_complex_lock.json`. |
| PDB and SIFTS | [RCSB PDB](https://www.rcsb.org/), [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) | Structural examples and structure-to-sequence residue correspondence. Mapping requires the actual inference sequence and saved score indices, not only a matching protein name. |

Public databases may change. Keep the downloaded release, retrieval date and checksum together with each analysis manifest rather than silently replacing archived references with the current release. Original data remain subject to their providers' terms. The GEO/ProteomeXchange accessions above identify study sources; the sample selections, pairing tables, processed inputs and model weights needed for exact reproduction are separate deliverables described in [INPUTS.md](docs/INPUTS.md).

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

The workflow order and input dependencies are described in [WORKFLOWS.md](docs/WORKFLOWS.md). These commands are not a request to rerun every experiment automatically. The launcher has no all-experiments command that could silently start long jobs.

## Figures

`python run.py plot benchmark` regenerates the benchmark and ablation figures from bundled aggregate tables. It retains the current **Combined molecular weight** label. `python run.py example` produces the independent candidate-rank example.

Other plot builders require their archived result tables. They are included as source workflows, not as a promise that all manuscript panels can be recreated from this small repository alone. Original figure numbers may reflect the figure-builder version rather than the final manuscript numbering. Hand-drawn framework diagrams and PyMOL sessions are not reconstructed by these statistical plotting scripts. See [PLOTTING.md](docs/PLOTTING.md).

## Reproducibility and release status

This repository packages the study's code and archived supporting outputs. Machine-specific defaults and non-English commentary were cleaned in a release copy; model tensor calculations, parameter keys, archived numerical values and existing checkpoint identity checks were retained. The original research workspace was not modified. Long-running training and cohort analyses require the external inputs listed in [INPUTS.md](docs/INPUTS.md).

`docs/source_manifest.json` records source checksums. `model/triage_server_candidate/source_manifest.json` distinguishes the original runtime hashes from the portable copy. Unit tests use test fixtures; their passing does not establish clinical validity or reproduce the full training runs.

The existing repository [Apache-2.0 license](LICENSE) is retained. Third-party resources remain subject to their original terms; the bundled alignment helper has its notice in `third_party/`. No publication DOI is assigned here. Checkpoint download links are not provided in this snapshot; the included CPU example and source-data plotting workflow do not require checkpoints.
