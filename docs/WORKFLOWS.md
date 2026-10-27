# Workflow guide

## 1. Molecular model and input preparation

`model/triage_server_candidate/` contains the checkpoint-compatible molecular model and its original training/inference entry points. The directory name is retained to preserve runtime selection and import compatibility; it is not a server address.

`model/osa_adapter/` contains sequence/input preparation, unordered pair registry construction, both-order inference, disease context features, training, prediction and experiment implementations. The obsolete alternate backbone directory is not included. The earlier ESM-only adaptation training script is not included as an active recipe; its resulting checkpoint is a required external asset for the current OSA workflow.

Typical discovery order:

1. Acquire the public datasets and create the sample/assay manifests.
2. Run discovery QC, differential analysis and identifier mapping.
3. Lock the candidate universe and prepare sequence/embedding inputs.
4. Run both-order molecular pair inference.
5. Fit the OSA specialization using the selected versioned training contract.
6. Produce complete pair rankings before consulting external evidence.

Use `python run.py list` to find each stage and `python run.py stage NAME --help` for its native arguments. Some auxiliary scripts retain their original input-directory contracts. Do not silently mix output stages from different candidate universes or training versions.

## 2. Controlled experiments

```bash
python run.py experiment ablations --device cuda:0
python run.py experiment strong-controls --device cuda:0
python run.py experiment audit
```

These use the existing trainer's experiment flags, not a new experimental protocol. A configured data workspace, checkpoints and model outputs are prerequisites. Check `--help`, `--variants`, `--seeds`, `--experiment-dir` and the selected lock before running. Supplement completion may include L2 inference and should not be assumed to be CPU-only.

Some aggregate ablation values were supplied as completed-result summaries. A plotting table is not proof that a corresponding original implementation is present in this repository. Only the actual supported trainer variants should be treated as executable reproduction recipes; the figure source tables preserve the recorded values without inventing implementation details.

## 3. Matched external comparisons

```bash
python run.py experiment external-controls
python run.py experiment published-comparison
```

The first command addresses Full versus converged constrained static fusion versus direct anchor under shared external endpoints. The second compares published OSA signatures using the study's common evaluation protocol. They answer different questions and must not be substituted for one another.

Inspect the original flags for permutation counts, bootstrap counts and result directories. Default Stage 38 settings retain 10,000 permutations and 5,000 bootstrap iterations. External results are not used to select a favorable model or revise discovery training targets.

## 4. Cross-participant sensitivity analyses

```bash
python run.py experiment cross-participant --device cuda:0
python run.py experiment external-controls --comparison-ranking results/discovery_preprocess_v2/39_cross_participant_adaptation_v1
python run.py experiment published-comparison --comparison-ranking results/discovery_preprocess_v2/39_cross_participant_adaptation_v1 --comparison-method Full
```

The ranking directory must refer to the completed Stage 39 run expected by the original scripts. Confirm that its `run_summary.json` and ranking files exist. Do not use this example directory name as evidence that a run has finished. Use each script's reports to check candidate universe, measurable background and paired comparison rules.

## 5. Candidate evidence and robustness

The stage registry includes CPAP reversal, adipose tissue views, single-cell context, proteomic coverage, known interactions, pathway linkage, ranking stability and shortlist-rule sensitivity. These are evidence streams, not interchangeable independent validation cohorts. Treatment reversal is a molecular endpoint, not a clinical treatment-benefit prediction.

`pathway_module_layer/scripts/run_all.py` links discovery pathway activity to candidate modules. Supply the exact versioned GMT files specified in its configuration; do not silently download the current gene-set release to reproduce an archived analysis.

## 6. Figures and audit

### Equal-input fusion controls (Stage 50)

The source is `model/osa_adapter/osa_fusion_capacity_experiments.py`, with
`config/osa_fusion_capacity_control_v1.json`. It compares the structured
architecture against parameter-matched, same-start and wider MLP controls.
The common-objective `Full` is distinct from the original-objective
`Native_full` reference. Training selection uses discovery validation only;
external evaluation is a separate phase.

```bash
python run.py stage fusion-capacity --check-env --device cpu
python run.py stage fusion-capacity --smoke --device cpu
python run.py stage fusion-capacity --device cuda:0 --resume
python run.py stage fusion-capacity --external-only --resume
```

These training commands require the archived Stage 31 fold inputs/checkpoints,
Stage 38 ranking background and stage-specific expression inputs recorded in
the config and preflight checks. The two public archives contain derived
outputs, not all training inputs. Smoke mode checks execution and is not a
formal result. `--resume` verifies existing contracts and output fingerprints.
Use `--root` after the stage name when selecting a separate analysis workspace.

On Linux, `bash scripts/run_osa_fusion_capacity_controls.sh check|smoke|formal`
combines preflight, tests and the two phases. It defaults to CPU. To use a
GPU, explicitly set `OSA_DEVICE=cuda:0` and, if needed, `CUDA_VISIBLE_DEVICES`
to the physical GPU you intend to use. `OSA_PYTHON` can select an interpreter.

For plotting completed results without training, follow
[`source_data/README.md`](../source_data/README.md).

### Other figures and checks

```bash
python run.py example
python run.py plot benchmark
python run.py audit
python run.py test
```

The example and benchmark plots can run from the small bundled aggregate tables. Larger biological plots need archived source tables. Model unit tests use explicitly generated test fixtures, not fabricated study data. Repository audit checks portable text, Python syntax and snapshot hashes; it cannot validate scientific conclusions or detect every possible secret.
