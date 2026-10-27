# Plotting code

All plot entry points are collected under `python run.py plot NAME`.

| Name | Source | Inputs |
|---|---|---|
| `candidate` | Principal example | Bundled candidate-evidence TSVs |
| `fusion-capacity` | Equal-input fusion-control plot | `Fusion_Control_Source_Data/` extracted from Supplementary Data 2 |
| `benchmark` | Molecular benchmark and OSA ablation builder | Bundled JSON and strong-control TSV |
| `published` | Published-signature comparison builder | Archived Stage 36 v2 outputs |
| `bio` | Expression, treatment, pathway and network builders | Archived biological analysis stages and metadata |
| `coverage` | Cohort and measurement coverage | Archived metadata and coverage outputs |
| `supp` | Supplementary control/evidence builders | Archived supplementary analysis outputs |
| `v4` | Original V4 combined result builder | Original repository-relative result tree |
| `legacy` | Earlier combined result builder | Original stage-specific result tree |

The model/analysis code and the plotting code remain separate internally. Plotting reads completed results; it should not retrain the model or tune selection rules. The portable benchmark builder reads the exported aggregate JSON instead of a private spreadsheet. Its values are unchanged.

The five manuscript builders were ported from their source workflow. Their internal figure IDs are retained for provenance and can precede final manuscript renumbering or manual layout revisions. They are not represented as byte-identical reproductions of every final submitted image. The statistical builders do not recreate the author-drawn framework slides or molecular structure rendering sessions.

The principal example exports editable PDF/SVG and a 600-dpi PNG. A logarithmic color scale is used only to display rank magnitude; the integer ranks are printed in the cells. No rows are dropped or statistical intervals invented.

The licensed alignment helper checks rendered axes geometry. PDF text/collision review and scientific review remain necessary after figure revisions; a geometry check is not a validation of the analysis.
