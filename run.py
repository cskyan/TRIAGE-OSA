#!/usr/bin/env python3
"""Single entry point for TRIAGE-OSA workflows and the bundled example."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent
TRAIN = 'model/osa_adapter/train_triage_osa.py'
EXPERIMENTS = {
    'ablations': '--experiment-suite',
    'strong-controls': '--complete-supplements',
    'external-controls': '--external-strong-controls',
    'published-comparison': '--published-osa-comparison',
    'cross-participant': '--cross-participant-adaptation',
    'audit': '--audit-experiments',
    'self-test': '--experiment-self-test',
}
STAGES = {
    'downloads': 'scripts/data/osa_geo_server_pipeline.py',
    'download-audit': 'scripts/data/audit_osa_downloads.py',
    'download-verify': 'scripts/data/verify_osa_downloads.py',
    'manifests': 'scripts/data/build_manifest_source.py',
    'discovery-qc': 'scripts/discovery/osa_discovery_preprocess_qc.py',
    'discovery-differential': 'scripts/discovery/osa_discovery_differential.py',
    'identifier-mapping': 'scripts/discovery/osa_discovery_identifier_mapping.py',
    'candidate-ranking': 'scripts/discovery/osa_discovery_candidate_ranking.py',
    'model-inputs': 'model/osa_adapter/prepare_osa_model_inputs.py',
    'pair-registry': 'model/osa_adapter/build_osa_pair_registry.py',
    'pair-inference': 'model/osa_adapter/run_osa_symmetric_inference.py',
    'context-ranking': 'scripts/discovery/osa_pair_context_ranking.py',
    'cpap-preprocess': 'scripts/validation/osa_cpap_validation_preprocess.py',
    'cpap-endpoint': 'scripts/validation/osa_cpap_reversal_endpoint.py',
    'cpap-signflip': 'scripts/validation/osa_cpap_subject_signflip.py',
    'adipose-preprocess': 'scripts/validation/osa_adipose_validation_preprocess.py',
    'adipose-endpoint': 'scripts/validation/osa_adipose_cross_tissue_endpoint.py',
    'single-cell-preprocess': 'scripts/validation/osa_scrna_validation_preprocess.py',
    'single-cell-endpoint': 'scripts/validation/osa_scrna_cell_context_endpoint.py',
    'proteome-preprocess': 'scripts/validation/osa_pediatric_proteome_preprocess.py',
    'proteome-endpoint': 'scripts/validation/osa_pediatric_proteome_endpoint.py',
    'known-ppi': 'scripts/validation/osa_known_ppi_complex_recovery.py',
    'known-ppi-references': 'scripts/validation/osa_known_ppi_reference_prepare.py',
    'independent-validation': 'scripts/validation/osa_independent_validation.py',
    'ranking-stability': 'scripts/candidates/osa_complete_ranking_stability.py',
    'shortlist-sensitivity': 'scripts/candidates/osa_shortlist_rule_sensitivity.py',
    'pathway-linkage': 'scripts/candidates/osa_candidate_pathway_linkage.py',
    'direction-feasibility': 'scripts/candidates/osa_direction_feasibility.py',
    'capacity-control': 'scripts/candidates/osa_representation_capacity_control.py',
    'fusion-capacity': 'model/osa_adapter/osa_fusion_capacity_experiments.py',
    'evidence-closeout': 'scripts/candidates/osa_public_evidence_closeout.py',
    'selection-check': 'scripts/candidates/check_candidate_selection_strength.py',
    'auc-diagnostic': 'scripts/candidates/osa_auc_protocol_diagnostic.py',
    'structure-audit': 'scripts/checks/audit_osa_structure_provenance.py',
    'pathway-modules': 'pathway_module_layer/scripts/run_all.py',
}
PLOTS = {name: 'plotting/figures/' + name + '.py'
         for name in ['benchmark', 'bio', 'published', 'coverage', 'supp']}
PLOTS.update({'v4': 'plotting/figures/osa_v4_publication_figures.py',
              'fusion-capacity': 'plotting/figures/plot_osa_fusion_capacity_controls.py',
              'legacy': 'plotting/figures/osa_publication_figures.py',
              'candidate': 'examples/candidate_evidence/run_example.py'})

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print the command without running it.')
    parser.add_argument('--data-root', default=os.environ.get('OSA_DATA_ROOT'),
                        help='Data workspace containing config/, metadata/, and results/.')
    parser.add_argument('command', choices=['list', 'example', 'plot', 'train', 'predict', 'experiment', 'stage', 'test', 'audit'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = args.arguments
    if rest and rest[0] == '--':
        rest = rest[1:]
    if args.command == 'list':
        for label, values in [('Experiments', EXPERIMENTS), ('Stages', STAGES), ('Plots', PLOTS)]:
            print(label + ': ' + ', '.join(values))
        return 0
    env = os.environ.copy()
    module_dirs = [REPO / 'scripts' / group for group in
                   ['data', 'discovery', 'validation', 'candidates', 'checks']]
    env['PYTHONPATH'] = os.pathsep.join(
        [str(path) for path in module_dirs] +
        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    if args.data_root:
        env['OSA_DATA_ROOT'] = str(Path(args.data_root).expanduser().resolve())
    flags = []
    if args.command in ['experiment', 'stage', 'plot']:
        registry = {'experiment': EXPERIMENTS, 'stage': STAGES, 'plot': PLOTS}[args.command]
        if not rest or rest[0] not in registry:
            parser.error('Choose one of: ' + ', '.join(registry))
        name, rest = rest[0], rest[1:]
        if args.command == 'experiment':
            script, flags = TRAIN, [registry[name]]
        else:
            script = registry[name]
    else:
        script = {'example': PLOTS['candidate'], 'train': TRAIN,
                  'predict': 'model/osa_adapter/predict_triage_osa.py',
                  'audit': 'tools/audit_repository.py', 'test': None}[args.command]
    if args.command in ['train', 'predict', 'experiment'] and args.data_root:
        flags += ['--root', env['OSA_DATA_ROOT']]
    if args.command == 'test':
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'] + rest
    else:
        command = [sys.executable, str(REPO / script)] + flags + rest
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(REPO), env=env)

if __name__ == '__main__':
    raise SystemExit(main())
