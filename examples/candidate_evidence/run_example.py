"""Plot the archived candidate ranks without fitting or changing any model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'plotting/qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/candidate_evidence')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / 'data'
    source = folder / '15_candidate_method_comparison.tsv'
    frame = pd.read_csv(source, sep='\t')
    rules = pd.read_csv(folder / '24_candidate_rule_review.tsv', sep='\t')
    columns = ['triage_osa_rank_primary_250', 'static_rank', 'anchor_rank', 'ppi_rank']
    if len(frame) != 15 or not frame.pair_id.is_unique:
        raise ValueError('Expected the complete original 15-edge shortlist, with unique pair IDs.')
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 1) or np.any(values > 31125):
        raise ValueError('All ranks must be finite and within the 31,125-pair universe.')
    joined = frame.merge(rules[['pair_id', 'direction_selected', 'source_grouped_selected',
                                'direction_source_grouped_selected']], on='pair_id', how='left', validate='one_to_one')
    if joined.isna().any().any():
        raise ValueError('The example requires complete evidence-rule annotations.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / 'candidate_summary.tsv', sep='\t', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(183/25.4, 145/25.4))
    fig.subplots_adjust(left=.30, right=.84, top=.92, bottom=.15)
    norm = LogNorm(vmin=1, vmax=10000)
    grid = ax.imshow(values, aspect='auto', cmap='Blues_r', norm=norm, interpolation='nearest')
    ax.set_xticks(range(4), ['Full', 'Constrained\nstatic', 'Direct\nanchor', 'PPI only'])
    ax.set_yticks(range(len(frame)), frame.gene_A + ' - ' + frame.gene_B)
    ax.tick_params(axis='both', length=0, pad=6)
    for row in range(len(frame)):
        for col in range(4):
            value = values[row, col]
            ax.text(col, row, str(int(value)), ha='center', va='center', fontsize=8,
                    color='white' if norm(value) < .48 else '#17212b')
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.88, .30, .025, .46])
    colorbar = fig.colorbar(grid, cax=cax, ticks=[1, 10, 100, 1000, 10000])
    colorbar.ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
    colorbar.set_label('Rank (lower is earlier)', labelpad=8)
    ax.set_title('Candidate ranks under four prioritization methods', fontsize=9, pad=12)
    ax.set_xlabel('Same 31,125-pair background; original 15-edge shortlist', labelpad=13)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax], json_out=out / 'candidate_ranks.alignment.json', strict=True)
    fig.savefig(out / 'candidate_ranks.pdf')
    fig.savefig(out / 'candidate_ranks.svg')
    fig.savefig(out / 'candidate_ranks.png', dpi=600)
    plt.close(fig)
    summary = {'candidate_edges': len(frame),
               'candidate_proteins': len(set(frame.gene_A) | set(frame.gene_B)),
               'ranking_background_pairs': 31125,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'training_performed': False,
               'ranking_recomputed': False,
               'rows_excluded': 0,
               'purpose': 'Trace archived rankings and rule sensitivity, not validate physical interactions.'}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Example complete:', out)

if __name__ == '__main__':
    main()
