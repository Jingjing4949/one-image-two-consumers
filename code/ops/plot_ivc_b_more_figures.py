"""Turbidity, selector-outcome and per-class figures from the saved summaries."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(r'.')
S = BASE / 'outputs/ivc_b_remote_summaries_20261012'
FIG = BASE / 'outputs/IVC_B_manuscript_working/manuscript/figures'
NAME = {'gray_world_mean': 'Grey-world', 'shallow_uwnet': 'Shallow-UWnet', 'funie_s20260913': 'FUnIE-1',
        'funie_s20261013': 'FUnIE-2', 'lite_s20260913': 'Lite-1', 'lite_s20261013': 'Lite-2',
        'deep_wavenet_euvp': 'Deep-WaveNet', 'uiec2net': 'UIEC2-Net', 'uir_polykernel': 'PolyKernel',
        'syreanet': 'SyreaNet', 'semi_uir': 'Semi-UIR', 'phaseformer_uieb': 'Phaseformer', 'ushape_lsui': 'U-shape', 'tacl': 'TACL'}
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
ex = json.loads((S / 'paper_extras/b_paper_extras.json').read_text(encoding='utf-8'))


def turbidity():
    rows = ex['tub']['by_stage']
    stage = [r['stage'] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a.plot(stage, [r['identity_ssim'] for r in rows], 'o-', color='#1f4e79', label='SSIM')
    a.plot(stage, [r['identity_lpips'] for r in rows], 's-', color='#b85c38', label='LPIPS')
    a.plot(stage, [r['identity_task'] for r in rows], 'D-', color='#2f6b4f', label='Foreground IoU')
    a.set_xlabel('Turbidity stage')
    a.set_ylabel('Original image score')
    a.set_title('(a) Original image against the stage-1 reference', fontsize=9)
    a.legend(frameon=False)
    b.bar(stage, [r['best_task_change'] for r in rows], color='#2f6b4f', width=0.6)
    b.axhline(0, color='black', lw=0.6)
    b.set_xlabel('Turbidity stage')
    b.set_ylabel('Mean foreground-IoU change')
    b.set_title('(b) Best of the 14 enhancers at each stage', fontsize=9)
    for ax in (a, b):
        ax.set_xticks(stage)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'tub_turbidity.pdf')
    plt.close(fig)


def selectors():
    labels = [('fixed_training_fold', 'Fixed\nPolyKernel'), ('cnn', 'Gain\nmodel'), ('ridge', '52-d\nridge'),
              ('probe', 'Confidence\nprobe'), ('joint_oracle', 'Joint-sum\nreference')]
    sel = ex['suim']['selectors']
    x = np.arange(len(labels))
    good = [sel[k]['safe_improved_groups'] for k, _ in labels]
    bad = [sel[k]['harmed_groups'] for k, _ in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    b1 = ax.bar(x - w / 2, good, w, color='#2f6b4f', label='Safe-improved groups')
    b2 = ax.bar(x + w / 2, bad, w, color='#b85c38', label='Harmed groups')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 15, f'{int(r.get_height()):,}', ha='center', fontsize=7.5)
    ax.axhline(sel['joint_safe_capacity'], color='grey', ls='--', lw=0.8)
    ax.text(len(labels) - 0.5, sel['joint_safe_capacity'] + 20, f"Opportunity count {sel['joint_safe_capacity']:,}", ha='right', fontsize=7.5, color='grey')
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in labels])
    ax.set_ylabel('SUIM-E groups (of 1,465)')
    ax.set_ylim(0, 1300)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'selector_groups.pdf')
    plt.close(fig)


def classes():
    full = json.loads((S / 'finish_suim_full.json').read_text(encoding='utf-8'))['class_iou_delta']
    head = ['Waterbody', 'Divers', 'Plants', 'Wrecks', 'Robots', 'Reefs and\ninvertebrates', 'Fish and\nvertebrates', 'Sea-floor\nand rocks']
    acts = list(full)
    m = np.array([full[a] for a in acts], dtype=float)
    lim = np.nanmax(np.abs(m))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(m, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f'{m[i, j]:+.3f}', ha='center', va='center', fontsize=6.5,
                    color='white' if abs(m[i, j]) > 0.6 * lim else 'black')
    ax.set_xticks(range(len(head)))
    ax.set_xticklabels(head, fontsize=7.5)
    ax.set_yticks(range(len(acts)))
    ax.set_yticklabels([NAME[a] for a in acts], fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Group-mean IoU change', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'suim_classes.pdf')
    plt.close(fig)


if __name__ == '__main__':
    turbidity()
    selectors()
    classes()
    print('wrote', FIG)
