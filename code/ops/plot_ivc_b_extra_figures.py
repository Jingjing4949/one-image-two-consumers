"""Manuscript figures from the pulled paper extras: TUB examples and held-out scene gains."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(r'.')
SRC = BASE / 'outputs/ivc_b_remote_summaries_20261012/paper_extras'
FIG = BASE / 'outputs/IVC_B_manuscript_working/manuscript/figures'
LABEL = {'identity': 'Original', 'gray_world_mean': 'Grey-world', 'funie_s20260913': 'FUnIE-1', 'funie_s20261013': 'FUnIE-2',
         'lite_s20260913': 'Lite-1', 'lite_s20261013': 'Lite-2', 'semi_uir': 'Semi-UIR', 'phaseformer_uieb': 'Phaseformer',
         'deep_wavenet_euvp': 'Deep-WaveNet', 'uir_polykernel': 'PolyKernel', 'shallow_uwnet': 'Shallow-UWnet',
         'syreanet': 'SyreaNet', 'ushape_lsui': 'U-shape', 'tacl': 'TACL', 'uiec2net': 'UIEC2-Net'}


def examples():
    meta = json.loads((SRC / 'b_fig_examples/meta.json').read_text(encoding='utf-8'))['examples']
    fig, axes = plt.subplots(len(meta), 6, figsize=(12, 1.75 * len(meta)))
    for row, ex in zip(axes, meta):
        sc = ex['scores']
        quality_best = max(sc, key=lambda a: sc[a]['ssim'] - sc[a]['lpips'])
        task_best = max(sc, key=lambda a: sc[a]['fg_iou'])
        panels = [('identity', 'Input (original)'), (None, 'Stage-1 reference'), ('mask', 'Object mask'),
                  (quality_best, 'Quality-best'), (task_best, 'Task-best'), ('uir_polykernel', 'PolyKernel')]
        for ax, (key, title) in zip(row, panels):
            if key is None:
                path = SRC / 'b_fig_examples' / f"{ex['stem']}__reference.png"
            else:
                path = SRC / 'b_fig_examples' / f"{ex['stem']}__{key}.png"
            ax.imshow(np.asarray(Image.open(path)), cmap='gray' if key == 'mask' else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if key in sc:
                s = sc[key]
                name = LABEL[key] if title in ('Quality-best', 'Task-best') else ''
                head = f'{title}: {name}' if name else title
                ax.set_title(f"{head}\nSSIM {s['ssim']:.3f}  LPIPS {s['lpips']:.3f}  IoU {s['fg_iou']:.3f}", fontsize=7.5)
            else:
                ax.set_title(title, fontsize=7.5)
        row[0].set_ylabel(f"Scene {ex['experiment']}, stage {ex['stage']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'tub_examples.pdf')
    fig.savefig(FIG / 'tub_examples.png', dpi=180)
    plt.close(fig)
    return meta


def holdout_scenes():
    data = json.loads((SRC / 'b6_holdout/breakdown.json').read_text(encoding='utf-8'))
    names = [('ridge', '52-d ridge'), ('probe', 'Confidence probe')]
    metrics = ['SSIM gain', 'LPIPS reduction', 'Fg-IoU gain']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for m, ax in enumerate(axes):
        for k, (key, label) in enumerate(names):
            vals = np.array([v[m] for v in data['scenes'][key].values()])
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(vals, k + jitter, s=18, label=label)
            ax.plot([vals.mean()] * 2, [k - 0.25, k + 0.25], color='black', lw=1.5)
        ax.axvline(0, color='grey', lw=0.8, ls='--')
        ax.set_title(metrics[m], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([n[1] for n in names], fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Held-out TUB scenes: per-scene gain versus the original image (bar = scene mean)', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'holdout_scenes.pdf')
    plt.close(fig)
    return data


if __name__ == '__main__':
    meta = examples()
    data = holdout_scenes()
    print('examples', [(m['id'], m['turbidity']) for m in meta])
    print('match', data['matches_saved_confirmation'], data['check'])
