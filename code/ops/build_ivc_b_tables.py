"""Generate LaTeX tables for the manuscript and supplement directly from the saved summaries."""
import json
from pathlib import Path

BASE = Path(r'.')
S = BASE / 'outputs/ivc_b_remote_summaries_20261012'
X = S / 'paper_extras'
OUT = BASE / 'outputs/IVC_B_manuscript_working/manuscript/tables'
NAME = {'identity': 'Original', 'gray_world_mean': 'Grey-world', 'funie_s20260913': 'FUnIE, seed 1', 'funie_s20261013': 'FUnIE, seed 2',
        'lite_s20260913': 'Lite, seed 1', 'lite_s20261013': 'Lite, seed 2', 'semi_uir': 'Semi-UIR', 'phaseformer_uieb': 'Phaseformer',
        'deep_wavenet_euvp': 'Deep-WaveNet', 'uir_polykernel': 'PolyKernel', 'shallow_uwnet': 'Shallow-UWnet',
        'syreanet': 'SyreaNet', 'ushape_lsui': 'U-shape', 'tacl': 'TACL', 'uiec2net': 'UIEC2-Net'}
EP = {'lraspp/raw': 'LR-ASPP, raw', 'deeplabv3/raw': 'DeepLabV3, raw', 'lraspp/mixed': 'LR-ASPP, mixed',
      'deeplabv3/mixed': 'DeepLabV3, mixed', 'lraspp/raw/seed2': 'LR-ASPP, raw, seed 2'}
COMPAT = {('suim', 'lraspp/raw'): 'raw15_lraspp', ('suim', 'deeplabv3/raw'): 'raw15_deeplabv3',
          ('suim', 'lraspp/mixed'): 'mixed15_lraspp', ('suim', 'deeplabv3/mixed'): 'mixed15_deeplabv3',
          ('tub', 'lraspp/raw'): 'tub_raw_lraspp', ('tub', 'deeplabv3/raw'): 'tub_raw_deeplabv3',
          ('tub', 'lraspp/mixed'): 'tub_mixed_lraspp', ('tub', 'deeplabv3/mixed'): 'tub_mixed_deeplabv3'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def n(v):
    return f'{v:,}'.replace(',', '{,}')


def f(v, d=3, sign=False):
    if v is None:
        return '--'
    if abs(v) < 0.5 * 10 ** -d:
        return f'{0:.{d}f}'
    s = f'{v:+.{d}f}' if sign else f'{v:.{d}f}'
    return s.replace('-', '$-$').replace('+', '$+$') if sign else s.replace('-', '$-$')


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding='utf-8')


def endpoints(ex):
    lines = [r'\begin{table}[pos=htbp]\centering\small',
             r'\caption{Task-endpoint sensitivity on the development sets. Same is the image fraction on which the quality-best and task-best actions coincide. Opportunity counts groups with a tolerated, strictly better alternative. Coverage is the mixed-integer maximum under the mean-effect bars; the harm columns are the solved minimum other-objective harm at the quality and task maxima. Mean change is the average over the 14 non-identity actions of the group-mean task change. A dash marks a problem without a solution; n/c marks quantities not computed for the seed-2 endpoint.}\label{tab:endpoints}',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrr}\toprule',
             r'Endpoint & Same & Opportunity & Coverage & Task harm & Quality harm & Mean change \\\midrule']
    for ds, label in (('suim', 'SUIM-E, 1{,}465 groups'), ('tub', 'TUB development, 22 scenes')):
        lines.append(r'\multicolumn{7}{l}{\emph{' + label + r'}} \\')
        for key, e in ex[ds]['endpoints'].items():
            tc = e['task_change_vs_identity']
            mean = sum(tc.values()) / len(tc)
            cname = COMPAT.get((ds, key))
            if cname:
                c = load(S / f'compat_{cname}.json')['anchors']['identity']
                cov = n(c['max_joint']) if c['max_joint'] else '0'
                th = '--' if c['min_task_harm_at_quality_max'] is None else n(c['min_task_harm_at_quality_max'])
                qh = '--' if c['min_quality_harm_at_task_max'] is None else n(c['min_quality_harm_at_task_max'])
            else:
                cov = th = qh = 'n/c'
            lines.append(f"{EP[key]} & {100 * e['decoupled']['same_action_image_fraction']:.1f}\\% & {n(e['safe_groups']['joint'])} & {cov} & {th} & {qh} & {f(mean, 3, True)} \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('endpoints.tex', '\n'.join(lines))


def agreement(ag):
    lines = [r'\begin{table}[pos=htbp]\centering\small',
             r'\caption{Agreement of per-image task gains, each measured against the original image under the same endpoint. Seed compares the two LR-ASPP raw seeds; architecture compares LR-ASPP and DeepLabV3 raw. Sign agreement is computed where the first endpoint moves by more than 0.01.}\label{tab:agreement}',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrr}\toprule',
             r' & \multicolumn{2}{c}{SUIM-E} & \multicolumn{2}{c}{TUB} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Statistic & Seed & Architecture & Seed & Architecture \\\midrule']
    rows = [('Median $|\\Delta_1-\\Delta_2|$', 'median_abs', 3), ('Pairs differing by $>0.01$', 'frac_abs_above_tol', 'pct'),
            ('Per-image Pearson $r$', 'pearson', 2), ('Sign agreement', 'sign_agreement_where_first_exceeds_tol', 'pct'),
            ('Harm-label agreement', 'harm_label_agreement', 'pct'), ('Action-mean Pearson $r$', 'action_mean_gain_pearson', 3),
            ('Group any-harm agreement', 'group_any_harm_agreement', 'pct')]
    for label, key, d in rows:
        vals = []
        for ds in ('suim', 'tub'):
            for cmp in ('lraspp_seed1_vs_seed2', 'lraspp_vs_deeplab'):
                v = ag[ds][cmp][key]
                vals.append(f'{100 * v:.1f}\\%' if d == 'pct' else f'{v:.{d}f}')
        lines.append(label + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('agreement.tex', '\n'.join(lines))


def selectors(ex):
    lines = [r'\begin{table}[pos=htbp]\centering\small',
             r'\caption{Group-level outcome of each development policy on the left-out folds, lraspp raw endpoint. Safe-improved groups have no image beyond the 0.01 tolerance on any coordinate and at least one image switched to a tolerated, strictly better action. Harmed groups contain at least one image beyond the tolerance. The joint-sum reference picks the largest summed gain per image and uses the labels.}\label{tab:selectors}',
             r'\begin{tabular}{lrrrr}\toprule',
             r' & \multicolumn{2}{c}{SUIM-E (1{,}465 groups)} & \multicolumn{2}{c}{TUB (22 scenes)} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Policy & Safe-improved & Harmed & Safe-improved & Harmed \\\midrule']
    labels = [('fixed_training_fold', 'Fixed action of the training folds'), ('cnn', 'MobileNetV3 gain model'),
              ('ridge', '52-d ridge'), ('probe', 'Confidence probe'), ('joint_oracle', 'Joint-sum reference (labels)')]
    for key, label in labels:
        a, b = ex['suim']['selectors'][key], ex['tub']['selectors'][key]
        lines.append(f"{label} & {n(a['safe_improved_groups'])} & {n(a['harmed_groups'])} & {b['safe_improved_groups']} & {b['harmed_groups']} \\\\")
    lines.append(r'\midrule')
    lines.append(f"Opportunity count & {n(ex['suim']['selectors']['joint_safe_capacity'])} & -- & {ex['tub']['selectors']['joint_safe_capacity']} & -- \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('selectors.tex', '\n'.join(lines))


def dists(ex):
    lines = [r'\begin{table}[pos=htbp]\centering\scriptsize',
             r'\caption{Group-mean DISTS and LPIPS reduction of each action relative to the original image (positive is closer to the reference), development sets.}\label{tab:dists}',
             r'\begin{tabular}{lrrrr}\toprule',
             r' & \multicolumn{2}{c}{SUIM-E} & \multicolumn{2}{c}{TUB} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Action & DISTS & LPIPS & DISTS & LPIPS \\\midrule']
    for a in ex['suim']['dists']['per_action']:
        s, t = ex['suim']['dists']['per_action'][a], ex['tub']['dists']['per_action'][a]
        lines.append(f"{NAME[a]} & {f(s['dists_reduction'], 3, True)} & {f(s['lpips_reduction'], 3, True)} & {f(t['dists_reduction'], 3, True)} & {f(t['lpips_reduction'], 3, True)} \\\\")
    lines.append(r'\midrule')
    s, t = ex['suim']['dists'], ex['tub']['dists']
    lines.append(f"Spearman across actions & \\multicolumn{{2}}{{c}}{{{s['spearman_lpips_vs_dists_across_actions']:.3f}}} & \\multicolumn{{2}}{{c}}{{{t['spearman_lpips_vs_dists_across_actions']:.3f}}} \\\\")
    lines.append(f"Same quality-best action & \\multicolumn{{2}}{{c}}{{{100 * s['same_quality_best_action_fraction']:.1f}\\%}} & \\multicolumn{{2}}{{c}}{{{100 * t['same_quality_best_action_fraction']:.1f}\\%}} \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('dists.tex', '\n'.join(lines))


def classes():
    full = load(S / 'finish_suim_full.json')['class_iou_delta']
    head = ['BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR']
    lines = [r'\begin{table}[pos=htbp]\centering\scriptsize',
             r'\caption{Group-mean per-class IoU change versus the original image on SUIM-E, raw LR-ASPP endpoint. Classes: BW waterbody, HD divers, PF plants, WR wrecks, RO robots, RI reefs and invertebrates, FV fish and vertebrates, SR sea-floor and rocks.}\label{tab:classes}',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{l' + 'r' * 8 + r'}\toprule', 'Action & ' + ' & '.join(head) + r' \\\midrule']
    cols = list(zip(*full.values()))
    for a, v in full.items():
        lines.append(NAME[a] + ' & ' + ' & '.join(f(x, 3, True) for x in v) + r' \\')
    lines.append(r'\midrule')
    lines.append('Mean of 14 & ' + ' & '.join(f(sum(c) / len(c), 3, True) for c in cols) + r' \\')
    lines.append('Actions $<0$ & ' + ' & '.join(str(sum(x < 0 for x in c)) for c in cols) + r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('classes.tex', '\n'.join(lines))


def stages(ex):
    lines = [r'\begin{table}[pos=htbp]\centering\small',
             r'\caption{TUB development by turbidity stage (88 images per stage; 84 at stage 10). Original-image scores, the fraction of images whose quality-best and task-best actions coincide, the number of images with a jointly tolerated, strictly better alternative, and the largest mean task change of any enhancer.}\label{tab:stages}',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{rrrrrrl}\toprule',
             r'Stage & SSIM & LPIPS & Fg-IoU & Same & Safe alt. & Best enhancer change \\\midrule']
    for r in ex['tub']['by_stage']:
        lines.append(f"{r['stage']} & {r['identity_ssim']:.3f} & {r['identity_lpips']:.3f} & {r['identity_task']:.3f} & {100 * r['same_best_fraction']:.0f}\\% & {r['images_with_joint_safe_alternative']} & {f(r['best_task_change'], 3, True)} ({NAME[r['best_task_change_action']]}) \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('stages.tex', '\n'.join(lines))


def scenes():
    d = load(X / 'b6_holdout/breakdown.json')
    lines = [r'\begin{table}[pos=htbp]\centering\small',
             r'\caption{Per-scene gains of the ridge and probe on the 11 held-out TUB scenes, relative to the original image. The convolution ensemble selected the original image for all 395 images, so its gains are zero in every scene.}\label{tab:scenes}',
             r'\begin{tabular}{lrrrrrr}\toprule',
             r' & \multicolumn{3}{c}{52-d ridge} & \multicolumn{3}{c}{Confidence probe} \\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Scene & SSIM & LPIPS red. & Fg-IoU & SSIM & LPIPS red. & Fg-IoU \\\midrule']
    keys = sorted(d['scenes']['ridge'], key=lambda k: int(k[3:]))
    for k in keys:
        r, p = d['scenes']['ridge'][k], d['scenes']['probe'][k]
        lines.append(f"{k[3:]} & " + ' & '.join(f(v, 3, True) for v in r + p) + r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('scenes.tex', '\n'.join(lines))


def provenance():
    pool = load(BASE / 'outputs/ivc_b1_action_pool_v1.json')
    lines = [r'\begin{table}[pos=htbp]\centering\scriptsize',
             r'\caption{Provenance of the nine public actions: training data stated by the source, and the inference deviations used here. Weights and source archives are identified by SHA-256 in the study records.}\label{tab:provenance}',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{p{0.15\linewidth}p{0.3\linewidth}p{0.45\linewidth}}\toprule',
             r'Action & Training data (source) & Deviation in this study \\\midrule']
    for key, v in pool['third_party'].items():
        dev = v['deviation'] or 'None'
        clean = lambda s: (s.replace('->', 'to').replace('_', r'\_').replace('~', r'$\sim$')
                           .replace('512x512', r'$512\times512$').replace('256x256', r'$256\times256$'))
        lines.append(f"{NAME[key]} & {clean(v['train_data'])} & {clean(dev)} \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    write('provenance.tex', '\n'.join(lines))


def main():
    ex = load(X / 'b_paper_extras.json')
    ag = load(X / 'b_gain_agreement.json')
    endpoints(ex)
    agreement(ag)
    selectors(ex)
    dists(ex)
    classes()
    stages(ex)
    scenes()
    provenance()
    print('tables', sorted(p.name for p in OUT.glob('*.tex')))


if __name__ == '__main__':
    main()
