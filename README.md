# One image, two consumers: code and result summaries

Code, frozen protocols and result summaries for the manuscript
"One image, two consumers: quality-task compatibility of fifteen underwater enhancers"
by Jingjing Fan, Yunan Zhang and Xueguang Wang.

The archive contains no images, masks, references or network weights. Every table and
figure in the manuscript can be rebuilt from `summaries/` with the scripts in `code/ops/`.
Re-running the experiments themselves requires the datasets and third-party enhancers listed below.

## Layout

| Folder | Content |
| --- | --- |
| `code/experiments/` | Enhancement, scoring, segmentation training, compatibility (MILP), selectors, confirmation and post-hoc analyses |
| `code/ops/` | Table and figure builders used for the manuscript |
| `protocols/` | Frozen study protocols, written in Chinese, as recorded before the corresponding runs |
| `metadata/` | Action-pool provenance, contamination screen, SUIM-E folds and estimated groups, hashes used by the held-out overlap screen |
| `summaries/` | JSON summaries behind every reported number |
| `MANIFEST.json` | SHA-256 of every file in this archive |

## Data and third-party models

* SUIM-E (Qi et al., 2022) and SUIM (Islam et al., 2020): obtain from the authors' releases.
* TUB (Ismiroglou et al., 2026): https://huggingface.co/datasets/vapaau/TUB, CC BY 4.0.
* LSUI and UIEB are used only for the input-overlap screen; UIEB does not permit redistribution.
* Public enhancers: repositories, archive hashes, weight hashes and inference deviations are listed in
  `metadata/ivc_b1_action_pool_v1.json`. Their source code and weights are not redistributed here.

## Environment

Python 3.11, PyTorch 2.7.1 (CUDA 11.8), torchvision, lpips 0.1 (AlexNet), piq 0.8.0 (DISTS),
scikit-image, SciPy (HiGHS MILP), NumPy, Pillow. Experiment scripts use the server layout
`/root/ivcb/{code,data,out}`; adjust the path constants at the top of each script.
The figure and table builders in `code/ops/` resolve paths relative to the working directory and expect
`outputs/ivc_b_remote_summaries_20261012/` to hold the contents of `summaries/`.

## Pipeline

1. `prepare_ivc_b3_suime_folds.py`, `prepare_ivc_b_tub.py`: folds, estimated groups, TUB development split and held-out scene list.
2. `run_ivc_b2_suime.py` (with `ivc_b_actions.py`, `ivc_b_third_party.py`): stored outputs, SSIM and LPIPS. `run_ivc_b_dists.py`: DISTS.
3. `run_ivc_b3_seg.py` (queued by `ivc_b_queue.py`): LR-ASPP and DeepLabV3 endpoints, raw, mixed and matched regimes, second seed.
4. `ivc_b_compat.py`, `run_ivc_b_frontier.py`, `ivc_b_decompose.py`, `run_ivc_b_finish_dev.py`: opportunity, coverage, minimum harm, retraining decomposition, label-assisted references, strata.
5. `run_ivc_b5_confidence.py`, `run_ivc_b5_cnn.py`, `run_ivc_b5_probe.py`: development selectors. `run_ivc_b_latency.py`: request time.
6. `run_ivc_b_holdout.py`, `run_ivc_b_holdout_score.py`: once-only held-out confirmation, guarded by the SHA-256 of the frozen protocol.
7. `run_ivc_b_paper_extras.py`, `run_ivc_b_gain_agreement.py`, `run_ivc_b_holdout_breakdown.py`, `run_ivc_b_rebuttal.py`, `export_ivc_b_figure_examples.py`: post-hoc summaries on saved outputs; no fitting or tuning.

## Licence

Code is released under the MIT licence (see `LICENSE`). Summary files contain derived numbers only;
the underlying datasets remain under their own licences.
