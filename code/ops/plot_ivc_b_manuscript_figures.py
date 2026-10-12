"""Figures for the current fifteen-action manuscript. Numbers come from the pulled summaries."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r".")
SRC = ROOT / "outputs" / "ivc_b_remote_summaries_20261012"
FIG = ROOT / "outputs" / "IVC_B_manuscript_working" / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

curve = json.loads((SRC / "finish_suim_fast.json").read_text(encoding="utf-8"))["category_curve"]["curve"]
size = [0] + [row["pool_size"] for row in curve]
quality = [0] + [row["quality"] for row in curve]
task = [0] + [row["task"] for row in curve]
joint = [0] + [row["joint"] for row in curve]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(size, quality, "o-", label="Quality-safe groups", color="#1f4e79")
ax.plot(size, task, "s-", label="Task-safe groups", color="#b85c38")
ax.plot(size, joint, "D-", label="Jointly safe groups", color="#2f6b4f")
ax.set_xlabel("Non-identity actions in the pool")
ax.set_ylabel("SUIM-E groups (of 1,465)")
ax.set_xlim(0, 14)
ax.set_ylim(0, 1600)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "pool_growth.pdf")
plt.close()

labels = ["Gain model", "52-d ridge", "Confidence probe"]
ssim = [0.0, -0.0337, -0.0092]
lpips = [0.0, -0.1577, -0.0359]
iou = [0.0, -0.1049, -0.0151]
x = np.arange(len(labels))
w = 0.24
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(x - w, ssim, w, label="SSIM gain", color="#1f4e79")
ax.bar(x, lpips, w, label="LPIPS reduction", color="#b85c38")
ax.bar(x + w, iou, w, label="Foreground-IoU gain", color="#2f6b4f")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(-0.185, 0.012)
ax.set_ylabel("Scene-macro gain versus the original image")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "holdout_gains.pdf")
plt.close()
print("wrote", FIG)
