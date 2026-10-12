"""Graphical abstract, about 2.5:1, using only reported results."""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(r".\outputs\IVC_B_manuscript_working\manuscript\figures")
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")
blocks = [
    (0.3, "6.5%", "SUIM-E images where\nquality and task\npick the same action"),
    (3.6, "314 to 1,189", "Jointly safe groups\nas 14 enhancers\nare added"),
    (6.9, "0 of 3", "Held-out selectors\nthat clear SSIM,\nLPIPS and IoU"),
]
for x, title, body in blocks:
    ax.add_patch(plt.Rectangle((x, 0.35), 2.8, 3.3, fill=False, linewidth=1.2, color="#1f4e79"))
    ax.text(x + 1.4, 2.7, title, ha="center", va="center", fontsize=16, color="#1f4e79")
    ax.text(x + 1.4, 1.35, body, ha="center", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "graphical_abstract.pdf")
fig.savefig(OUT / "graphical_abstract.png", dpi=200)
plt.close()
print("ga", OUT / "graphical_abstract.pdf")
