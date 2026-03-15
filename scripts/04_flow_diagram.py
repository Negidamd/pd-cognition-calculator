#!/usr/bin/env python3
"""
04_flow_diagram.py — CONSORT-style flow diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# NOTE: Update BASE to your local PPMI data directory
BASE = os.environ.get("PPMI_NOMOGRAM_DIR", ".")

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def draw_box(x, y, w, h, text, color='#deebf7', edgecolor='#2171b5', fontsize=9):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.15",
                                   facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontfamily='Arial', wrap=True, linespacing=1.3)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2171b5', lw=1.5))

def draw_exclusion(x_main, y_from, y_to, x_exc, text, n_exc):
    y_mid = (y_from + y_to) / 2
    # Horizontal line
    ax.annotate('', xy=(x_exc - 1.2, y_mid), xytext=(x_main, y_mid),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
    draw_box(x_exc + 0.5, y_mid, 3.0, 0.6, f'{text}\n(n = {n_exc})',
             color='#fee0d2', edgecolor='#d62728', fontsize=8)

# Boxes
draw_box(4, 11, 4, 0.7, 'PPMI participants\n(N = 7,781)', fontsize=10)
draw_arrow(4, 10.65, 4, 10.15)

draw_box(4, 9.8, 4, 0.7, 'Parkinson\'s disease cohort\n(N = 1,970)', fontsize=10)
draw_arrow(4, 9.45, 4, 8.95)

draw_box(4, 8.6, 4, 0.7, 'With cognitive state assessment\n(N = 1,520)', fontsize=10)
draw_arrow(4, 8.25, 4, 7.75)

draw_box(4, 7.4, 4.2, 0.7, 'Normal cognition at first assessment\n(N = 1,328)', fontsize=10)
draw_arrow(4, 7.05, 4, 6.55)

draw_box(4, 6.2, 4.2, 0.7, '≥2 cognitive assessments\n(N = 1,180)', fontsize=10)
draw_arrow(4, 5.85, 4, 5.35)

draw_box(4, 5.0, 4.2, 0.7, 'With baseline predictor data\n(N = 1,152)', fontsize=10)
draw_arrow(4, 4.65, 4, 4.0)

# Final split
draw_arrow(4, 4.0, 2.5, 3.3)
draw_arrow(4, 4.0, 5.5, 3.3)

draw_box(2.5, 2.9, 3.0, 0.8, 'Converters\n(n = 441)\nMCI: 430\nDementia: 11',
         color='#fee0d2', edgecolor='#d62728', fontsize=9)
draw_box(5.5, 2.9, 3.0, 0.8, 'Non-converters\n(n = 711)\nCensored',
         color='#c7e9c0', edgecolor='#2ca02c', fontsize=9)

# Exclusion boxes
draw_exclusion(4, 10.65, 10.15, 7.5, 'Non-PD cohorts\n(HC, prodromal, SWEDD)', 5811)
draw_exclusion(4, 9.45, 8.95, 7.5, 'No cognitive assessment', 450)
draw_exclusion(4, 8.25, 7.75, 7.5, 'MCI/dementia at baseline', 192)
draw_exclusion(4, 7.05, 6.55, 7.5, 'Single assessment only', 148)
draw_exclusion(4, 5.85, 5.35, 7.5, 'Missing time data', 28)

# Median follow-up annotation
ax.text(4, 2.0, 'Median follow-up: 4.0 years (IQR 1.0–6.0)',
        ha='center', va='center', fontsize=9, fontfamily='Arial',
        style='italic', color='#525252')

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "flow_diagram.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "flow_diagram.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/flow_diagram.png/pdf")
