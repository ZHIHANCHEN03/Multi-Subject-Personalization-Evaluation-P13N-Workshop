import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

# ---------------------------------------------------------
# 1. Overall Agreement by Dimension (Bar Chart)
# ---------------------------------------------------------
dimensions = ['Preference', 'Existence', 'Interaction', 'Appearance']
rates = [95.1, 89.1, 71.9, 69.0]

fig, ax = plt.subplots(figsize=(6, 5))
colors = ['#2980b9', '#27ae60', '#f39c12', '#e67e22']
bars = ax.bar(dimensions, rates, color=colors, width=0.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='black')

ax.set_ylabel('Cross-Model Agreement (%)')
ax.set_ylim(0, 110)
# Rotate x labels slightly to fit the narrower aspect ratio
ax.set_xticks(range(len(dimensions)))
ax.set_xticklabels(dimensions, rotation=25, ha='right')
ax.set_title('Agreement by Evaluation Dimension')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(OUTPUT_DIR / 'overall_agreement.pdf')
plt.savefig(OUTPUT_DIR / 'overall_agreement.png')
plt.close()

# ---------------------------------------------------------
# 2. Preference Agreement by Subject Count (Line Chart)
# ---------------------------------------------------------
subj_counts = [2, 4, 6, 8]
subj_rates = [91.1, 93.4, 97.7, 98.1]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(subj_counts, subj_rates, marker='o', markersize=10, linewidth=3, color='#2980b9')

# Annotate points
for i, (count, rate) in enumerate(zip(subj_counts, subj_rates)):
    # Shift the first point to the right, and the last point to the left
    if i == 0:
        offset_x = 15
    elif i == 3:
        offset_x = -15
    else:
        offset_x = 0
        
    ax.annotate(f'{rate:.1f}%',
                xy=(count, rate),
                xytext=(offset_x, 10),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='#2980b9', fontsize=10)

ax.set_xlabel('Number of Subjects')
ax.set_ylabel('Preference Agreement (%)')
ax.set_xticks(subj_counts)
ax.set_ylim(85, 102)
ax.set_title('Agreement vs. Scene Complexity')
ax.grid(True, linestyle='--', alpha=0.7)

plt.savefig(OUTPUT_DIR / 'agreement_by_subject.pdf')
plt.savefig(OUTPUT_DIR / 'agreement_by_subject.png')
plt.close()

# ---------------------------------------------------------
# 3. Preference Agreement by Class Tag (Horizontal Bar)
# ---------------------------------------------------------
tags = ['Occlusion w/ Interaction', 'No Occlusion w/o Interaction', 'Occlusion w/o Interaction']
# Original keys: occlusion_interaction (97.2), no_interaction_no_occlusion (95.3), occlusion_no_interaction (92.8)
tag_rates = [97.2, 95.3, 92.8]

fig, ax = plt.subplots(figsize=(6, 5))
y_pos = np.arange(len(tags))
bars = ax.barh(y_pos, tag_rates, color=['#2980b9', '#27ae60', '#f39c12'], height=0.5)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(-30, 0),  # Inside the bar
                textcoords="offset points",
                ha='center', va='center', fontweight='bold', color='white')

ax.set_yticks(y_pos)
ax.set_yticklabels(tags)
ax.set_xlabel('Preference Agreement (%)')
ax.set_xlim(80, 100)
ax.set_title('Agreement by Spatial Relationship')
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.savefig(OUTPUT_DIR / 'agreement_by_tag.pdf')
plt.savefig(OUTPUT_DIR / 'agreement_by_tag.png')
plt.close()

print(f"Figures saved to {OUTPUT_DIR}")
