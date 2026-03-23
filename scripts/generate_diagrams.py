import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Adjust figure size or font sizes here if needed.
# The user requested larger fonts for figure 2 (benchmark_design).
plt.rcParams['font.size'] = 14

def create_benchmark_design(filename):
    # Set font to match CVPR style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Subject Pool Box
    pool_box = patches.FancyBboxPatch((0.5, 1), 2, 3, boxstyle="round,pad=0.1", linewidth=2, edgecolor='#1f77b4', facecolor='#e6f2ff')
    ax.add_patch(pool_box)
    ax.text(1.5, 3.5, "Subject\nPool", fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(1.5, 2.5, "Identity A\nIdentity B\n...\nIdentity Z", fontsize=12, ha='center', va='center')

    # Arrow to Prompts
    ax.annotate('', xy=(3.5, 2.5), xytext=(2.5, 2.5), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))

    # Prompts Box
    prompt_box = patches.FancyBboxPatch((3.5, 1), 2.5, 3, boxstyle="round,pad=0.1", linewidth=2, edgecolor='#2ca02c', facecolor='#e6ffe6')
    ax.add_patch(prompt_box)
    ax.text(4.75, 3.5, "Complexity\nScaling", fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(4.75, 2.5, "2 Subjects\n4 Subjects\n6 Subjects\n8 Subjects\n10 Subjects", fontsize=12, ha='center', va='center')

    # Arrows to Scenes
    ax.annotate('', xy=(7, 4), xytext=(6, 2.5), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))
    ax.annotate('', xy=(7, 2.5), xytext=(6, 2.5), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))
    ax.annotate('', xy=(7, 1), xytext=(6, 2.5), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))

    # Scene Types
    scene1 = patches.FancyBboxPatch((7, 3.5), 2.5, 1, boxstyle="round,pad=0.1", linewidth=2, edgecolor='#ff7f0e', facecolor='#fff2e6')
    ax.add_patch(scene1)
    ax.text(8.25, 4, "Neutral\n(No Interaction)", fontsize=12, fontweight='bold', ha='center', va='center')

    scene2 = patches.FancyBboxPatch((7, 2), 2.5, 1, boxstyle="round,pad=0.1", linewidth=2, edgecolor='#ff7f0e', facecolor='#fff2e6')
    ax.add_patch(scene2)
    ax.text(8.25, 2.5, "Occlusion\n(Spatial Overlap)", fontsize=12, fontweight='bold', ha='center', va='center')

    scene3 = patches.FancyBboxPatch((7, 0.5), 2.5, 1, boxstyle="round,pad=0.1", linewidth=2, edgecolor='#ff7f0e', facecolor='#fff2e6')
    ax.add_patch(scene3)
    ax.text(8.25, 1, "Interaction\n(Physical Contact)", fontsize=12, fontweight='bold', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_scr_illustration(filename):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Generated Image Representation
    gen_box = patches.Rectangle((0.5, 1), 3, 2, linewidth=2, edgecolor='#9467bd', facecolor='#f2e6ff')
    ax.add_patch(gen_box)
    ax.text(2, 3.2, "Generated Image (4 Subjects)", fontsize=11, fontweight='bold', ha='center', va='center')
    
    # 4 Subjects inside
    colors = ['#2ca02c', '#2ca02c', '#d62728', '#2ca02c'] # 3 pass, 1 fail
    labels = ['Subj 1\nDINO: 0.65', 'Subj 2\nDINO: 0.55', 'Subj 3\nDINO: 0.15', 'Subj 4\nDINO: 0.70']
    
    for i in range(4):
        x = 0.8 + (i%2)*1.4
        y = 1.2 + (i//2)*0.8
        sub = patches.Rectangle((x, y), 1.2, 0.6, linewidth=1, edgecolor=colors[i], facecolor='white')
        ax.add_patch(sub)
        ax.text(x+0.6, y+0.3, labels[i], fontsize=8, ha='center', va='center', color=colors[i])

    # Arrow to Threshold
    ax.annotate('', xy=(4.5, 2), xytext=(3.5, 2), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))

    # Threshold Box
    thresh_box = patches.Rectangle((4.5, 1.5), 1.5, 1, linewidth=2, edgecolor='gray', facecolor='#f0f0f0')
    ax.add_patch(thresh_box)
    ax.text(5.25, 2, "Threshold\n$\\tau = 0.4$", fontsize=10, fontweight='bold', ha='center', va='center')

    # Arrow to Result
    ax.annotate('', xy=(7, 2), xytext=(6, 2), arrowprops=dict(facecolor='gray', shrink=0.05, width=2))

    # Result Box
    res_box = patches.Rectangle((7, 1), 1.5, 2, linewidth=2, edgecolor='#d62728', facecolor='#ffe6e6')
    ax.add_patch(res_box)
    ax.text(7.75, 2.5, "1 Collapsed", fontsize=10, fontweight='bold', color='#d62728', ha='center', va='center')
    ax.text(7.75, 2.0, "3 Passed", fontsize=10, fontweight='bold', color='#2ca02c', ha='center', va='center')
    ax.text(7.75, 1.3, "SCR = 1/4\n= 25%", fontsize=12, fontweight='bold', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    out_dir1 = "/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/figures"
    out_dir2 = "/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/images"
    
    create_benchmark_design(f"{out_dir1}/benchmark_design.pdf")
    create_benchmark_design(f"{out_dir2}/benchmark_design.pdf")
    
    create_scr_illustration(f"{out_dir1}/scr_illustration.pdf")
    create_scr_illustration(f"{out_dir2}/scr_illustration.pdf")
