import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import os

def create_advanced_teaser():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    output_dir = os.path.join(base_dir, 'Paper', 'latex_source', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.05)

    # --- Panel 1: Quantitative Plot ---
    ax0 = fig.add_subplot(gs[0])
    N = [2, 4, 6, 8, 10]
    dinov2 = [0.425, 0.235, 0.164, 0.126, 0.110]
    scr = [48.9, 81.7, 90.7, 94.7, 96.0]
    
    color_dino = '#1f77b4'
    color_scr = '#d62728'
    
    ax0.plot(N, dinov2, marker='o', linestyle='-', color=color_dino, linewidth=3.5, markersize=10, label='Identity Fidelity (DINOv2) ↓')
    ax0.set_xlabel('Number of Interacting Subjects', fontsize=15, fontweight='bold')
    ax0.set_ylabel('DINOv2 Score', color=color_dino, fontsize=15, fontweight='bold')
    ax0.tick_params(axis='y', labelcolor=color_dino, labelsize=12)
    ax0.tick_params(axis='x', labelsize=12)
    ax0.set_xticks(N)
    ax0.grid(True, linestyle='--', alpha=0.6)

    ax0_twin = ax0.twinx()
    ax0_twin.plot(N, scr, marker='s', linestyle='-', color=color_scr, linewidth=3.5, markersize=10, label='Subject Collapse Rate (SCR) ↑')
    ax0_twin.set_ylabel('Subject Collapse Rate (%)', color=color_scr, fontsize=15, fontweight='bold')
    ax0_twin.tick_params(axis='y', labelcolor=color_scr, labelsize=12)
    
    lines_1, labels_1 = ax0.get_legend_handles_labels()
    lines_2, labels_2 = ax0_twin.get_legend_handles_labels()
    ax0.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=12, framealpha=0.9)
    
    ax0.set_title("The Illusion of Scalability:\nMetrics Collapse as N Increases", fontsize=18, fontweight='bold', pad=15)

    # --- Panel 2: 2 Subjects (Success) ---
    ax1 = fig.add_subplot(gs[1])
    img2_path = os.path.join(base_dir, 'results/mosaic/11_interaction/42.png')
    img2 = mpimg.imread(img2_path)
    ax1.imshow(img2)
    ax1.axis('off')
    ax1.set_title("N=2: Successful Composition\n(Identities Maintained)", fontsize=16, color='#2ca02c', fontweight='bold', pad=10)
    ax1.text(0.5, -0.1, 'Prompt: "A black woman and a\nwestern woman shaking hands."', 
             transform=ax1.transAxes, ha='center', va='top', fontsize=13, style='italic')
    
    rect = patches.Rectangle((0,0), img2.shape[1], img2.shape[0], linewidth=6, edgecolor='#2ca02c', facecolor='none')
    ax1.add_patch(rect)

    # --- Panel 3: 8 Subjects (Collapse) ---
    ax2 = fig.add_subplot(gs[2])
    img8_path = os.path.join(base_dir, 'results/mosaic/56_interaction/42.png')
    img8 = mpimg.imread(img8_path)
    ax2.imshow(img8)
    ax2.axis('off')
    ax2.set_title("N=8: Catastrophic Collapse\n(Attention Leakage & Clones)", fontsize=16, color='#d62728', fontweight='bold', pad=10)
    ax2.text(0.5, -0.1, 'Prompt: "Eight people of varying ethnicities\nand genders hugging in a circle."', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=13, style='italic')
    
    rect2 = patches.Rectangle((0,0), img8.shape[1], img8.shape[0], linewidth=6, edgecolor='#d62728', facecolor='none')
    ax2.add_patch(rect2)
    
    circle = patches.Ellipse((img8.shape[1]*0.55, img8.shape[0]*0.55), width=img8.shape[1]*0.7, height=img8.shape[0]*0.4, linewidth=3, edgecolor='#d62728', linestyle='--', facecolor='none')
    ax2.add_patch(circle)
    
    ax2.text(img8.shape[1]*0.55, img8.shape[0]*0.88, "Identity Bleeding\n& Homogenization", color='white', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='#d62728', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    
    out_pdf = os.path.join(output_dir, 'teaser.pdf')
    out_png = os.path.join(output_dir, 'teaser.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved advanced teaser to {out_pdf} and {out_png}")

if __name__ == '__main__':
    create_advanced_teaser()
