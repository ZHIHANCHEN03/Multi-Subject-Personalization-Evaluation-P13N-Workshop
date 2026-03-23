import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import os

def load_data():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/eval_outputs'
    df_mosaic = pd.read_csv(f'{base_dir}/mosaic_summary.csv')
    df_mosaic['Model'] = 'MOSAIC'
    df_xverse = pd.read_csv(f'{base_dir}/xverse_summary.csv')
    df_xverse['Model'] = 'XVerse'
    df_psr = pd.read_csv(f'{base_dir}/psr_summary.csv')
    df_psr['Model'] = 'PSR'
    return pd.concat([df_mosaic, df_xverse, df_psr], ignore_index=True)

def plot_radar(df, output_dir):
    df_filtered = df[(df['scene_type'] == 'all') & (df['subject_count'] == 2)]
    
    metrics = ['clip_t', 'clip_i', 'dinov2', 'scr@0.4']
    labels = ['CLIP-T\n(Semantic)', 'CLIP-I\n(Style)', 'DINOv2\n(Identity)', 'SCR@0.4\n(Collapse)']
    
    models = ['MOSAIC', 'XVerse', 'PSR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    for i, model in enumerate(models):
        model_data = df_filtered[df_filtered['Model'] == model]
        if len(model_data) == 0: continue
        
        values = []
        for m in metrics:
            val = model_data[m].values[0]
            # Normalize to 0-1 scale relative to typical max values for radar aesthetics
            if m == 'clip_t': val = val / 0.35
            if m == 'clip_i': val = val / 0.85
            if m == 'dinov2': val = val / 0.55
            if m == 'scr@0.4': 
                # SCR is lower the better (0 to 1). We invert it to 1-SCR 
                # so that a larger radar area means better performance.
                val = 1.0 - val
            values.append(val)
            
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12, fontweight='bold')
    ax.set_yticklabels([]) # Hide radial ticks
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig_radar_metrics.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {output_path}")

if __name__ == '__main__':
    df = load_data()
    output_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/images'
    plot_radar(df, output_dir)
