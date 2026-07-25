# Section 4.2.3

## LaTeX-Ready Body

Copy the block below directly into `main.tex` under `\subsubsection{MIE Breakdown Analysis}`.

```tex
We next ask where the gains of the learned evaluator actually come from. A consistent pattern emerges across all six checkpoints: every model performs better on the seen-generator subset than on the unseen-generator subset, confirming that cross-generator generalization remains a non-trivial challenge. However, the size of this gap depends strongly on the tuning regime. The smallest seen-to-unseen drop is achieved by \texttt{qwen35\_4b\_lora\_layer} at $-0.098$, whereas the largest drop appears in \texttt{2b layer\_only} at $-0.221$. This shows that additional model capacity alone is not enough; how that capacity is adapted is equally important.

The LoRA-versus-layer comparison reinforces this conclusion. At 2B, LoRA-layer tuning improves pairwise accuracy by 0.061 and macro-F1 by 0.116 relative to \texttt{2B layer\_only}. At 4B, it still yields gains of 0.046 in pairwise accuracy and 0.044 in macro-F1 over \texttt{4B layer\_only}. Even at 0.8B, where the pairwise gain is nearly neutral, LoRA-layer tuning improves macro-F1 by 0.128. These results suggest that the primary benefit of LoRA-layer tuning is not merely stronger ranking, but consistently sharper diagnostic discrimination.

The category-level breakdown points to the same conclusion. \texttt{Existence} remains the easiest dimension, whereas \texttt{Appearance} and especially \texttt{Interaction} are substantially harder, particularly on the unseen-generator subset. As shown in Figure~\ref{fig:mie_breakdown}, the strongest checkpoint achieves the best balance between overall alignment and fine-grained diagnostic sensitivity under generator shift. Overall, the breakdown analysis clarifies that the best-performing evaluator is not simply the largest model, but the model that combines additional capacity with the right adaptation strategy and preserves diagnostic sensitivity under distribution shift.
```

## LaTeX Figure Import

The current file [section_4_2_3_mie_breakdown.png](file:///Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/paper_data/images/section_4_2_3_mie_breakdown.png) is already a composed three-panel figure, so the recommended import style is a single `figure` environment.

```tex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/section_4_2_3_mie_breakdown.png}
    \caption{Breakdown analysis of MIE variants. Left: seen-to-unseen generalization gap. Middle: LoRA-layer gains over layer-only tuning at different model scales. Right: category-level F1 for the strongest checkpoint. LoRA-layer tuning improves diagnostic quality across scales, while interaction remains the hardest binding dimension, especially under generator shift.}
    \label{fig:mie_breakdown}
\end{figure}
```

## Main Data Sources

- `paper_data/section_4_2_3_breakdown/mie_seen_unseen_table.csv`
- `paper_data/section_4_2_3_breakdown/mie_lora_vs_layer_table.csv`
- `paper_data/section_4_2_3_breakdown/mie_scaling_table.csv`
- `paper_data/section_4_2_3_breakdown/mie_category_by_dataset.csv`
- `paper_data/images/section_4_2_3_mie_breakdown.png`
