# Section 4.2.2

## LaTeX-Ready Body

Copy the block below directly into `main.tex` under `\subsubsection{MIE Aligns Better with Human Preference}`.

```tex
If Section~4.2.1 establishes that existing metrics are fundamentally misaligned with human preference on \texttt{MIB-Gold}, the next question is whether this benchmark can be used to train a better evaluator. Our results show that it can. Across all six exported checkpoints, the strongest variant is \texttt{qwen35\_4b\_lora\_layer}, which achieves an overall pairwise accuracy of 0.922, including 0.982 on the seen-generator subset and 0.884 on the unseen-generator subset. This substantially exceeds the strongest third-party baseline, showing that supervision derived from \texttt{MIB} can be translated into a materially stronger human-aligned metric.

The gains are not limited to pairwise ranking. The same 4B LoRA-layer model reaches a macro-F1 of 0.818, indicating that the evaluator is not merely learning a shallow preference score, but capturing a meaningful portion of the fine-grained diagnostic structure underlying human judgments. This matters because the binding problem in multi-subject personalization is inherently multi-dimensional: a useful evaluator must jointly reason about existence, appearance, and interaction rather than collapse everything into a single weak aesthetic signal.

More broadly, the six-checkpoint comparison now supports a full scaling narrative. LoRA-layer variants consistently outperform their layer-only counterparts, and the largest LoRA-based model achieves the strongest overall human alignment. As shown in Figure~\ref{fig:mie_alignment}, the advantage is visible not only in pairwise accuracy but also in the category-level F1 breakdown. Thus, \texttt{MIB} does not only expose the failure of existing metrics; it also enables the training of a new evaluator that tracks human preference far more faithfully in multi-subject settings.
```

## LaTeX Figure Import

The current file [section_4_2_2_mie_alignment.png](file:///Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/paper_data/images/section_4_2_2_mie_alignment.png) is already a composed two-panel figure, so it should be imported as a single figure block.

```tex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/section_4_2_2_mie_alignment.png}
    \caption{Human alignment of MIE variants on \texttt{MIB-Gold}. Left: overall, seen-generator, and unseen-generator pairwise accuracy. Right: category-level F1 across existence, appearance, and interaction. The 4B LoRA-layer evaluator is the strongest overall variant and remains clearly above third-party baselines under generator shift.}
    \label{fig:mie_alignment}
\end{figure}
```

## Main Data Sources

- `paper_data/section_4_2_2_mie_alignment/mie_overall_metrics.csv`
- `paper_data/section_4_2_2_mie_alignment/mie_category_metrics.csv`
- `paper_data/section_4_2_2_mie_alignment/mie_vs_human_summary.json`
- `paper_data/images/section_4_2_2_mie_alignment.png`
