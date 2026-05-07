# Section 4.1.2

## LaTeX-Ready Body

Copy the block below directly into `main.tex` under `\subsubsection{MIB-Gold Results}`.

```tex
While Section~4.1.1 shows that our SOP yields reliable large-scale silver supervision, \texttt{MIB-Gold} serves a different role: it is the human-controlled benchmark that tests whether multi-subject evaluation remains reliable once ambiguity is no longer filtered by model-model agreement alone. In total, \texttt{MIB-Gold} contains 4,020 raw pair groups. Of these, 1,500 come from the \texttt{Nano Banana + Mosaic} subset, while the remaining 2,520 are drawn from a broader cross-platform pool. After preference-consistency filtering, the retained-pair rate is 94.1\% for the \texttt{Nano Banana + Mosaic} subset and 90.4\% for the broader cross-platform subset, indicating that difficulty is not uniform across generator sources.

This difficulty is structured rather than noisy. As shown in Figure~\ref{fig:mib_gold_benchmark}, human preference consistency rises from 87.0\% at level 2 to 94.9\% at level 6, and remains high at level 8. Even as scenes become denser and compositionally more demanding, annotators still converge on a stable preference signal after consistency control. Thus, \texttt{MIB-Gold} does not simply introduce additional annotation variance; it reveals a meaningful difficulty distribution while preserving a dependable human judgment target.

Taken together, these results establish \texttt{MIB-Gold} as more than a small human-labeled subset. It is the benchmark layer that makes the rest of our evaluation possible: difficult enough to expose non-trivial disagreement, yet reliable enough to serve as the reference set for testing both existing metrics and learned evaluators.
```

## LaTeX Figure Import

The current file [section_4_1_2_gold_benchmark.png](file:///Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/paper_data/images/section_4_1_2_gold_benchmark.png) is already a composed two-panel figure, so the recommended usage is to import it as one complete figure.

```tex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/section_4_1_2_gold_benchmark.png}
    \caption{Human annotation results on \texttt{MIB-Gold}. Left: preference consistency by benchmark level. Right: preference consistency by spatial relationship. The lower retention in the broader cross-platform subset shows that benchmark difficulty is systematic rather than annotation noise.}
    \label{fig:mib_gold_benchmark}
\end{figure}
```

## Main Data Sources

- `paper_data/section_4_1_2_mib_gold/gold_human_annotation_summary.json`
- `paper_data/section_4_1_2_mib_gold/gold_summary_by_level.csv`
- `paper_data/section_4_1_2_mib_gold/gold_summary_by_class_tag.csv`
- `paper_data/images/section_4_1_2_gold_benchmark.png`
