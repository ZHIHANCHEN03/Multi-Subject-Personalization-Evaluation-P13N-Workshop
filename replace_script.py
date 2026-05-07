import re

file_path = "/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper_Neruips/V3/MIBE__Multi_subject_Interaction_Benchmark_and_Evaluator_for_Personalized_Generation/neurips_2026.tex"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

new_section_3_2 = r"""\subsection{Evaluator Modeling: MIE}

The primary goal of MIBE is to establish a rigorous evaluation framework for multi-subject personalized image generation. To demonstrate the reusability of our benchmark and provide an immediate, out-of-the-box tool for the community, we instantiate the \textbf{Multi-subject Interaction Evaluator (MIE)}. Trained exclusively on MIB-Silver and assessed on MIB-Gold, MIE bridges the gap between human judgment and automated metrics.

\paragraph{Input-Output Formulation.}
For each task, let $p$ denote the prompt, $R=\{r_1,\dots,r_N\}$ denote the set of reference images, and let $y$ denote a candidate generation. MIE is a reference-conditioned multimodal evaluator that maps the triplet $(R,p,y)$ to two types of outputs:
\begin{enumerate}
    \item A scalar quality score $s_\theta(R,p,y)\in\mathbb{R}$ used for global pairwise ranking.
    \item A three-dimensional diagnostic prediction vector of logits
    $\hat d_\theta(R,p,y) = (\hat d_{\text{exist}}, \hat d_{\text{app}}, \hat d_{\text{inter}})$,
    corresponding to \emph{Existence}, \emph{Appearance}, and \emph{Interaction}.
\end{enumerate}

\paragraph{Why a Dual-Head Design?} 
A purely diagnostic model with binary labels cannot capture relative failure severity (e.g., missing one subject vs. missing all). Conversely, a purely scalar reward model acts as an opaque black box, offering no actionable feedback on the nature of the failure. MIE solves this via a dual-head architecture. The \textbf{ranking head} provides a continuous global signal for pairwise comparison, which is essential for leaderboard benchmarking. Simultaneously, the \textbf{diagnostic head} explicitly attributes errors to Existence, Appearance, or Interaction, providing interpretable feedback for model developers. 

By optimizing these jointly ($\mathcal{L} = \alpha \mathcal{L}_{\text{rank}} + \beta \mathcal{L}_{\text{diag}}$), the scalar score is forced to ground itself in concrete binding failures rather than superficial aesthetics. This design mirrors the human annotation procedure directly, preserving both ranking separability and interpretability.

\paragraph{Why Lightweight Tuning?} 
To ensure that the evaluator remains practical and highly reusable, we adopt a parameter-efficient fine-tuning strategy (LoRA and layer-only updates) over off-the-shelf vision-language backbones (e.g., Qwen3.5-VLM). This design choice proves the representativeness and high quality of our dataset: one does not need to pretrain a massive multimodal model from scratch; simply fine-tuning a lightweight VLM on MIB-Silver yields an evaluator that highly correlates with humans. This guarantees that MIB-Silver is a highly reusable resource for future metric alignment in the community.
"""

new_section_4 = r"""\section{Results}
All training and evaluation are conducted on a single NVIDIA A100 GPU, with 16 vCPUs and 251 GB of system memory.

\subsection{Dataset Robustness and Representativeness}
To ensure MIB serves as a reliable community standard, we first validate the quality of our large-scale annotations. Our 60K \texttt{MIB-Silver} set, generated via SOP-guided dual-VLM consensus, achieves a remarkable 95.1\% preference agreement across 59,852 matched tasks. This confirms that our factorized prompt design and error-first SOP successfully anchor large-scale, scalable supervision. 

Furthermore, the \texttt{MIB-Gold} set is genuinely challenging. After consistency filtering, the retained-pair rate is 94.1\% on seen generators and 90.4\% on unseen generators. Crucially, human preference consistency rises from 87.0\% at 2 subjects to 94.9\% at 6 subjects, remaining high even at 8 subjects. This proves that MIB's difficulty is structurally meaningful rather than noisy: denser scenes are undeniably harder for generative models, yet human judgments stay coherent. This establishes \texttt{MIB-Gold} as a highly representative testbed spanning diverse generator failures.

\subsection{The Illusion of Existing Metrics on MIB-Gold}
To establish a comprehensive performance ceiling, we select a representative suite of baselines spanning four distinct paradigms: (1) Low-level Reconstruction (PSNR, SSIM); (2) Semantic Alignment (DINOv2, CLIP, SigLIP); (3) General Human Preference (PickScore, ImageReward, HPS v2.1); and (4) Identity-Specific Metrics (SCR). We compare these automatic metrics against 4,991 valid human pairwise preference annotations on \texttt{MIB-Gold}.

The comparison reveals a fundamental gap in current evaluation paradigms. General-purpose preference scorers completely fail to proxy human binding judgments: HPS v2.1 (0.520) is barely above random guess, PickScore (0.486) falls slightly below random, and PSNR (0.399) is substantially misaligned. While identity-specialized metrics like SCR and DINOv2 show non-trivial alignment on single-dimensional facets, baseline performance degrades sharply as the number of requested subjects increases. This confirms that multi-subject binding breaks the assumptions behind standard image-quality surrogates, underscoring the critical need for a unified, multi-dimensional evaluator.

\subsection{MIE Restores Human Alignment and Generalization}
\label{sec:mie_alignment}
MIB does not merely reveal that existing metrics fail; it enables the training of a robust evaluator. Across six exported MIE checkpoints, the strongest variant is \texttt{qwen35\_4b\_lora\_layer}, which achieves an overall pairwise accuracy of 0.922 against human annotations. Remarkably, it reaches 0.982 on seen generators and maintains a robust 0.884 on unseen generators, substantially exceeding the strongest third-party baselines even under distribution shift. 

The gains extend beyond pairwise ranking. The \texttt{4B lora\_layer} model reaches a macro-F1 of 0.818 on fine-grained diagnostics. As shown in Figure~\ref{fig:mie_alignment}, MIE successfully captures the meaningful diagnostic structure underlying human judgments across Existence, Appearance, and Interaction, proving it is not merely learning a shallow aesthetic preference.

\paragraph{MIE Breakdown and Scaling Analysis.} 
We analyze the source of MIE's gains. The full scaling story consistently shows that parameter-efficient tuning (LoRA-layer) dominates layer-only adaptation. At the 2B scale, adding LoRA-layer tuning improves pairwise accuracy by 0.061 and macro-F1 by 0.116. Even at the 4B scale, LoRA-layer still yields gains of 0.046 in accuracy and 0.044 in macro-F1 over layer-only tuning. Furthermore, the generalization gap (unseen minus seen) is smallest for the \texttt{4B lora\_layer} variant ($-0.098$) and largest for \texttt{2B layer\_only} ($-0.221$). 

The category-level breakdown (Figure~\ref{fig:mie_breakdown}) confirms our human annotation observations (Appendix~\ref{sec:appendix_failure_modes}): \emph{Existence} is the easiest dimension to evaluate, while \emph{Interaction} remains the hardest binding dimension, particularly on unseen generators. Ultimately, these results demonstrate the reusability of MIB: by combining our dataset with the right adaptation strategy, the community can train scalable evaluators that maintain sharp diagnostic discrimination and strong human alignment.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/section_4_2_3_mie_breakdown.png}
    \caption{Breakdown analysis of MIE variants. Left: seen-to-unseen generalization gap. Middle: LoRA-layer gains over layer-only tuning at different model scales. Right: category-level F1 for the strongest checkpoint. The results show that LoRA-layer tuning improves diagnostic quality across scales and that interaction remains the hardest binding dimension, especially under generator shift.}
    \label{fig:mie_breakdown}
\end{figure}
"""

# Regex to find section 3.2 and section 4
pattern = re.compile(r"\\subsection\{Evaluator Modeling\}.*?\\section\{Limitations\}", re.DOTALL)

# Replacement text
replacement = new_section_3_2 + "\n\n" + new_section_4 + "\n\n\\section{Limitations}"

# Pass a lambda to re.sub so it doesn't process escape sequences
new_content = pattern.sub(lambda match: replacement, content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Replacement successful.")
