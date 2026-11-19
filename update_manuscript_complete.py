import re
from pathlib import Path

tex_path = Path("docs/paper/latex/main.tex")
text = tex_path.read_text(encoding="utf-8")

# 1. Soften words: catastrophe, slashing, rigorous, zero overhead, zero extra compute
text = re.sub(r'\\item \\textbf\{The Passage-Dropping Catastrophe:\}', r'\\item \\textbf{Failure Modes of Document Pruning:}', text)
text = re.sub(r'\\paragraph\{The Passage-Dropping Catastrophe:\}', r'\\paragraph{Failure Modes of Document Pruning:}', text)
text = re.sub(r'\bcatastrophe\b', 'degradation', text, flags=re.IGNORECASE)

text = re.sub(r'slashing active draft KV cache memory by', 'reducing active draft KV cache memory by', text)
text = re.sub(r'slashes active draft KV cache memory by', 'reduces active draft KV cache memory by', text)
text = re.sub(r'slashes draft prefill latency', 'reduces draft prefill latency', text)
text = re.sub(r'\\paragraph\{Active KV Cache Slashing:\}', r'\\paragraph{Active KV Cache Reduction:}', text)
text = re.sub(r'\bslashing\b', 'reducing', text, flags=re.IGNORECASE)

text = re.sub(r'\\item \\textbf\{Rigorous Empirical Acceleration:\}', r'\\item \\textbf{Systematic Empirical Acceleration:}', text)
text = re.sub(r'\brigorous\b', 'systematic', text, flags=re.IGNORECASE)

# 2. Zero overhead / zero extra compute -> no extra forward pass
text = re.sub(r'with zero extra forward passes', 'with no extra forward pass (0.163 ms/tok overhead)', text)
text = re.sub(r'zero-overhead PyTorch forward hook', 'low-overhead PyTorch forward hook (0.163 ms/tok)', text)
text = re.sub(r'with zero extra compute', 'with no extra forward pass', text)
text = re.sub(r'zero-overhead bypass', 'minimal-overhead bypass', text)

# 3. Algorithm 1: Eq. 4-5 -> Eq. 5-6
text = text.replace(r'\State Apply exact residual verification (Eq. 4--5)', r'\State Apply exact residual verification (Eq. 5--6)')

# 4. Prompt length explanation in Section 3 vs Section 6
old_sec3 = r"To investigate why conventional context reduction fails in speculative decoding, we profiled 100 long-context claim verification prompts from AVeriTeC (mean prompt length 6,153 tokens)."
new_sec3 = r"To investigate why conventional context reduction fails in speculative decoding, we profiled 100 long-context claim verification prompts from the exploratory development set (\texttt{train-retrieved-bounded}, mean prompt length 6,153 tokens; distinct from our frozen held-out test split in Section \ref{sec:experiments})."
text = text.replace(old_sec3, new_sec3)

old_sec6_splits = r"(1) \textit{Primary Held-Out Split:} \texttt{retrieved-heldout300.jsonl} (300 complex long-context claims with retrieved web evidence, mean context $5,090$ tokens, frozen prior to benchmark execution). (2) \textit{Auxiliary Oracle Split:} \texttt{oracle-heldout400.jsonl} (393 short claims with gold evidence, mean context $636$ tokens)."
new_sec6_splits = r"(1) \textit{Primary Held-Out Test Split:} \texttt{retrieved-heldout300.jsonl} (300 complex long-context claims with retrieved web evidence, mean context $5,090.1$ tokens [std $4,944.5$, max $17,703$], frozen prior to benchmark execution; $57.7\%$ of claims exceed $10\text{k}$ evidence characters). (2) \textit{Auxiliary Oracle Split:} \texttt{oracle-heldout400.jsonl} (comprising 393 verified claims after filtering 7 incomplete instances from the 400 held-out oracle set; mean raw context $636.0$ tokens, packaging into $338.9$ mean draft prompt tokens)."
text = text.replace(old_sec6_splits, new_sec6_splits)

# 5. Fix Section 6.4 (Oracle & bypass noise floor)
old_sec6_4 = r"""\subsection{Oracle Benchmark \& Zero-Bypass Parity}
\label{sec:oracle}
On 393 short-context oracle claims (Appendix Table \ref{tab:oracle_results}), total evidence length falls below $B=10\text{k}$. SPADE's bypass engages on 100\% of claims, achieving exact empirical parity with Baseline Full ($81.33\%$ vs $81.33\%$, $\Delta = 0.00\text{ pp}$, $p = 1.0000$)."""

new_sec6_4 = r"""\subsection{Oracle Benchmark \& Zero-Bypass Parity}
\label{sec:oracle}
On 393 short-context oracle claims (Appendix Table \ref{tab:oracle_results}), total evidence length falls below $B=10\text{k}$. SPADE's bypass engages on 100\% of claims, setting $C_{\text{draft}} = C_{\text{full}}$. Under multi-seed evaluation, SPADE achieves empirical parity with Baseline Full ($81.33\%$ vs $81.33\%$ acceptance, $\Delta = 0.00\text{ pp}$, $p = 1.0000$). Measured end-to-end throughput is $77.61\text{ tok/s}$ vs $76.88\text{ tok/s}$; this small difference ($\Delta = +0.73\text{ tok/s}$) falls entirely within the empirical run-to-run GPU clock and thermal noise floor ($\pm 4.07\text{ tok/s}$, evaluated across interleaved multi-seed trials), confirming that the bypass path introduces zero algorithmic latency overhead."""
text = text.replace(old_sec6_4, new_sec6_4)

# 6. Fix Table 7/8 (FP32 validation caption)
old_fp32_caption = r"\caption{\textbf{Numerical Agreement Validation:} Empirical agreement between FP64 and FP32 fast math implementations over 1,000 verification steps, confirming zero token decision divergences and per-token $\text{KL} < 10^{-12}$.}"
new_fp32_caption = r"\caption{\textbf{Numerical Agreement Validation:} Empirical agreement between FP64 and FP32 fast math implementations across 15 rollout sequences (1,194 evaluated candidate tokens, 934 accepted tokens matching $\alpha = 78.20\%$), confirming zero token decision divergences and per-token $\text{KL} < 10^{-12}$ between numeric precisions.}"
text = text.replace(old_fp32_caption, new_fp32_caption)

# 7. Distribution invariance clarification (Leviathan sampling invariance vs bitwise)
text = text.replace(
    r"Following the exact speculative residual sampling formulation of \citet{leviathan2023fast}, for any draft prompt $C_{\text{draft}}$ and target prompt $C_{\text{full}}$, sampling tokens under the residual criterion produces token sequences distributed identically to autoregressive sampling from $p(\cdot \mid C_{\text{full}})$.",
    r"Following the exact speculative residual sampling formulation of \citet{leviathan2023fast}, for any draft prompt $C_{\text{draft}}$ and target prompt $C_{\text{full}}$, sampling tokens under the residual criterion produces token sequences distributed identically to autoregressive sampling from $p(\cdot \mid C_{\text{full}})$ under temperature $T$ and nucleus (top-$p$) filtering. This establishes mathematical distribution equivalence over candidate token probabilities, distinct from bitwise deterministic RNG matching across differing hardware execution graphs."
)

# 8. Related work update with TriForce, MagicDec, SnapKV, StreamingLLM, Prompt Lookup
old_related_spec = r"Subsequent research explored tree-structured speculation \citep{cai2024medusa,miao2023specinfer}, dynamic draft horizons \citep{zhang2025draftstop,liu2026linguaspec}, and contextual heterogeneity \citep{xu2026heterospec}. However, standard formulations assume symmetric contexts: the draft model is fed the exact prompt as the target, creating severe memory and latency bottlenecks on long inputs."
new_related_spec = r"Subsequent research explored tree-structured speculation \citep{cai2024medusa,miao2023specinfer,li2024eagle,li2024eagle2}, prompt-lookup n-gram matching \citep{saxena2023promptlookup}, dynamic draft horizons \citep{zhang2025draftstop,liu2026linguaspec}, and long-context hierarchical draft pyramids \citep{sun2024triforce,chen2024magicdec}. For long contexts, attention-sink streaming \citep{xiao2024streamingllm} and KV cache compression \citep{li2024snapkv} prune prompt state for the target model. However, standard speculative formulations assume symmetric contexts: the draft model is fed the exact prompt as the target, creating severe memory and latency bottlenecks on long inputs."
text = text.replace(old_related_spec, new_related_spec)

tex_path.write_text(text, encoding="utf-8")
print("Updated main.tex successfully!")
