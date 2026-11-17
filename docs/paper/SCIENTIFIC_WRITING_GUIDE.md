# Masterclass: Scientific Paper Writing & LaTeX Standards for Top-Tier NLP Conferences (ACL / EMNLP)

> **Target Audience:** Researchers and Software Engineers aiming for publication in ACL, EMNLP, NAACL, ICML, or NeurIPS.  
> **Core Principle:** A scientific paper is not a chronological diary of implementation steps; it is a **rigorous, deductive scientific argument** validated by controlled empirical evidence.

---

## 1. The Rhetorical Architecture of a Paper (The "Hourglass Model")

Every top-tier paper follows a geometric rhetorical structure: starting broad, narrowing down to an exact mathematical bottleneck, presenting the proposed solution, and then expanding back out to empirical and real-world implications.

```
       [ Broad Context: Importance of LLM Efficiency in Fact-Checking ]
                            \                              /
                             \   The Unsolved Gap:        /
                              \   Speculative Divergence /
                               \  on Evidence Tokens    /
                                [  Diagnostic Discovery  ]
                                [ (M3: 82K+ Token Traces) ]
                                [   SPADE Formulation     ]
                                [ (Biasing + Dynamic K(t))]
                               / [ Theoretical Guarantee ] \
                              /  [  (Theorem 1 Invariance) ]\
                             /                               \
                            /   Empirical Verification:       \
                           /    100 Claims Benchmark Matrix    \
                          /                                     \
                         [ Broader Implications & Serving Engines ]
```

---

## 2. Section-by-Section Scientific Logic

### 2.1. Title & Abstract
- **The Title Formula:** `[Catchy Acronym]: [Principle / Mechanism] for [Desired Outcome] in [Target Domain]`
  - *Example:* **SPADE: Evidence-Adaptive Speculative Decoding for Efficient and Distribution-Preserving Fact-Checking Summarization**.
- **The 5-Sentence Abstract Formula:**
  1. **Context & Stakes:** Why does this problem matter to the field?
  2. **The Critical Bottleneck:** What fails in current SOTA methods, and why?
  3. **The Proposed Method:** What is the core algorithmic innovation?
  4. **The Theoretical Guarantee:** Why is this mathematically sound (e.g., Theorem 1)?
  5. **The Quantitative Punchline:** Give the exact numbers ($4.04\times$ reduction, $+1.07\%$ evidence gain, $100\%$ format validity).

### 2.2. Introduction: The 4-Anchor Flow
A winning Introduction answers four questions across 1.5–2 pages:
1. **What is the problem?** LLMs are crucial for fact-checking summarization, but autoregressive generation is bounded by memory bandwidth.
2. **Why don't existing baselines solve it?** Speculative decoding treats all tokens identically. Small draft models diverge on evidence-dense factual entities.
3. **What is our diagnostic discovery?** Show the empirical observation upfront! Over $72.7\%$ of rejections occur at positions 0 and 1, and factual tokens have lower acceptance ($65.9\%$) than punctuation ($88.1\%$).
4. **What are our concrete contributions?** Summarize them in a bulleted list:
   - *Diagnostic Benchmark:* First token-level rejection analysis ($82,850+$ traces).
   - *Algorithmic Innovation:* SPADE (Evidence n-gram indexer + Entropy-gated logit biasing).
   - *Dynamic Speculative Horizon:* $K(t) \in \{2, 4, 6\}$ eliminating wasted draft computation.
   - *Empirical & Theoretical Validation:* Rigorous evaluation on RTX 5090 GPU with exact distribution preservation.

### 2.3. Methodology: Formal Mathematical Rigor
- **Never describe algorithms purely in prose or code snippets.** Use standard mathematical notation:
  - Sequences: $x_{<t} = (x_1, \dots, x_{t-1})$.
  - Models & Distributions: $\mathcal{M}_p, \mathcal{M}_q$, with conditional probabilities $p(\cdot \mid x)$ and $q(\cdot \mid x)$.
  - Vector operations: Logits $z_t \in \mathbb{R}^{|\mathcal{V}|}$, Shannon entropy $\mathcal{H}(q) = -\sum_{v} q(v) \log q(v)$.
- **State Invariants as Formal Theorems:**
  - Explicitly cite and prove distribution preservation. If reviewers suspect heuristic approximations, the paper will be rejected. Theorem 1 guarantees exact fidelity to the target distribution.
- **Provide Algorithm Pseudocode:**
  - An `algorithm` environment with inputs, outputs, clear line numbers, and concise comments.

### 2.4. Experiments & Results: Causality & Ablations
- **Table 1: The Main Matrix:** Include all baselines (Autoregressive Greedy, Stochastic, Speculative Greedy, Exact Sampling at different $K$, and Proposed Methods).
- **Table 2 & 3: Causal Ablations:**
  - *Table 2 (Mechanism Isolation):* Show Evidence-Overlap vs. Non-Overlap tokens to prove the biasing actually worked where intended.
  - *Table 3 (Horizon Isolation):* Compare Fixed $K=6$ vs. Dynamic $K(t)$ to prove the adaptive controller eliminates wasted proposals.
- **Self-Contained Captions:** Reviewers skim papers by looking at tables and figures first. Every caption must state:
  1. What is being compared.
  2. The dataset and model setup.
  3. The core conclusion (e.g., *"SPADE yields the lowest target verification round count (2,311 rounds) while eliminating 1,376 wasted proposals"*).

### 2.5. Diagnostic Analysis: Revealing the "Why"
Top papers don't just report "X is better than Y." They explain **why** X is better:
- **Geometric Decay of Rejection:** Plot the histogram of rejection positions $k=0 \dots 5$.
- **Semantic Token Sensitivity:** Group tokens by regex/heuristics into Numbers, Punctuation, Dates, and Entities to demonstrate domain-specific divergence.

### 2.6. Intellectual Honesty: Limitations & Discussion
- Reviewers reject papers that oversell or conceal bottlenecks.
- Be transparent about PyTorch eager execution overhead vs. fused CUDA serving engines (vLLM / TensorRT-LLM).
- Frame the software architecture as a necessary diagnostic vehicle to collect probability traces, and outline the exact roadmap to fused serving.

---

## 3. LaTeX Formatting & Presentation Standards

### 3.1. The Golden Rules of Tables (`booktabs`)
1. **Never use vertical lines (`|`).** Professional typesetting uses whitespace to separate columns.
2. **Use three horizontal line commands:** `\toprule` (top), `\midrule` (under headers), and `\bottomrule` (bottom).
3. **Bold the best result in each column** (`\textbf{...}`).
4. **Separate units from values:** Put units in the header column (e.g., `Throughput (tok/s)`), not inside every single cell.

```latex
% Correct ACL Table Format
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Draft $K$} & \textbf{Accept. $\alpha$} & \textbf{Acc/Rnd} & \textbf{Target Rounds} \\
\midrule
B2 (Exact) & 4 & 61.35\% & 2.44 & 2,771 \\
SPADE (Fixed) & 4 & 62.33\% & 2.48 & 2,754 \\
SPADE (Dynamic) & Dyn $\{2,4,6\}$ & \textbf{55.62\%} & \textbf{2.98} & \textbf{2,311} \\
\bottomrule
\end{tabular}
\caption{\textbf{Core Speculative Performance:} SPADE Dynamic $K(t)$ achieves the lowest target rounds.}
\end{table}
```

### 3.2. Math Notation Conventions
- **Probability:** Use `\Pr` or `p(\cdot)`.
- **Vectors / Matrices:** Bold lowercase for vectors ($\mathbf{x}$), bold uppercase for matrices ($\mathbf{W}$).
- **Sets / Alphabets:** Calligraphic font (`\mathcal{V}` for vocabulary, $\mathcal{E}$ for evidence).
- **Expectation / Variance:** Blackboard bold (`\mathbb{E}`, `\mathbb{R}`).
- **Definitions / Theorems:** Use `amsthm` environments (`\newtheorem{theorem}{Theorem}`).

### 3.3. BibTeX Cleanliness (`custom.bib`)
- **Capitalization Locks:** BibTeX automatically lowercases titles unless protected by curly braces `{}`:
  - *Wrong:* `title = {Fast inference from transformers via speculative decoding}`
  - *Correct:* `title = {Fast Inference from {Transformers} via Speculative Decoding}`
  - *Keywords to protect:* `{Qwen}`, `{AVeriTeC}`, `{GPU}`, `{CUDA}`, `{LLM}`, `{BERT}`, `{FEVER}`.
- **Conference Names:** Use clean, consistent proceedings names (e.g., `Proceedings of the 40th International Conference on Machine Learning (ICML)`).

### 3.4. Overleaf Compilation Checklist
When uploading to Overleaf:
1. Ensure all figures referenced in `\includegraphics{figures/...}` exist in the `figures/` folder.
2. Use `\usepackage{microtype}` for optimal kerning and paragraph justification.
3. Check for `overfull \hbox` warnings in equations and wide tables (use `\small` or `resizebox` if needed).
4. Run `bibtex` $\rightarrow$ `pdflatex` $\rightarrow$ `pdflatex` to resolve all cross-references (`\Cref`).
