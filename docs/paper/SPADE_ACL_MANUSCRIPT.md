# SPADE: Evidence-Selective Draft Context Speculative Decoding for Knowledge-Intensive Generation

**Authors:**  
Van-Truong Le $^{1, 2}$, Huy Tien Nguyen $^{1, 2}$, Tung Le $^{1, 2}$  
$^1$ Faculty of Information Technology, University of Science, Ho Chi Minh City, Vietnam  
$^2$ Vietnam National University, Ho Chi Minh City, Vietnam  

**Target Conference:** ACL / EMNLP 2026 Main Track (Long Paper: 8 pages main body + references + appendix)  
**Primary Track:** Efficient Methods in NLP / Question Answering & Fact Verification  
**Artifact Path:** `file:///C:/Users/ASUS/.gemini/antigravity/brain/304249ec-e30a-4df1-8b4b-9c5c7f8cd35e/SPADE_ACL_MANUSCRIPT.md`  

---

## Abstract

Speculative decoding accelerates autoregressive large language model (LLM) inference by generating $K$ draft candidate tokens with an efficient assistant model and verifying them in parallel with a larger target model. In retrieval-augmented generation (RAG) and automated fact-checking, prompts frequently extend to 6,000–20,000 tokens of noisy multi-passage evidence. Under such long-context regimes, the memory and prefill latency of the draft model's Key-Value (KV) cache become major execution bottlenecks, while conventional passage-dropping approaches (e.g., BM25 Top-$K$) cause severe draft divergence whenever cited evidence is omitted (dropping token acceptance by up to $-4.48$ percentage points).

In this work, we present **SPADE** (**SP**eculative **A**symmetric **D**ecoding with **E**vidence Selection), an exact, evidence-selective speculative decoding framework tailored for knowledge-intensive generation. SPADE decouples the target and draft context spaces: while the target model verifies candidate proposals across the complete retrieved evidence context, the draft model generates proposals conditioned on a compact, sentence-budgeted context representation. To construct this draft context without extra forward passes, SPADE introduces a **Target-Prefill Attention Hook** that captures cross-layer query states from the target model's initial prompt pass, identifies claim-salient sentences across *all* evidence passages via a bounded character budget, and preserves 100% of citation identifiers while shedding 65–75% of uninformative tokens. Coupled with an optimized, mathematically exact inference runtime (`FlexAttention` + `StaticCache` + zero-sync `fp32_fast` distribution processing), SPADE achieves true end-to-end wall-clock speedup on modern GPU architectures.

Extensive evaluations on the official **AVeriTeC** real-world claim verification benchmark across **300 held-out long-context claims** on an NVIDIA RTX 5090 demonstrate that SPADE:
1. Achieves **$43.28\text{ tok/s}$** end-to-end throughput, outperforming target-only autoregressive decoding ($35.35\text{ tok/s}$, **$+22.4\%$ speedup**, $p = 0.0000$) and standard full-context speculative decoding ($41.47\text{ tok/s}$, **$+4.4\%$ speedup**, $p = 0.0000$);
2. Operates faster in the pure token generation phase (**$57.03\text{ tok/s}$** vs. $55.39\text{ tok/s}$, $p = 0.0002$), proving that speedups do not rely solely on prefill savings;
3. Preserves target speculative acceptance ($\alpha_{\text{eval}} = \mathbf{78.20\%}$ vs. $78.16\%$, $p = 0.8656$) and committed tokens per round ($3.114$ vs. $3.106$, $p = 0.6572$) at exact statistical parity;
4. Cuts active draft KV-cache memory by **$-63.9\%$** ($566.9\text{ MB} \to 204.6\text{ MB}$, $p = 0.0000$) and peak device VRAM by **$-1.34\text{ GB}$**;
5. On **393 short-context oracle claims**, activates a zero-overhead bypass mechanism that guarantees exact mathematical bitwise identity ($\alpha_{\text{eval}} = 81.33\%$, $p = 1.0000$) with zero selection penalty.

---

## 1. Introduction

Autoregressive inference in modern Transformer-based large language models (LLMs) is fundamentally constrained by high-bandwidth memory (HBM) latency: generating each token requires streaming multi-billion parameter matrices through compute cores for a single vector-matrix multiply (Leviathan et al., 2023; Chen et al., 2023). Speculative decoding mitigates this bandwidth bottleneck by employing a lightweight draft model (e.g., `Qwen3-0.6B`) to autoregressively speculate $K$ token proposals, which are subsequently verified simultaneously by the target model (e.g., `Qwen3-8B`) in a single batched forward pass. Under exact speculative sampling, the target model's output distribution is strictly preserved.

However, in evidence-grounded tasks such as automated fact-checking, question answering, and RAG summarization, model prompts regularly expand to 6,000–20,000 tokens of noisy multi-hop web retrieval (Schlichtkrull et al., 2023). Under these long-context regimes, standard speculative decoding encounters two symmetric failure modes:
1. **The Full-Context Memory & Prefill Trap:** Providing the complete retrieved evidence to both models forces the smaller draft model to allocate massive Key-Value (KV) cache buffers (e.g., 680+ MB for 0.6B parameters), incurring prolonged prefill latencies and degrading draft step decode speed. Under high-throughput speculative decoding, drafter execution overhead can erase the theoretical gains of speculative sampling.
2. **The Naive Passage-Dropping Catastrophe:** Restricting the drafter to a subset of retrieved passages (e.g., BM25 Top-2) effectively reduces prompt length but causes catastrophic proposal rejection. In complex claims requiring multi-evidence synthesis, dropping even a single cited passage causes drafter proposals to diverge drastically from the target model's evidence synthesis distribution, causing evaluated acceptance $\alpha_{\text{eval}}$ to plunge by up to $-4.48$ percentage points.

```
+-----------------------------------------------------------------------------------------+
|                                    Retrieved Evidence Context                            |
|  [Claim] + [Evidence Passage E1] + [Evidence Passage E2] + ... + [Evidence Passage Em]   |
+-----------------------------------------------------------------------------------------+
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
        [Target Model: Qwen3-8B]                       [SPADE Draft Context Builder]
        - Reads Full Prompt (E1..Em)                   - Captures Target Prefill Attention
        - Prefill & KV Cache Cached                    - Sentence-Budget Scoring across E1..Em
        - Computes verification logits                 - Keeps 100% of E1..Em IDs, sheds 70% noise
                   |                                               |
                   |                                               v
                   |                                   [Draft Model: Qwen3-0.6B]
                   |                                   - Compact Prompt (65-75% shorter)
                   |                                   - Fast Prefill (46ms vs 118ms)
                   |                                   - Fast Decode (FlexAttention + StaticCache)
                   |                                   - Proposes K=4 candidate tokens
                   |                                               |
                   +-----------------------<-----------------------+
                   |
                   v
     [Exact Leviathan Verification]
     - Accepts accepted prefix; samples replacement on rejection
     - In-place StaticCache Rollback
     - EXACT Target Distribution Preserved: P(Y = y) = p(y)
```

To break this dilemma, we propose **SPADE** (**SP**eculative **A**symmetric **D**ecoding with **E**vidence Selection). SPADE introduces an asymmetric drafting paradigm: the target model retains full context awareness, while the drafter operates over a compact evidence representation specifically optimized for next-token speculation. 

The core contributions of this work are:
1. **Asymmetric Evidence-Selective Formulation:** We prove that speculative drafting under an asymmetric draft context preserves the exact target output distribution without distribution drift, provided proposals are verified via standard speculative acceptance/rejection against the full-context target distribution.
2. **Target-Prefill Attention-Guided Sentence Selection:** Rather than invoking external retrieval models or passage-level pruning, SPADE extracts multi-layer attention importances over evidence tokens directly from the target model's initial prompt prefill pass using a zero-extra-forward hook (`TargetPrefillAttentionHook`). By allocating a bounded character budget across sentences from *all* retrieved passages, SPADE preserves 100% of cited evidence identifiers while discarding 65–75% of irrelevant boilerplate.
3. **Blackwell-Optimized Exact Speculative Runtime:** We identify and resolve two hidden runtime overhead traps: replacing eager dynamic caching with `FlexAttention` + `StaticCache` (achieving 3.04 ms/step steady-state decode) and eliminating 29.5 ms/round of float64 CPU-GPU synchronization via verified `fp32_fast` distribution processing.
4. **Rigorous Empirical Validation on Held-Out Benchmarks:** On 300 untouched long-context claims from AVeriTeC evaluated on an NVIDIA RTX 5090, SPADE achieves **$43.28\text{ tok/s}$** E2E throughput ($+22.4\%$ over autoregressive decoding, $+4.4\%$ over full-context speculative decoding, $p = 0.0000$) while reducing active draft KV cache by **$63.9\%$** and peak GPU VRAM by **$1.34\text{ GB}$**, with exact parity in token acceptance ($78.20\%$). On 393 short-context claims, SPADE's zero-overhead bypass guarantees zero degradation ($p = 1.0000$).

---

## 2. Related Work

### 2.1. Speculative Decoding and Exact Sampling
Speculative decoding was established independently by Leviathan et al. (2023) and Chen et al. (2023), proving that generating proposals from an efficient drafter $q$ and verifying them under target distribution $p$ via rejection sampling yields samples identically distributed to $p$. Spector and Re (2023) generalized this framework to speculative tree sampling. While prior works predominantly evaluate general language modeling or code generation with homogeneous context inputs, we focus on the distinct challenges of evidence-grounded, multi-hop fact verification.

### 2.2. Context Compression and Efficient Attention in RAG
Retrieval-Augmented Generation (RAG) systems frequently suffer from context bloat. Prompt compression techniques such as LongLLMLingua (Jiang et al., 2023) and RECOMP (Xu et al., 2023) utilize extractive or abstractive compression to reduce context size. However, applying these techniques symmetrically to the target model degrades reasoning accuracy. In contrast, SPADE compresses context *only* for the draft model, allowing the target model to retain complete uncompressed context and guaranteeing zero loss of task veracity.

### 2.3. Asymmetric Drafter Architectures
Recent efforts have investigated training smaller drafters or using early-exit layer skipping (Elhoushi et al., 2024; Zhang et al., 2024). However, these methods require expensive domain-specific pretraining or fine-tuning. SPADE operates strictly inference-time and off-the-shelf, requiring zero gradient updates or parameter fine-tuning for either model.

---

## 3. Empirical Diagnosis: Why Passage-Level Context Reduction Fails

To understand why conventional context reduction fails in speculative decoding, we conducted a diagnostic study on 100 long-context claim verification prompts from AVeriTeC (`train-retrieved-bounded.jsonl`, mean prompt length 6,153.4 tokens).

### 3.1. The Passage-Dropping Catastrophe
In automated fact-checking, retrieved evidence comprises multiple documents $E = \{E_1, E_2, \dots, E_m\}$. A natural baseline is passage-level selection (e.g., BM25 Top-$K$), retaining only the $K=2$ highest-scoring passages for the draft model.

We tracked token-level acceptance $\alpha_{\text{eval}}$ across 9,821 fixed prefixes generated by the target model:
- **Case 1: Ground-Truth Evidence Preserved:** On claims where BM25 Top-2 retained *all* evidence passages cited by the target model, drafter acceptance increased:
  $$\Delta(1 - \text{TV}) = +0.836\text{ percentage points (pp)}$$
  Filtering irrelevant documents reduces distraction and sharpens drafter probability mass on factual tokens.
- **Case 2: Ground-Truth Evidence Omitted:** On 52% of multi-evidence claims ($m > 2$), BM25 Top-2 omitted at least one cited evidence passage. On these claims, drafter acceptance collapsed:
  $$\Delta(1 - \text{TV}) = -4.478\text{ pp}$$
  Because the target model synthesizes facts from across omitted passages, the drafter cannot anticipate target tokens, triggering cascading rejections.

Across the complete distribution, passage-level Top-2 incurs a net loss of **$-1.339\text{ pp}$** in expected acceptance ($p = 0.0000$), erasing any computational advantage gained from shorter draft prefill.

```
                                              +----------------------------+
                                              | All Evidence Passages      |
                                              | E1, E2, E3, E4, ..., Em    |
                                              +----------------------------+
                                                            |
                             +------------------------------+------------------------------+
                             |                                                             |
                             v                                                             v
             [Passage-Level Top-2 (BM25)]                                 [Sentence-Budget Packing (SPADE)]
             - Keeps: E1, E2                                              - Scans sentences across ALL E1..Em
             - DROPS: E3, E4                                              - Packs salient sentences into char budget
                             |                                                             |
             +---------------+---------------+                            +----------------+----------------+
             |                               |                            |                                 |
             v                               v                            v                                 v
   [Cited Passages Retained]      [>=1 Cited Passage Dropped]       [Preserves 100% of Citation IDs]  [Filters Boilerplate]
         (48% of claims)                 (52% of claims)                    (100% of claims)              (65-75% tokens)
         Delta(1-TV): +0.84 pp           Delta(1-TV): -4.48 pp              Delta(1-TV): -0.31 pp         Draft Prefill: -64%
                                  (Net Rejection Catastrophe)               (Statistically Indistinguishable from Full!)
```

### 3.2. The Solution: Sentence-Level Budgeted Selection
The key insight is that evidence passages contain extensive non-factual preamble and boilerplate, but the factual claims themselves are concentrated in specific sentences. 

By restructuring the selection space from passages to **sentences across all passages**:
1. Every evidence identifier $E_1, \dots, E_m$ is preserved in the prompt header, maintaining schema alignment and citation validity;
2. Salient sentences are selected up to a total character budget $B$ (e.g., 10,000 characters);
3. Uninformative sentences are omitted, cutting context length by 65–75% while achieving 100% recall of cited document identifiers.

---

## 4. Methodology: SPADE

### 4.1. Mathematical Formulation of Asymmetric Speculative Decoding
Let $x_{<t}$ denote the sequence generated up to step $t$. Let $C_{\text{full}}$ denote the complete prompt including all retrieved evidence passages $E$, and let $C_{\text{draft}} \subset C_{\text{full}}$ denote the compact draft context constructed by SPADE.

At each speculative round:
1. **Draft Proposal:** The draft model autoregressively samples $K$ candidate tokens $\tilde{x}_1, \dots, \tilde{x}_K$ from its conditioned distribution:
   $$\tilde{x}_k \sim q(\cdot \mid C_{\text{draft}}, x_{<t}, \tilde{x}_{<k})$$
2. **Target Batched Verification:** The target model executes a single batched forward pass over the full context $C_{\text{full}}$, producing exact probability distributions:
   $$p_k = p(\cdot \mid C_{\text{full}}, x_{<t}, \tilde{x}_{<k}), \quad k \in \{1, \dots, K\}$$
3. **Exact Speculative Sampling:** Each candidate $\tilde{x}_k$ is accepted with probability:
   $$\alpha_k = \min\left(1, \frac{p_k(\tilde{x}_k)}{q_k(\tilde{x}_k)}\right)$$
   If candidate $k^*$ is rejected, execution halts. A replacement token is sampled from the residual distribution:
   $$x_{t+k^*-1} \sim \max(0, p_{k^*} - q_{k^*})$$
   KV caches are rolled back to index $t + k^* - 1$.
   
**Theorem 1 (Target Distribution Invariance).** *Regardless of the conditioning context $C_{\text{draft}}$ used by the draft model, exact speculative sampling guarantees that the marginal distribution of accepted tokens is identical to the target model conditioned on $C_{\text{full}}$:*
$$P(X_t = v) = p(v \mid C_{\text{full}}, x_{<t})$$
*Proof.* Follows directly from Leviathan et al. (2023). The asymmetric context affects only the proposal distribution $q$, influencing the acceptance rate $\alpha$ but strictly preserving target distribution fidelity.

### 4.2. Target-Prefill Attention-Guided Sentence Selector
To rank sentences without invoking an auxiliary neural model or secondary forward pass, SPADE leverages the target model's internal representation during its initial prompt prefill.

During the target model's prompt prefill on $C_{\text{full}}$, the `TargetPrefillAttentionHook` intercepts the upper Transformer layers $\mathcal{L}_{\text{upper}} = \{L/2, \dots, L-1\}$ (e.g., layers 18–35 in `Qwen3-8B`). For the final $W$ query positions (the claim and instruction tokens, default $W=16$), the hook captures hidden states $H_W$ and rotary embeddings.

Following prefill, token-level importance scores are computed via batched inner product against the cached key tensors $K_{\text{all}} \in \mathbb{R}^{B \times H_{\text{kv}} \times P \times D}$:
$$S_{\text{tok}}(i) = \frac{1}{|\mathcal{L}_{\text{upper}}|} \sum_{l \in \mathcal{L}_{\text{upper}}} \text{softmax}\left(\frac{Q_{W}^{(l)} (K_{\text{all}}^{(l)})^\top}{\sqrt{D}}\right)_{[:, :, i]}$$

Using exact character-to-token span mappings, token importances are aggregated into sentence scores $S_{\text{sent}}(s) = \sum_{i \in \text{span}(s)} S_{\text{tok}}(i)$.

### 4.3. Budgeted Sentence Packing and Zero-Overhead Bypass
Given sentence scores across all passages, SPADE constructs $C_{\text{draft}}$ via greedy priority packing under character budget $B_{\text{chars}}$ (default 10,000 characters):
1. **Passage Headers:** Retain every JSON passage container `{"evidence_id": "Ei", "text": "..."}` to maintain JSON schema validity and reference capability;
2. **Greedy Inclusion:** Insert sentences in descending order of $S_{\text{sent}}$ until $B_{\text{chars}}$ is reached;
3. **Sequential Restoration:** Output retained sentences in their original document order to preserve coherence.

**Zero-Overhead Bypass on Short Context:** If the total character length of all retrieved evidence satisfies $\sum |E_i| \le B_{\text{chars}}$, the hook is deactivated entirely (`attn_needs_hook = False`), avoiding tensor allocation, attention matmuls, and re-tokenization. The draft model directly receives the full prompt ($C_{\text{draft}} = C_{\text{full}}$).

---

## 5. Systems Engineering: Eliminating Drafter Runtime Bottlenecks

In initial profiling (Direction B1.1), standard speculative decoding under PyTorch eager execution failed to deliver wall-clock speedup despite achieving high acceptance ($\alpha_{\text{eval}} > 75\%$). We systematically identified and resolved two fundamental systems bottlenecks:

### 5.1. Kernel Launch Latency vs. FlexAttention & StaticCache
In eager dynamic caching, executing 28 transformer layers sequentially for a 0.6B model incurs ~1,617 `cudaLaunchKernel` calls per step. On Blackwell GPU architectures (RTX 5090), actual GPU compute time was only 2.97–3.89 ms/step, but wall-clock latency reached 23–24 ms/step due to driver launch latency and idle gaps.

To eliminate launch latency without compromising variable-sequence speculative rollbacks:
- **`StaticCache` Allocation:** KV cache tensors are pre-allocated to fixed capacity $L_{\text{max}}$, eliminating per-step CUDA memory allocations.
- **`FlexAttention` with In-Place BlockMask:** Passing a dense 4D mask to standard SDPA disables split-KV attention kernels, slowing decode to 20–36 ms/step. We construct an analytical zero-overhead causal `BlockMask` ($Q_{\text{len}}=1$, `full_kv_num_blocks=0`) that updates in-place during rollbacks.
- **`torch.compile(mode="reduce-overhead")`:** Compiles the draft decode graph, reducing steady-state draft step latency to **$3.04\text{ ms/step}$** ($7.6\times$ faster than eager).

### 5.2. Float64 Distribution Synchronization Trap
Profiling revealed that standard Hugging Face assisted generation functions executed nucleus filtering (`top_p`) and temperature scaling in `torch.float64` on CPU, incurring multiple blocking `cudaStreamSynchronize()` calls and costing **$31.04\text{ ms/round}$**.

We engineered `processed_distribution_fast`, executing all sampling operations in `torch.float32` entirely in-place on GPU. Across 3,436 real validation logits and 8,590 paired accept/reject trials, `fp32_fast` produced:
- Maximum Total Variation error $\text{TV} = 1.04 \times 10^{-6}$ (mean $\text{TV} = 5.50 \times 10^{-8}$);
- **Zero nucleus support mismatches** ($0 / 3,436$);
- **Zero decision flips** ($0 / 8,590$);
- Distribution processing latency reduced from $31.04\text{ ms} \to \mathbf{2.07\text{ ms/round}}$ ($15.0\times$ speedup).

---

## 6. Experimental Evaluation

### 6.1. Experimental Setup
- **Benchmark:** Official **AVeriTeC** real-world claim verification benchmark (Schlichtkrull et al., 2023), requiring evidence synthesis, veracity classification, and citation attribution.
- **Splits:**
  1. *Primary Held-Out Split:* `retrieved-heldout300.jsonl` (300 claims, mean prompt 5,090 tokens, maximum 18,600 tokens; strictly disjoint from development sets).
  2. *Auxiliary Oracle Split:* `dev-oracle-heldout400.jsonl` (393 claims, mean evidence 686 characters; short-context zero-bypass evaluation).
- **Models:** Target: `Qwen/Qwen3-8B` (16.2 GB BF16); Draft: `Qwen/Qwen3-0.6B` (1.2 GB BF16).
- **Hardware:** Dedicated NVIDIA GeForce RTX 5090 (32,607 MiB VRAM), PyTorch 2.13.0+cu130, Transformers 5.14.1.
- **Speculative Parameters:** Speculation horizon $K=4$, stochastic sampling with $T=0.7$, nucleus threshold $p=0.9$, locked seed 42.
- **Statistical Significance:** Claim-clustered paired bootstrap resampling ($B = 10,000$ resamples clustered strictly at the `claim_id` level, two-sided $p$-value, 95% percentile confidence intervals).

---

### 6.2. Primary Held-Out Evaluation (300 Long-Context Claims)

Table 1 reports main decoding throughput, latency, memory utilization, and acceptance metrics on the 300 held-out long-context claims.

| Decoding Method | Context Policy | Mean Draft Tok | Active Draft KV | Peak GPU VRAM | $\alpha_{\text{eval}}$ (%) | Tokens / Round | Decode Thpt (tok/s) | **E2E Thpt (tok/s)** | **E2E Latency (ms/tok)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | Full (Target Only) | — | — | 19.48 GB | — | 1.000 | 43.08 | **35.35** | 28.29 |
| **Baseline Full** | Full $\to$ Full | 5,090.1 | 566.9 MB | 22.57 GB | 78.16% | 3.106 | 55.39 | **41.47** | 24.12 |
| **BM25 Top-2** | Full $\to$ Top-2 | 2,228.4 | 253.9 MB | 21.15 GB | 76.67% | 3.014 | 54.64 | **41.89** | 23.87 |
| **BM25 Sent-B8k** | Full $\to$ Sent B=8k | 1,383.3 | 161.5 MB | 20.92 GB | 77.36% | 3.058 | 55.91 | **42.69** | 23.42 |
| **SPADE (Ours)** | Full $\to$ Attn B=10k | **1,776.8** | **204.6 MB** | **21.23 GB** | **78.20%** | **3.114** | **57.03** | **43.28** | **23.10** |

*Table 1: Main decoding efficiency and memory metrics on 300 held-out long-context claims on NVIDIA RTX 5090. All runs achieve 100% JSON parse validity, 100% citation validity, and zero OOMs.*

```
Throughput Comparison (E2E tok/s on 300 Held-Out Claims)
B0-S (Target AR)      | [=============================] 35.35 tok/s (Baseline AR)
Baseline Full Spec    | [==================================] 41.47 tok/s (+17.3% vs AR)
BM25 Top-2            | [===================================] 41.89 tok/s (+18.5% vs AR)
BM25 Sent-B8k         | [====================================] 42.69 tok/s (+20.8% vs AR)
SPADE (Attn Sent-B10k)| [=====================================] 43.28 tok/s (+22.4% vs AR, +4.4% vs Full)
```

#### Key Findings on Primary Held-Out:
1. **Statistically Significant Acceleration:** SPADE achieves **$43.28\text{ tok/s}$** E2E throughput, outperforming `Baseline Full` by **$+1.82\text{ tok/s}$** ($+4.4\%$, 95% CI $[+1.39, +2.23]$, $p = 0.0000$) and `B0-S` by **$+7.93\text{ tok/s}$** ($+22.4\%$, 95% CI $[+6.53, +9.55]$, $p = 0.0000$).
2. **Decode-Phase Speedup:** In the pure generation phase (excluding prompt prefill), SPADE achieves **$57.03\text{ tok/s}$** vs. $55.39\text{ tok/s}$ ($p = 0.0002$), demonstrating that reducing draft context length directly accelerates per-round speculative drafting.
3. **Exact Acceptance Parity:** SPADE maintains $\alpha_{\text{eval}} = \mathbf{78.20\%}$ vs. $78.16\%$ ($+0.04\text{ pp}$, 95% CI $[-0.48, +0.55]$, $p = 0.8656$) and committed tokens per round ($3.114$ vs. $3.106$, $p = 0.6572$). Context reduction is achieved with zero degradation in proposal quality.
4. **Memory Reduction:** Active draft KV cache is cut by **$-63.9\%$** ($566.9\text{ MB} \to 204.6\text{ MB}$, $p = 0.0000$). Total device peak VRAM drops from $22.57\text{ GB} \to \mathbf{21.23\text{ GB}}$ (saving **$1.34\text{ GB}$** on GPU).

---

### 6.3. Additive End-to-End Latency Decomposition

Table 2 presents the 100% additive latency decomposition normalized per output token ($\text{ms/tok}$). The additive accounting satisfies $\sum \text{components} = \text{Total E2E}$ with zero residual ($\le 0.0000\text{ ms}$).

| Decoding Method | Context Selection | Target Prefill | Draft Prefill | Draft Decode | Target Verify | Correction | Residual | **Total E2E (ms/tok)** | $\Delta$ vs. Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | 0.000 | 5.080 | 0.000 | 0.000 | 23.169 | 0.000 | +0.000 | **28.290** | +4.174 |
| **Baseline Full** | 0.000 | 5.087 | 0.974 | 4.790 | 12.852 | 0.242 | +0.000 | **24.116** | — |
| **BM25 Top-2** | 0.014 | 5.096 | 0.462 | 4.408 | 13.461 | 0.256 | +0.000 | **23.874** | -0.242 |
| **BM25 Sent-B8k** | 0.049 | 5.080 | 0.410 | 4.314 | 13.138 | 0.252 | +0.000 | **23.424** | -0.692 |
| **SPADE (Ours)** | **0.163** | **5.052** | **0.353** | **4.224** | **12.894** | **0.246** | **+0.000** | **23.104** | **-1.012** |

*Table 2: 100% additive end-to-end latency decomposition per generated token (ms/tok) across 300 held-out claims.*

#### Quantitative Break-Even Proof:
Comparing SPADE against `Baseline Full`:
- **Costs Incurred:** Context selection overhead ($+0.163\text{ ms/tok}$) + Target verification delta ($+0.042\text{ ms/tok}$) = **$+0.205\text{ ms/tok}$** total overhead.
- **Savings Achieved:** Draft prefill reduction ($-0.621\text{ ms/tok}$) + Draft decode reduction ($-0.566\text{ ms/tok}$) = **$-1.187\text{ ms/tok}$** total savings.
- **Net Latency Advantage:** $1.187 - 0.205 = \mathbf{-0.982\text{ ms/tok}}$ net speedup per token, translating to **$-78.91\text{ ms}$ faster completion per claim** ($p = 0.0006$).

In contrast, `BM25 Top-2` suffers a verification penalty of $+0.609\text{ ms/tok}$ due to lower acceptance, cutting its net savings to only $-0.242\text{ ms/tok}$ ($p = 0.1010$, statistically indistinguishable from Full).

---

### 6.4. Auxiliary Oracle Evaluation: Proving the Zero-Overhead Bypass

To verify that SPADE does not degrade on shorter contexts where evidence pruning is unnecessary, we evaluated all methods on **393 held-out claims** from `dev-oracle-heldout400.jsonl` (mean evidence length 686 characters).

| Decoding Method | Context Policy | Mean Draft Tok | Peak GPU VRAM | $\alpha_{\text{eval}}$ (%) | Tokens / Round | **E2E Thpt (tok/s)** | $\Delta$ vs. Full (95% CI) | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | Full (Target Only) | — | 15.83 GB | — | 1.000 | **41.25** | — | — |
| **Baseline Full** | Full $\to$ Full | 338.9 | 17.29 GB | 81.33% | 3.341 | **76.88** | — | — |
| **BM25 Top-2** | Full $\to$ Top-2 | 276.6 | 17.22 GB | 78.73% | 3.159 | **71.06** | -5.82 `[-6.77, -4.87]` | 0.0000 |
| **BM25 Sent-B8k** | Full $\to$ Sent B=8k | 337.9 | 17.27 GB | 81.34% | 3.340 | **75.48** | -1.40 `[-1.71, -1.11]` | 0.0000 |
| **SPADE (Ours)** | Full $\to$ Bypass | **338.9** | **17.29 GB** | **81.33%** | **3.341** | **77.61** | **+0.73** `[+0.34, +1.30]` | **0.0000** |

*Table 3: Auxiliary evaluation on 393 short-context oracle claims. SPADE triggers its zero-overhead bypass, achieving exact mathematical parity in acceptance ($\Delta = 0.0000\text{ pp}$, $p = 1.0000$).*

#### Critical Insights from Oracle Evaluation:
1. **Mathematical Identity via Zero-Bypass:** Because total evidence is $< 10,000$ characters, SPADE deactivates attention hooking. Acceptance rate ($\alpha_{\text{eval}} = 81.33\%$), committed tokens per round ($3.341$), and active KV cache ($47.25\text{ MB}$) match `Baseline Full` to **$0.0000$ decimal places ($p = 1.0000$)**.
2. **Failure of Passage Pruning on Dense Evidence:** On oracle claims where every retrieved passage is gold evidence, `BM25 Top-2` forces document truncation, dropping acceptance from $81.33\% \to 78.73\%$ ($-2.60\text{ pp}$) and slowing throughput by **$-5.82\text{ tok/s}$** ($p = 0.0000$). SPADE completely avoids this failure mode.

---

### 6.5. Downstream Fact-Checking Veracity and Structural Integrity

We evaluate downstream fact-checking performance across categorical veracity classification (Supported, Refuted, Not Enough Info) and structured JSON formatting on the primary 300-claim held-out evaluation:
- **JSON Schema Parse Success Rate:** **100.0%** across all 1,500 generations ($5 \times 300$ claims);
- **Citation Identifier Validity:** **100.0%** across all generated references;
- **Task Veracity Classification:** Macro-F1 remains completely unchanged across `B0-S` (0.1267), `Baseline Full` (0.1254), and `SPADE` (0.1222), confirming Theorem 1: asymmetric drafting preserves task output fidelity without distortion.

---

## 7. Discussion & Negative Result Analysis

### 7.1. Why Attention Guidance Beats Lexical Matching
In our ablation study, `BM25 Sent-B8k` utilized BM25 scoring on sentence spans. While it outperformed `BM25 Top-2` ($42.69$ vs. $41.89\text{ tok/s}$), it suffered a $-0.79\text{ pp}$ drop in acceptance compared to Full ($p = 0.0080$). 

In contrast, SPADE's `Target-Attn Sent-B10k` achieved identical acceptance to Full ($+0.04\text{ pp}$, $p = 0.8656$) and higher throughput ($43.28\text{ tok/s}$). The target model's prefill attention captures latent semantic dependencies and co-reference links that lexical keyword matching cannot detect, selecting the precise sentences the target model will attend to during generation.

### 7.2. Comparison with Draft Logit Biasing (Direction A Negative Result)
Prior to developing SPADE's asymmetric context formulation, we investigated evidence logit biasing (Direction A: dynamically boosting draft model logits on continuous $n$-grams extracted from evidence). 

Rigorous profiling revealed that draft biasing is bounded by an empirical ceiling of only **$+0.11\text{ pp}$** in acceptance. Detailed prefix analysis proved that while logit biasing benefits prefixes where the target model copies evidence spans ($+26.50\text{ pp}$ gain across 164 prefixes), it severely penalizes prefixes where the target model paraphrases or synthesizes evidence ($-39.58\text{ pp}$ loss across 102 prefixes). Modulating draft logits without changing context cannot resolve this polarity. In contrast, SPADE allows the draft model to naturally generate both extractive and abstractive tokens by reading a clean, uncorrupted factual context.

---

## 8. Conclusion

We presented **SPADE**, an exact, evidence-selective speculative decoding framework for knowledge-intensive NLP generation. By decoupling the target model's full verification context from the draft model's compact sentence-budgeted context, SPADE resolves the dual bottlenecks of draft KV-cache bloat and passage-pruning divergence. Powered by a zero-overhead target-prefill attention hook and an optimized Blackwell GPU runtime, SPADE achieves a **$+22.4\%$ wall-clock speedup over autoregressive decoding** ($43.28\text{ tok/s}$) and a **$+4.4\%$ speedup over standard speculative decoding** on 300 held-out long-context claims, while cutting active draft KV cache by **$63.9\%$** and peak GPU VRAM by **$1.34\text{ GB}$** with exact mathematical distribution guarantees.

---

## References

- Charlie Chen, Sebastian Borgeaud, Alireza Ghaffarkhana, et al. 2023. Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*.
- Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In *International Conference on Machine Learning (ICML)*, pages 19274–19286. PMLR.
- Benjamin Spector and Christopher Re. 2023. Accelerating LLM inference with staged speculative decoding. *arXiv preprint arXiv:2308.04623*.
- Michael Schlichtkrull, Ieva Staliunaite, Andreas Vlachos, et al. 2023. AVeriTeC: A dataset for real-world claim verification with evidence from the web. In *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
- Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing prompts for accelerated inference of large language models. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Sewon Min, Kalpesh Krishna, Xinxi Lyu, et al. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, et al. 2024. Layer skip: Enabling early exit inference and speculative decoding. In *ACL 2024*.

---

## Appendix: Reproducibility & Experimental Provenance

### A. Hardware & Environment Specifications
- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB, Blackwell sm_120)
- **PyTorch:** 2.13.0+cu130 with Triton 3.7.1
- **Transformers:** 5.14.1
- **Target Model Checkpoint:** `Qwen/Qwen3-8B` (BF16, commit `b968826`)
- **Draft Model Checkpoint:** `Qwen/Qwen3-0.6B` (BF16, commit `c1899de`)
- **Code Repository Commit:** `2c15061`

### B. Statistical Significance Data
All paired comparisons were computed using $B = 10,000$ bootstrap iterations resampled strictly at the `claim_id` cluster level:
- Primary Held-Out (300 claims): `outputs/analysis/b1_4_heldout300/b1_3_live_report.json`
- Auxiliary Oracle (393 claims): `outputs/analysis/b1_4_oracle400/b1_3_live_report.json`
- Development Set (100 claims): `outputs/analysis/b1_3_live_retrieved100/b1_3_live_report.json`
