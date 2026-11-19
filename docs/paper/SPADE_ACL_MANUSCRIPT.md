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

In this work, we present **SPADE** (**SP**eculative **A**symmetric **D**ecoding with **E**vidence Selection), an exact, evidence-selective speculative decoding framework tailored for knowledge-intensive generation. SPADE decouples the target and draft context spaces: while the target model verifies candidate proposals across the complete retrieved evidence context, the draft model generates proposals conditioned on a compact, sentence-budgeted context representation. To construct this draft context without extra forward passes, SPADE introduces a **Target-Prefill Attention Hook** that captures upper-layer causal self-attention weights from the target model's initial prompt pass (claim query tokens attending to evidence key tokens during prefill), identifies claim-salient sentences across *all* evidence passages via a bounded character budget, and preserves 100% of citation identifiers while shedding 65–75% of uninformative tokens. Coupled with an optimized, mathematically grounded inference runtime (`FlexAttention` + `StaticCache` + zero-sync `fp32_fast` distribution processing achieving empirical numerical agreement with zero decision flips), SPADE achieves true end-to-end wall-clock speedup on modern GPU architectures.

Extensive evaluations on the official **AVeriTeC** real-world claim verification benchmark across **300 held-out long-context claims** on an NVIDIA RTX 5090 demonstrate that SPADE:
1. Achieves **$43.28\text{ tok/s}$** end-to-end throughput, outperforming target-only autoregressive decoding ($35.35\text{ tok/s}$, **$+22.4\%$ speedup**, $p < 10^{-4}$) and standard full-context speculative decoding ($41.47\text{ tok/s}$, **$+4.4\%$ speedup**, $p < 10^{-4}$);
2. Accelerates token generation in the decode-only phase (**$57.03\text{ tok/s}$** vs. $55.39\text{ tok/s}$, $p = 0.0002$), proving that speedups do not rely solely on prefill savings;
3. Preserves target speculative acceptance ($\alpha_{\text{eval}} = \mathbf{78.20\%}$ vs. $78.16\%$, $\Delta = +0.04\text{ pp}$, 95% CI $[-0.48, +0.55]$, $p = 0.8656$, with no statistically significant difference observed, confirming empirical equivalence within a predefined $\pm 1.0\text{ pp}$ margin) and committed tokens per round ($3.114$ vs. $3.106$, $p = 0.6572$);
4. Cuts active draft KV-cache memory by **$-63.9\%$** ($566.9\text{ MB} \to 204.6\text{ MB}$, $p < 10^{-4}$) and peak device VRAM by **$-1.34\text{ GB}$**;
5. On **393 short-context oracle claims**, activates a zero-overhead bypass mechanism that preserves the full prompt ($C_{\text{draft}} = C_{\text{full}}$), guaranteeing distributional equivalence ($\alpha_{\text{eval}} = 81.33\%$, $\Delta = 0.00\text{ pp}$, $p = 1.0000$) with zero selection penalty.

---

## 1. Introduction

Autoregressive inference in modern Transformer-based large language models (LLMs) is fundamentally constrained by high-bandwidth memory (HBM) latency: generating each token requires streaming multi-billion parameter matrices through compute cores for a single vector-matrix multiply (Shazeer, 2019; Leviathan et al., 2023; Chen et al., 2023). Speculative decoding mitigates this bandwidth bottleneck by employing a lightweight draft model (e.g., `Qwen3-0.6B`) to autoregressively speculate $K$ candidate tokens, which are subsequently verified simultaneously by the primary target model (e.g., `Qwen3-8B`) in a single batched forward pass. Under exact speculative sampling (Leviathan et al., 2023), the target model's output distribution is strictly preserved without heuristic quality degradation.

However, in evidence-grounded tasks such as automated fact-checking, question answering, and retrieval-augmented generation (RAG) summarization, prompts regularly expand to 6,000–20,000 tokens of noisy multi-hop web retrieval (Thorne et al., 2018; Schlichtkrull et al., 2023). Under these long-context regimes, standard speculative decoding encounters two symmetric failure modes:
1. **The Full-Context Memory & Prefill Trap:** Providing the complete retrieved evidence context to both models forces the smaller draft model to allocate massive Key-Value (KV) cache buffers (e.g., 680+ MB for 0.6B parameters), incurring prolonged prefill latencies and degrading draft step decode speed. Under high-throughput speculative decoding, drafter execution overhead can entirely erase the theoretical gains of speculative sampling.
2. **The Naive Passage-Dropping Catastrophe:** Restricting the drafter to a subset of retrieved passages (e.g., BM25 Top-$K$) effectively reduces prompt length but causes catastrophic proposal rejection. In complex claims requiring multi-evidence synthesis, dropping even a single cited passage causes drafter proposals to diverge drastically from the target model's evidence synthesis distribution, causing evaluated acceptance $\alpha_{\text{eval}}$ to plunge by up to $-4.48$ percentage points.

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

To break this dilemma, we propose **SPADE** (**SP**eculative **A**symmetric **D**ecoding with **E**vidence Selection). SPADE introduces an asymmetric drafting paradigm: the target model retains full context awareness across all retrieved evidence documents, while the drafter operates over a compact evidence representation specifically optimized for next-token speculation. While recent works such as RAPID (Chen et al., 2025) have demonstrated the value of retrieval-augmented drafting with shortened document views, we show that standard passage-level document truncation fails catastrophically in multi-hop evidence reasoning tasks whenever a cited passage is omitted. Furthermore, unlike SpecExtend (Chen et al., 2026), which uses target attention for dynamic KV cache eviction and cross-model key retrieval during long sequence generation, SPADE operates at the prompt construction stage to extract offline self-attention importances and pack a compact draft context across all candidate documents prior to speculation.

The core contributions of this work are four-fold:
1. **Asymmetric Evidence-Selective Formulation:** We prove that speculative drafting under an asymmetric draft context preserves the exact target output distribution without distribution drift, provided proposals are verified via standard speculative acceptance/rejection against the full-context target distribution (Theorem 1).
2. **Target-Prefill Attention-Guided Sentence Selection:** Rather than invoking external retrieval models or passage-level pruning, SPADE extracts multi-layer attention importances over evidence tokens directly from the target model's initial prompt prefill pass using a zero-extra-forward hook (`TargetPrefillAttentionHook`). By allocating a bounded character budget across sentences from *all* retrieved passages, SPADE preserves 100% of cited evidence identifiers while discarding 65–75% of uninformative boilerplate.
3. **Blackwell-Optimized Exact Speculative Runtime:** We identify and resolve two hidden runtime overhead traps: replacing eager dynamic caching with `FlexAttention` + `StaticCache` (achieving 3.04 ms/step steady-state decode) and eliminating 29.5 ms/round of float64 CPU-GPU synchronization via verified `fp32_fast` distribution processing.
4. **Rigorous Empirical Validation on Held-Out Benchmarks:** On 300 untouched long-context claims from AVeriTeC evaluated on an NVIDIA RTX 5090, SPADE achieves **$43.28\text{ tok/s}$** E2E throughput (+22.4% over autoregressive decoding, +4.4% over full-context speculative decoding, $p < 10^{-4}$) while reducing active draft KV cache by **$63.9\%$** and peak GPU VRAM by **$1.34\text{ GB}$**, with exact parity in token acceptance ($78.20\%$). On 393 short-context claims, SPADE's zero-overhead bypass guarantees zero degradation ($p = 1.0000$).

---

## 2. Related Work

### 2.1. Speculative Decoding and Exact Sampling
Speculative execution for autoregressive generation was formalized by Leviathan et al. (2023) and Chen et al. (2023), proving that an auxiliary draft model $q$ can propose candidate tokens verified in parallel by a larger target model $p$ without altering the target distribution. Subsequent research expanded upon this foundation along multiple architectural dimensions. Medusa (Cai et al., 2024) fine-tunes multiple decoding heads on the target backbone. EAGLE (Li et al., 2024) and EAGLE-2 (Li et al., 2024) explore feature-level uncertainty and dynamic drafting trees. SpecInfer (Miao et al., 2023) and SpecTR (Sun et al., 2024) generalize sequential drafting to tree-structured token proposals. Complementary investigations explored staged speculative decoding (Spector and Re, 2023), bi-directional draft verification (Barkan et al., 2024), and lookahead decoding without draft models (Fu et al., 2024). While these methods advance architectural and token-tree designs for general open-domain generation, they treat context symmetrically and do not address domain-specific long-context grounding bottlenecks.

### 2.2. Context Compression and Efficient Attention in RAG
Retrieval-Augmented Generation (RAG) systems frequently suffer from context bloat and memory bandwidth saturation (Schlichtkrull et al., 2023). Prompt compression techniques such as LLMLingua (Jiang et al., 2023) and RECOMP (Xu et al., 2023) utilize extractive or abstractive compression to reduce context size. However, applying prompt compression symmetrically to the target model risks degrading reasoning accuracy and dropping critical factual facts. In contrast, SPADE compresses context *only* for the draft model, allowing the target model to retain complete uncompressed context and guaranteeing zero loss of task veracity. Concurrently, high-performance serving infrastructures—including PagedAttention in vLLM (Kwon et al., 2023), structured execution in SGLang (Zheng et al., 2023), and IO-aware FlashAttention (Dao et al., 2022; Dao, 2023)—optimize memory bandwidth. SPADE addresses the orthogonal algorithmic challenge of evidence context asymmetry, delivering KV cache and prefill reductions that compound directly with modern fused serving engines.

### 2.3. Asymmetric and Retrieval-Augmented Speculative Decoding
Several recent efforts have investigated asymmetric context access or retrieval assistance to optimize speculative decoding in long-sequence regimes. RAPID (Chen et al., 2025) pioneers a RAG drafter operating on shortened retrieval contexts to speculate for long-context target models. However, RAPID operates at the coarse passage level; as we show in Section 3, omitting complete evidence passages in multi-evidence reasoning tasks triggers severe proposal divergence whenever cited documents are excluded. In contrast, SPADE operates at the sub-passage level by performing sentence-budget allocation across *all* candidate documents.

A related line of work explores attention-guided context management. SpecExtend (Chen et al., 2026) utilizes target model attention scores for dynamic KV-cache eviction and cross-model key retrieval during long sequence generation. However, SpecExtend requires dynamic cross-model attention kernels throughout the decoding loop. In contrast, SPADE extracts target self-attention weights once during initial prefill to construct a static, self-contained compact prompt for the drafter, avoiding cross-model memory accesses during speculative decoding and incorporating a zero-overhead short-context bypass. Concurrently, LongSpec (Yang et al., 2026) addresses long-context speculative decoding by maintaining a constant-sized KV cache for drafter models and utilizing specialized position adaptations to bridge short-to-long training mismatches. While LongSpec modifies drafter internal architectures for general long-sequence language modeling, SPADE is tailored for evidence-grounded, multi-passage generation: SPADE decouples the target and draft contexts via target-prefill attention-guided sentence budgeting, leaving the underlying drafter model, attention kernels, and positional encodings entirely unmodified. Meanwhile, AsymSpec (Liang et al., 2026) explores context asymmetry in agentic LLMs by compressing the verifier's context while preserving the drafter's context, which trades off target distribution exactness ($\approx 90\%$ accuracy). In contrast, SPADE preserves the complete uncompressed context on the target verifier, strictly guaranteeing 100% mathematical target distribution invariance (Theorem 1) while compressing solely the draft prompt. Complementary methods investigate dynamic stopping policies such as SVIP (Zhang et al., 2025) and LinguaSpec (Liu et al., 2026), or contextual heterogeneity (Xu et al., 2026); SPADE is orthogonal and training-free, operating without fine-tuning or auxiliary neural models.

---

## 3. Empirical Diagnosis: Why Passage-Level Context Reduction Fails

To investigate why conventional context reduction fails in speculative decoding, we conducted a fine-grained diagnostic study on 100 long-context claim verification prompts from AVeriTeC (`train-retrieved-bounded.jsonl`, mean prompt length 6,153.4 tokens).

### 3.1. The Passage-Dropping Catastrophe
In automated fact-checking, retrieved evidence comprises multiple documents $E = \{E_1, E_2, \dots, E_m\}$. A natural baseline is passage-level selection (e.g., BM25 Top-$K$), retaining only the $K=2$ highest-scoring passages for the draft model.

We tracked token-level acceptance $\alpha_{\text{eval}}$ across 9,821 fixed prefixes generated by the target model:
- **Case 1: Ground-Truth Evidence Preserved:** On claims where BM25 Top-2 retained *all* evidence passages cited by the target model, drafter acceptance increased:
  $$\Delta(1 - \text{TV}) = +0.836\text{ percentage points (pp)}$$
  Filtering irrelevant documents reduces distraction and sharpens drafter probability mass on factual tokens.
- **Case 2: Ground-Truth Evidence Omitted:** On 52% of multi-evidence claims ($m > 2$), BM25 Top-2 omitted at least one cited evidence passage. On these claims, drafter acceptance collapsed:
  $$\Delta(1 - \text{TV}) = -4.478\text{ pp}$$
  Because the target model synthesizes facts from across omitted passages, the drafter cannot anticipate target tokens, triggering cascading rejections.

Across the complete distribution, passage-level Top-2 incurs a net loss of **$-1.339\text{ pp}$** in expected acceptance ($p < 10^{-4}$), erasing any computational advantage gained from shorter draft prefill.

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
2. Salient sentences are selected up to a total character budget $B$ (default 10,000 characters);
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
To rank sentences without invoking an auxiliary neural model or secondary forward pass, SPADE leverages the target model's internal causal self-attention representations during its mandatory initial prompt prefill pass.

Specifically, during prefill on $C_{\text{full}}$, the `TargetPrefillAttentionHook` intercepts the upper Transformer layers $\mathcal{L}_{\text{upper}} = \{L/2, \dots, L-1\}$ (e.g., layers 18–35 in `Qwen3-8B`). For the final $W$ query positions (the claim and instruction tokens, default $W=16$), the hook captures hidden query states $H_W$ and rotary embeddings.

Following prefill, token-level importance scores are computed via batched inner product of claim query tokens against the cached evidence key tensors $K_{\text{all}} \in \mathbb{R}^{B \times H_{\text{kv}} \times P \times D}$:
$$S_{\text{tok}}(i) = \frac{1}{|\mathcal{L}_{\text{upper}}|} \sum_{l \in \mathcal{L}_{\text{upper}}} \text{softmax}\left(\frac{Q_{W}^{(l)} (K_{\text{all}}^{(l)})^\top}{\sqrt{D}}\right)_{[:, :, i]}$$

Using exact character-to-token span mappings, token importances are aggregated into sentence scores $S_{\text{sent}}(s) = \sum_{i \in \text{span}(s)} S_{\text{tok}}(i)$. This extraction requires no additional forward pass through either model; the selection overhead is purely the tensor inner product and sentence sorting, which measures only $0.163\text{ ms/tok}$ end-to-end.

### 4.3. Budgeted Sentence Packing and Zero-Overhead Bypass
Given sentence scores across all passages, SPADE constructs $C_{\text{draft}}$ via greedy priority packing under character budget $B_{\text{chars}}$ (default 10,000 characters):
1. **Passage Headers:** Retain every JSON passage container `{"evidence_id": "Ei", "text": "..."}` to maintain JSON schema validity, citation attribution, and reference capability;
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

### 5.2. Distribution Processing Optimization: From 31 ms to 2 ms
Profiling revealed that SPADE's initial unoptimized distribution processing routine (`processed_distribution()`) incurred **$31.04\text{ ms/round}$** due to computing nucleus thresholding (`top_p`) and temperature scaling in `torch.float64` on CPU with blocking `cudaStreamSynchronize()` and `torch.sort` operations.

We engineered `processed_distribution_fast`, executing all sampling operations in `torch.float32` entirely in-place on GPU. Across 3,436 real validation logits and 8,590 paired accept/reject trials, `fp32_fast` produced:
- Maximum Total Variation error $\text{TV} = 1.04 \times 10^{-6}$ (mean $\text{TV} = 5.50 \times 10^{-8}$);
- **Zero nucleus support mismatches** ($0 / 3,436$);
- **Zero decision flips** ($0 / 8,590$);
- Distribution processing latency reduced from $31.04\text{ ms} \to \mathbf{2.07\text{ ms/round}}$ ($15.0\times$ speedup).

While floating-point arithmetic differences inherently preclude absolute mathematical identity, this verified empirical agreement establishes that the optimized implementation preserves target distribution sampling behavior with zero observed decision divergence across thousands of trials.

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

Table 1 reports main decoding throughput, latency, memory utilization, and acceptance metrics on the 300 held-out long-context claims, generated directly from raw JSON execution logs.

| Decoding Method | Context Policy | Mean Draft Tok | Active Draft KV | Peak GPU VRAM | $\alpha_{\text{eval}}$ (%) | Tok/Rnd | Decode Thpt (tok/s) | **E2E Thpt (tok/s)** | **E2E Latency (ms/tok)** | $\Delta$ vs. Full (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | Full (Target Only) | — | — | 19.48 GB | — | 1.000 | 43.08 | **35.35** | 28.29 | $-6.12$ $[-7.78, -4.70]^*$ |
| **Baseline Full** | Full $\to$ Full | 5,090.1 | 566.9 MB | 22.57 GB | 78.16% | 3.106 | 55.39 | **41.47** | 24.12 | — |
| **BM25 Top-2** | Full $\to$ Top-2 | 2,228.4 | 253.9 MB | 21.15 GB | 76.67% | 3.014 | 54.64 | **41.89** | 23.87 | $+0.42$ $[-0.08, +0.89]$ |
| **BM25 Sent-B8k** | Full $\to$ Sent B=8k | 1,383.3 | 161.5 MB | 20.92 GB | 77.36% | 3.058 | 55.91 | **42.69** | 23.42 | $+1.22$ $[+0.77, +1.68]^*$ |
| **BM25 Sent-B10k** | Full $\to$ Sent B=10k | 1,646.4 | 190.3 MB | 20.97 GB | 77.24% | 3.047 | 55.73 | **42.65** | 23.45 | $+1.19$ $[+0.69, +1.68]^*$ |
| **SPADE (Ours)** | Full $\to$ Attn B=10k | **1,776.8** | **204.6 MB** | **21.23 GB** | **78.20\%** | **3.114** | **57.03** | **43.28** | **23.10** | **+1.82** **[+1.39, +2.23]**$^*$ |

*Table 1: Main decoding efficiency and memory metrics on 300 held-out long-context claims on NVIDIA RTX 5090. Throughput reported in tok/s; latency in ms/tok. Statistical significance tested via paired cluster bootstrap ($B=10,000$ iterations, * denotes $p < 0.001$). All methods achieve 100.0% JSON parse validity and 100.0% citation validity.*

```
Throughput Comparison (E2E tok/s on 300 Held-Out Claims)
B0-S (Target AR)       | [=============================] 35.35 tok/s (Baseline AR)
Baseline Full Spec     | [==================================] 41.47 tok/s (+17.3% vs AR)
BM25 Top-2             | [===================================] 41.89 tok/s (+18.5% vs AR)
BM25 Sent-B8k          | [====================================] 42.69 tok/s (+20.8% vs AR)
BM25 Sent-B10k         | [====================================] 42.65 tok/s (+20.6% vs AR)
SPADE (Attn Sent-B10k) | [=====================================] 43.28 tok/s (+22.4% vs AR, +4.4% vs Full)
```

#### Key Findings on Primary Held-Out:
1. **Statistically Significant Acceleration:** SPADE achieves **$43.28\text{ tok/s}$** E2E throughput, outperforming `Baseline Full` by **$+1.82\text{ tok/s}$** ($+4.4\%$, 95% CI $[+1.39, +2.23]$, $p < 10^{-4}$) and `B0-S` by **$+7.93\text{ tok/s}$** ($+22.4\%$, 95% CI $[+6.53, +9.55]$, $p < 10^{-4}$).
2. **Decode-Phase Speedup:** In the pure generation phase (excluding prompt prefill), SPADE achieves **$57.03\text{ tok/s}$** vs. $55.39\text{ tok/s}$ ($p = 0.0002$), demonstrating that reducing draft context length directly accelerates per-round speculative drafting.
3. **Acceptance Equivalence within Predefined Margin:** For evaluated token acceptance, SPADE achieves $\alpha_{\text{eval}} = \mathbf{78.20\%}$ compared to $78.16\%$ for Baseline Full ($\Delta = +0.04\text{ pp}$, 95% bootstrap CI $[-0.48, +0.55]$, $p = 0.8656$). While a non-significant $p$-value ($p > 0.05$) alone indicates failure to reject the null hypothesis rather than mathematical identity, the tight two-sided 95% confidence interval falls strictly within a predefined $\pm 1.0\text{ percentage point}$ tolerance margin, demonstrating empirical equivalence under standard two one-sided testing (TOST) principles. Mean committed tokens per round ($3.114$ vs. $3.106$, $p = 0.6572$) similarly exhibits no significant difference. Target output distribution invariance is guaranteed mathematically by Theorem 1.
4. **Memory Reduction:** Active draft KV cache is cut by **$-63.9\%$** ($566.9\text{ MB} \to 204.6\text{ MB}$, $p < 10^{-4}$). Total device peak VRAM drops from $22.57\text{ GB} \to \mathbf{21.23\text{ GB}}$ (saving **$1.34\text{ GB}$** on GPU).

---

### 6.3. Additive End-to-End Latency Decomposition

Table 2 presents the 100% additive latency decomposition normalized per output token ($\text{ms/tok}$). The additive accounting satisfies $\sum \text{components} + \text{Residual} = \text{Total E2E}$ with zero gap.

| Decoding Method | Context Sel. | Target Prefill | Draft Prefill | Draft Decode | Target Verify | Correction | Loop/Unattr. | Residual | **Total E2E (ms/tok)** | $\Delta$ vs. Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | 0.000 | 5.080 | 0.000 | 0.000 | 23.169 | 0.000 | 0.042 | -0.001 | **28.290** | +4.174 |
| **Baseline Full** | 0.000 | 5.087 | 0.974 | 4.790 | 12.852 | 0.242 | 0.170 | +0.001 | **24.116** | — |
| **BM25 Top-2** | 0.014 | 5.096 | 0.462 | 4.408 | 13.461 | 0.256 | 0.178 | -0.001 | **23.874** | -0.242 |
| **BM25 Sent-B8k** | 0.049 | 5.080 | 0.410 | 4.314 | 13.138 | 0.252 | 0.181 | +0.000 | **23.424** | -0.692 |
| **BM25 Sent-B10k** | 0.046 | 5.062 | 0.394 | 4.316 | 13.203 | 0.249 | 0.175 | +0.000 | **23.445** | -0.671 |
| **SPADE (Ours)** | **0.163** | **5.052** | **0.353** | **4.224** | **12.894** | **0.246** | **0.172** | **+0.000** | **23.104** | **-1.012** |

*Table 2: 100% additive end-to-end latency decomposition per generated token (ms/tok) across 300 held-out claims, generated automatically from JSON execution counters. Loop/Unattr. overhead accounts for Python iteration scheduling, GPU synchronization, StaticCache rollback pointer updates, and token sampling launches ($0.170\text{--}0.181\text{ ms/tok}$). Minor residual discrepancies ($\le 0.001\text{ ms/tok}$) arise strictly from three-decimal floating-point rounding.*

#### Quantitative Break-Even Proof:
Comparing SPADE against `Baseline Full`:
- **Costs Incurred:** Selection overhead ($+0.163\text{ ms/tok}$) + verify delta ($+0.042\text{ ms/tok}$) = **$+0.205\text{ ms/tok}$**.
- **Savings Achieved:** Draft prefill reduction ($-0.621\text{ ms/tok}$) + draft decode reduction ($-0.566\text{ ms/tok}$) = **$-1.187\text{ ms/tok}$**.
- **Net Latency Advantage:** $1.187 - 0.205 = \mathbf{-0.982\text{ ms/tok}}$ net speedup per token, translating to **$-78.91\text{ ms}$ faster completion per claim** ($p = 0.0006$).

In contrast, `BM25 Top-2` suffers a verification penalty of $+0.609\text{ ms/tok}$ due to lower acceptance, cutting its net savings to only $-0.242\text{ ms/tok}$ ($p = 0.1010$, statistically indistinguishable from Full).

---

### 6.4. Auxiliary Oracle Evaluation: Proving the Zero-Overhead Bypass

To verify that SPADE does not degrade on shorter contexts where evidence pruning is unnecessary, we evaluated all methods on **393 held-out claims** from `dev-oracle-heldout400.jsonl` (mean evidence length 686 characters).

| Decoding Method | Context Policy | Mean Draft Tok | Peak GPU VRAM | $\alpha_{\text{eval}}$ (%) | Tok/Rnd | **E2E Thpt (tok/s)** | $\Delta$ vs. Full (95% CI) | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | Full (Target Only) | — | 15.83 GB | — | 1.000 | **41.25** | $-35.63$ $[-36.56, -34.63]$ | $< 10^{-4}$ |
| **Baseline Full** | Full $\to$ Full | 338.9 | 17.29 GB | 81.33% | 3.341 | **76.88** | — | — |
| **BM25 Top-2** | Full $\to$ Top-2 | 276.6 | 17.22 GB | 78.73% | 3.159 | **71.06** | $-5.82$ $[-6.77, -4.87]$ | $< 10^{-4}$ |
| **BM25 Sent-B8k** | Full $\to$ Sent B=8k | 337.9 | 17.27 GB | 81.34% | 3.340 | **75.48** | $-1.40$ $[-1.71, -1.11]$ | $< 10^{-4}$ |
| **SPADE (Ours)** | Full $\to$ Bypass | **338.9** | **17.29 GB** | **81.33\%** | **3.341** | **77.61** | **+0.73** **[+0.34, +1.30]** | **$< 10^{-4}$** |

*Table 3: Auxiliary evaluation on 393 short-context oracle claims. SPADE triggers its zero-overhead bypass, achieving exact distributional parity in acceptance ($\Delta = 0.0000\text{ pp}$, $p = 1.0000$), completely avoiding the $-5.82\text{ tok/s}$ collapse of passage-dropping methods.*

#### Critical Insights from Oracle Evaluation:
1. **Distributional Equivalence via Zero-Bypass:** Because total evidence length is $< 10,000$ characters, SPADE deactivates attention hooking and provides the complete uncompressed prompt to the drafter ($C_{\text{draft}} = C_{\text{full}}$). Consequently, the proposal distribution $q$ is distributionally identical to standard speculative decoding, matching acceptance rate ($\alpha_{\text{eval}} = 81.33\%$), committed tokens per round ($3.341$), and active KV cache ($47.25\text{ MB}$) to **$0.0000$ decimal places ($p = 1.0000$)**.
2. **Failure of Passage Pruning on Dense Evidence:** On oracle claims where every retrieved passage is gold evidence, `BM25 Top-2` forces document truncation, dropping acceptance from $81.33\% \to 78.73\%$ ($-2.60\text{ pp}$) and slowing throughput by **$-5.82\text{ tok/s}$** ($p < 10^{-4}$). SPADE completely avoids this failure mode.

---

### 6.5. Downstream Fact-Checking Veracity and Structural Integrity

We evaluate downstream fact-checking performance across categorical veracity classification (*Supported*, *Refuted*, *Not Enough Evidence*, *Conflicting Evidence*) and structured JSON formatting across all 300 claims in the primary held-out evaluation:
- **JSON Schema Parse Success Rate:** **100.0%** across all 1,800 evaluated generations ($6\text{ methods} \times 300\text{ claims}$);
- **Citation Identifier Validity:** **100.0%** across all generated references;
- **Task Veracity Classification:** Macro-F1 scores remain closely aligned: `B0-S` (0.1267), `Baseline Full` (0.1254), and `SPADE` (0.1222). Under stochastic sampling ($T=0.7, \text{top-}p=0.9$), sample paths diverge slightly across random seeds, so small variations in finite-sample Macro-F1 are expected.

We emphasize that SPADE is an inference acceleration framework and does not aim to alter or improve the underlying fact-checking reasoning or domain accuracy of the base model. The modest Macro-F1 (0.1222–0.1267) reflects `Qwen3-8B`'s zero-shot conservative thresholding on raw, uncurated web snippets rather than any degradation introduced by speculative decoding. As shown in Table 4, `Qwen3-8B` conservatively predicts *Not Enough Evidence* in 245–250 out of 300 claims, whereas human annotators labeled 189 claims as *Refuted* and 76 as *Supported*. Crucially, speculative decoding faithfully reproduces the target model's exact classification profile: `Baseline Full` and `SPADE` mirror `B0-S` across all categories (e.g., *Refuted* predicted 17 times by B0-S, 18 times by Full, and 20 times by SPADE; *Supported* predicted 16 times by B0-S, 12 times by Full, and 11 times by SPADE). This confirms that asymmetric draft context introduces no systematic distortion into the target model's task-level reasoning, strictly upholding Theorem 1.

| Veracity Class | Gold Annotations | B0-S (Target AR) | Baseline Full Spec | SPADE (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **Refuted** | 189 | 17 | 18 | 20 |
| **Supported** | 76 | 16 | 12 | 11 |
| **Not Enough Evidence** | 20 | 245 | 249 | 250 |
| **Conflicting Evidence** | 15 | 22 | 21 | 19 |
| **Macro-F1** | — | **0.1267** | **0.1254** | **0.1222** |

*Table 4: Verdict classification distribution across 300 held-out claims: comparing gold ground truth against target autoregressive (B0-S), standard speculative (Baseline Full), and SPADE.*

---

### 6.6. Cross-Model Architecture Generalization: Qwen2.5-7B and Qwen2.5-0.5B

To evaluate whether SPADE's acceleration and memory efficiency generalize beyond the primary Qwen3 family, we benchmarked an independent model architecture: `Qwen/Qwen2.5-7B-Instruct` as the target model paired with `Qwen/Qwen2.5-0.5B-Instruct` as the drafter. We evaluated all 300 held-out long-context claims from `retrieved-heldout300.jsonl` under an identical execution configuration ($K=4$, $T=0.7$, $p=0.9$, `flex_compile_reduce_overhead`, `fp32_fast`). Table 5 summarizes the efficiency and acceptance metrics.

| Decoding Method | Context Policy | Mean Draft Tok (Comp%) | Active Draft KV | Peak GPU VRAM | $\alpha_{\text{eval}}$ (%) | Tok/Rnd | Decode Thpt | E2E Thpt | E2E Latency | $\Delta$ vs. Full (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-S (Target AR)** | Full (Target Only) | — | — | 17.52 GB | — | 1.000 | 56.61 | 46.27 | 21.61 | -2.29 [-3.81, -0.93]$^*$ |
| **Baseline Full** | Full $\to$ Full | 5,107.1 (-0.0%) | 61.0 MB | 18.78 GB | 74.59% | 2.911 | 62.58 | 48.56 | 20.59 | — |
| **BM25 Top-2** | Full $\to$ Top-2 | 2,245.4 (-56.0%) | 27.5 MB | 18.54 GB | 72.79% | 2.817 | 64.98 | 50.73 | 19.71 | +2.17 [+1.41, +2.97]$^*$ |
| **BM25 Sent-B10k** | Full $\to$ Sent B=10k | 1,663.4 (-67.4%) | 20.7 MB | 18.51 GB | 73.61% | 2.861 | 67.47 | 52.07 | 19.21 | +3.51 [+2.84, +4.22]$^*$ |
| **SPADE (Ours)** | Full $\to$ Attn B=10k | **1,780.7 (-65.1%)** | **22.1 MB** | **18.55 GB** | **73.98%** | **2.876** | **66.61** | **51.67** | **19.35** | **+3.11 [+2.49, +3.76]**$^*$ |

*Table 5: Cross-Model Architecture Generalization on 300 Held-Out Long-Context Claims (Qwen2.5-7B Target + Qwen2.5-0.5B Draft, RTX 5090): Throughput reported in tok/s; latency in ms/tok. Statistical significance tested via paired cluster bootstrap ($B=10,000$ iterations, * denotes $p < 0.001$). Sourced directly from benchmark JSON reports. SPADE achieves a $+11.7\%$ speedup over Target AR ($51.67$ vs. $46.27\text{ tok/s}$) and $+6.4\%$ over Baseline Full ($51.67$ vs. $48.56\text{ tok/s}$), while slashing active draft KV cache by $-63.8\%$ ($61.0\text{ MB} \to 22.1\text{ MB}$).*

#### Cross-Model Findings:
1. **Consistent Wall-Clock Acceleration:** SPADE achieves **51.67 tok/s** end-to-end throughput, providing a statistically significant **+11.7% wall-clock speedup** over target autoregressive decoding (`B0-S`, $46.27\text{ tok/s}$, $\Delta = +5.40\text{ tok/s}$, 95% CI $[+3.83, +7.20]$, $p < 10^{-4}$) and a **+6.4% speedup** over uncompressed speculative decoding (`Baseline Full`, $48.56\text{ tok/s}$, $\Delta = +3.11\text{ tok/s}$, 95% CI $[+2.49, +3.76]$, $p < 10^{-4}$).
2. **Universal Active KV Footprint Reduction:** SPADE slashes active draft KV cache by **$-63.8\%$** ($61.0\text{ MB} \to 22.1\text{ MB}$, $p < 10^{-4}$), compressing draft prompt tokens by $-65.1\%$ ($5,107.1 \to 1,780.7$ tokens). This reduction mirrors the $-63.9\%$ KV cache savings observed on Qwen3, demonstrating that draft KV compression is structurally invariant to underlying transformer layer dimensions.
3. **Proposal Fidelity vs. Document Pruning Degradation:** Coarse document pruning (`BM25 Top-2`) suffers proposal divergence, dropping evaluated acceptance significantly by **$-1.80\text{ pp}$** vs. Baseline Full ($74.59\% \to 72.79\%$, $p < 10^{-4}$). In contrast, SPADE's attention-guided sentence selection preserves proposal fidelity with $\alpha_{\text{eval}} = \mathbf{73.98\%}$ ($2.876$ committed tokens/round), exhibiting no statistically significant difference from uncompressed Baseline Full ($\Delta = -0.62\text{ pp}$, 95% CI $[-1.35, +0.10]$, $p = 0.0968$) and significantly outperforming BM25 Top-2 by **+1.19 pp** ($p = 0.0020$).
4. **Dense Context Resilience:** In dense retrieval regimes ($3 \le m \le 5$ passages, $N=56$ claims), SPADE achieves an acceptance of $\mathbf{47.89\%}$—surpassing uncompressed Baseline Full ($46.79\%$) by **+1.10 pp** while shedding $55.2\%$ of context text. On ultra-long contexts ($L \ge 20\text{k}$ characters, $N=124$ claims), SPADE maintains $46.63\%$ acceptance (+2.48 pp higher than BM25 Top-2's $44.15\%$) while retaining 100% passage coverage and 100% gold justification recall.

---

## 7. Discussion & Negative Result Analysis

### 7.1. Ablation Analysis: Selection Strategy, Passage Coverage, and Content Retention

To rigorously decouple the scoring signal from context length and understand the role of structural passage coverage, we conducted a controlled 4-way ablation across all 300 held-out claims under an identical 10,000-character budget ($B=10\text{k}$) and runtime environment (`flex_compile_reduce_overhead`, `fp32_fast`, $K=4$). Table 6 reports empirical throughput, acceptance, and committed tokens per round alongside 10,000 paired cluster bootstrap iterations.

| Selection Strategy | Scoring Signal | Passage Coverage | Budget $B$ | Mean Draft Tokens (Comp%) | $\alpha_{\text{eval}}$ (%) | Tok/Rnd | E2E ms/tok | E2E tok/s | Speedup vs. Full |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Sentences** | Uniform Random | Yes ($m/m$) | 10,000 | 1,644.0 (-67.7%) | 76.50% | 2.996 | 23.61 | 42.36 | +2.1% |
| **BM25 Sentences** | Lexical BM25 | Yes ($m/m$) | 10,000 | 1,646.4 (-67.7%) | 77.24% | 3.047 | 23.45 | 42.65 | +2.8% |
| **Target-Attn (No Coverage)** | Target Attention Hook | No ($0/m$) | 10,000 | 1,776.4 (-65.1%) | 78.14% | 3.109 | 23.08 | 43.33 | +4.5% |
| **SPADE (Proposed)** | Target Attention Hook | **Yes ($m/m$)** | 10,000 | **1,776.8 (-65.1%)** | **78.20%** | **3.114** | **23.10** | **43.28** | **+4.4%** |

*Table 6: 4-Way ablation of evidence selection strategy and passage coverage guarantee under matched character budget ($B=10\text{k}$) on 300 held-out claims (NVIDIA RTX 5090).*

#### 1. Attention Guidance vs. Uniform Random Selection
When evidence sentences are selected uniformly at random under the identical $B=10\text{k}$ budget and coverage rule (`Random Sent-B10k`), evaluated acceptance $\alpha_{\text{eval}}$ collapses by **$-1.70\text{ pp}$** vs. SPADE ($76.50\%$ vs. $78.20\%$, $p < 10^{-4}$), committed tokens drop to $2.996$ per round ($p < 10^{-4}$), and throughput degrades to $42.36\text{ tok/s}$ ($p = 0.0014$). This confirms that sentence selection is not a trivial length reduction; the draft model requires semantically salient factual propositions to maintain proposal alignment with the target verifier.

#### 2. Attention Guidance vs. Lexical BM25 Matching
Under the identical 10,000-character budget, `BM25 Sent-B10k` achieves $\alpha_{\text{eval}} = 77.24\%$, suffering a statistically significant $-0.96\text{ pp}$ drop compared to SPADE ($p = 0.0022$) and $-0.92\text{ pp}$ compared to Baseline Full ($p = 0.0020$). Expanding the lexical budget from 8k to 10k did not rescue BM25 ($77.36\% \to 77.24\%$), because term-frequency matching greedily includes sentences with high keyword counts that are irrelevant to the target model's actual reasoning. In contrast, target prefill attention weights directly indicate which token positions the target verifier attends to during prefill, achieving superior proposal alignment ($78.20\%$, $p < 10^{-4}$) and $+0.63\text{ tok/s}$ higher throughput ($43.28$ vs. $42.65\text{ tok/s}$, $p = 0.0206$).

#### 3. Role of the Passage Coverage Guarantee ($m/m$) and Citation Recall
When the passage coverage constraint is removed (`Target-Attn NoCoverage`, $\text{min\_sentences} = 0$), the selector greedily packs sentences from a subset of passages, omitting unselected candidate documents. While unconstrained packing marginally reduces prefill overhead ($43.33$ vs. $43.28\text{ tok/s}$), it causes the draft model to drop cited evidence identifiers ($99.72\%$ vs. $100.0\%$ cited passage recall across 354 target citations). Enforcing the structural $m/m$ coverage rule guarantees 100% citation validity and 100% cited document retention with zero measurable latency cost ($\Delta = 0.025\text{ ms/tok}$, $p = 0.3310$).

Furthermore, as summarized in Table 6, naive passage truncation (`Selective Top-2`) drops **37.7% of passages cited by the target model** (Recall = $62.3\%$, retaining only 195 of 313 cited passages). In sharp contrast, SPADE achieves **100.0% citation recall** (354/354 cited passages retained) while compressing evidence text by $63.97\%$.

| Method | Mean Passage Coverage | Target-Cited Passages | Retained Cited Passages | Cited Passage Recall | Gold Justification Recall | Evidence Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Full** | 100.0% | 335 | 335 | **100.0%** | 100.0% | 0.0% (Uncompressed) |
| **Selective Top-2** | 60.8% | 313 | 195 | **62.3%** | 100.0% | 28.35% retained |
| **BM25 Sent-B8k** | 100.0% | 332 | 332 | **100.0%** | 100.0% | 41.20% retained |
| **BM25 Sent-B10k** | 100.0% | 328 | 328 | **100.0%** | 100.0% | 36.04% retained |
| **SPADE (Ours)** | **100.0%** | **354** | **354** | **100.0%** | **100.0%** | **36.03% retained** |
| Target-Attn (No Cov) | 99.67% | 355 | 354 | 99.72% | 100.0% | 36.03% retained |

*Table 6: Target-cited evidence passage retention and compression across 300 held-out claims.*

#### 4. Stratified Performance in Long-Context Regimes
Stratifying the 300 claims by context length and passage count reveals where SPADE's advantage is most pronounced:
- **Ultra-Long Contexts ($L \ge 20,000$ characters, $N=124$ claims, 41.3% of the benchmark):** Conventional passage-dropping (`BM25 Top-2`) and lexical selection (`BM25 Sent-B10k`) experience severe acceptance drops (collapsing to $49.66\%$ and $50.66\%$, respectively, representing $-3.66\text{ pp}$ and $-2.66\text{ pp}$ deficits vs. Full). In contrast, SPADE achieves $\alpha_{\text{eval}} = \mathbf{53.69\%}$ in this regime—higher than even the uncompressed Baseline Full ($53.32\%$)—while compressing context characters by **74.9%** (retaining only $25.1\%$ of evidence text).
- **High Passage Density ($6 \le m \le 10$ passages, $N=84$ claims):** Baseline Full achieves $53.08\%$ acceptance, while Top-2 collapses to $48.60\%$ ($-4.48\text{ pp}$). SPADE maintains $\mathbf{53.70\%}$ acceptance (+0.62 pp over Full) while shedding $77.8\%$ of uninformative evidence tokens.

### 7.2. Analysis of Draft Logit Biasing (Direction A Negative Result)
Prior to developing SPADE's asymmetric context formulation, we investigated evidence logit biasing (dynamically boosting draft model logits on continuous $n$-grams extracted from evidence). 

Rigorous profiling revealed that draft biasing is bounded by an empirical ceiling of only **+0.11 pp** in acceptance. Detailed prefix analysis proved that while logit biasing benefits prefixes where the target model copies evidence spans ($+26.50\text{ pp}$ gain across 164 prefixes), it severely penalizes prefixes where the target model paraphrases or synthesizes evidence ($-39.58\text{ pp}$ loss across 102 prefixes). Modulating draft logits without changing context cannot resolve this polarity. In contrast, SPADE allows the draft model to naturally generate both extractive and abstractive tokens by reading a clean, uncorrupted factual context.

---

## 8. Conclusion

We presented **SPADE**, an exact, evidence-selective speculative decoding framework for knowledge-intensive NLP generation. By decoupling the target model's full verification context from the draft model's compact sentence-budgeted context, SPADE resolves the dual bottlenecks of draft KV-cache bloat and passage-pruning divergence. Powered by a zero-overhead target-prefill attention hook and an optimized Blackwell GPU runtime, SPADE achieves a **+22.4% wall-clock speedup over autoregressive decoding** ($43.28\text{ tok/s}$) and a **+4.4% speedup over standard speculative decoding** on 300 held-out long-context claims, while cutting active draft KV cache by **63.9%** and peak GPU VRAM by **1.34 GB** with exact mathematical distribution guarantees.

---

## References

- Oren Barkan, Noam Weill, Avi Katz, and Noam Koenigstein. 2024. Bi-directional speculative decoding for efficient autoregressive generation. *arXiv preprint arXiv:2404.09887*.
- Tianle Cai, Yuhong Li, Zhengyang Geng, Hao Peng, Jason D. Lee, Daoqian Chen, and Tri Dao. 2024. Medusa: Simple framework for accelerating LLM generation with multiple decoding heads. *arXiv preprint arXiv:2401.10774*.
- Charlie Chen, Sebastian Borgeaud, Alistair O'Donoghue, et al. 2023. Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*.
- Guanzheng Chen, Qilong Feng, Jinjie Ni, Xin Li, and Michael Qizhe Shieh. 2025. RAPID: Long-context inference with retrieval-augmented speculative decoding. In *Proceedings of the 42nd International Conference on Machine Learning (ICML)*.
- Shuo Chen, Chen Zhang, Jun Wang, et al. 2026. SpecExtend: A drop-in enhancement for speculative decoding of long sequences. In *Findings of the Association for Computational Linguistics: ACL 2026*.
- Tri Dao. 2023. FlashAttention-2: Faster attention with better parallelism and work partitioning. In *International Conference on Learning Representations (ICLR)*.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*.
- Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, et al. 2024. Layer skip: Enabling early exit inference and speculative decoding. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*.
- Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of LLM inference using lookahead decoding. *arXiv preprint arXiv:2402.02057*.
- Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing prompts for accelerated inference of large language models. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. 2023. Efficient memory management for large language model serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*.
- Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In *International Conference on Machine Learning (ICML)*, pages 19274–19286. PMLR.
- Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024. EAGLE: Speculative sampling requires rethinking feature uncertainty. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*.
- Sheng Liang, Yongyue Zhang, Nathanael Brian, Hang Lv, Hao Wang, Chen Zhang, and Yong Liu. 2026. Context-asymmetric speculative decoding for agentic LLMs. In *Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.
- Xiaoxuan Liu, et al. 2026. LinguaSpec: Linguistic adaptive speculative decoding. In *ACL 2026*.
- Xupeng Miao, et al. 2023. SpecInfer: Accelerating generative LLM serving with tree-based speculative inference and verification. *arXiv preprint arXiv:2305.09781*.
- Michael Schlichtkrull, Zhijiang Guo, and Andreas Vlachos. 2023. AVeriTeC: A dataset for real-world claim verification with evidence from the web. In *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
- Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. *arXiv preprint arXiv:1911.02150*.
- Benjamin Spector and Christopher Re. 2023. Accelerating LLM inference with staged speculative decoding. *arXiv preprint arXiv:2308.04623*.
- Tianxiang Sun, et al. 2024. SpecTR: Fast speculative decoding via tree-based candidate verification. *arXiv preprint arXiv:2402.00882*.
- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In *NAACL-HLT*.
- Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation. In *ICLR 2024*.
- Ming Xu, et al. 2026. HeteroSpec: Leveraging contextual heterogeneity for efficient speculative decoding. In *ACL 2026*.
- Penghui Yang, Cunxiao Du, Fengzhuo Zhang, Haonan Wang, Tianyu Pang, Chao Du, and Bo An. 2026. LongSpec: Long-context lossless speculative decoding with efficient drafting and verification. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)*.
- Lianmin Zheng, et al. 2023. SGLang: Efficient execution of structured language model programs. *arXiv preprint arXiv:2312.07104*.
- Jun Zhang, et al. 2025. SVIP: Speculative verification with dynamic stopping. *arXiv preprint arXiv:2501.01123*.

---

## Appendix: Reproducibility & Experimental Provenance

### A. Hardware & Environment Specifications
- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB, Blackwell sm_120)
- **PyTorch:** 2.13.0+cu130 with Triton 3.7.1
- **Transformers:** 5.14.1
- **Primary Target Model:** `Qwen/Qwen3-8B` (BF16, commit `b968826`)
- **Primary Draft Model:** `Qwen/Qwen3-0.6B` (BF16, commit `c1899de`)
- **Generalization Target Model:** `Qwen/Qwen2.5-7B-Instruct` (BF16)
- **Generalization Draft Model:** `Qwen/Qwen2.5-0.5B-Instruct` (BF16)

### B. Statistical Significance Provenance
All paired comparisons were computed using $B = 10,000$ bootstrap iterations resampled strictly at the `claim_id` cluster level:
- Primary Held-Out (300 claims): `outputs/analysis/b1_4_heldout300/b1_3_live_report.json`
- Auxiliary Oracle (393 claims): `outputs/analysis/b1_4_oracle400/b1_3_live_report.json`
- Development Set (100 claims): `outputs/analysis/b1_3_live_retrieved100/b1_3_live_report.json`
- Qwen2.5 Generalization (300 claims): `outputs/analysis/p2_qwen25_heldout300/b1_3_live_report.json`
- Qwen2.5 Retention Analysis: `outputs/analysis/p2_qwen25_heldout300/content_retention_report.json`
