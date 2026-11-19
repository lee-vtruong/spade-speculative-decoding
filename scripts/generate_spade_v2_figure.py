"""Generate modern publication-grade architecture diagram for SPADE (Evidence-Selective Speculative Decoding)."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_spade_architecture(output_path: str):
    fig, ax = plt.subplots(figsize=(16, 9.2), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # Background styling
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Color Palette (Nature/ACL style)
    c_blue_bg = '#EBF3FB'
    c_blue_border = '#1976D2'
    c_purple_bg = '#F3E5F5'
    c_purple_border = '#7B1FA2'
    c_green_bg = '#E8F5E9'
    c_green_border = '#2E7D32'
    c_amber_bg = '#FFF8E1'
    c_amber_border = '#F57F17'
    c_red_bg = '#FFEBEE'
    c_red_border = '#C62828'
    c_slate_bg = '#ECEFF1'
    c_slate_border = '#455A64'

    # Title Banner
    ax.text(8.0, 8.8, "SPADE: Evidence-Selective Draft Context Speculative Decoding Architecture", 
            ha='center', va='center', fontsize=17, fontweight='bold', color='#0D47A1')

    # =========================================================================
    # Panel 1: Input Prompts & Evidence Passages (Top-Left)
    # =========================================================================
    p1 = patches.FancyBboxPatch((0.5, 4.7), 4.6, 3.7, boxstyle="round,pad=0.2", 
                                ec=c_slate_border, fc=c_slate_bg, lw=2)
    ax.add_patch(p1)
    ax.text(2.8, 8.1, "1. Knowledge-Intensive Evidence Context", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=c_slate_border)

    ax.text(2.8, 7.35, "Full Prompt Context $C_{full}$ (5k - 20k tokens):\n• User Claim $c$ + Multi-hop Web Evidence\n• Evidence Documents $\\mathcal{E} = \\{E_1, E_2, \\dots, E_m\\}$", 
            ha='center', va='center', fontsize=9.2, color='#263238',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#CFD8DC', lw=1.2))

    ax.text(2.8, 6.1, "The Speculative Grounding Dilemma:\n"
                       "• Full Drafter Context -> KV bloat & slow prefill\n"
                       "• Passage Pruning (Top-K) -> Drops cited evidence\n"
                       "  causes -4.48 pp acceptance collapse on 52% claims!", 
            ha='center', va='center', fontsize=8.8, color='#C62828',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFEBEE', edgecolor='#EF9A9A', lw=1.2))

    # =========================================================================
    # Panel 2: Target Model Prefill & Attention Hook (Top-Middle)
    # =========================================================================
    p2 = patches.FancyBboxPatch((5.5, 4.7), 4.8, 3.7, boxstyle="round,pad=0.2", 
                                ec=c_green_border, fc=c_green_bg, lw=2)
    ax.add_patch(p2)
    ax.text(7.9, 8.1, "2. Target Prefill & Attention Hook", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=c_green_border)

    ax.text(7.9, 7.35, "Target Model $\\mathcal{M}_p$ (Qwen3-8B BF16):\n• Full prompt prefill over complete $C_{full}$\n• Retains 100% context for faithful verification", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#A5D6A7', lw=1.2))

    ax.text(7.9, 5.95, "TargetPrefillAttentionHook (Zero Extra Forward):\n"
                       "• Intercepts upper causal self-attention layers\n"
                       "• Claim query tokens attend to evidence key tokens\n"
                       "• Computes token salience $S_{tok}(i) = \\frac{1}{|L|} \\sum Q_W K_i^\\top$\n"
                       "• Aggregates into sentence scores $S_{sent}(s)$", 
            ha='center', va='center', fontsize=8.8, color='#004D40',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#E0F2F1', edgecolor='#80CBC4', lw=1.2))

    # =========================================================================
    # Panel 3: Sentence-Budget Context Packing (Top-Right)
    # =========================================================================
    p3 = patches.FancyBboxPatch((10.7, 4.7), 4.8, 3.7, boxstyle="round,pad=0.2", 
                                ec=c_amber_border, fc=c_amber_bg, lw=2)
    ax.add_patch(p3)
    ax.text(13.1, 8.1, "3. Budgeted Sentence Packing", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=c_amber_border)

    ax.text(13.1, 7.35, "Sentence-Budget Context Packing ($B_{chars}=10k$):\n• Preserves 100% of Passage Headers (E1..Em)\n• Keeps schema alignment & citation validity", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#E65100',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#FFE0B2', lw=1.2))

    ax.text(13.1, 5.95, "Compact Draft Context $C_{draft}$:\n"
                       "• Greedily packs top $S_{sent}$ across ALL passages\n"
                       "• Restores natural document sentence order\n"
                       "• Sheds 65% - 75% noisy context tokens!\n"
                       "• Zero-Bypass: Bypasses hook when $\\sum |E_i| \\leq B$", 
            ha='center', va='center', fontsize=8.8, color='#BF360C',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF3E0', edgecolor='#FFCC80', lw=1.2))

    # =========================================================================
    # Panel 4: Hardware-Optimized Draft Speculation (Bottom-Left)
    # =========================================================================
    p4 = patches.FancyBboxPatch((0.5, 0.5), 7.0, 3.8, boxstyle="round,pad=0.2", 
                                ec=c_purple_border, fc=c_purple_bg, lw=2)
    ax.add_patch(p4)
    ax.text(4.0, 4.0, "4. Fast Draft Speculation (Qwen3-0.6B on $C_{draft}$)", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=c_purple_border)

    ax.text(4.0, 3.25, "Draft Execution on Compact Evidence ($C_{draft}$):\n"
                       "• Active Draft KV-cache cut by -63.9% (566.9MB -> 204.6MB)\n"
                       "• Draft prefill latency reduced by -64% (92ms -> 33ms)", 
            ha='center', va='center', fontsize=9.2, color='#4A148C',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#D1C4E9', lw=1.2))

    ax.text(4.0, 1.8, "Blackwell Systems Optimization (RTX 5090):\n"
                       "• StaticCache: Pre-allocated tensor buffers (zero dynamic malloc)\n"
                       "• FlexAttention: In-place causal BlockMask (Q_len=1)\n"
                       "• torch.compile(mode=\"reduce-overhead\"): 3.04 ms/step decode\n"
                       "• Autoregressively speculates K=4 candidate tokens $\\tilde{x}_1, \\dots, \\tilde{x}_K$", 
            ha='center', va='center', fontsize=8.8, color='#311B92',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#EDE7F6', edgecolor='#B39DDB', lw=1.2))

    # =========================================================================
    # Panel 5: Target Batched Verification & Lossless Sampling (Bottom-Right)
    # =========================================================================
    p5 = patches.FancyBboxPatch((8.0, 0.5), 7.5, 3.8, boxstyle="round,pad=0.2", 
                                ec=c_blue_border, fc=c_blue_bg, lw=2)
    ax.add_patch(p5)
    ax.text(11.75, 4.0, "5. Exact Speculative Verification & Rollback", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=c_blue_border)

    ax.text(11.75, 3.25, "Target Parallel Verification on Full Context ($C_{full}$):\n"
                        "• Single batched forward pass evaluates candidate prefix simultaneously\n"
                        "• Computes true target conditional logits $p_1, \\dots, p_{K+1}$", 
            ha='center', va='center', fontsize=9.2, color='#0D47A1',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#90CAF9', lw=1.2))

    ax.text(11.75, 1.8, "Exact Residual Speculative Sampling (Leviathan et al.):\n"
                        "• Accept candidate with $\\alpha_k = \\min(1, p_k(\\tilde{x}_k) / q_k(\\tilde{x}_k))$\n"
                        "• On reject at $k^*$: sample residual $x \\sim \\max(0, p - q)$ & rollback KV cache\n"
                        "• Zero-Sync fp32_fast: GPU in-place nucleus sampling (2.07 ms vs 31.04 ms)\n"
                        "• Theorem 1 (Target Distribution Invariance): $P(X_t=y) = p(y \\mid C_{full}, x_{<t})$ strictly!", 
            ha='center', va='center', fontsize=8.8, fontweight='semibold', color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFCDD2', edgecolor='#E53935', lw=1.2))

    # =========================================================================
    # Directed Connecting Arrows
    # =========================================================================
    arrow_kw = dict(arrowstyle="simple,tail_width=2,head_width=7,head_length=8", color='#1565C0')
    arrow_amber = dict(arrowstyle="simple,tail_width=2,head_width=7,head_length=8", color='#E65100')
    arrow_green = dict(arrowstyle="simple,tail_width=2,head_width=7,head_length=8", color='#2E7D32')
    arrow_purple = dict(arrowstyle="simple,tail_width=2,head_width=7,head_length=8", color='#7B1FA2')

    # Arrow 1 -> 2 (Full context to Target prefill)
    ax.annotate("", xy=(5.5, 6.55), xytext=(5.1, 6.55), arrowprops=arrow_green)
    # Arrow 2 -> 3 (Attention scores to Sentence Packer)
    ax.annotate("", xy=(10.7, 6.55), xytext=(10.3, 6.55), arrowprops=arrow_amber)
    # Arrow 3 -> 4 (Compact prompt to Drafter)
    ax.annotate("", xy=(5.2, 4.3), xytext=(12.0, 4.7), 
                arrowprops=dict(arrowstyle="simple,tail_width=2,head_width=7,head_length=8", color='#7B1FA2',
                                connectionstyle="arc3,rad=-0.25"))
    # Arrow 4 -> 5 (Proposals to Verifier)
    ax.annotate("", xy=(8.0, 2.4), xytext=(7.5, 2.4), arrowprops=arrow_kw)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    draw_spade_architecture("docs/paper/latex/figures/spade_overview.png")
