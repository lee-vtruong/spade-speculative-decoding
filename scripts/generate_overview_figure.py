"""Generate publication-grade architecture diagram for SPADE."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_spade_overview(output_path: str):
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Background subtle styling
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Color Palette (Publication style)
    c_evidence = '#E3F2FD'   # Soft Blue
    c_evidence_b = '#1976D2'
    c_draft = '#EDE7F6'      # Soft Purple
    c_draft_b = '#673AB7'
    c_target = '#E8F5E9'     # Soft Green
    c_target_b = '#388E3C'
    c_adaptive = '#FFF3E0'   # Soft Amber
    c_adaptive_b = '#F57C00'
    c_verify = '#FCE4EC'     # Soft Pink
    c_verify_b = '#C2185B'
    c_text = '#212121'

    # Title Banner
    ax.text(7.5, 8.1, "SPADE: Evidence-Adaptive Speculative Decoding Architecture", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#1A237E')

    # =========================================================================
    # Panel 1: Evidence Indexing (Left)
    # =========================================================================
    p1 = patches.FancyBboxPatch((0.5, 4.3), 3.5, 3.2, boxstyle="round,pad=0.2", 
                                ec=c_evidence_b, fc=c_evidence, lw=2)
    ax.add_patch(p1)
    ax.text(2.25, 7.2, "1. Retrieved Evidence & Indexing", ha='center', va='center', 
            fontsize=12, fontweight='bold', color=c_evidence_b)

    ax.text(2.25, 6.4, "Evidence Passages:\n[E1] \"Claim refuted in Oct 1994...\"\n[E2] \"Official census data indicates...\"", 
            ha='center', va='center', fontsize=9.5, style='italic', color='#37474F',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#ffffff', edgecolor='#B0BEC5', lw=1))

    ax.text(2.25, 5.0, "EvidenceIndexer (Trie / Hash Table)\nO(1) n-gram prefix continuation lookup\nS_E(x_<t) -> Valid Candidate Tokens", 
            ha='center', va='center', fontsize=9.5, fontweight='semibold', color='#0D47A1',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#BBDEFB', edgecolor=c_evidence_b, lw=1.2))

    # =========================================================================
    # Panel 2: Draft Model & Adaptive Biasing (Middle Left)
    # =========================================================================
    p2 = patches.FancyBboxPatch((4.5, 4.3), 4.8, 3.2, boxstyle="round,pad=0.2", 
                                ec=c_draft_b, fc=c_draft, lw=2)
    ax.add_patch(p2)
    ax.text(6.9, 7.2, "2. Draft Model & Dynamic Biasing", ha='center', va='center', 
            fontsize=12, fontweight='bold', color=c_draft_b)

    ax.text(6.9, 6.4, "Draft Model (Qwen3-0.6B)\nPrefix context x_<t -> Raw Logits z_t", 
            ha='center', va='center', fontsize=10, fontweight='semibold', color='#311B92',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#B39DDB', lw=1))

    ax.text(6.9, 5.1, "Dynamic Entropy-Gated Biasing:\nz'_t(v) = z_t(v) + beta(m) * sigma((H(q) - mu) / tau)\nTargeted boost for candidate tokens in S_E", 
            ha='center', va='center', fontsize=9.5, color='#4A148C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1C4E9', edgecolor=c_draft_b, lw=1.2))

    # =========================================================================
    # Panel 3: Dynamic Speculative Horizon K(t) (Middle Right)
    # =========================================================================
    p3 = patches.FancyBboxPatch((9.8, 4.3), 4.7, 3.2, boxstyle="round,pad=0.2", 
                                ec=c_adaptive_b, fc=c_adaptive, lw=2)
    ax.add_patch(p3)
    ax.text(12.15, 7.2, "3. Dynamic Speculative Horizon K(t)", ha='center', va='center', 
            fontsize=12, fontweight='bold', color=c_adaptive_b)

    ax.text(12.15, 6.4, "Adaptive Controller Policy:\nK(t) in {2, 4, 6} (Zero Extra Prefill Overhead)", 
            ha='center', va='center', fontsize=10, fontweight='semibold', color='#E65100',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#FFE0B2', lw=1))

    ax.text(12.15, 5.1, "• K = 6 (83.8%): Low entropy / Evidence match\n• K = 2 (10.8%): Immediate rejection throttling\n• K = 4 ( 5.5%): Moderate uncertainty", 
            ha='center', va='center', fontsize=9.5, color='#BF360C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCC80', edgecolor=c_adaptive_b, lw=1.2))

    # =========================================================================
    # Panel 4: Target Verification & Lossless Residual Sampling (Bottom Full)
    # =========================================================================
    p4 = patches.FancyBboxPatch((0.5, 0.5), 14.0, 3.3, boxstyle="round,pad=0.2", 
                                ec=c_target_b, fc=c_target, lw=2)
    ax.add_patch(p4)
    ax.text(7.5, 3.4, "4. Target Parallel Verification & Lossless Residual Sampling (Theorem 1)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color=c_target_b)

    # Sub-boxes in Target verification
    ax.text(3.5, 2.3, "Target Model (Qwen3-8B)\nParallel evaluation of (x_1 ... x_K) in 1 Forward Pass\nYields distributions p_1, ..., p_K and extra p_{K+1}", 
            ha='center', va='center', fontsize=9.5, fontweight='semibold', color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#A5D6A7', lw=1.2))

    ax.text(8.5, 2.3, "Exact Speculative Sampling Rule:\nalpha = min(1, p(x) / q'(x))\nRejected -> sample from residual: max(0, p(x) - q'(x))\nGuarantees Pr(X=x) = p(x) strictly!", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#E53935', lw=1.2))

    ax.text(13.0, 2.3, "Final Generated Output:\n{\n  \"summary\": \"Refuted...\",\n  \"citations\": [\"E1\"],\n  \"verdict\": \"Refuted\"\n}", 
            ha='center', va='center', fontsize=8.5, family='monospace', color='#263238',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#ECEFF1', edgecolor='#90A4AE', lw=1))

    # Bottom summary badge
    ax.text(7.5, 1.0, "Result: 4.04x Target Call Reduction | +1.07% Evidence Acceptance Gain | 100% JSON & Citation Correctness", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2E7D32', edgecolor='#1B5E20', lw=1.5))

    # =========================================================================
    # Connecting Arrows
    # =========================================================================
    arrow_props = dict(facecolor='#455A64', edgecolor='#37474F', width=2, headwidth=8, headlength=8)
    
    # 1 -> 2
    ax.annotate('', xy=(4.5, 5.7), xytext=(4.0, 5.7), arrowprops=arrow_props)
    ax.text(4.25, 6.0, "S_E", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1565C0')

    # 2 -> 3
    ax.annotate('', xy=(9.8, 5.7), xytext=(9.3, 5.7), arrowprops=arrow_props)
    ax.text(9.55, 6.0, "q'(x)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#4A148C')

    # 3 -> 4 (Downward arrow)
    ax.annotate('', xy=(7.5, 3.8), xytext=(7.5, 4.3), arrowprops=arrow_props)
    ax.text(7.9, 4.05, "Proposed (x_1..x_K)", ha='left', va='center', fontsize=9, fontweight='bold', color='#E65100')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overview figure saved to {output_path}")

if __name__ == "__main__":
    draw_spade_overview("docs/paper/latex/figures/spade_overview.png")
