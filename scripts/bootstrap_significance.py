import json
import random
import math

def run_bootstrap_analysis():
    b2_trace_file = "outputs/runs/full-b2-k4-lim100/token_traces.jsonl"
    m4_trace_file = "outputs/runs/full-m4-k4-lim100/token_traces.jsonl"
    
    def load_claim_stats(trace_path):
        claims = {}
        with open(trace_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                cid = rec['claim_id']
                if cid not in claims:
                    claims[cid] = {'ev_eval': 0, 'ev_acc': 0, 'non_eval': 0, 'non_acc': 0, 'all_eval': 0, 'all_acc': 0}
                
                if not rec.get('reached_by_verifier', False):
                    continue
                    
                claims[cid]['all_eval'] += 1
                is_acc = 1 if rec.get('status') == 'accepted' else 0
                if is_acc:
                    claims[cid]['all_acc'] += 1
                    
                overlap = rec.get('evidence_overlap', {}).get('lexical_overlap', 0.0)
                if overlap > 0:
                    claims[cid]['ev_eval'] += 1
                    if is_acc:
                        claims[cid]['ev_acc'] += 1
                else:
                    claims[cid]['non_eval'] += 1
                    if is_acc:
                        claims[cid]['non_acc'] += 1
        return claims

    b2_claims = load_claim_stats(b2_trace_file)
    m4_claims = load_claim_stats(m4_trace_file)
    
    common_claims = sorted(list(set(b2_claims.keys()) & set(m4_claims.keys())))
    n_claims = len(common_claims)
    print(f"Total evaluated claims in both runs: {n_claims}")
    
    # 1. Pooled Numbers
    b2_ev_eval = sum(b2_claims[c]['ev_eval'] for c in common_claims)
    b2_ev_acc = sum(b2_claims[c]['ev_acc'] for c in common_claims)
    m4_ev_eval = sum(m4_claims[c]['ev_eval'] for c in common_claims)
    m4_ev_acc = sum(m4_claims[c]['ev_acc'] for c in common_claims)
    
    b2_rate = b2_ev_acc / b2_ev_eval
    m4_rate = m4_ev_acc / m4_ev_eval
    obs_delta = m4_rate - b2_rate
    
    print(f"\nPOOLED EVIDENCE OVERLAP ACCEPTANCE:")
    print(f"  B2 Baseline (K=4): {b2_rate*100:.2f}% ({b2_ev_acc}/{b2_ev_eval})")
    print(f"  SPADE M4 (K=4):    {m4_rate*100:.2f}% ({m4_ev_acc}/{m4_ev_eval})")
    print(f"  Observed Delta:    {obs_delta*100:+.2f}%")
    
    # 2. Claim-level paired statistics
    claim_ev_deltas = []
    wins, losses, ties = 0, 0, 0
    
    for c in common_claims:
        if b2_claims[c]['ev_eval'] > 0 and m4_claims[c]['ev_eval'] > 0:
            b2_r = b2_claims[c]['ev_acc'] / b2_claims[c]['ev_eval']
            m4_r = m4_claims[c]['ev_acc'] / m4_claims[c]['ev_eval']
            d = m4_r - b2_r
            claim_ev_deltas.append(d)
            if d > 0.0001: wins += 1
            elif d < -0.0001: losses += 1
            else: ties += 1
            
    mean_claim_delta = sum(claim_ev_deltas) / len(claim_ev_deltas)
    std_claim_delta = math.sqrt(sum((x - mean_claim_delta)**2 for x in claim_ev_deltas) / (len(claim_ev_deltas) - 1))
    
    print(f"\nCLAIM-LEVEL PAIRED ANALYSIS (N={len(claim_ev_deltas)} claims with evidence proposals):")
    print(f"  SPADE Wins:   {wins} ({wins/len(claim_ev_deltas)*100:.1f}%)")
    print(f"  SPADE Losses: {losses} ({losses/len(claim_ev_deltas)*100:.1f}%)")
    print(f"  Ties:         {ties} ({ties/len(claim_ev_deltas)*100:.1f}%)")
    print(f"  Mean Claim-level Delta: {mean_claim_delta*100:+.2f}% (std: {std_claim_delta*100:.2f}%)")
    
    # Paired t-test
    se = std_claim_delta / math.sqrt(len(claim_ev_deltas))
    t_stat = mean_claim_delta / se if se > 0 else 0
    print(f"  Paired t-statistic: t = {t_stat:.3f} (df = {len(claim_ev_deltas)-1})")
    
    # 3. Cluster Bootstrap by Claim (10,000 resamples)
    random.seed(42)
    B = 10000
    boot_deltas = []
    
    for _ in range(B):
        # sample claim indices with replacement
        sample = random.choices(common_claims, k=n_claims)
        boot_b2_acc = sum(b2_claims[c]['ev_acc'] for c in sample)
        boot_b2_eval = sum(b2_claims[c]['ev_eval'] for c in sample)
        boot_m4_acc = sum(m4_claims[c]['ev_acc'] for c in sample)
        boot_m4_eval = sum(m4_claims[c]['ev_eval'] for c in sample)
        
        r_b2 = boot_b2_acc / boot_b2_eval if boot_b2_eval > 0 else 0
        r_m4 = boot_m4_acc / boot_m4_eval if boot_m4_eval > 0 else 0
        boot_deltas.append(r_m4 - r_b2)
        
    boot_deltas.sort()
    ci_lower = boot_deltas[int(0.025 * B)]
    ci_upper = boot_deltas[int(0.975 * B)]
    boot_mean = sum(boot_deltas) / B
    boot_se = math.sqrt(sum((x - boot_mean)**2 for x in boot_deltas) / (B - 1))
    
    p_value_one_tailed = sum(1 for x in boot_deltas if x <= 0) / B
    p_value_two_tailed = sum(1 for x in boot_deltas if abs(x - boot_mean) >= abs(obs_delta)) / B
    
    print(f"\nCLUSTER BOOTSTRAP RESAMPLING (B=10,000 resamples):")
    print(f"  Observed Delta:                 {obs_delta*100:+.2f}%")
    print(f"  Bootstrap Mean Delta:           {boot_mean*100:+.2f}%")
    print(f"  Bootstrap Standard Error (SE):  {boot_se*100:.2f}%")
    print(f"  95% Confidence Interval:        [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]")
    print(f"  Empirical p-value (one-tailed): p = {p_value_one_tailed:.4f}")
    print(f"  Empirical p-value (two-tailed): p = {p_value_two_tailed:.4f}")
    
    # 4. Overall acceptance bootstrap
    boot_overall_deltas = []
    for _ in range(B):
        sample = random.choices(common_claims, k=n_claims)
        boot_b2_acc = sum(b2_claims[c]['all_acc'] for c in sample)
        boot_b2_eval = sum(b2_claims[c]['all_eval'] for c in sample)
        boot_m4_acc = sum(m4_claims[c]['all_acc'] for c in sample)
        boot_m4_eval = sum(m4_claims[c]['all_eval'] for c in sample)
        r_b2 = boot_b2_acc / boot_b2_eval if boot_b2_eval > 0 else 0
        r_m4 = boot_m4_acc / boot_m4_eval if boot_m4_eval > 0 else 0
        boot_overall_deltas.append(r_m4 - r_b2)
        
    boot_overall_deltas.sort()
    ci_overall_lower = boot_overall_deltas[int(0.025 * B)]
    ci_overall_upper = boot_overall_deltas[int(0.975 * B)]
    p_overall_one = sum(1 for x in boot_overall_deltas if x <= 0) / B
    
    print(f"\nOVERALL EVALUATED ACCEPTANCE BOOTSTRAP (B=10,000):")
    obs_all_delta = (sum(m4_claims[c]['all_acc'] for c in common_claims)/sum(m4_claims[c]['all_eval'] for c in common_claims)) - \
                    (sum(b2_claims[c]['all_acc'] for c in common_claims)/sum(b2_claims[c]['all_eval'] for c in common_claims))
    print(f"  Observed Overall Delta:         {obs_all_delta*100:+.2f}%")
    print(f"  95% Confidence Interval:        [{ci_overall_lower*100:+.2f}%, {ci_overall_upper*100:+.2f}%]")
    print(f"  Empirical p-value (one-tailed): p = {p_overall_one:.4f}")

if __name__ == '__main__':
    run_bootstrap_analysis()
