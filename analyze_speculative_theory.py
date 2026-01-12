"""
Analyze why throughput keeps improving in experimental results vs theoretical predictions.

Compares experimental results with theoretical model to understand why there's no plateau.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils.analyze_speculative_results import load_speculative_results, compute_averages


def calculate_theoretical_throughput(n_max, alpha, L_target, L_draft):
    """
    Calculate theoretical throughput for given parameters.
    
    Args:
        n_max: Draft step size
        alpha: Acceptance rate
        L_target: Target model latency per token (seconds)
        L_draft: Draft model latency per token (seconds)
        
    Returns:
        Theoretical throughput (tokens/second)
    """
    # Cost per iteration
    cost = n_max * L_draft + L_target
    
    # Expected tokens per iteration
    expected_tokens = alpha * n_max + 1
    
    # Throughput
    throughput = expected_tokens / cost
    
    return throughput


def estimate_latencies_from_data(df: pd.DataFrame):
    """
    Estimate L_target and L_draft from experimental data.
    
    For baseline (no speculative decoding), latency = L_target
    For speculative decoding, we can estimate from the data.
    """
    # Estimate L_target from latency (this is the average per-token latency)
    # In speculative decoding, this includes both draft and target model work
    avg_latency = df['latency_sec_per_token'].mean()
    
    # Estimate L_draft: if we have acceptance rate data, we can work backwards
    # For now, use a rough estimate: draft is typically 5-10x faster
    # We'll refine this with acceptance rate data if available
    
    return avg_latency


def analyze_acceptance_rates(df: pd.DataFrame):
    """
    Analyze acceptance rates if available in the data.
    """
    if 'acceptance_rate' in df.columns:
        acceptance_by_draft = df.groupby('draft_steps')['acceptance_rate'].agg(['mean', 'std', 'count'])
        return acceptance_by_draft
    else:
        print("Warning: acceptance_rate not found in data. Need to re-run experiments with updated measurement code.")
        return None


def compare_theory_vs_experiment(results_dir: str = "results/speculative_performance"):
    """
    Compare theoretical predictions with experimental results.
    """
    print("=" * 80)
    print("Theoretical vs Experimental Analysis")
    print("=" * 80)
    
    # Load experimental data
    df = load_speculative_results(results_dir)
    averages = compute_averages(df)
    
    print("\nExperimental Results:")
    print("-" * 80)
    print(f"{'Draft Steps':<12} {'Throughput (exp)':<18} {'Latency (exp)':<18}")
    print("-" * 80)
    for _, row in averages.iterrows():
        print(f"{int(row['draft_steps']):<12} "
              f"{row['avg_throughput']:<18.2f} "
              f"{row['avg_latency']:<18.4f}")
    
    # Check if we have acceptance rate data
    has_acceptance = 'acceptance_rate' in df.columns
    
    if has_acceptance:
        print("\n" + "=" * 80)
        print("Acceptance Rate Analysis")
        print("=" * 80)
        acceptance_by_draft = analyze_acceptance_rates(df)
        print(acceptance_by_draft)
        
        # Use actual acceptance rates for theoretical calculation
        print("\n" + "=" * 80)
        print("Theoretical Throughput (using actual acceptance rates)")
        print("=" * 80)
        
        # Estimate L_target and L_draft
        # L_target: baseline latency (without speculative decoding)
        # For now, use the latency from smallest draft step as approximation
        baseline_latency = averages[averages['draft_steps'] == averages['draft_steps'].min()]['avg_latency'].values[0]
        L_target = baseline_latency
        
        # Estimate L_draft: assume draft is 10x faster (typical ratio)
        L_draft = L_target / 10
        
        print(f"\nEstimated parameters:")
        print(f"  L_target ≈ {L_target*1000:.2f} ms/token")
        print(f"  L_draft ≈ {L_draft*1000:.2f} ms/token")
        print(f"  Ratio: {L_target/L_draft:.1f}x")
        
        print(f"\n{'Draft Steps':<12} {'α (actual)':<15} {'Throughput (exp)':<18} {'Throughput (theory)':<20} {'Match':<10}")
        print("-" * 80)
        
        for draft_steps in sorted(df['draft_steps'].unique()):
            exp_data = averages[averages['draft_steps'] == draft_steps].iloc[0]
            exp_throughput = exp_data['avg_throughput']
            
            # Get actual acceptance rate for this draft step
            draft_data = df[df['draft_steps'] == draft_steps]
            if 'acceptance_rate' in draft_data.columns:
                alpha = draft_data['acceptance_rate'].mean()
                theory_throughput = calculate_theoretical_throughput(draft_steps, alpha, L_target, L_draft)
                match = "OK" if abs(exp_throughput - theory_throughput) / exp_throughput < 0.2 else "NO"
                print(f"{draft_steps:<12} {alpha:<15.3f} {exp_throughput:<18.2f} {theory_throughput:<20.2f} {match:<10}")
    else:
        print("\n" + "=" * 80)
        print("Why Throughput Keeps Improving: Possible Explanations")
        print("=" * 80)
        
        print("\n1. **Acceptance rate might be staying high**")
        print("   - If acceptance rate doesn't decrease much with n_max, throughput keeps increasing")
        print("   - Need to check actual acceptance rates from timings")
        
        print("\n2. **Draft model is very fast relative to target**")
        print("   - If L_draft << L_target, even low acceptance rates can help")
        print("   - The condition alpha * L_target > L_draft might be satisfied for all n_max")
        
        print("\n3. **Theoretical model assumes constant acceptance rate**")
        print("   - With constant acceptance rate, if alpha * L_target > L_draft, throughput increases unbounded")
        print("   - In practice, acceptance rate decreases with n_max, creating an optimum")
        print("   - But if acceptance rate stays high enough, no optimum is reached in your range")
        
        print("\n4. **Batch verification efficiency**")
        print("   - Target model verifies all draft tokens in parallel (batch)")
        print("   - This makes verification cost constant regardless of n_max")
        print("   - So cost = n_max * L_draft + L_target (constant)")
        print("   - If acceptance rate stays high, more tokens = better throughput")
        
        print("\n5. **Need to check acceptance rates**")
        print("   - Re-run experiments with updated measurement code that captures draft_n and draft_n_accepted")
        print("   - This will show if acceptance rate is actually decreasing")


def main():
    """Main analysis function."""
    results_dir = "results/speculative_performance"
    compare_theory_vs_experiment(results_dir)


if __name__ == "__main__":
    main()
