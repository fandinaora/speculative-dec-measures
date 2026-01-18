"""
Analyze speculative decoding experiment results and plot metrics by draft step size.

Computes average prefill time, latency per token, and throughput for each draft step size,
then creates a grouped bar plot.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_speculative_results(results_dir: str = "results/speculative_performance") -> pd.DataFrame:
    """
    Load all CSV files from speculative decoding experiment.
    
    Args:
        results_dir: Directory containing the CSV files
        
    Returns:
        Combined DataFrame with all results
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all CSV files matching the pattern
    csv_files = list(results_path.glob("*_draft_steps_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No draft_steps CSV files found in {results_dir}\n"
            f"Expected files matching pattern: *_draft_steps_*.csv"
        )
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} samples")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Drop rows where draft_steps is NaN (can happen when mixing old and new CSV files)
    combined_df = combined_df.dropna(subset=['draft_steps'])
    
    # Ensure draft_steps is integer
    combined_df['draft_steps'] = combined_df['draft_steps'].astype(int)
    
    return combined_df


def compute_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average metrics for each draft step size.
    
    Args:
        df: DataFrame with all results
        
    Returns:
        DataFrame with one row per draft_steps, containing averages
    """
    # Group by draft_steps and compute averages
    averages = df.groupby('draft_steps').agg({
        'prefill_time_sec': 'mean',
        'latency_sec_per_token': 'mean',
        'throughput_tokens_per_sec': 'mean',
        'sample_id': 'count'  # Count samples per draft step
    }).reset_index()
    
    # Rename columns
    averages.columns = ['draft_steps', 'avg_prefill_time', 'avg_latency', 'avg_throughput', 'num_samples']
    
    # Sort by draft_steps
    averages = averages.sort_values('draft_steps')
    
    return averages


def plot_metrics(averages: pd.DataFrame, output_file: str = None):
    """
    Create three separate bar plots, one for each metric (prefill time, latency, throughput).
    
    Uses separate plots because the scales are very different.
    
    Args:
        averages: DataFrame with average metrics per draft step
        output_file: Optional path to save the plot
    """
    draft_steps = averages['draft_steps'].values
    
    # Prepare data for plotting
    prefill_times = averages['avg_prefill_time'].values
    latencies = averages['avg_latency'].values
    throughputs = averages['avg_throughput'].values
    
    # Create figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Set bar width and positions
    bar_width = 0.6
    x = np.arange(len(draft_steps))
    
    # Plot 1: Prefill Time
    ax1 = axes[0]
    bars1 = ax1.bar(x, prefill_times, bar_width, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Draft Steps (n_max)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Prefill Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Average Prefill Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(draft_steps)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, prefill_times)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Latency per Token
    ax2 = axes[1]
    bars2 = ax2.bar(x, latencies, bar_width, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Draft Steps (n_max)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Latency per Token (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Latency per Token', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(draft_steps)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, latencies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Throughput
    ax3 = axes[2]
    bars3 = ax3.bar(x, throughputs, bar_width, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Draft Steps (n_max)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Throughput (tokens/second)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Throughput', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(draft_steps)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars3, throughputs)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Average Metrics by Draft Step Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def print_summary(averages: pd.DataFrame):
    """Print summary statistics."""
    print("Summary: Average Metrics by Draft Step Size")
    print(f"{'Draft Steps':<12} {'Prefill Time (s)':<18} {'Latency (s/token)':<18} {'Throughput (tok/s)':<18} {'Samples':<10}")
  
    
    for _, row in averages.iterrows():
        print(f"{int(row['draft_steps']):<12} "
              f"{row['avg_prefill_time']:<18.4f} "
              f"{row['avg_latency']:<18.4f} "
              f"{row['avg_throughput']:<18.4f} "
              f"{int(row['num_samples']):<10}")


def compute_total_evaluation_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total evaluation time for each draft step size.
    
    Total time = sum of (prefill_time + total_generation_time) for all samples
    with the same draft_steps value.
    
    Args:
        df: DataFrame with all results
        
    Returns:
        DataFrame with draft_steps and total_time_sec
    """
    # Calculate total time per sample (prefill + generation)
    # Note: total_generation_time_sec already includes the generation time for all tokens
    # We need to add prefill_time_sec to get the complete time per sample
    df['total_time_per_sample'] = df['prefill_time_sec'] + df['total_generation_time_sec']
    
    # Group by draft_steps and sum the total time
    total_times = df.groupby('draft_steps')['total_time_per_sample'].agg(['sum', 'count']).reset_index()
    total_times.columns = ['draft_steps', 'total_time_sec', 'num_samples']
    
    # Sort by draft_steps
    total_times = total_times.sort_values('draft_steps')
    
    return total_times


def plot_evaluation_time(total_times: pd.DataFrame, output_file: str = None):
    """
    Plot total evaluation time vs draft step size.
    
    Args:
        total_times: DataFrame with draft_steps and total_time_sec
        output_file: Optional path to save the plot
    """
    draft_steps = total_times['draft_steps'].values
    total_times_sec = total_times['total_time_sec'].values
    total_times_min = total_times_sec / 60.0  # Convert to minutes for readability
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    bars = ax.bar(draft_steps.astype(str), total_times_min, 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_xlabel('Draft Steps (n_max)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Evaluation Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Total Time to Evaluate Dataset vs Draft Step Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add value labels on top of bars
    for i, (bar, time_min, time_sec) in enumerate(zip(bars, total_times_min, total_times_sec)):
        height = bar.get_height()
        # Show both minutes and seconds for clarity
        if time_min < 1:
            label = f'{time_sec:.1f}s'
        else:
            label = f'{time_min:.1f}m\n({time_sec:.0f}s)'
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def print_evaluation_time_summary(total_times: pd.DataFrame):
    """Print summary of total evaluation times."""
    print("Total Evaluation Time by Draft Step Size")
    print(f"{'Draft Steps':<12} {'Total Time (min)':<18} {'Total Time (sec)':<18} {'Samples':<10}")
    
    for _, row in total_times.iterrows():
        time_min = row['total_time_sec'] / 60.0
        print(f"{int(row['draft_steps']):<12} "
              f"{time_min:<18.2f} "
              f"{row['total_time_sec']:<18.1f} "
              f"{int(row['num_samples']):<10}")


def compute_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average acceptance rate for each draft step size.
    
    Acceptance rate is the ratio of accepted draft tokens to total draft tokens.
    Only available when speculative decoding is enabled (draft_n > 0).
    
    Args:
        df: DataFrame with all results (must contain 'acceptance_rate' column)
        
    Returns:
        DataFrame with draft_steps and avg_acceptance_rate
    """
    # Filter rows where acceptance_rate is available (not NaN)
    df_with_acceptance = df[df['acceptance_rate'].notna()].copy()
    
    if len(df_with_acceptance) == 0:
        raise ValueError("No acceptance_rate data found. Ensure speculative decoding is enabled.")
    
    # Group by draft_steps and compute average acceptance rate
    acceptance_stats = df_with_acceptance.groupby('draft_steps').agg({
        'acceptance_rate': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    acceptance_stats.columns = ['draft_steps', 'avg_acceptance_rate', 'std_acceptance_rate', 'num_samples']
    
    # Sort by draft_steps
    acceptance_stats = acceptance_stats.sort_values('draft_steps')
    
    return acceptance_stats


def plot_acceptance_rate(acceptance_stats: pd.DataFrame, output_file: str = None):
    """
    Plot average acceptance rate vs draft step size.
    
    Args:
        acceptance_stats: DataFrame with draft_steps and avg_acceptance_rate
        output_file: Optional path to save the plot
    """
    draft_steps = acceptance_stats['draft_steps'].values
    avg_acceptance = acceptance_stats['avg_acceptance_rate'].values
    std_acceptance = acceptance_stats['std_acceptance_rate'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart with error bars
    bars = ax.bar(draft_steps.astype(str), avg_acceptance, 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6,
                   yerr=std_acceptance, capsize=5, error_kw={'linewidth': 2, 'ecolor': '#34495e'})
    
    ax.set_xlabel('Draft Steps (n_max)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Acceptance Rate', fontsize=12, fontweight='bold')
    ax.set_title('Speculative Decoding: Acceptance Rate vs Draft Step Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (bar, acceptance, std) in enumerate(zip(bars, avg_acceptance, std_acceptance)):
        height = bar.get_height()
        label = f'{acceptance:.3f}\n±{std:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set y-axis limit to accommodate labels (acceptance rate is between 0 and 1, add space for labels)
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def print_acceptance_rate_summary(acceptance_stats: pd.DataFrame):
    """Print summary of acceptance rates."""
    print("\nAcceptance Rate by Draft Step Size")
    print(f"{'Draft Steps':<12} {'Avg Accept Rate':<18} {'Std Dev':<18} {'Samples':<10}")
    
    for _, row in acceptance_stats.iterrows():
        print(f"{int(row['draft_steps']):<12} "
              f"{row['avg_acceptance_rate']:<18.4f} "
              f"{row['std_acceptance_rate']:<18.4f} "
              f"{int(row['num_samples']):<10}")


def main():
    """Main function to analyze and plot speculative decoding results."""
    results_dir = "results/speculative_performance"
    

    # Load results
    df = load_speculative_results(results_dir)
    
    print(f"\nTotal samples loaded: {len(df)}")
    print(f"Draft step sizes found: {sorted(df['draft_steps'].unique())}")
    
    averages = compute_averages(df)
    
    # Print summary
    print_summary(averages)
    
    # Create plot
    output_file = Path(results_dir) / "speculative_metrics_by_draft_steps.png"
    plot_metrics(averages, output_file=str(output_file))
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
