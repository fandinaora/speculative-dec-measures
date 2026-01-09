"""
Plotting utilities for visualizing benchmark results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


def load_benchmark_data(results_dir: str = "results") -> pd.DataFrame:
    """
    Load all benchmark CSV files and combine them into a single DataFrame.
    
    Args:
        results_dir: Directory containing the CSV files (default: "results")
    
    Returns:
        Combined DataFrame with all benchmark results
    """
    results_path = Path(results_dir)
    
    # If relative path doesn't exist, try looking in parent directory (project root)
    if not results_path.exists() and not results_path.is_absolute():
        # Try parent directory (for when running from utils/ subdirectory)
        parent_path = Path(__file__).parent.parent / results_dir
        if parent_path.exists():
            results_path = parent_path
        else:
            raise FileNotFoundError(
                f"Results directory not found: {results_dir}\n"
                f"Tried: {Path(results_dir).absolute()}\n"
                f"Tried: {parent_path.absolute()}"
            )
    elif not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all CSV files matching the pattern
    csv_files = list(results_path.glob("benchmark_max_tokens_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No benchmark CSV files found in {results_dir}")
    
    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def plot_benchmark_distributions(
    results_dir: str = "results",
    output_file: Optional[str] = None,
    figsize: tuple = (15, 5)
) -> None:
    """
    Create violin plots for TTFT, latency, and throughput distributions grouped by max_tokens.
    
    Args:
        results_dir: Directory containing the CSV files (default: "results")
        output_file: Path to save the plot (if None, displays interactively)
        figsize: Figure size (width, height) in inches
    """
    # Load data
    df = load_benchmark_data(results_dir)
    
    # Ensure max_tokens is treated as categorical for proper ordering
    df['max_tokens'] = df['max_tokens'].astype(int)
    
    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Benchmark Metrics Distribution by Max Tokens', fontsize=16, fontweight='bold')
    
    # Plot 1: TTFT (Time To First Token)
    ax1 = axes[0]
    ttft_data = df[['max_tokens', 'server_ttft_sec']].dropna()
    if not ttft_data.empty:
        sns.violinplot(
            data=ttft_data,
            x='max_tokens',
            y='server_ttft_sec',
            ax=ax1,
            palette='husl',
            inner='quart',  # Show quartiles inside
            linewidth=1.5
        )
        ax1.set_xlabel('Max Tokens', fontsize=11)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Time To First Token (TTFT)', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=0)
    
    # Plot 2: Latency per token
    ax2 = axes[1]
    latency_data = df[['max_tokens', 'latency_sec_per_token']].dropna()
    if not latency_data.empty:
        sns.violinplot(
            data=latency_data,
            x='max_tokens',
            y='latency_sec_per_token',
            ax=ax2,
            palette='husl',
            inner='quart',
            linewidth=1.5
        )
        ax2.set_xlabel('Max Tokens', fontsize=11)
        ax2.set_ylabel('Time per Token (seconds)', fontsize=12)
        ax2.set_title('Latency per Token', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=0)
    
    # Plot 3: Throughput
    ax3 = axes[2]
    throughput_data = df[['max_tokens', 'throughput_tokens_per_sec']].dropna()
    if not throughput_data.empty:
        sns.violinplot(
            data=throughput_data,
            x='max_tokens',
            y='throughput_tokens_per_sec',
            ax=ax3,
            palette='husl',
            inner='quart',
            linewidth=1.5
        )
        ax3.set_xlabel('Max Tokens', fontsize=11)
        ax3.set_ylabel('Tokens per Second', fontsize=12)
        ax3.set_title('Throughput', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_combined_violin(
    results_dir: str = "results",
    output_file: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Create a single plot with all three metrics (TTFT, latency, throughput) as separate violin plots
    side by side, all on the same figure. Alternative visualization approach.
    
    Args:
        results_dir: Directory containing the CSV files (default: "results")
        output_file: Path to save the plot (if None, displays interactively)
        figsize: Figure size (width, height) in inches
    """
    # Load data
    df = load_benchmark_data(results_dir)
    df['max_tokens'] = df['max_tokens'].astype(int)
    
    # Prepare data for plotting
    plot_data = []
    
    # TTFT data
    for max_tokens in sorted(df['max_tokens'].unique()):
        values = df[df['max_tokens'] == max_tokens]['server_ttft_sec'].dropna()
        for val in values:
            plot_data.append({'max_tokens': max_tokens, 'metric': 'TTFT (s)', 'value': val})
    
    # Latency data
    for max_tokens in sorted(df['max_tokens'].unique()):
        values = df[df['max_tokens'] == max_tokens]['latency_sec_per_token'].dropna()
        for val in values:
            plot_data.append({'max_tokens': max_tokens, 'metric': 'Latency (s/token)', 'value': val})
    
    # Throughput data
    for max_tokens in sorted(df['max_tokens'].unique()):
        values = df[df['max_tokens'] == max_tokens]['throughput_tokens_per_sec'].dropna()
        for val in values:
            plot_data.append({'max_tokens': max_tokens, 'metric': 'Throughput (tok/s)', 'value': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for better violin plots
    sns.violinplot(
        data=plot_df,
        x='metric',
        y='value',
        hue='max_tokens',
        ax=ax,
        palette='husl',
        inner='quart'
    )
    
    ax.set_title('Benchmark Metrics Distribution by Max Tokens', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(title='Max Tokens', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_average_metrics(
    results_dir: str = "results",
    output_file: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot average TTFT, latency, and throughput for each max_tokens value.
    Computes averages over all samples for each max_tokens setting.
    
    Args:
        results_dir: Directory containing the CSV files (default: "results")
        output_file: Path to save the plot (if None, displays interactively)
        figsize: Figure size (width, height) in inches
    """
    # Load data
    df = load_benchmark_data(results_dir)
    
    # Ensure max_tokens is treated as integer for proper ordering
    df['max_tokens'] = df['max_tokens'].astype(int)
    
    # Group by max_tokens and compute averages
    averages = df.groupby('max_tokens').agg({
        'server_ttft_sec': 'mean',
        'latency_sec_per_token': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()
    
    # Sort by max_tokens for proper x-axis ordering
    averages = averages.sort_values('max_tokens')
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create figure with subplots (3 metrics, could use different scales)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle('Average Metrics by Max Tokens', fontsize=16, fontweight='bold')
    
    # Plot 1: TTFT
    ax1 = axes[0]
    ax1.plot(averages['max_tokens'], averages['server_ttft_sec'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='TTFT')
    ax1.set_ylabel('Time To First Token (s)', fontsize=12)
    ax1.set_title('Average TTFT', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Latency
    ax2 = axes[1]
    ax2.plot(averages['max_tokens'], averages['latency_sec_per_token'], 
             marker='s', linewidth=2, markersize=8, color='#A23B72', label='Latency')
    ax2.set_ylabel('Latency per Token (s)', fontsize=12)
    ax2.set_title('Average Latency per Token', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Throughput
    ax3 = axes[2]
    ax3.plot(averages['max_tokens'], averages['throughput_tokens_per_sec'], 
             marker='^', linewidth=2, markersize=8, color='#F18F01', label='Throughput')
    ax3.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax3.set_xlabel('Max Tokens', fontsize=12)
    ax3.set_title('Average Throughput', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()



if __name__ == "__main__":
    # Configuration - modify these as needed
    results_dir = "results"
    output_file = None  # Set to a filename to save, or None to display
    
    plot_benchmark_distributions(results_dir, output_file)
