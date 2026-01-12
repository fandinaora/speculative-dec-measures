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
    
    # Find all CSV files matching the pattern (support both old and new naming)
    # Old pattern: benchmark_max_tokens_*.csv
    # New pattern: {experiment_name}_max_tokens_*.csv or {experiment_name}_draft_steps_*.csv
    csv_files = list(results_path.glob("*_max_tokens_*.csv"))
    csv_files.extend(results_path.glob("*_draft_steps_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in {results_dir}\n"
            f"Expected files matching pattern: *_max_tokens_*.csv or *_draft_steps_*.csv"
        )
    
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
    Create violin plots for prefill time, latency, and throughput distributions grouped by max_tokens.
    
    Args:
        results_dir: Directory containing the CSV files (default: "results")
        output_file: Path to save the plot (if None, displays interactively)
        figsize: Figure size (width, height) in inches
    """
    # Load data
    df = load_benchmark_data(results_dir)
    
    # Determine grouping key (draft_steps or max_tokens)
    if 'draft_steps' in df.columns:
        grouping_key = 'draft_steps'
        x_label = 'Draft Steps (n_max)'
    else:
        grouping_key = 'max_tokens'
        x_label = 'Max Tokens'
    
    # Drop rows where grouping key is NaN (can happen when mixing old and new CSV files)
    df = df.dropna(subset=[grouping_key])
    
    # Ensure grouping key is treated as integer for proper ordering
    df[grouping_key] = df[grouping_key].astype(int)
    
    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title = f'Benchmark Metrics Distribution by {x_label}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Prefill Time (prompt processing time) - in seconds
    ax1 = axes[0]
    prefill_data = df[[grouping_key, 'prefill_time_sec']].dropna().copy()
    if not prefill_data.empty:
        sns.violinplot(
            data=prefill_data,
            x=grouping_key,
            y='prefill_time_sec',
            ax=ax1,
            palette='husl',
            inner='quart',  # Show quartiles inside
            linewidth=1.5
        )
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Prefill Time (Prompt Processing)', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=0)
        ax1.set_ylim(bottom=0)  # Prevent KDE smoothing from showing negative values
    
    # Plot 2: Latency per token
    ax2 = axes[1]
    latency_data = df[[grouping_key, 'latency_sec_per_token']].dropna()
    if not latency_data.empty:
        sns.violinplot(
            data=latency_data,
            x=grouping_key,
            y='latency_sec_per_token',
            ax=ax2,
            palette='husl',
            inner='quart',
            linewidth=1.5
        )
        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel('Time per Token (seconds)', fontsize=12)
        ax2.set_title('Latency per Token', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=0)
        ax2.set_ylim(bottom=0)  # Prevent KDE smoothing from showing negative values
    
    # Plot 3: Throughput
    ax3 = axes[2]
    throughput_data = df[[grouping_key, 'throughput_tokens_per_sec']].dropna()
    if not throughput_data.empty:
        sns.violinplot(
            data=throughput_data,
            x=grouping_key,
            y='throughput_tokens_per_sec',
            ax=ax3,
            palette='husl',
            inner='quart',
            linewidth=1.5
        )
        ax3.set_xlabel(x_label, fontsize=11)
        ax3.set_ylabel('Tokens per Second', fontsize=12)
        ax3.set_title('Throughput', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=0)
        ax3.set_ylim(bottom=0)  # Prevent KDE smoothing from showing negative values
    
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
    Create a single plot with all three metrics (prefill time, latency, throughput) as separate violin plots
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
    
    # Prefill time data - convert to milliseconds for better visibility
    for max_tokens in sorted(df['max_tokens'].unique()):
        values = df[df['max_tokens'] == max_tokens]['prefill_time_sec'].dropna() * 1000.0  # Convert to ms
        for val in values:
            plot_data.append({'max_tokens': max_tokens, 'metric': 'Prefill Time (ms)', 'value': val})
    
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
    Plot average prefill time, latency, and throughput for each max_tokens value.
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
        'prefill_time_sec': 'mean',
        'latency_sec_per_token': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()
    
    # Sort by max_tokens for proper x-axis ordering
    averages = averages.sort_values('max_tokens')
    
    # Convert prefill time to milliseconds for better visibility
    averages['prefill_time_ms'] = averages['prefill_time_sec'] * 1000.0
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create figure with subplots (3 metrics, could use different scales)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle('Average Metrics by Max Tokens', fontsize=16, fontweight='bold')
    
    # Plot 1: Prefill Time (in milliseconds)
    ax1 = axes[0]
    ax1.plot(averages['max_tokens'], averages['prefill_time_ms'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Prefill Time')
    ax1.set_ylabel('Prefill Time (ms)', fontsize=12)
    ax1.set_title('Average Prefill Time', fontsize=13, fontweight='bold')
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
