"""
Script to plot benchmark results from CSV files.
"""
from utils.plotting import (
    plot_benchmark_distributions,
    plot_average_metrics,
    
)


if __name__ == "__main__":
    results_dir = "results"  # Directory containing CSV files
    output_file = None  # Set to a filename to save, or None to display
    
    # Choose which plot to generate:
    # - "distributions": Violin plots showing distributions (default)
    # - "averages": Three subplots with average metrics
   
    plot_type = "distributions"  # Change this to switch plot types
    
    try:
        if plot_type == "distributions":
            plot_benchmark_distributions(results_dir, output_file)
        elif plot_type == "averages":
            plot_average_metrics(results_dir, output_file)
        else:
            print(f"Unknown plot type: {plot_type}")
            print("Available types: 'distributions', 'averages', 'averages_combined'")
            exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the benchmark first to generate CSV files.")
        exit(1)
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
