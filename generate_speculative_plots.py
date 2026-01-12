"""
Standalone script to generate evaluation time plots for speculative performance results.
Run this after speculative_performance experiment completes.
"""
from pathlib import Path
from utils.analyze_speculative_results import (
    load_speculative_results, 
    compute_averages, 
    plot_metrics,
    compute_total_evaluation_time,
    plot_evaluation_time
)


if __name__ == "__main__":
    # Configuration - change this if your results are in a different directory
    results_dir = Path("results") / "speculative_performance"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please check that you have run the speculative_performance experiment first.")
        exit(1)
    
    print(f"Loading results from: {results_dir}")
    
    try:
        # Load results
        df = load_speculative_results(str(results_dir))
        print(f"Loaded {len(df)} results")
        
        # Generate average metrics bar plot
        print("\nGenerating average metrics bar plot...")
        averages = compute_averages(df)
        output_file = results_dir / "speculative_metrics_by_draft_steps.png"
        plot_metrics(averages, output_file=str(output_file))
        print(f"✓ Average metrics plot saved to: {output_file}")
        
        # Generate evaluation time plot
        print("\nGenerating evaluation time plot...")
        total_times = compute_total_evaluation_time(df)
        eval_time_output = results_dir / "evaluation_time_by_draft_steps.png"
        plot_evaluation_time(total_times, output_file=str(eval_time_output))
        print(f"✓ Evaluation time plot saved to: {eval_time_output}")
        
        print("\n✅ All plots generated successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find results file.")
        print(f"Details: {e}")
        print(f"\nMake sure the speculative_performance experiment has completed and saved results.")
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
