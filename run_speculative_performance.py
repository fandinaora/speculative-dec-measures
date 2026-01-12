"""
Entry point for running performance benchmarks.
"""
from experiments.performance import PerformanceExperiment
from utils.analyze_speculative_results import (
    load_speculative_results, 
    compute_averages, 
    plot_metrics,
    compute_total_evaluation_time,
    plot_evaluation_time
)
from pathlib import Path


if __name__ == "__main__":
    # Configuration
    config = {
        'server_url': "http://127.0.0.1:8081",
        'model_name': "qwen2.5-7b-instruct",
        'output_dir': "results",
        'experiment_name': "speculative_performance",
        'speculative_dec_params': {
            # Base speculative decoding params (n_max will be overridden by draft_steps_list)
            "speculative.n_min": 2,
            "speculative.p_min": 0.5
        }
    }
    
    # Create experiment instance
    experiment = PerformanceExperiment(**config)
    
    # Execute experiment: iterates over draft steps with fixed max_tokens
    results = experiment.execute(
        load_kwargs={
            'dataset': "openai_humaneval",
            'num_samples': 100,
            'seed': 2026
        },
        run_kwargs={
            'draft_steps_list': [5, 10, 25, 50, 100, 200, 512],  # Iterate over draft steps (speculative.n_max)
            'fixed_max_tokens': 512,  # Fixed max_tokens for generation
            'randomize_order': False,  # Set to False to disable randomization
            #'random_seed': 42 
        },
        save=True,
        plot=True  # Set to True to generate plots
    )
    
    # Print summary
    #experiment.print_summary()
    
    print("Benchmarking complete!")
    
    # Generate bar plot with average metrics by draft step size
  
    try:
        # Load results from the experiment directory
        results_dir = Path(config['output_dir']) / config['experiment_name']
        
        # Load and compute averages
        df = load_speculative_results(str(results_dir))
        averages = compute_averages(df)
        
        # Generate bar plot
        output_file = results_dir / "speculative_metrics_by_draft_steps.png"
        plot_metrics(averages, output_file=str(output_file))
        
        print(f"\nAverage metrics bar plot saved to: {output_file}")
        
        # Also generate evaluation time plot
        print("\nGenerating evaluation time plot...")
        total_times = compute_total_evaluation_time(df)
        eval_time_output = results_dir / "evaluation_time_by_draft_steps.png"
        plot_evaluation_time(total_times, output_file=str(eval_time_output))
        print(f"Evaluation time plot saved to: {eval_time_output}")
        
    except Exception as e:
        print(f"\nWarning: Could not generate average metrics bar plot: {e}")
        print("(This is optional - the main violin plots were already generated)")
    