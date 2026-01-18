"""
Entry point for running performance benchmarks with speculative decoding.
"""
import argparse
from experiments.performance import PerformanceExperiment
from utils.analyze_speculative_results import (
    load_speculative_results, 
    compute_averages, 
    plot_metrics,
    compute_total_evaluation_time,
    plot_evaluation_time
)
from pathlib import Path
from utils.logging_config import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks with speculative decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        '--server_url', 
        type=str, 
        default="http://127.0.0.1:8081",
        help="URL of the llama.cpp server"
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="qwen2.5-7b-instruct",
        help="Name of the model being tested"
    )
    
    # Output configuration
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="results",
        help="Base directory for saving results"
    )
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default="speculative_performance",
        help="Name of the experiment (used in output filenames)"
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="openai_humaneval",
        help="Dataset to use for benchmarking"
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=100,
        help="Number of samples to benchmark"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=2026,
        help="Random seed for reproducibility"
    )
    
    # Speculative decoding configuration
    parser.add_argument(
        '--draft_steps', 
        type=str, 
        default="5,10,25,50,100,200,512",
        help="Comma-separated list of draft step sizes to test (e.g., '5,10,25,50')"
    )
    parser.add_argument(
        '--fixed_max_tokens', 
        type=int, 
        default=512,
        help="Fixed max_tokens for generation (used for all draft step sizes)"
    )
    parser.add_argument(
        '--spec_n_min', 
        type=int, 
        default=2,
        help="Minimum draft tokens (speculative.n_min)"
    )
    parser.add_argument(
        '--spec_p_min', 
        type=float, 
        default=0.5,
        help="Minimum probability threshold (speculative.p_min)"
    )
    parser.add_argument(
        '--randomize_order', 
        action='store_true',
        default=False,
        help="Randomize test order to reduce bias"
    )
    parser.add_argument(
        '--no_randomize_order', 
        action='store_false',
        dest='randomize_order',
        help="Disable randomization of test order (default)"
    )
    
    # Output options
    parser.add_argument(
        '--no_save', 
        action='store_false',
        dest='save',
        default=True,
        help="Disable saving results to files"
    )
    parser.add_argument(
        '--no_plot', 
        action='store_false',
        dest='plot',
        default=True,
        help="Disable generating plots"
    )
    
    # Logging options
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help="Path to log file (default: {output_dir}/{experiment_name}/{experiment_name}.log)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Parse draft_steps list from comma-separated string
    draft_steps_list = [int(x.strip()) for x in args.draft_steps.split(',')]
    
    # Configuration
    config = {
        'server_url': args.server_url,
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name,
        'speculative_dec_params': {
            "speculative.n_min": args.spec_n_min,
            "speculative.p_min": args.spec_p_min
        }
    }
    
    # Setup logging
    setup_logging(
        log_file=args.log_file,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir
    )
    
    # Create experiment instance
    experiment = PerformanceExperiment(**config)
    
    # Execute experiment: iterates over draft steps with fixed max_tokens
    results = experiment.execute(
        load_kwargs={
            'dataset': args.dataset,
            'num_samples': args.num_samples,
            'seed': args.seed
        },
        run_kwargs={
            'draft_steps_list': draft_steps_list,
            'fixed_max_tokens': args.fixed_max_tokens,
            'randomize_order': args.randomize_order,
        },
        save=args.save,
        plot=args.plot
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
    