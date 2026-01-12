"""
Entry point for running performance benchmarks.
"""
import argparse
from experiments.performance import PerformanceExperiment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for code completion models.",
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
        default="performance",
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
    
    # Benchmark configuration
    parser.add_argument(
        '--max_tokens', 
        type=str, 
        default="20,40,60",
        help="Comma-separated list of max_tokens values to test (e.g., '20,40,60')"
    )
    parser.add_argument(
        '--randomize_order', 
        action='store_true',
        default=True,
        help="Randomize test order to reduce bias"
    )
    parser.add_argument(
        '--no_randomize_order', 
        action='store_false',
        dest='randomize_order',
        help="Disable randomization of test order"
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
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Parse max_tokens list from comma-separated string
    max_tokens_list = [int(x.strip()) for x in args.max_tokens.split(',')]
    
    # Configuration
    config = {
        'server_url': args.server_url,
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name
    }
    
    # Create experiment instance
    experiment = PerformanceExperiment(**config)
    
    # Execute experiment
    results = experiment.execute(
        load_kwargs={
            'dataset': args.dataset,
            'num_samples': args.num_samples,
            'seed': args.seed
        },
        run_kwargs={
            'max_tokens_list': max_tokens_list,
            'randomize_order': args.randomize_order,
        },
        save=args.save,
        plot=args.plot
    )
    
    print("Benchmarking complete!")
    
