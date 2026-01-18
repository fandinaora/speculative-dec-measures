"""
Entry point for running evaluation experiments using the OOP structure.
"""
from typing import Dict, Any
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.evaluation import EvaluationExperiment
from utils.logging_config import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run code quality evaluation experiments.",
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
        default="evaluation",
        help="Name of the experiment (used in output filenames)"
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="openai_humaneval",
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=100,
        help="Number of samples to evaluate (max 164 for HumanEval)"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=2026,
        help="Random seed for reproducibility"
    )
    
    # Generation configuration
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=512,
        help="Maximum tokens to generate for each completion"
    )
    parser.add_argument(
        '--api_seed',
        type=int,
        default=None,
        help="Seed for API generation calls (default: None, no seed)"
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
        '--plot', 
        action='store_true',
        default=False,
        help="Enable generating plots (not yet implemented)"
    )
    
    # Logging options
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help="Path to log file (default: {output_dir}/{experiment_name}/{experiment_name}.log)"
    )
    
    return parser.parse_args()


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:

    result_list = results.get('results', [])
    total_samples = len(result_list)
    
    if total_samples == 0:
        return {
            'total_samples': 0,
            'avg_bleu_score': None,
            'exact_match_rate': None,
            'unit_test_pass_rate': None,
            'exact_matches': 0,
            'unit_test_passed': 0
        }
    
    df = pd.DataFrame(result_list)
    
    # Calculate averages/rates using pandas
    avg_bleu_score = df['bleu_score'].mean() if 'bleu_score' in df.columns else None
    exact_matches = df['exact_match'].sum() if 'exact_match' in df.columns else 0
    exact_match_rate = (exact_matches / total_samples) if total_samples > 0 else 0.0
    
    unit_test_passed = df['unit_test_passed'].sum() if 'unit_test_passed' in df.columns else 0
    unit_test_pass_rate = (unit_test_passed / total_samples) if total_samples > 0 else 0.0
    
    return {
        'total_samples': total_samples,
        'avg_bleu_score': float(avg_bleu_score) if avg_bleu_score is not None and not pd.isna(avg_bleu_score) else None,
        'exact_match_rate': float(exact_match_rate),
        'unit_test_pass_rate': float(unit_test_pass_rate),
        'exact_matches': int(exact_matches),
        'unit_test_passed': int(unit_test_passed)
    }


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Configuration
    config = {
        'server_url': args.server_url,
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name,
    }
    
    # Setup logging
    setup_logging(
        log_file=args.log_file,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir
    )
    
    # Create experiment instance
    experiment = EvaluationExperiment(**config)
    
    # Execute experiment
    results = experiment.execute(
        load_kwargs={
            'dataset': args.dataset,
            'num_samples': args.num_samples, 
            'seed': args.seed
        },
        run_kwargs={
            'max_tokens': args.max_tokens,
            'api_seed': args.api_seed
        },
        save=args.save,
        plot=args.plot  
    )
    
    print("Evaluation complete!")
    
    # Analyze results
    analysis = analyze_results(results)
    
    if analysis['avg_bleu_score'] is not None:
        print(f"Average BLEU score: {analysis['avg_bleu_score']:.4f}")
    else:
        print("Average BLEU score: N/A (no BLEU scores available)")
    
    if analysis['exact_match_rate'] is not None:
        print(f"Exact match rate: {analysis['exact_matches']}/{analysis['total_samples']} = {analysis['exact_match_rate']:.2%}")
    else:
        print("Exact match rate: N/A")
    
    if analysis['unit_test_pass_rate'] is not None:
        print(f"Unit test pass rate: {analysis['unit_test_passed']}/{analysis['total_samples']} = {analysis['unit_test_pass_rate']:.2%}")
    else:
        print("Unit test pass rate: N/A")
    
    # Save analysis results to JSON file with all average/rate metrics
    output_dir = Path(config['output_dir'])
    experiment_name = config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for this experiment
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare summary with all averages/rates
    summary = {
        'experiment_name': experiment_name,
        'model_name': config['model_name'],
        'total_samples': analysis['total_samples'],
        'metrics': {
            'avg_bleu_score': analysis['avg_bleu_score'],
            'exact_match_rate': analysis['exact_match_rate'],
            'unit_test_pass_rate': analysis['unit_test_pass_rate'],
        },
        'counts': {
            'exact_matches': analysis['exact_matches'],
            'unit_test_passed': analysis['unit_test_passed']
        }
    }
    
    # Remove None values from metrics for cleaner JSON
    summary['metrics'] = {k: v for k, v in summary['metrics'].items() if v is not None}
    
    # Save to JSON file
    summary_file = experiment_dir / f"{experiment_name}_analysis.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis results saved to: {summary_file}")
    