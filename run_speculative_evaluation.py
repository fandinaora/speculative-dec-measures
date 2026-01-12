"""
Entry point for running evaluation experiments with speculative decoding.

This script runs evaluation with a single draft step size (n_max) and computes
all evaluation metrics (exact match, BLEU, unit tests).
"""
from typing import Dict, Any
import json
from pathlib import Path
import pandas as pd
from experiments.evaluation import EvaluationExperiment


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze evaluation results and compute summary statistics."""
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
    # Configuration
    # Set the draft step size (n_max) here
    DRAFT_STEP_SIZE = 50  # Change this to test different draft step sizes
    
    config = {
        'server_url': "http://127.0.0.1:8081",
        'model_name': "qwen2.5-7b-instruct",
        'output_dir': "results",
        'experiment_name': f"evaluation_speculative_n{DRAFT_STEP_SIZE}",  # Include n_max in experiment name
        'speculative_dec_params': {
            "speculative.n_max": DRAFT_STEP_SIZE,  # Single draft step size
            "speculative.n_min": 2,
            "speculative.p_min": 0.5
        }
    }
    
    print(f"Running Evaluation Experiment with Speculative Decoding")
    print(f"Draft Step Size (n_max): {DRAFT_STEP_SIZE}")
    
    # Create experiment instance
    experiment = EvaluationExperiment(**config)
    
    # Execute experiment
    results = experiment.execute(
        load_kwargs={
            'dataset': "openai_humaneval",
            'num_samples': 100,  # Adjust as needed
            'seed': 2026
        },
        run_kwargs={
            'max_tokens': 512
        },
        # metrics=None means compute all available metrics (exact_match, bleu, unit_tests)
        save=True,
        plot=False  # Evaluation plotting not yet implemented
    )
    
    print("Evaluation complete!")
    
    # Analyze results
    analysis = analyze_results(results)
    
    print(f"\nTotal samples: {analysis['total_samples']}")
    
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
    
    # Save analysis results to JSON file
    output_dir = Path(config['output_dir'])
    experiment_name = config['experiment_name']
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare summary with all metrics
    summary = {
        'experiment_name': experiment_name,
        'model_name': config['model_name'],
        'draft_step_size': DRAFT_STEP_SIZE,
        'speculative_dec_params': config['speculative_dec_params'],
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
    print(f"Full results saved to: {experiment_dir}")
