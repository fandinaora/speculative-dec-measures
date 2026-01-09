"""
Entry point for running evaluation experiments using the OOP structure.
"""
from experiments.evaluation import EvaluationExperiment


if __name__ == "__main__":
    # Configuration
    config = {
        'server_url': "http://127.0.0.1:8081",
        'model_name': "qwen2.5-7b-instruct",
        'output_dir': "results/evaluation",
        'use_official_humaneval': True,  # Set to True to use official human-eval package (requires all 164 samples)
        'num_samples_per_problem': 1,  # Number of completions per problem. For pass@k, set to at least k (e.g., 3 for pass@3)
        'temperature': 0.2  # Temperature for additional pass@k samples only. First sample always uses temp=0 for deterministic metrics
    }
    
    # Create experiment instance
    experiment = EvaluationExperiment(**config)
    
    # Execute experiment
    results = experiment.execute(
        load_kwargs={
            'dataset': "openai_humaneval",
            'num_samples': 164,  # Set to None to load full dataset (required if use_official_humaneval=True)
            'seed': 2026
        },
        run_kwargs={
            'max_tokens': 512  # Use higher max_tokens for quality evaluation
        },
        save=True,
        plot=False  # Set to True when plotting is implemented
    )
    
    print("Evaluation complete!")
    print(f"Evaluated {results.get('num_samples', 0)} samples")
