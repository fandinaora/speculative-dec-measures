"""
Entry point for running performance benchmarks using the OOP structure.
This is an alternative to bench_code_completion.py that uses the Experiment class.
"""
from experiments.performance import PerformanceExperiment


if __name__ == "__main__":
    # Configuration
    config = {
        'server_url': "http://127.0.0.1:8081",
        'model_name': "qwen2.5-7b-instruct",
        'output_dir': "results/warmap"
    }
    
    # Create experiment instance
    experiment = PerformanceExperiment(**config)
    
    # Execute experiment: exacutes, and saves the results
    results = experiment.execute(
        load_kwargs={
            'dataset': "openai_humaneval",
            'num_samples': 100,
            'seed': 2026
        },
        run_kwargs={
            'max_tokens_list': [40, 60, 80]
        },
        save=True,
        plot=False  # Set to True to generate plots
    )
    
    # Print summary
    #experiment.print_summary()
    
    print("\n" + "="*80)
    print("Benchmarking complete!")
    print("="*80)
    
