"""
Entry point for running performance benchmarks.
"""
from experiments.performance import PerformanceExperiment


if __name__ == "__main__":
    # Configuration
    config = {
        'server_url': "http://127.0.0.1:8081",
        'model_name': "qwen2.5-7b-instruct",
        'output_dir': "results",
        'experiment_name': "performance"
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
            'max_tokens_list': [20, 40, 60],
            'randomize_order': True,  # Set to False to disable randomization
            #'random_seed': 42 
        },
        save=True,
        plot=True  # Set to True to generate plots
    )
    
    # Print summary
    #experiment.print_summary()
    
    print("Benchmarking complete!")
    
