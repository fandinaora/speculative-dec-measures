import csv
from typing import Dict, Any, List
from utils.measurements import measure_server_side_metrics
from utils.data_loading import load_and_sample_dataset
from utils.prompt_formatting import format_code_completion_prompt


SERVER_URL = "http://127.0.0.1:8081"
MODEL_NAME = "qwen2.5-7b-instruct"


def run_benchmark_single(example_prompt: str, max_tokens: int, sample_id: int) -> Dict[str, Any]:
    """
    Run benchmarks for a single prompt and max_tokens value using server-side metrics.
    Returns a dictionary with all server-side measurements.
    
    Args:
        example_prompt: The prompt text from the dataset
        max_tokens: Maximum tokens to generate
        sample_id: ID of the sample for tracking
    
    Returns:
        Dictionary with server-side metrics including:
        - sample_id: Sample identifier
        - max_tokens: Maximum tokens setting
        - server_ttft_sec: Server-side time to first token (seconds)
        - tokens_generated: Number of tokens generated
        - latency_sec_per_token: Average latency per token (seconds)
        - throughput_tokens_per_sec: Throughput in tokens per second
    """
    content = format_code_completion_prompt({"example_prompt": example_prompt})
    
    # Get server-side metrics (TTFT, latency, throughput, etc.) and generated content
    metrics = measure_server_side_metrics(content, max_tokens, SERVER_URL, MODEL_NAME)
    
    # Add metadata, prompt, and generated content
    result = {
        'sample_id': sample_id,
        'max_tokens': max_tokens,
        'prompt': content,  # The full formatted prompt sent to the LLM
        **metrics
    }
    
    return result


def run_benchmark(examples: List[Dict[str, Any]], max_tokens_list: List[int] = [20, 40, 60], interleaved: bool = False) -> Dict[int, List[Dict[str, Any]]]:
    """
    Runs the complete benchmark for each max_tokens value.
    Returns a dictionary mapping max_tokens to a list of individual results.
    
    Args:
        examples: List of example dictionaries from the dataset. Each dictionary must contain a 'prompt' key.
        max_tokens_list: List of max_tokens values to test
        interleaved: If True, interleave runs (sample 1 with all max_tokens, then sample 2, etc.)
                    This reduces warmup effects. If False, run all samples for each max_tokens sequentially.
    
    Returns:
        Dictionary mapping max_tokens to list of result dictionaries
    """
    print(f"\nRunning benchmarks on {len(examples)} examples.")
    if interleaved:
        print("Using interleaved execution to reduce warmup effects.")
    
    all_results = {max_tokens: [] for max_tokens in max_tokens_list}
    
    if interleaved:
        # Interleaved: for each example, run all max_tokens values
        for i, example in enumerate(examples):
            prompt = example.get('prompt', '')
            if not prompt:
                print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                continue
            
            for max_tokens in max_tokens_list:
                try:
                    result = run_benchmark_single(prompt, max_tokens, sample_id=i)
                    all_results[max_tokens].append(result)
                except Exception as e:
                    print(f"    Error processing example {i} with max_tokens={max_tokens}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(examples)} examples...")
    else:
        # Sequential: all samples for max_tokens=20, then all for max_tokens=40, etc.
        for max_tokens in max_tokens_list:
            print(f"\n  Processing max_tokens={max_tokens}.")
            results = []
            
            for i, example in enumerate(examples):
                prompt = example.get('prompt', '')
                if not prompt:
                    print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                    continue
                
                try:
                    result = run_benchmark_single(prompt, max_tokens, sample_id=i)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Processed {i + 1}/{len(examples)} examples...")
                        
                except Exception as e:
                    print(f"    Error processing example {i}: {e}")
                    continue
            
            all_results[max_tokens] = results
            print(f"  Completed max_tokens={max_tokens}: {len(results)} successful samples")
    
    return all_results


def save_results_to_csv(all_results: Dict[int, List[Dict[str, Any]]], output_dir: str = "results") -> None:
    """
    Save benchmark results to CSV files, one file per max_tokens value.
    
    Args:
        all_results: Dictionary mapping max_tokens to list of result dictionaries
        output_dir: Directory to save CSV files (default: "results")
    """
    from pathlib import Path
    
    # Create output directory if it doesn't exist (including parent directories)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save one CSV file per max_tokens value
    for max_tokens, results in all_results.items():
        if not results:
            continue
            
        csv_filename = output_path / f"benchmark_max_tokens_{max_tokens}.csv"
        
        # Dynamically get all unique keys from results to create CSV headers
        # Collect all keys from all result dictionaries
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Put sample_id first if it exists, then sort the rest alphabetically
        fieldnames = []
        if 'sample_id' in all_keys:
            fieldnames.append('sample_id')
            all_keys.remove('sample_id')
        fieldnames.extend(sorted(all_keys))
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Write all fields that exist in the result
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"  Saved results to {csv_filename} ({len(results)} rows)")



if __name__ == "__main__":
    dataset = "openai_humaneval"
    num_samples = 100
    seed = 2026
    max_tokens_list = [20]
    
    # Load and sample random examples from HumanEval dataset (only loads sampled examples)
    #retunrs a list of dicts 
    try:
        sampled_examples = load_and_sample_dataset(
            source=dataset,
            n=num_samples,
            seed=seed
        )
    except ImportError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        exit(1)
    
    # Run benchmarks
    all_results = run_benchmark(sampled_examples, max_tokens_list=max_tokens_list)
    
    # Print results
    #print_benchmark_results(all_results)
    
    print("Saving results to CSV files...")
    save_results_to_csv(all_results, output_dir="results/warmap")
    print("Benchmarking complete!")
 


