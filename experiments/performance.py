"""
Performance benchmark experiment for measuring LLM latency, throughput, and TTFT.
"""
import csv
import statistics
import random
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import Experiment
from utils.data_loading import load_and_sample_dataset
from utils.measurements import measure_server_side_metrics
from utils.prompt_formatting import format_code_completion_prompt
from utils.plotting import plot_benchmark_distributions


class PerformanceExperiment(Experiment):
    """
    Performance benchmark experiment that measures:
    - Prefill time (prompt processing time)
    - Latency per token
    - Throughput (tokens per second)
    - Generated content
    """
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8081",
        model_name: str = "qwen2.5-7b-instruct",
        output_dir: str = "results",
        experiment_name: str = None,
        speculative_dec_params: Dict[str, Any] = None
    ):
        """
        Initialize performance experiment.
        
        Args:
            server_url: URL of the llama.cpp server
            model_name: Name of the model to use
            output_dir: Directory to save results
            experiment_name: Name for this experiment (used in output filenames)
            speculative_dec_params: Optional dictionary of speculative decoding parameters to pass to chat_payload.
                                   Examples: {"speculative.n_max": 16, "speculative.n_min": 3, "speculative.p_min": 0.75}
        """
        super().__init__(server_url, model_name, output_dir, experiment_name)
        self.speculative_dec_params = speculative_dec_params or {}
    
    def load_data(
        self,
        dataset: str = "openai_humaneval",
        num_samples: int = 100,
        seed: int = 2026,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load and sample dataset for benchmarking.
        
        Args:
            dataset: Hugging Face dataset identifier
            num_samples: Number of samples to load
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to load_and_sample_dataset
            
        Returns:
            List of example dictionaries from the dataset
        """
        return load_and_sample_dataset(
            source=dataset,
            n=num_samples,
            seed=seed,
            **kwargs
        )
    
    def run(
        self,
        examples: List[Dict[str, Any]],
        max_tokens_list: List[int] = None,
        temperature: float = 0.0,
        randomize_order: bool = True,
        random_seed: Optional[int] = None,
        draft_steps_list: List[int] = None,
        fixed_max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run performance benchmarks for each max_tokens value or draft_steps value.
        
        Args:
            examples: List of example dictionaries from the dataset
            max_tokens_list: List of max_tokens values to test (default: [20, 40, 60])
            temperature: Sampling temperature (0.0 = deterministic, >0.0 = more diverse). Default: 0.0
            randomize_order: If True, randomize the order for each example to avoid caching effects.
                           Default: True
            random_seed: Random seed for reproducible randomization. If None, uses system random state.
                        Default: None
            draft_steps_list: If provided, iterate over draft steps (speculative.n_max) instead of max_tokens.
                            Each step will use fixed_max_tokens for generation.
            fixed_max_tokens: Fixed max_tokens value to use when draft_steps_list is provided (required if draft_steps_list is provided)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with 'results' key mapping max_tokens or draft_steps to list of result dictionaries
        """
        # Determine iteration mode
        if draft_steps_list is not None:
            # Iterate over draft steps
            if fixed_max_tokens is None:
                raise ValueError("fixed_max_tokens must be provided when draft_steps_list is provided")
            if not isinstance(fixed_max_tokens, int) or fixed_max_tokens <= 0:
                raise ValueError("fixed_max_tokens must be a positive integer")
            if not draft_steps_list:
                raise ValueError("draft_steps_list cannot be empty")
            if not all(isinstance(ds, int) and ds > 0 for ds in draft_steps_list):
                raise ValueError("draft_steps_list must contain positive integers")
            
            iteration_list = draft_steps_list
            iteration_key = 'draft_steps'
            max_tokens_value = fixed_max_tokens
        else:
            # Default: iterate over max_tokens
            if max_tokens_list is None:
                max_tokens_list = [20, 40, 60]
            if not max_tokens_list:
                raise ValueError("max_tokens_list cannot be empty")
            if not all(isinstance(mt, int) and mt > 0 for mt in max_tokens_list):
                raise ValueError("max_tokens_list must contain positive integers")
            
            iteration_list = max_tokens_list
            iteration_key = 'max_tokens'
            max_tokens_value = None
        
        # Validate inputs
        if not examples:
            raise ValueError("examples list cannot be empty")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")
        
        print(f"\nRunning benchmarks on {len(examples)} examples.")
        if iteration_key == 'draft_steps':
            print(f"  Iterating over draft steps: {iteration_list}, max_tokens fixed at {fixed_max_tokens}")
        else:
            print(f"  Iterating over max_tokens: {iteration_list}")
        
        if randomize_order:
            if random_seed is not None:
                print(f"  Randomizing order for each sample (seed={random_seed}) to avoid caching effects.")
                random.seed(random_seed)
            else:
                print(f"  Randomizing order for each sample to avoid caching effects.")
        
        # Initialize results dictionary
        all_results = {key: [] for key in iteration_list}
        
        # Process each example (sample) first, then all iteration values for that sample
        for i, example in enumerate(examples):
            prompt = example.get('prompt', '')
            if not prompt:
                print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                continue
            
            # Create randomized order for this example
            iteration_order = iteration_list.copy()
            if randomize_order:
                if random_seed is not None:
                    random.seed(random_seed + i)
                random.shuffle(iteration_order)
            
            print(f"\n  Processing example {i+1}/{len(examples)} ({iteration_key} order: {iteration_order})")
            
            # Run all iteration values for this example in randomized order
            for iteration_value in iteration_order:
                try:
                    if iteration_key == 'draft_steps':
                        # Create speculative_dec_params for this draft step
                        spec_params = self.speculative_dec_params.copy()
                        spec_params["speculative.n_max"] = iteration_value
                        result = self._run_single(
                            prompt,
                            fixed_max_tokens,
                            sample_id=i,
                            temperature=temperature,
                            speculative_dec_params=spec_params
                        )
                        result['draft_steps'] = iteration_value
                    else:
                        result = self._run_single(
                            prompt,
                            iteration_value,
                            sample_id=i,
                            temperature=temperature
                        )
                    # Append result to the correct list
                    all_results[iteration_value].append(result)
                        
                except Exception as e:
                    print(f"    Error processing example {i} with {iteration_key}={iteration_value}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(examples)} examples...")
        
        # Print summary
        for key in sorted(iteration_list):
            results = all_results[key]
            print(f"  Completed {iteration_key}={key}: {len(results)} successful samples")
        
        # Return in format compatible with base class expectations
        return {'results': all_results, 'iteration_key': iteration_key}
    
    def _run_single(
        self,
        example_prompt: str,
        max_tokens: int,
        sample_id: int,
        temperature: float = 0.0,
        seed: int = 42,
        speculative_dec_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark for a single prompt and max_tokens value.
        
        Args:
            example_prompt: The prompt text from the dataset
            max_tokens: Maximum tokens to generate
            sample_id: ID of the sample for tracking
            temperature: Sampling temperature (0.0 = deterministic, >0.0 = more diverse). Default: 0.0
            seed: Random seed for reproducibility. Default: 42
            speculative_dec_params: Optional override for speculative decoding parameters.
                                   If None, uses self.speculative_dec_params
            
        Returns:
            Dictionary with all server-side measurements
            
        Raises:
            ValueError: If prompt formatting fails
            requests.RequestException: If API call fails
        """
        try:
            content = format_code_completion_prompt({"example_prompt": example_prompt})
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Failed to format prompt: {e}") from e
        
        # Use provided params or fall back to instance params
        spec_params = speculative_dec_params if speculative_dec_params is not None else self.speculative_dec_params
        
        # Get server-side metrics (TTFT, latency, throughput, etc.) and generated content
        try:
            metrics = measure_server_side_metrics(
                content,
                max_tokens,
                self.server_url,
                self.model_name,
                temperature=temperature,
                seed=seed,
                **spec_params
            )
        except Exception as e:
            raise RuntimeError(f"Failed to measure server-side metrics: {e}") from e
        
        # Add metadata, prompt, and generated content
        result = {
            'sample_id': sample_id,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'prompt': content,  # The full formatted prompt sent to the LLM
            **metrics
        }
        
        return result
    
    def _save_to_csv(
        self,
        results: Dict[str, Any],
        subdirectory: str = None,
        **kwargs
    ) -> None:
        """
        Save benchmark results to CSV files, one file per max_tokens value.
        
        Args:
            results: Results dictionary with 'results' key mapping max_tokens to list of result dictionaries
            subdirectory: Optional subdirectory within output_dir. If None, uses experiment_name
            **kwargs: Additional arguments
        """
        # Extract the actual results dictionary
        all_results = results.get('results', results)
        if not isinstance(all_results, dict):
            raise ValueError("results must be a dictionary with 'results' key or a dict mapping keys to lists")
        
        # Determine iteration key (max_tokens or draft_steps)
        iteration_key = results.get('iteration_key', 'max_tokens')
        
        # Determine output path (consistent with ResultSaver pattern)
        output_path = self.output_dir
        if subdirectory:
            output_path = output_path / subdirectory
        else:
            output_path = output_path / self.experiment_name
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save one CSV file per iteration value
        for key_value, result_list in all_results.items():
            if not result_list:
                print(f"  Warning: No results for {iteration_key}={key_value}, skipping CSV file")
                continue
            
            # Use experiment_name in filename for consistency
            csv_filename = output_path / f"{self.experiment_name}_{iteration_key}_{key_value}.csv"
            
            # Dynamically get all unique keys from results to create CSV headers
            all_keys = set()
            for result in result_list:
                all_keys.update(result.keys())
            
            # Put important fields first, then sort the rest alphabetically
            fieldnames = []
            priority_fields = ['sample_id', iteration_key, 'max_tokens', 'temperature']
            # Remove duplicates while preserving order
            priority_fields = [f for f in priority_fields if f in all_keys]
            for field in priority_fields:
                if field in all_keys:
                    fieldnames.append(field)
                    all_keys.remove(field)
            fieldnames.extend(sorted(all_keys))
            
            try:
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in result_list:
                        row = {field: result.get(field, '') for field in fieldnames}
                        writer.writerow(row)
                
                print(f"  Saved results to {csv_filename} ({len(result_list)} rows)")
            except IOError as e:
                print(f"  Error saving CSV file {csv_filename}: {e}")
                raise
    
    def _plot_results(
        self,
        results: Dict[str, Any],
        output_file: str = None,
        **kwargs
    ) -> None:
        """
        Plot benchmark results using the plotting utilities.
        
        Args:
            results: Results dictionary with 'results' key mapping max_tokens to list of result dictionaries
            output_file: Optional file path to save the plot. If None, automatically generates a filename.
            **kwargs: Additional plotting arguments (can include 'subdirectory')
        """
        # For plotting, we need to use the CSV files, so we'll use the plotting utility
        # which reads from the results directory
        subdirectory = kwargs.get('subdirectory', None)
        if subdirectory:
            results_dir = self.output_dir / subdirectory
        else:
            results_dir = self.output_dir / self.experiment_name
        
        # Generate default output file path if not provided
        if output_file is None:
            output_file = str(results_dir / f"{self.experiment_name}_benchmark_plot.png")
        
        # Ensure the directory exists
        results_dir.mkdir(exist_ok=True, parents=True)
        
        plot_benchmark_distributions(
            results_dir=str(results_dir),
            output_file=output_file,
            **{k: v for k, v in kwargs.items() if k != 'subdirectory'}
        )
    
    def print_summary(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a comprehensive summary of benchmark results with statistics.
        
        Args:
            results: Results dictionary (if None, uses self.results)
        """
        if results is None:
            results = self.results
        
        if results is None:
            print("No results to summarize.")
            return
        
        # Extract the actual results dictionary
        all_results = results.get('results', results)
        if not isinstance(all_results, dict):
            print("Invalid results format.")
            return
        
        if not all_results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print(f"Benchmark Results (model: {self.model_name}, server: {self.server_url})")
        print("="*80)
        
        print("\nAggregated Metrics (averages across all examples):")
        print("max_tokens | #examples | avg_generated | avg_prefill(s) | avg_latency(s/tok) | avg_throughput(tok/s)")
        print("-" * 100)
        
        for max_tokens in sorted(all_results.keys()):
            result_list = all_results[max_tokens]
            if not result_list:
                continue
            
            # Safely extract metrics with defaults
            num_examples = len(result_list)
            tokens_generated = [r.get('tokens_generated', 0) for r in result_list]
            prefill_values = [r.get('prefill_time_sec', 0.0) for r in result_list]
            latency_values = [r.get('latency_sec_per_token', 0.0) for r in result_list]
            throughput_values = [r.get('throughput_tokens_per_sec', 0.0) for r in result_list]
            
            # Calculate averages
            avg_generated = statistics.mean(tokens_generated) if tokens_generated else 0.0
            avg_prefill = statistics.mean(prefill_values) if prefill_values else 0.0
            avg_latency = statistics.mean(latency_values) if latency_values else 0.0
            avg_throughput = statistics.mean(throughput_values) if throughput_values else 0.0
            
            print(
                f"{max_tokens:10d} | {num_examples:9d} | "
                f"{avg_generated:13.1f} | {avg_prefill:13.3f} | "
                f"{avg_latency:17.3f} | {avg_throughput:21.3f}"
            )
        
        print("\nDetailed Statistics:")
        print("-" * 100)
        for max_tokens in sorted(all_results.keys()):
            result_list = all_results[max_tokens]
            if not result_list:
                continue
            
            # Safely extract metrics
            prefill_values = [r.get('prefill_time_sec', 0.0) for r in result_list if 'prefill_time_sec' in r]
            latency_values = [r.get('latency_sec_per_token', 0.0) for r in result_list if 'latency_sec_per_token' in r]
            throughput_values = [r.get('throughput_tokens_per_sec', 0.0) for r in result_list if 'throughput_tokens_per_sec' in r]
            
            if not prefill_values and not latency_values:
                print(f"\nmax_tokens={max_tokens}: No valid metrics found")
                continue
            
            print(f"\nmax_tokens={max_tokens}:")
            
            # Show prefill time (should be constant for same prompt)
            if prefill_values:
                print(f"  Prefill Time (s) - should be constant for same prompt:")
                print(f"    min={min(prefill_values):.3f}, max={max(prefill_values):.3f}, "
                      f"avg={statistics.mean(prefill_values):.3f}, "
                      f"median={statistics.median(prefill_values):.3f}")
                if len(prefill_values) > 1:
                    std_dev = statistics.stdev(prefill_values)
                    print(f"    std={std_dev:.3f}")
                    # Calculate coefficient of variation to assess consistency
                    mean_val = statistics.mean(prefill_values)
                    cv = std_dev / mean_val if mean_val > 0 else 0
                    print(f"    CV={cv:.1%} (coefficient of variation - lower is better)")
                    
                    # Check for cache usage if available
                    cached_tokens = [r.get('cached_tokens') for r in result_list if 'cached_tokens' in r and r.get('cached_tokens') is not None]
                    if cached_tokens:
                        cache_used_count = len([c for c in cached_tokens if c is not None and c > 0])
                        print(f"    Cache info: {cache_used_count}/{len(cached_tokens)} requests used cache (cached_tokens > 0)")
                        if cache_used_count > 0:
                            print(f"      Note: Cache usage may explain prefill time variation")
            
            if latency_values:
                print(f"  Latency (s/tok):")
                print(f"    min={min(latency_values):.3f}, max={max(latency_values):.3f}, "
                      f"avg={statistics.mean(latency_values):.3f}, "
                      f"median={statistics.median(latency_values):.3f}")
                if len(latency_values) > 1:
                    print(f"    std={statistics.stdev(latency_values):.3f}")
            
            if throughput_values:
                print(f"  Throughput (tok/s):")
                print(f"    min={min(throughput_values):.1f}, max={max(throughput_values):.1f}, "
                      f"avg={statistics.mean(throughput_values):.1f}, "
                      f"median={statistics.median(throughput_values):.1f}")
                if len(throughput_values) > 1:
                    print(f"    std={statistics.stdev(throughput_values):.1f}")
