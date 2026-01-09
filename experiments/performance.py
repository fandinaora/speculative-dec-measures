"""
Performance benchmark experiment for measuring LLM latency, throughput, and TTFT.
"""
import csv
from typing import Dict, Any, List
from pathlib import Path

from .base import Experiment
from utils.data_loading import load_and_sample_dataset
from utils.measurements import measure_server_side_metrics
from utils.prompt_formatting import format_code_completion_prompt
from utils.plotting import plot_benchmark_distributions


class PerformanceExperiment(Experiment):
    """
    Performance benchmark experiment that measures:
    - Time To First Token (TTFT)
    - Latency per token
    - Throughput (tokens per second)
    - Generated content
    """
    
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
        max_tokens_list: List[int] = [20, 40, 60],
        **kwargs
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Run performance benchmarks for each max_tokens value.
        
        Args:
            examples: List of example dictionaries from the dataset
            max_tokens_list: List of max_tokens values to test
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping max_tokens to list of result dictionaries
        """
        print(f"\nRunning benchmarks on {len(examples)} examples.")
        
        all_results = {}
        
        for max_tokens in max_tokens_list:
            print(f"\n  Processing max_tokens={max_tokens}.")
            results = []
            
            for i, example in enumerate(examples):
                prompt = example.get('prompt', '')
                if not prompt:
                    print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                    continue
                
                try:
                    result = self._run_single(prompt, max_tokens, sample_id=i)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Processed {i + 1}/{len(examples)} examples...")
                        
                except Exception as e:
                    print(f"    Error processing example {i}: {e}")
                    continue
            
            all_results[max_tokens] = results
            print(f"  Completed max_tokens={max_tokens}: {len(results)} successful samples")
        
        return all_results
    
    def _run_single(
        self,
        example_prompt: str,
        max_tokens: int,
        sample_id: int
    ) -> Dict[str, Any]:
        """
        Run benchmark for a single prompt and max_tokens value.
        
        Args:
            example_prompt: The prompt text from the dataset
            max_tokens: Maximum tokens to generate
            sample_id: ID of the sample for tracking
            
        Returns:
            Dictionary with all server-side measurements
        """
        content = format_code_completion_prompt({"example_prompt": example_prompt})
        
        # Get server-side metrics (TTFT, latency, throughput, etc.) and generated content
        metrics = measure_server_side_metrics(
            content,
            max_tokens,
            self.server_url,
            self.model_name
        )
        
        # Add metadata, prompt, and generated content
        result = {
            'sample_id': sample_id,
            'max_tokens': max_tokens,
            'prompt': content,  # The full formatted prompt sent to the LLM
            **metrics
        }
        
        return result
    
    def _save_to_csv(
        self,
        all_results: Dict[int, List[Dict[str, Any]]],
        subdirectory: str = None,
        **kwargs
    ) -> None:
        """
        Save benchmark results to CSV files, one file per max_tokens value.
        
        Args:
            all_results: Dictionary mapping max_tokens to list of result dictionaries
            subdirectory: Optional subdirectory within output_dir
            **kwargs: Additional arguments
        """
        output_path = self.output_dir
        if subdirectory:
            output_path = output_path / subdirectory
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save one CSV file per max_tokens value
        for max_tokens, results in all_results.items():
            if not results:
                continue
                
            csv_filename = output_path / f"benchmark_max_tokens_{max_tokens}.csv"
            
            # Dynamically get all unique keys from results to create CSV headers
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
                    row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(row)
            
            print(f"  Saved results to {csv_filename} ({len(results)} rows)")
    
    def _plot_results(
        self,
        all_results: Dict[int, List[Dict[str, Any]]],
        output_file: str = None,
        **kwargs
    ) -> None:
        """
        Plot benchmark results using the plotting utilities.
        
        Args:
            all_results: Dictionary mapping max_tokens to list of result dictionaries
            output_file: Optional file path to save the plot
            **kwargs: Additional plotting arguments
        """
        # For plotting, we need to use the CSV files, so we'll use the plotting utility
        # which reads from the results directory
        results_dir = str(self.output_dir)
        if 'subdirectory' in kwargs:
            results_dir = str(self.output_dir / kwargs['subdirectory'])
        
        plot_benchmark_distributions(
            results_dir=results_dir,
            output_file=output_file,
            **{k: v for k, v in kwargs.items() if k != 'subdirectory'}
        )
    
    def print_summary(self, results: Dict[int, List[Dict[str, Any]]] = None) -> None:
        """
        Print a summary of benchmark results.
        
        Args:
            results: Results dictionary (if None, uses self.results)
        """
        if results is None:
            results = self.results
        
        if results is None:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print(f"Benchmark Results (model: {self.model_name}, server: {self.server_url})")
        print("="*80)
        
        print("\nAggregated Metrics (averages across all examples):")
        print("max_tokens | #examples | avg_generated | avg_TTFB(s) | avg_latency(s/tok) | avg_throughput(tok/s)")
        print("-" * 100)
        
        for max_tokens in sorted(results.keys()):
            result_list = results[max_tokens]
            if not result_list:
                continue
                
            num_examples = len(result_list)
            avg_generated = sum(r['tokens_generated'] for r in result_list) / num_examples
            avg_ttfb = sum(r['server_ttft_sec'] for r in result_list) / num_examples
            avg_latency = sum(r['latency_sec_per_token'] for r in result_list) / num_examples
            avg_throughput = sum(r['throughput_tokens_per_sec'] for r in result_list) / num_examples
            
            print(
                f"{max_tokens:10d} | {num_examples:9d} | "
                f"{avg_generated:13.1f} | {avg_ttfb:11.3f} | "
                f"{avg_latency:17.3f} | {avg_throughput:21.3f}"
            )
        
        print("\nDetailed Statistics:")
        print("-" * 100)
        for max_tokens in sorted(results.keys()):
            result_list = results[max_tokens]
            if not result_list:
                continue
                
            ttfb_values = [r['server_ttft_sec'] for r in result_list]
            latency_values = [r['latency_sec_per_token'] for r in result_list]
            
            print(f"\nmax_tokens={max_tokens}:")
            print(f"  TTFB: min={min(ttfb_values):.3f}s, max={max(ttfb_values):.3f}s, avg={sum(ttfb_values)/len(ttfb_values):.3f}s")
            print(f"  Latency: min={min(latency_values):.3f}s/tok, max={max(latency_values):.3f}s/tok, avg={sum(latency_values)/len(latency_values):.3f}s/tok")
