"""
Evaluation experiment for measuring the quality of LLM responses.
"""
from typing import Dict, Any, List
from pathlib import Path

from .base import Experiment
from utils.data_loading import load_and_sample_dataset
from utils.measurements import measure_server_side_metrics
from utils.prompt_formatting import format_code_completion_prompt
from utils.evaluation_metrics import (
    exact_match,
    bleu_score,
    extract_completion_only,
    run_unit_tests
)


class EvaluationExperiment(Experiment):
    """
    Evaluation experiment that measures response quality using various metrics:
    - Code quality metrics (exact match, BLEU, etc.)
    - Functional correctness
    - Optional: Official HumanEval pass@k metrics
    """
    
    def __init__(self, use_official_humaneval: bool = False, num_samples_per_problem: int = 1, temperature: float = 0.0, **kwargs):
        """
        Initialize evaluation experiment.
        
        Args:
            use_official_humaneval: If True, use official human-eval package for pass@k.
                                   Requires all 164 HumanEval problems to be evaluated.
            num_samples_per_problem: Number of completions to generate per problem (for pass@k).
                                    Default is 1. For pass@k, you need at least k samples.
            temperature: Sampling temperature for generation (0.0 = deterministic, >0.0 = diverse).
                        For pass@k, use temperature > 0 (e.g., 0.2-0.8) to get diverse samples.
                        Default: 0.0 (deterministic)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.use_official_humaneval = use_official_humaneval
        self.num_samples_per_problem = num_samples_per_problem if num_samples_per_problem else 1
        self.temperature = temperature if temperature is not None else 0.0
    
    def execute(
        self,
        load_kwargs: Dict[str, Any] = None,
        run_kwargs: Dict[str, Any] = None,
        save: bool = True,
        plot: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete evaluation workflow.
        Overrides base execute to compute pass@k AFTER saving results.
        
        Args:
            load_kwargs: Keyword arguments for load_data()
            run_kwargs: Keyword arguments for run()
            save: Whether to save results after running
            plot: Whether to plot results after running
            **kwargs: Additional options
            
        Returns:
            Experiment results dictionary with pass@k computed
        """
        if load_kwargs is None:
            load_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}
        
        # Load data
        print("Loading data...")
        data = self.load_data(**load_kwargs)
        print(f"Loaded {len(data)} samples")
        
        # Run experiment (generates samples and computes BLEU, exact match, etc.)
        print("\nRunning experiment...")
        results = self.run(data, **run_kwargs)
        self.results = results
        
        # Save results FIRST (before computing pass@k)
        if save:
            print("\nSaving results...")
            self.save_results(**kwargs)
        
        # Compute pass@k AFTER saving (if using official HumanEval)
        if self.use_official_humaneval:
            print("\nComputing pass@k metric...")
            pass_at_k = self._run_official_humaneval(results.get('results', []), k=1)
            results['pass@k'] = pass_at_k
            self.results = results
            
            # Update summary file with pass@k
            if save:
                self._update_summary_with_pass_at_k(pass_at_k, **kwargs)
        
        # Plot results
        if plot:
            print("\nGenerating plots...")
            self.plot(**kwargs)
        
        return results
    
    def _update_summary_with_pass_at_k(self, pass_at_k: float, **kwargs):
        """Update the summary JSON file with pass@k value."""
        import json
        from pathlib import Path
        
        subdirectory = kwargs.get('subdirectory', 'evaluation')
        output_path = self.output_dir
        if subdirectory:
            output_path = output_path / subdirectory
        
        summary_filename = output_path / "evaluation_summary.json"
        
        # Read existing summary or create new one
        if summary_filename.exists():
            with open(summary_filename, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Update with pass@k
        summary['pass@k'] = pass_at_k
        
        # Write back
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Updated evaluation summary with pass@k: {pass_at_k:.4f}")
    
    def load_data(
        self,
        dataset: str = "openai_humaneval",
        num_samples: int = None,
        seed: int = 2026,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load dataset for evaluation. Can load full dataset or a sample.
        
        Args:
            dataset: Hugging Face dataset identifier
            num_samples: Number of samples to load (None = load all)
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
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation experiment: generate responses and evaluate quality.
        
        Args:
            examples: List of example dictionaries from the dataset
            max_tokens: Maximum tokens to generate for each response
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results with metrics for each sample
        """
        print(f"\nRunning evaluation on {len(examples)} examples.")
        print(f"Using max_tokens={max_tokens}")
        
        # Check if using official HumanEval and validate sample count
        if self.use_official_humaneval:
            if len(examples) != 164:
                raise ValueError(
                    f"Official HumanEval evaluation requires all 164 problems. "
                    f"Got {len(examples)} examples. Set num_samples=None to load all problems."
                )
            print("Using official HumanEval package for pass@k calculation")
        
        results = []
        
        # Generate multiple samples per problem if needed for pass@k
        for i, example in enumerate(examples):
            prompt = example.get('prompt', '')
            if not prompt:
                print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                continue
            
            task_id = example.get('task_id', f'sample_{i}')
            
            # Generate num_samples_per_problem completions for this problem
            for sample_idx in range(self.num_samples_per_problem):
                try:
                    # Use temperature=0 for first sample (deterministic metrics)
                    # Use temperature>0 for additional samples (diverse pass@k)
                    use_temp = self.temperature if (sample_idx > 0 and self.use_official_humaneval) else 0.0
                    
                    # Generate response
                    result = self._generate_response(
                        prompt, 
                        task_id, 
                        max_tokens, 
                        sample_id=f"{i}_{sample_idx}" if self.num_samples_per_problem > 1 else i,
                        temperature=use_temp
                    )
                    
                    # Evaluate quality (only evaluate first sample for custom metrics to avoid duplicates)
                    if sample_idx == 0:
                        evaluation = self._evaluate_response(result, example)
                        result.update(evaluation)
                    else:
                        # For additional samples, just extract completion (skip expensive metrics)
                        generated_content = result.get('generated_content', '')
                        generated_code = extract_completion_only(generated_content)
                        if not generated_code:
                            generated_code = generated_content.strip()
                        result['extracted_completion'] = generated_code
                        result['canonical_solution'] = example.get('canonical_solution', '')
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"    Error processing example {i}, sample {sample_idx}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(examples)} problems ({len(results)} total completions)...")
        
        print(f"\nCompleted evaluation: {len(results)} successful samples")
        
        # Note: pass@k will be computed after saving results (in execute method)
        return {
            'results': results,
            'num_samples': len(results),
            'max_tokens': max_tokens,
            'pass@k': None  # Will be computed after saving
        }
    
    def _generate_response(
        self,
        prompt: str,
        task_id: str,
        max_tokens: int,
        sample_id: int,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: The prompt text from the dataset
            task_id: Task identifier from the dataset
            max_tokens: Maximum tokens to generate
            sample_id: ID of the sample for tracking
            temperature: Sampling temperature (0.0 = deterministic, >0.0 = diverse)
            
        Returns:
            Dictionary with generated response and metadata
        """
        content = format_code_completion_prompt({"example_prompt": prompt})
        
        # Get response from server (temperature passed explicitly)
        metrics = measure_server_side_metrics(
            content,
            max_tokens,
            self.server_url,
            self.model_name,
            temperature=temperature
        )
        
        result = {
            'sample_id': sample_id,
            'task_id': task_id,
            'prompt': prompt,
            'formatted_prompt': content,
            'max_tokens': max_tokens,
            **metrics
        }
        
        return result
    
    def _evaluate_response(
        self,
        result: Dict[str, Any],
        example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response.
        
        Args:
            result: Result dictionary with generated content
            example: Original example from dataset (may contain ground truth and tests)
            
        Returns:
            Dictionary with evaluation metrics
        """
        generated_content = result.get('generated_content', '')
        prompt = example.get('prompt', '')
        ground_truth = example.get('canonical_solution', '')
        test_code = example.get('test', '')
        
        # Extract the actual code from the generated response
        # Use extract_completion_only for HumanEval format (just function body)
        generated_code = extract_completion_only(generated_content)
        
        if not generated_code:
            # Fallback to extract_code_from_response if extract_completion_only fails
            generated_code = extract_completion_only(generated_content)
            if not generated_code:
                # If we can't extract code, use the full generated content
                generated_code = generated_content
        
        # Calculate exact match
        exact_match_score = False
        if ground_truth:
            exact_match_score = exact_match(generated_code, ground_truth)
        
        # Calculate BLEU score
        bleu = 0.0
        if ground_truth:
            bleu = bleu_score(generated_code, ground_truth)
        
        # Run unit tests
        unit_test_result = {'passed': False, 'error': 'No test code available', 'output': ''}
        if test_code:
            unit_test_result = run_unit_tests(
                prompt=prompt,
                generated_code=generated_code,
                test_code=test_code,
                timeout=10
            )
        
        evaluation = {
            'exact_match': exact_match_score,
            'bleu_score': bleu,
            'unit_test_passed': unit_test_result['passed'],
            'unit_test_error': unit_test_result.get('error', ''),
            'extracted_completion': generated_code,  # Store extracted completion
            'canonical_solution': ground_truth,  # Store ground truth for reference
        }
        
        return evaluation
    
    def _save_to_csv(
        self,
        results: Dict[str, Any],
        subdirectory: str = "evaluation",
        **kwargs
    ) -> None:
        """
        Save evaluation results to CSV file.
        
        Args:
            results: Results dictionary containing 'results' key with list of evaluations
            subdirectory: Optional subdirectory within output_dir
            **kwargs: Additional arguments
        """
        output_path = self.output_dir
        if subdirectory:
            output_path = output_path / subdirectory
        output_path.mkdir(exist_ok=True, parents=True)
        
        result_list = results.get('results', [])
        if not result_list:
            print("No results to save.")
            return
        
        csv_filename = output_path / "evaluation_results.csv"
        
        # Get all unique keys from results
        all_keys = set()
        for result in result_list:
            all_keys.update(result.keys())
        
        # Put sample_id first if it exists, then sort the rest
        fieldnames = []
        if 'sample_id' in all_keys:
            fieldnames.append('sample_id')
            all_keys.remove('sample_id')
        if 'task_id' in all_keys:
            fieldnames.append('task_id')
            all_keys.remove('task_id')
        fieldnames.extend(sorted(all_keys))
        
        import csv
        import json
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in result_list:
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"  Saved evaluation results to {csv_filename} ({len(result_list)} rows)")
        
        # Save summary with pass@k if available
        summary = {
            'num_samples': results.get('num_samples', len(result_list)),
            'max_tokens': results.get('max_tokens', ''),
            'pass@k': results.get('pass@k'),
        }
        
        # Remove None values
        summary = {k: v for k, v in summary.items() if v is not None}
        
        if summary:
            summary_filename = output_path / "evaluation_summary.json"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved evaluation summary to {summary_filename}")
    
    def _plot_results(
        self,
        results: Dict[str, Any],
        output_file: str = None,
        **kwargs
    ) -> None:
        """
        Plot evaluation results.
        
        Args:
            results: Results dictionary
            output_file: Optional file path to save the plot
            **kwargs: Additional plotting arguments
        """
        # TODO: Implement plotting for evaluation metrics
        # This could show distributions of BLEU scores, exact match rates, etc.
        print("Plotting evaluation results...")
        print("(Plotting functionality to be implemented)")
        
        # Example structure:
        # - Bar chart of average scores per metric
        # - Distribution plots for each metric
        # - Comparison plots if evaluating multiple models/configs
    
    def _run_official_humaneval(self, results: List[Dict[str, Any]], k: int = 3) -> float:
        """
        Run official HumanEval evaluation to compute pass@k.
        
        Args:
            results: List of result dictionaries with extracted completions
            k: Value of k for pass@k calculation (default: 3)
            
        Returns:
            pass@k score as float
        """
        try:
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportError:
            print("⚠ Warning: human-eval package not found. Cannot compute official pass@k.")
            print("  Install with: pip install human-eval")
            return None
        
        import json
        import tempfile
        from pathlib import Path
        
        # Prepare completions in HumanEval format
        completions = []
        for result in results:
            task_id = result.get('task_id', '')
            completion = result.get('extracted_completion', '')
            if task_id and completion:
                completions.append({
                    "task_id": task_id,
                    "completion": completion
                })
        
        if not completions:
            print("⚠ Warning: No completions found for official evaluation")
            return None
        
        # Save to temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for completion in completions:
                f.write(json.dumps(completion) + '\n')
            temp_file = f.name
        
        try:
            print(f"\nRunning official HumanEval evaluation (pass@{k})...")
            eval_results = evaluate_functional_correctness(
                temp_file,
                k=[k],
                n_workers=4,
                timeout=3.0
            )
            
            pass_at_k_key = f"pass@{k}"
            if pass_at_k_key in eval_results:
                pass_at_k = eval_results[pass_at_k_key]
                print(f"✓ pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
                return pass_at_k
            else:
                print(f"⚠ Warning: Could not find pass@{k} in evaluation results")
                return None
                
        except Exception as e:
            print(f"✗ Error running official evaluation: {e}")
            return None
        finally:
            # Clean up temp file
            try:
                Path(temp_file).unlink()
            except:
                pass