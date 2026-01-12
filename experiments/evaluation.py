"""
Evaluation experiment for measuring the quality of LLM responses.
"""
from typing import Dict, Any, List, Optional

from .base import Experiment
from .result_saver import ResultSaver
from utils.data_loading import load_and_sample_dataset
from utils.measurements import measure_server_side_metrics
from utils.prompt_formatting import format_code_completion_prompt
from utils.evaluation_metrics import (
    Metric,
    exact_match,
    bleu_score,
    extract_completion_only,
    run_unit_tests
)


class EvaluationExperiment(Experiment):
    """
    Evaluation experiment that measures response quality using various metrics:
    - Code quality metrics (exact match, BLEU, etc.)
    - Functional correctness (unit tests)
    """
    
    def __init__(
        self,
        speculative_dec_params: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize evaluation experiment.
        
        Args:
            speculative_dec_params: Optional dictionary of speculative decoding parameters to pass to chat_payload.
                                   Examples: {"speculative.n_max": 16, "speculative.n_min": 3, "speculative.p_min": 0.75}
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        
        self.speculative_dec_params = speculative_dec_params or {}
        self.generated_results = None  # Store intermediate generation results (before evaluation)
        self.result_saver = ResultSaver(self.output_dir, self.experiment_name)
    
    def execute(
        self,
        load_kwargs: Dict[str, Any] = None,
        run_kwargs: Dict[str, Any] = None,
        save: bool = True,
        plot: bool = False,
        metrics: Optional[List[str]] = None,
        save_intermediate: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete evaluation workflow.
        
        Args:
            load_kwargs: Keyword arguments for load_data()
            run_kwargs: Keyword arguments for run() (passed to both generate() and evaluate())
                        Can include 'max_tokens', etc.
            save: Whether to save final results after running
            plot: Whether to plot results after running
            metrics: List of metric names (strings). Valid options: 'exact_match', 'bleu', 'unit_tests'.
                    If None, computes all available metrics
            save_intermediate: Whether to save intermediate generation results (before evaluation).
                           Defaults to True for safety (prevents losing generations if evaluation crashes)
            **kwargs: Additional options
            
        Returns:
            Experiment results dictionary 
        """
        # Pass metrics and save_intermediate to run_kwargs
        if run_kwargs is None:
            run_kwargs = {}
        if metrics is not None:
            run_kwargs['metrics'] = metrics
        run_kwargs['save_intermediate'] = save_intermediate
        
        # Call parent execute 
        results = super().execute(
            load_kwargs=load_kwargs,
            run_kwargs=run_kwargs,
            save=save,
            plot=plot,
            **kwargs
        )
        
        return results
    
    def load_data(
        self,
        dataset: str = "openai_humaneval",
        num_samples: int = 100,
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
        metrics: Optional[List[str]] = None,
        save_intermediate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation workflow: generate then evaluate.
        This orchestrates the two-phase approach (separation of generation and evaluation).
        
        Args:
            examples: List of example dictionaries from the dataset
            max_tokens: Maximum tokens to generate for each response
            metrics: List of metric names (strings). Valid options: 'exact_match', 'bleu', 'unit_tests'.
                    If None, computes all available metrics
            save_intermediate: Whether to save intermediate generation results (before evaluation).
                           Defaults to True for safety (prevents losing generations if evaluation crashes)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results with metrics for each sample
        """
        # Phase 1: Generate all responses
        print("Phase 1: Generating responses...")
        generated_results = self.generate(examples, max_tokens=max_tokens)
        self.generated_results = generated_results
        
        # Optionally save intermediate results (generations only)
        if save_intermediate:
            subdirectory = kwargs.get('subdirectory', None)
            self.result_saver.save_generated_results(generated_results, subdirectory=subdirectory)
        
        # Phase 2: Evaluate all generated responses
        print("\nPhase 2: Evaluating responses...")
        evaluated_results_dict = self.evaluate(generated_results, examples, metrics=metrics, max_tokens=max_tokens, **kwargs)
        
        # Save evaluated results (generation + metrics per sample) to JSONL file
        # This contains all metrics per sample and is saved right after evaluation
        subdirectory = kwargs.get('subdirectory', None)
        self.result_saver.save_evaluated_results(evaluated_results_dict.get('results', []), subdirectory=subdirectory)
        
        return evaluated_results_dict
    
    def generate(
        self,
        examples: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for all examples. Pure generation phase - no evaluation.
        
        Args:
            examples: List of example dictionaries from the dataset
            max_tokens: Maximum tokens to generate for each response
            **kwargs: Additional arguments (e.g., speculative_dec_params override)
            
        Returns:
            List of generated results (one per sample, with example_index for matching back)
        """
        
        generated_results = []
        
        # Get speculative_dec_params from kwargs or use instance default
        spec_params = kwargs.get('speculative_dec_params', self.speculative_dec_params)
        
        # Generate one response per example
        for i, example in enumerate(examples):
            prompt = example.get('prompt', '')
            if not prompt:
                print(f"    Warning: Example {i} has no 'prompt' field, skipping")
                continue
            
            task_id = example.get('task_id', f'sample_{i}')
            
            try:
                # Generate response
                result = self._generate_response(
                    prompt, 
                    task_id, 
                    max_tokens, 
                    sample_id=i,
                    temperature=0.0,  # Deterministic generation
                    speculative_dec_params=spec_params
                )
                
                # Store example index so we can match back during evaluation
                result['example_index'] = i
                
                generated_results.append(result)
                
            except Exception as e:
                print(f"    Error generating response for example {i}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                print(f"    Generated {i + 1}/{len(examples)} examples...")
        
        print(f"Completed generation: {len(generated_results)} successful responses")
        return generated_results
    
    def evaluate(
        self,
        generated_results: List[Dict[str, Any]],
        examples: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate generated responses with selectable metrics. Pure evaluation phase.
        
        Args:
            generated_results: List of generated results from generate()
            examples: Original example dictionaries from the dataset (for ground truth, tests, etc.)
            metrics: List of metric names (strings). Valid options: 'exact_match', 'bleu', 'unit_tests'.
                    If None, computes all available metrics
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results with metrics for each sample
        """
        if metrics is None:
            metrics = Metric.default_metrics()
        else:
            # Validate metrics if provided
            metrics = Metric.validate(metrics)
        
        print(f"Evaluating {len(generated_results)} generated responses...")
        print(f"Selected metrics: {', '.join(metrics)}")
        
        evaluated_results = []
        
        # Create a lookup map by task_id for efficient matching (more robust than index-based)
        examples_by_task_id = {ex.get('task_id', f'sample_{i}'): ex for i, ex in enumerate(examples)}
        
        # Evaluate each generated result
        for result in generated_results:
            task_id = result.get('task_id', '')
            
            # Find corresponding example by task_id (more robust than index)
            example = examples_by_task_id.get(task_id)
            if example is None:
                # Fallback to index if task_id lookup fails
                example_idx = result.get('example_index', 0)
                if example_idx < len(examples):
                    example = examples[example_idx]
                else:
                    print(f"    Warning: Could not find example for task_id={task_id}, skipping evaluation")
                    continue
            
            evaluated_result = result.copy()
            
            # Evaluate with selected metrics - convert strings to Metric enums for internal method
            metric_enums = [Metric(m) for m in metrics]  # Convert strings to enum values
            if (Metric.EXACT_MATCH in metric_enums or 
                Metric.BLEU in metric_enums or 
                Metric.UNIT_TESTS in metric_enums):
                evaluation = self._evaluate_response(result, example, metrics=metric_enums)
                evaluated_result.update(evaluation)
            
            evaluated_results.append(evaluated_result)
        
        # Get max_tokens from first result or kwargs
        max_tokens = kwargs.get('max_tokens', None)
        if max_tokens is None and generated_results:
            max_tokens = generated_results[0].get('max_tokens', 512)
        if max_tokens is None:
            max_tokens = 512
        
        return {
            'results': evaluated_results,
            'num_samples': len(evaluated_results),
            'max_tokens': max_tokens,
            'metrics_used': metrics
        }
    
    
    def _generate_response(
        self,
        prompt: str,
        task_id: str,
        max_tokens: int,
        sample_id: int,
        temperature: float = 0.0,
        speculative_dec_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: The prompt text from the dataset
            task_id: Task identifier from the dataset
            max_tokens: Maximum tokens to generate
            sample_id: ID of the sample for tracking
            temperature: Sampling temperature (0.0 = deterministic, >0.0 = diverse)
            speculative_dec_params: Optional override for speculative decoding parameters.
                                   If None, uses self.speculative_dec_params
            
        Returns:
            Dictionary with generated response and metadata
        """
        content = format_code_completion_prompt({"example_prompt": prompt})
        
        # Use provided params or fall back to instance params
        spec_params = speculative_dec_params if speculative_dec_params is not None else self.speculative_dec_params
        
        # Get response from server (temperature and speculative params passed)
        metrics = measure_server_side_metrics(
            content,
            max_tokens,
            self.server_url,
            self.model_name,
            temperature=temperature,
            **spec_params
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
        example: Dict[str, Any],
        metrics: Optional[List[Metric]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response with selectable metrics.
        Internal method - accepts Metric enum values for type safety.
        
        Args:
            result: Result dictionary with generated content
            example: Original example from dataset (may contain ground truth and tests)
            metrics: List of Metric enum values. If None, computes all available metrics
            
        Returns:
            Dictionary with evaluation metrics (only requested metrics included)
        """
        if metrics is None:
            metrics = [Metric.EXACT_MATCH, Metric.BLEU, Metric.UNIT_TESTS]
        
        generated_content = result.get('generated_content', '')
        prompt = example.get('prompt', '')
        ground_truth = example.get('canonical_solution', '')
        test_code = example.get('test', '')
        
        # Extract the actual code from the generated response
        # Use extract_completion_only for HumanEval format (just function body)
        generated_code = extract_completion_only(generated_content)
        
        # Fallback: if extraction fails, use the full generated content
        if not generated_code:
            generated_code = generated_content
        
        evaluation = {
            'extracted_completion': generated_code,  # Always store extracted completion
            'canonical_solution': ground_truth,  # Always store ground truth for reference
        }
        
        # Calculate exact match if requested
        if Metric.EXACT_MATCH in metrics:
            exact_match_score = False
            if ground_truth:
                exact_match_score = exact_match(generated_code, ground_truth)
            evaluation['exact_match'] = exact_match_score
        
        # Calculate BLEU score if requested
        if Metric.BLEU in metrics:
            bleu = 0.0
            if ground_truth:
                bleu = bleu_score(generated_code, ground_truth)
            evaluation['bleu_score'] = bleu
        
        # Run unit tests if requested
        if Metric.UNIT_TESTS in metrics:
            unit_test_result = {'passed': False, 'error': 'No test code available', 'output': ''}
            if test_code:
                unit_test_result = run_unit_tests(
                    prompt=prompt,
                    generated_code=generated_code,
                    test_code=test_code,
                    timeout=10
                )
            evaluation['unit_test_passed'] = unit_test_result['passed']
            evaluation['unit_test_error'] = unit_test_result.get('error', '')
            evaluation['unit_test_execution_time'] = unit_test_result.get('execution_time')
        
        return evaluation
    
    def _save_to_csv(
        self,
        results: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Save evaluation results to CSV file.
        Required by abstract base class - delegates to ResultSaver.
        
        Args:
            results: Results dictionary containing 'results' key with list of evaluations
            **kwargs: Additional arguments (can include 'subdirectory')
        """
        subdirectory = kwargs.get('subdirectory', None)
        self.result_saver.save_to_csv(results, subdirectory=subdirectory)
    
    def _plot_results(
        self,
        results: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Plot evaluation results.
        Required by abstract base class - TODO: implement plotting functionality.
        
        Args:
            results: Results dictionary
            **kwargs: Additional plotting arguments (can include 'output_file')
        """
        # TODO: Implement plotting for evaluation metrics
        # This could show distributions of BLEU scores, exact match rates, etc.
        print("Plotting evaluation results...")
        print("(Plotting functionality to be implemented)")