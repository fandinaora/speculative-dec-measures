"""
Simple script to run HumanEval using the official evaluation package.
This is a minimal wrapper around the official human-eval package.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from utils.data_loading import load_and_sample_dataset
from utils.prompt_formatting import format_code_completion_prompt
from utils.measurements import measure_server_side_metrics
from utils.evaluation_metrics import extract_completion_only, bleu_score


def main():
    """Run HumanEval benchmark with official evaluation."""
    # Configuration
    SERVER_URL = "http://127.0.0.1:8081"
    MODEL_NAME = "qwen2.5-7b-instruct"
    NUM_SAMPLES = 164 # Set to None to evaluate all 164 problems (required for official evaluation)
    NUM_SAMPLES_PER_PROBLEM = 3  # Number of completions per problem (for pass@k, set to at least k)
    TEMPERATURE = 0.2  # Temperature for additional pass@k samples only. First sample always uses temp=0 for deterministic metrics
    MAX_TOKENS = 512
    
    print("HumanEval Benchmark - Using Official Evaluation")
    print("\nStep 1: Install official human-eval package if needed...")

    # Check if human-eval is available
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        print("\n✓ Found human-eval package!")
        use_official = True
    except ImportError:
        print("human-eval package not found.")
        print("  Will generate completions file for manual evaluation.")
        use_official = False
    
    # Load dataset
    print(f"Loading {NUM_SAMPLES} samples from HumanEval...")
    try:
        examples = load_and_sample_dataset(
            source="openai_humaneval",
            n=NUM_SAMPLES,
            seed=2026
        )
        print(f"✓ Loaded {len(examples)} examples")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Generate completions
    print(f"Generating {NUM_SAMPLES_PER_PROBLEM} completions per problem...")
    completions = []
    # Store ground truth for BLEU calculation
    ground_truth_map = {}
    
    for i, example in enumerate(examples):
        task_id = example.get('task_id', f'HumanEval/{i}')
        prompt = example.get('prompt', '')
        canonical_solution = example.get('canonical_solution', '')
        
        if not prompt:
            continue
        
        # Store ground truth for later BLEU calculation
        if canonical_solution:
            ground_truth_map[task_id] = canonical_solution
        
        # Generate multiple completions per problem (for pass@k)
        for sample_idx in range(NUM_SAMPLES_PER_PROBLEM):
            try:
                # Format prompt
                formatted_prompt = format_code_completion_prompt({"example_prompt": prompt})
                
                # Use temperature=0 for first sample (deterministic), temperature>0 for additional samples (diverse pass@k)
                use_temp = TEMPERATURE if sample_idx > 0 else 0.0
                
                # Get completion
                metrics = measure_server_side_metrics(
                    formatted_prompt,
                    MAX_TOKENS,
                    SERVER_URL,
                    MODEL_NAME,
                    temperature=use_temp
                )
                
                generated_content = metrics.get('generated_content', '')
                
                # Extract function body (HumanEval format expects just the body)
                # extract_completion_only only takes the generated content, not the prompt
                generated_code = extract_completion_only(generated_content)
                if not generated_code:
                    generated_code = generated_content.strip()
                
                completions.append({
                    "task_id": task_id,
                    "completion": generated_code
                })
                
            except Exception as e:
                print(f"  Error on {task_id}, sample {sample_idx}: {e}")
                continue
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(examples)} problems ({len(completions)} total completions)...")
    
    # Save to JSONL file (format expected by official evaluator)
    output_file = Path("results/humaneval_completions.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for completion in completions:
            f.write(json.dumps(completion) + '\n')
    
    print(f"\n✓ Saved {len(completions)} completions to: {output_file}")
    
    # Run official evaluation if available
    if use_official:
        print("Running official HumanEval evaluation...")
        try:
            # The official evaluator may complain about missing problems, but we can still evaluate
            # what we have by catching the error and extracting results if available
            # Calculate k values based on samples per problem
            k_values = [1]
            if NUM_SAMPLES_PER_PROBLEM >= 3:
                k_values.append(3)
            if NUM_SAMPLES_PER_PROBLEM >= 5:
                k_values.append(5)
            
            results = evaluate_functional_correctness(
                str(output_file),
                k=k_values,
                n_workers=4,
                timeout=3.0
            )
            
            print("Results (Official HumanEval Evaluation):")
            # Print pass@k metrics first
            for metric, value in results.items():
                if metric.startswith("pass@") and isinstance(value, float):
                    print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        
            
            # Calculate BLEU scores (additional metric not in official evaluation)
            print("Calculating BLEU scores (additional metric)")
            bleu_scores = []
            for completion in completions:
                task_id = completion["task_id"]
                generated_code = completion["completion"]
                if task_id in ground_truth_map:
                    bleu = bleu_score(generated_code, ground_truth_map[task_id])
                    bleu_scores.append(bleu)
            
            if bleu_scores:
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                results["bleu_score_avg"] = avg_bleu
                results["bleu_score_count"] = len(bleu_scores)
                print(f"  Average BLEU score: {avg_bleu:.4f} ({avg_bleu*100:.2f}%)")
                print(f"  (Computed for {len(bleu_scores)} samples with ground truth)")
            
            # Save results
            results_file = Path("results/humaneval_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Saved results to: {results_file}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error: {error_msg}")
            
            # The evaluator may complain about missing problems, but we can work around it
            if "not attempted" in error_msg.lower():
                print("\n⚠ The official evaluator expects all 164 HumanEval problems.")
                print(f"  You have {len(completions)} samples. This is normal for partial evaluation.")
                print("\n  Options:")
                print("  1. Evaluate all 164 problems (set NUM_SAMPLES = None or 164)")
                print("  2. Use your custom evaluation (run_evaluation.py) for partial datasets")
                print("  3. Manually run evaluation on the completions file")
                print(f"\n  File saved at: {output_file}")
            else:
                print("\nYou can run the evaluation manually:")
                print(f"  python -m human_eval.evaluation {output_file}")
    else:
        print(f"\n{'='*80}")
        print("To run the official evaluation:")
        print("="*80)
        print("1. Install: pip install human-eval")
        print("   OR clone: git clone https://github.com/openai/human-eval.git")
        print("\n2. Run evaluation:")
        print(f"   python -m human_eval.evaluation {output_file}")
        print("\n   OR if you cloned the repo:")
        print(f"   cd human-eval")
        print(f"   python evaluate_functional_correctness.py --samples_file ../{output_file}")
        print("="*80)


if __name__ == "__main__":
    main()
