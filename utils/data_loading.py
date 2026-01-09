"""
Data loading utilities for benchmarking.
"""
from typing import Dict, Any, List
from datasets import load_dataset


def load_and_sample_dataset(source: str = "openai_humaneval", n: int = 100, seed: int = 2026) -> List[Dict[str, Any]]:
    """
    Load and sample n random examples from the given dataset using streaming.
    Only downloads the sampled examples, not the entire dataset.
    
    Args:
        source: Hugging Face dataset identifier
        n: Number of random samples to load (default: 100)
        seed: Random seed for reproducibility (default: 2026)
    
    Returns:
        List of dicts - materialized because we iterate multiple times in benchmarking
    """
    # Use streaming mode - this doesn't download the entire dataset to disk
    dataset_stream = load_dataset(source, split="test", streaming=True)
    # take(n) takes first n items from the shuffled stream
    sampled_dataset = dataset_stream.shuffle(seed=seed, buffer_size=n).take(n)
    examples = [dict(example) for example in sampled_dataset]
    return examples


if __name__ == "__main__":
    # Load one sample from HumanEval to demonstrate before/after dict() conversion
    dataset_stream = load_dataset("openai_humaneval", split="test", streaming=True)
    sampled_dataset = dataset_stream.shuffle(seed=2026, buffer_size=1).take(1)
    
    # Get the first (and only) example
    example = next(iter(sampled_dataset))
    
    print("\n" + "="*80)
    print("Sample BEFORE dict() conversion:")
    print("="*80)
    print(f"Type: {type(example)}")
    print(f"Value: {example}")
    print(f"Keys (if dict-like): {list(example.keys()) if hasattr(example, 'keys') else 'N/A'}")
    
    # Convert to dict
    example_dict = dict(example)
    
    print("\n" + "="*80)
    print("Sample AFTER dict() conversion:")
    print("="*80)
    print(f"Type: {type(example_dict)}")
    print(f"Value: {example_dict}")
    print(f"Keys: {list(example_dict.keys())}")
    print("="*80 + "\n")