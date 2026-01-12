"""
Result saver for evaluation experiments. Handles all file I/O operations.
Separates file I/O concerns from evaluation logic.
"""
from typing import Dict, Any, List
from pathlib import Path
import json
import csv


class ResultSaver:
    """
    Handles saving evaluation results in multiple formats (JSONL, CSV).
    """
    
    def __init__(self, output_dir: Path, experiment_name: str):
        """
        Initialize result saver.
        
        Args:
            output_dir: Base output directory
            experiment_name: Name of the experiment (used in filenames)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
    
    def get_output_path(self, subdirectory: str = None) -> Path:
        """
        Get the output path for saving results.
        
        Args:
            subdirectory: Optional subdirectory within output_dir. If None, uses experiment_name
            
        Returns:
            Path object for the output directory (created if it doesn't exist)
        """
        output_path = self.output_dir
        if subdirectory:
            output_path = self.output_dir / subdirectory
        else:
            output_path = self.output_dir / self.experiment_name
        output_path.mkdir(exist_ok=True, parents=True)
        return output_path
    
    def save_to_jsonl(
        self,
        results: List[Dict[str, Any]],
        filename_suffix: str,
        description: str,
        subdirectory: str = None
    ) -> None:
        """
        Save results to JSONL file.
                
        Args:
            results: List of result dictionaries to save
            filename_suffix: Suffix for filename (e.g., 'generated' or 'results')
            description: Description for print message
            subdirectory: Optional subdirectory within output_dir
        """
        output_path = self.get_output_path(subdirectory)
        jsonl_filename = output_path / f"{self.experiment_name}_{filename_suffix}.jsonl"
        
        # Save as JSONL: one JSON object per line
        with open(jsonl_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Saved {description} to {jsonl_filename} ({len(results)} samples)")
    
    def save_generated_results(self, generated_results: List[Dict[str, Any]], subdirectory: str = None) -> None:
        """
        Save intermediate generation results (before evaluation).
        Safety backup in case evaluation crashes.
        
        Args:
            generated_results: List of generated results to save
            subdirectory: Optional subdirectory within output_dir
        """
        self.save_to_jsonl(
            generated_results,
            filename_suffix='generated',
            description='intermediate generation results',
            subdirectory=subdirectory
        )
    
    def save_evaluated_results(self, evaluated_results: List[Dict[str, Any]], subdirectory: str = None) -> None:
        """
        Save evaluated results (generation + metrics per sample).
        
        Args:
            evaluated_results: List of evaluated results (each contains generation + metrics)
            subdirectory: Optional subdirectory within output_dir
        """
        self.save_to_jsonl(
            evaluated_results,
            filename_suffix='results',
            description='evaluated results (generation + metrics)',
            subdirectory=subdirectory
        )
    
    def save_to_csv(
        self,
        results: Dict[str, Any],
        subdirectory: str = None
    ) -> None:
        """
        Save evaluation results to CSV file.
        
        Args:
            results: Results dictionary containing 'results' key with list of evaluations
            subdirectory: Optional subdirectory within output_dir
        """
        output_path = self.get_output_path(subdirectory)
        
        result_list = results.get('results', [])
        if not result_list:
            print("No results to save.")
            return
        
        csv_filename = output_path / f"{self.experiment_name}_results.csv"
        
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
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in result_list:
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"  Saved evaluation results to {csv_filename} ({len(result_list)} rows)")
