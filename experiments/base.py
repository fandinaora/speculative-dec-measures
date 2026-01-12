"""
Base experiment class that defines the common interface for all experiments.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path


class Experiment(ABC):
    """
    Base class for all experiments. Defines the common workflow:
    1. Load data
    2. Run experiment
    3. Save results
    4. Plot results (optional)
    """
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8081",
        model_name: str = "qwen2.5-7b-instruct",
        output_dir: str = "results",
        experiment_name: str = None
    ):
        """
        Initialize experiment with configuration.
        
        Args:
            server_url: URL of the llama.cpp server
            model_name: Name of the model to use
            output_dir: Directory to save results
            experiment_name: Name for this experiment (used in output filenames)
                           If None, defaults to class name (e.g., 'evaluation')
        """
        self.server_url = server_url
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate default experiment name from class name if not provided
        if experiment_name is None:
            class_name = self.__class__.__name__
            # Remove 'Experiment' suffix if present and convert to lowercase
            experiment_name = class_name.lower().replace('experiment', '').strip('_') or class_name.lower()
        self.experiment_name = experiment_name
        self.results = None
    
    @abstractmethod
    def load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Load and prepare data for the experiment.
        
        Returns:
            List of data samples to process
        """
        pass
    
    @abstractmethod
    def run(self, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Run the experiment on the provided data.
        
        Args:
            data: List of data samples to process
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing experiment results
        """
        pass
    
    def save_results(self, results: Dict[str, Any] = None, **kwargs) -> None:
        """
        Save experiment results to files (CSV, JSON, etc.).
        
        Args:
            results: Results to save (if None, uses self.results)
            **kwargs: Additional save options
        """
        if results is None:
            results = self.results
        
        if results is None:
            raise ValueError("No results to save. Run the experiment first.")
        
        self._save_to_csv(results, **kwargs)
    
    @abstractmethod
    def _save_to_csv(self, results: Dict[str, Any], **kwargs) -> None:
        """
        Save results to CSV format. Implementation specific to each experiment.
        
        Args:
            results: Results dictionary to save
            **kwargs: Additional save options
        """
        pass
    
    def plot(self, results: Dict[str, Any] = None, **kwargs) -> None:
        """
        Plot experiment results.
        
        Args:
            results: Results to plot (if None, uses self.results)
            **kwargs: Additional plotting options
        """
        if results is None:
            results = self.results
        
        if results is None:
            raise ValueError("No results to plot. Run the experiment first.")
        
        self._plot_results(results, **kwargs)
    
    @abstractmethod
    def _plot_results(self, results: Dict[str, Any], **kwargs) -> None:
        """
        Generate plots for results. Implementation specific to each experiment.
        
        Args:
            results: Results dictionary to plot
            **kwargs: Additional plotting options
        """
        pass
    
    def execute(
        self,
        load_kwargs: Dict[str, Any] = None,
        run_kwargs: Dict[str, Any] = None,
        save: bool = True,
        plot: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete experiment workflow.
        
        Args:
            load_kwargs: Keyword arguments for load_data()
            run_kwargs: Keyword arguments for run()
            save: Whether to save results after running
            plot: Whether to plot results after running
            **kwargs: Additional options
            
        Returns:
            Experiment results dictionary
        """
        if load_kwargs is None:
            load_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}
        
        # Load data
        print("Loading data...")
        data = self.load_data(**load_kwargs)
        print(f"Loaded {len(data)} samples")
        
        # Run experiment
        print("\nRunning experiment...")
        results = self.run(data, **run_kwargs, **kwargs)
        self.results = results
        
        # Save results
        if save:
            print("\nSaving results...")
            self.save_results(**kwargs)
        
        # Plot results
        if plot:
            print("\nGenerating plots...")
            self.plot(**kwargs)
        
        return results
