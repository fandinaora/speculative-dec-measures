"""
Logging configuration utility for experiments.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    experiment_name: str = "experiment",
    output_dir: str = "results",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for an experiment with both file and console output.
    
    Args:
        log_file: Path to log file. If None, defaults to {output_dir}/{experiment_name}/{experiment_name}.log
        experiment_name: Name of the experiment (used for default log filename)
        output_dir: Base directory for output (used for default log path)
        log_level: Logging level (default: logging.INFO)
        
    Returns:
        Configured root logger
    """
    # Determine log file path
    if log_file is None:
        # Default: save logs to experiment directory
        log_dir = Path(output_dir) / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"{experiment_name}.log")
    else:
        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Log initialization message
    root_logger.info(f"Logging initialized. Log file: {log_file}")
    root_logger.info(f"Experiment: {experiment_name}")
    root_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return root_logger
