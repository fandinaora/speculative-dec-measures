"""
Prompt formatting utilities for benchmarking.
"""
import re
from pathlib import Path
from typing import Dict, Any


def format_code_completion_prompt(variables: Dict[str, Any], template_file: str = "prompts/prompt_template.txt") -> str:
    """
    Build the instruction string for the task.
    Loads prompt template from txt file and fills it with variables.
    Validates that all required variables are provided.
    
    Args:
        variables: Dictionary of variables to fill into the template (e.g., {"prompt": "...", "task": "..."})
        template_file: Path to the prompt template file (default: "prompts/prompt_template.txt")
    
    Returns:
        Formatted prompt string
    
    Raises:
        ValueError: If required variables are missing from the variables dict
        FileNotFoundError: If template file is not found
    """
    # Construct full path to template file
    template_path = Path(template_file)
    
    # If absolute path, use as-is
    if template_path.is_absolute():
        pass  # Use as-is
    # If it's just a filename (no slashes), look in prompts folder
    elif "/" not in template_file and "\\" not in template_file:
        template_path = Path("prompts") / template_file
    # Otherwise, it's already a relative path, use as-is
    else:
        template_path = Path(template_file)
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Template file not found: {template_path}\n"
        )
    
    # Find all placeholders in the template (e.g., {variable_name})
    # Pattern matches {variable_name} but not {{escaped}}
    placeholder_pattern = r'\{([^}]+)\}'
    required_vars = set(re.findall(placeholder_pattern, template))
    
    # Check if all required variables are provided
    missing_vars = required_vars - set(variables.keys())
    if missing_vars:
        raise ValueError(
            f"Missing required variables in template '{template_file}': {sorted(missing_vars)}\n"
            f"Provided variables: {sorted(variables.keys())}\n"
            f"Required variables: {sorted(required_vars)}"
        )
    
    # Only use variables that are actually in the template (ignore extra variables)
    template_vars = {key: value for key, value in variables.items() if key in required_vars}
    
    # Format the template with only the matching variables
    return template.format(**template_vars)
