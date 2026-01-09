"""
Evaluation metrics for code generation quality.
"""
import re
import subprocess
import tempfile
import os
import textwrap
from typing import Dict, Any, Optional


def normalize_code(code: str) -> str:
    """
    Normalize code for comparison by removing extra whitespace and normalizing formatting.
    
    Args:
        code: Code string to normalize
        
    Returns:
        Normalized code string
    """
    if not code:
        return ""
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Normalize whitespace (multiple spaces/tabs to single space)
    code = re.sub(r'\s+', ' ', code)
    
    # Remove comments (simple approach - may not catch all cases)
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Normalize newlines
    code = code.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra spaces
    code = re.sub(r' +', ' ', code).strip()
    
    return code


def exact_match(generated: str, ground_truth: str) -> bool:
    """
    Check if generated code exactly matches ground truth after normalization.
    
    Args:
        generated: Generated code string
        ground_truth: Ground truth code string
        
    Returns:
        True if codes match exactly (after normalization), False otherwise
    """
    if not generated or not ground_truth:
        return False
    
    gen_norm = normalize_code(generated)
    truth_norm = normalize_code(ground_truth)
    
    return gen_norm == truth_norm


def bleu_score(generated: str, ground_truth: str) -> float:
    """
    Calculate BLEU score between generated and ground truth code.
    Uses nltk if available, otherwise returns 0.0.
    
    Args:
        generated: Generated code string
        ground_truth: Ground truth code string
        
    Returns:
        BLEU score between 0.0 and 1.0, or 0.0 if nltk is not available
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Auto-download punkt_tab if missing (needed for word_tokenize)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Tokenize the code (split by whitespace and punctuation)
        gen_tokens = word_tokenize(generated.lower())
        truth_tokens = word_tokenize(ground_truth.lower())
        
        # Calculate BLEU score
        score = sentence_bleu([truth_tokens], gen_tokens)
        return float(score)
    except ImportError:
        # nltk not installed, try alternative
        try:
            from sacrebleu import BLEU
            bleu = BLEU()
            score = bleu.sentence_score(generated, [ground_truth])
            return score.score / 100.0  # Convert to 0-1 scale
        except ImportError:
            # Neither library available
            return 0.0
    except Exception as e:
        # Error in calculation (e.g., missing NLTK resources)
        # Silently return 0.0 - the warning is already printed by NLTK
        return 0.0


import re
from typing import Optional

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
TOPLEVEL_STOP_RE = re.compile(r"(?m)^\s*(def\s+|class\s+|if\s+__name__)")

def extract_completion_only(generated: str) -> Optional[str]:
    if not generated:
        return None
    text = generated.strip()

    # If the model ignored instructions and used a fenced block, take the first one.
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # If model included a def header, drop it.
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("def "):
            text = "\n".join(lines[i+1:]).lstrip("\n")
            break

    # Stop if it starts a new top-level block.
    m2 = TOPLEVEL_STOP_RE.search(text)
    if m2:
        text = text[:m2.start()].rstrip()

    # Ensure indentation (HumanEval expects body).
    body_lines = [ln.rstrip() for ln in text.splitlines()]
    body_lines = [ln for ln in body_lines if ln.strip() != ""]
    if not body_lines:
        return None
    if any(ln and not ln.startswith((" ", "\t")) for ln in body_lines):
        body_lines = [("    " + ln) if ln.strip() else ln for ln in body_lines]

    return "\n".join(body_lines).rstrip() + "\n"


def extract_function_name(prompt: str) -> Optional[str]:
    """
    Extract function name from prompt (function signature).
    
    Args:
        prompt: Function signature string (e.g., "def add(a, b):")
        
    Returns:
        Function name or None if not found
    """
    match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if match:
        return match.group(1)
    return None


def _indent_code_body(code: str, indent_level: int = 4) -> str:
    """
    Indent code body to proper level if it's not already indented.
    Preserves relative indentation within the code.
    
    This is used when combining HumanEval prompt (function signature) with
    generated function body. In HumanEval format, the prompt contains the
    function signature/docstring, and the body needs to be indented (4 spaces)
    to be valid Python code.
    
    Args:
        code: Code string to indent (function body)
        indent_level: Number of spaces to indent (default: 4 for Python)
        
    Returns:
        Properly indented code string
    """
    if not code:
        return code
    
    # Remove trailing whitespace but preserve structure
    code = code.rstrip()
    if not code:
        return code
    
    # Check if code is already indented by looking at first non-empty line
    for line in code.split('\n'):
        if line.strip():  # Found first non-empty line
            if line[0] in [' ', '\t']:  # Already indented
                return code
            break
    
    # Code starts at column 0, indent the whole string
    indent_str = ' ' * indent_level
    return textwrap.indent(code, indent_str)


def run_unit_tests(
    prompt: str,
    generated_code: str,
    test_code: str,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Run unit tests on the generated code.
    Handles HumanEval format where tests use 'candidate' as function name.
    
    Note: In HumanEval format, the prompt contains the function signature/docstring
    and the generated_code should be the function body. We combine them to create
    a complete function for testing. Indentation handling ensures the body is properly
    indented (4 spaces) even if the LLM generates it starting at column 0.
    
    Args:
        prompt: Function signature and docstring from HumanEval (e.g., "def add(a, b):\\n    \"\"\"Add two numbers.\"\"\"")
        generated_code: Generated function body (should be indented, but may not be)
        test_code: Test code to execute (may use 'candidate' as function name)
        timeout: Timeout in seconds for test execution
        
    Returns:
        Dictionary with:
        - 'passed': bool indicating if tests passed
        - 'error': str with error message if tests failed
        - 'output': str with test output
    """
    if not generated_code or not test_code:
        return {
            'passed': False,
            'error': 'Missing generated code or test code',
            'output': ''
        }
    
    # Combine prompt (function signature) and generated code (function body) to create complete function
    # Ensure generated code is properly indented (4 spaces for Python function body)
    indented_code = _indent_code_body(generated_code, indent_level=4)
    complete_code = prompt + '\n' + indented_code
    
    # Extract function name from prompt
    function_name = extract_function_name(prompt)
    
    # If test code uses 'candidate', create an alias
    # HumanEval tests use 'candidate' to refer to the function
    if function_name and 'candidate' in test_code:
        # Add alias: candidate = function_name
        test_code_with_alias = f"{complete_code}\n\ncandidate = {function_name}\n\n{test_code}"
    else:
        # Just combine the code and tests
        test_code_with_alias = f"{complete_code}\n\n{test_code}"
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        # Write the complete code with test
        f.write(test_code_with_alias)
        temp_file = f.name
    
    try:
        # Run the test file
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(temp_file)
        )
        
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        
        return {
            'passed': passed,
            'error': result.stderr if not passed else '',
            'output': output
        }
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': f'Test execution timed out after {timeout} seconds',
            'output': ''
        }
    except Exception as e:
        return {
            'passed': False,
            'error': f'Error running tests: {str(e)}',
            'output': ''
        }
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except Exception:
            pass
