"""
Evaluation metrics for code generation quality.
"""
import re
import subprocess
import tempfile
import os
import textwrap
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class Metric(str, Enum):
    """
    Available evaluation metrics.
    """
    EXACT_MATCH = 'exact_match'
    BLEU = 'bleu'
    UNIT_TESTS = 'unit_tests'
    
    @classmethod
    def default_metrics(cls) -> List[str]:
        """Get list of default metrics."""
        return [cls.EXACT_MATCH.value, cls.BLEU.value, cls.UNIT_TESTS.value]
    
    @classmethod
    def all_metrics(cls) -> List[str]:
        """Get list of all available metrics."""
        return [metric.value for metric in cls]
    
    @classmethod
    def validate(cls, metrics: List[str]) -> List[str]:
        """
        Validate metric names.
        
        Args:
            metrics: List of metric names (strings)
            
        Returns:
            List of validated metric values (strings)
            
        Raises:
            ValueError: If any metric is invalid
        """
        valid_metrics = cls.all_metrics()
        invalid = [m for m in metrics if m not in valid_metrics]
        
        if invalid:
            raise ValueError(
                f"Invalid metrics: {invalid}. "
                f"Valid options are: {', '.join(valid_metrics)}"
            )
        
        return list(metrics)  # Return copy to avoid mutation


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
        BLEU score between 0.0 and 1.0, if nltk is not available try sacrebleu
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
        logger.warning("nltk library not available for BLEU score calculation. Trying sacrebleu as fallback...")
        try:
            from sacrebleu import BLEU
            bleu = BLEU()
            score = bleu.sentence_score(generated, [ground_truth])
            logger.info("Using sacrebleu for BLEU score calculation")
            return score.score / 100.0  # Convert to 0-1 scale
        except ImportError:
            # Neither library available
            logger.error("Neither nltk nor sacrebleu libraries are available. BLEU score calculation disabled. Returning 0.0")
            return 0.0
        except Exception as e:
            # Error in sacrebleu calculation
            logger.error(f"Error calculating BLEU score with sacrebleu: {str(e)}. Returning 0.0")
            return 0.0
    except Exception as e:
        # Error in calculation (e.g., missing NLTK resources)
        logger.error(f"Error calculating BLEU score with nltk: {str(e)}. Returning 0.0")
        return 0.0


import re
from typing import Optional

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
TOPLEVEL_STOP_RE = re.compile(r"(?m)^\s*(def\s+|class\s+|if\s+__name__)")

def extract_completion_only(generated: str) -> Optional[str]:
    if not generated:
        return None
    # Use rstrip() instead of strip() to preserve leading indentation
    # Only remove trailing whitespace
    text = generated.rstrip()

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
    
    # Check if the code is already properly indented (first non-empty line should have at least 4 spaces)
    # If the first line is already indented, preserve the existing indentation
    first_line_spaces = len(body_lines[0]) - len(body_lines[0].lstrip())
    needs_indentation = first_line_spaces == 0
    
    if needs_indentation:
        # Code starts at column 0, add base indentation (4 spaces)
        body_lines = [("    " + ln) if ln.strip() else ln for ln in body_lines]
    # Otherwise, preserve existing indentation as-is

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
        timeout: Timeout in seconds for test execution (default: 10)
        
    Returns:
        Dictionary with:
        - 'passed': bool indicating if tests passed
        - 'error': str with error message if tests failed (empty if passed)
        - 'output': str with combined stdout and stderr output
        - 'stdout': str with standard output only
        - 'stderr': str with standard error only
        - 'execution_time': float with execution time in seconds (None if timeout/error)
        - 'return_code': int with process return code (None if exception)
    """
    import time
    
    if not generated_code or not test_code:
        return {
            'passed': False,
            'error': 'Missing generated code or test code',
            'output': '',
            'stdout': '',
            'stderr': 'Missing generated code or test code',
            'execution_time': None,
            'return_code': None
        }
    
    # Combine prompt (function signature) and generated code (function body) first
    indented_code = _indent_code_body(generated_code, indent_level=4)
    complete_code = prompt + '\n' + indented_code
    
    # Basic security check: prevent obviously dangerous code
    # Check the actual formatted code that will be executed
    dangerous_patterns = [
        '__import__', 'eval(', 'exec(', 'compile(',
        'open(', 'file(', '__builtins__', '__globals__',
        'os.system', 'subprocess.', 'shutil.', 'sys.exit'
    ]
    combined_code = (complete_code + '\n\n' + test_code).lower()
    for pattern in dangerous_patterns:
        if pattern in combined_code:
            # Allow if it's in a string (commented or string literal)
            # This is a simple check - could be improved with AST parsing
            if f'"{pattern}"' not in combined_code and f"'{pattern}'" not in combined_code:
                return {
                    'passed': False,
                    'error': f'Potentially dangerous code detected: {pattern}',
                    'output': '',
                    'stdout': '',
                    'stderr': f'Potentially dangerous code detected: {pattern}',
                    'execution_time': None,
                    'return_code': None
                }
    
    # Extract function name from prompt (we already created complete_code above)
    function_name = extract_function_name(prompt)
    
    # If test code uses 'candidate', create an alias
    # HumanEval tests use 'candidate' to refer to the function
    if function_name and 'candidate' in test_code:
        # Add alias: candidate = function_name
        test_code_with_alias = f"{complete_code}\n\ncandidate = {function_name}\n\n{test_code}"
    else:
        # Just combine the code and tests
        test_code_with_alias = f"{complete_code}\n\n{test_code}"
    
    # CRITICAL: HumanEval test code defines check(candidate) but doesn't call it
    # We need to actually call check(candidate) for the assertions to run
    # Check if 'check(' appears as a function call (not just in the definition)
    # Look for 'check(' that's not part of 'def check('
    test_code_lower = test_code_with_alias.lower()
    has_check_def = 'def check(' in test_code_lower
    # Count 'check(' occurrences - if only 1, it's just the definition
    # If more than 1, it's already being called
    check_call_count = test_code_with_alias.count('check(')
    
    if has_check_def and check_call_count == 1:
        # check() is defined but not called - add the call
        test_code_with_alias += "\ncheck(candidate)\n"
    
    # Create a temporary Python file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            # Write the complete code with test
            f.write(test_code_with_alias)
            temp_file = f.name
        
        # Determine Python executable (handle both 'python' and 'python3')
        python_cmd = ['python']
        try:
            # Try to find python3 first (common on Linux/Mac)
            subprocess.run(['python3', '--version'], capture_output=True, timeout=1, check=True)
            python_cmd = ['python3']
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to python (Windows or if python3 not found)
            python_cmd = ['python']
        
        # Run the test file with timing
        start_time = time.time()
        try:
            result = subprocess.run(
                python_cmd + [temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(temp_file) or os.getcwd(),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Disable buffering for better output capture
            )
            execution_time = time.time() - start_time
            return_code = result.returncode
            
        except subprocess.TimeoutExpired as e:
            execution_time = timeout  # Approximate - actual time might be slightly more
            return {
                'passed': False,
                'error': f'Test execution timed out after {timeout} seconds',
                'output': f'Timeout after {timeout} seconds',
                'stdout': getattr(e, 'stdout', '') or '',
                'stderr': getattr(e, 'stderr', '') or f'Timeout after {timeout} seconds',
                'execution_time': execution_time,
                'return_code': None
            }
        
        passed = return_code == 0
        output = result.stdout + result.stderr
        
        # Extract error message - prioritize stderr, but include stdout if stderr is empty
        error_msg = result.stderr.strip() if result.stderr.strip() else (result.stdout.strip() if not passed else '')
        
        return {
            'passed': passed,
            'error': error_msg,
            'output': output,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time,
            'return_code': return_code
        }
        
    except Exception as e:
        return {
            'passed': False,
            'error': f'Error running tests: {str(e)}',
            'output': '',
            'stdout': '',
            'stderr': f'Error running tests: {str(e)}',
            'execution_time': None,
            'return_code': None
        }
    finally:
        # Clean up temporary file
        if temp_file:
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                # File might have been deleted already or doesn't exist
                pass


def _docker_available() -> bool:
    """
    Check if Docker is installed and running.
    
    Returns:
        True if Docker is available, False otherwise
    """
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


def _run_python_file_in_docker(
    host_py_path: str,
    timeout: int,
    image: str = "python:3.11-slim",
) -> subprocess.CompletedProcess:
    """
    Runs host_py_path inside a container. Assumes Docker is installed and the daemon is running.
    Cross-platform: uses absolute host path for -v mount.
    
    Args:
        host_py_path: Path to Python file on host machine
        timeout: Timeout in seconds for execution
        image: Docker image to use (default: python:3.11-slim)
        
    Returns:
        CompletedProcess object with execution results
    """
    host_py_path = os.path.abspath(host_py_path)
    host_dir = os.path.dirname(host_py_path)
    script_name = os.path.basename(host_py_path)

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--pids-limit", "256",
        "--memory", "1g", "--memory-swap", "1g",
        "--cpus", "1.0",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "-v", f"{host_dir}:/work:ro",
        "-w", "/work",
        image,
        "python", "-I", script_name,  # -I: isolated mode
    ]

    # Host-side timeout still applies (kills docker client; container is removed because --rm)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PYTHONUNBUFFERED": "1"},
    )


def run_unit_tests_in_docker(
    prompt: str,
    generated_code: str,
    test_code: str,
    timeout: int = 10,
    image: str = "python:3.11-slim"
) -> Dict[str, Any]:
    """
    Run unit tests on the generated code inside a Docker container for enhanced security.
    Handles HumanEval format where tests use 'candidate' as function name.
    
    This function provides the same functionality as run_unit_tests() but executes the code
    in an isolated Docker container with security restrictions:
    - No network access
    - Limited memory (1GB)
    - Limited CPU (1 core)
    - Limited processes (256 max)
    - Read-only filesystem mount
    - No privilege escalation
    - All capabilities dropped
    
    Note: Requires Docker to be installed and running. Use _docker_available() to check.
    
    Args:
        prompt: Function signature and docstring from HumanEval (e.g., "def add(a, b):\\n    \"\"\"Add two numbers.\"\"\"")
        generated_code: Generated function body (should be indented, but may not be)
        test_code: Test code to execute (may use 'candidate' as function name)
        timeout: Timeout in seconds for test execution (default: 10)
        image: Docker image to use (default: python:3.11-slim)
        
    Returns:
        Dictionary with:
        - 'passed': bool indicating if tests passed
        - 'error': str with error message if tests failed (empty if passed)
        - 'output': str with combined stdout and stderr output
        - 'stdout': str with standard output only
        - 'stderr': str with standard error only
        - 'execution_time': float with execution time in seconds (None if timeout/error)
        - 'return_code': int with process return code (None if exception)
    """
    import time
    
    # Check if Docker is available
    if not _docker_available():
        logger.warning("Docker is not available. Falling back to direct execution.")
        return run_unit_tests(prompt, generated_code, test_code, timeout)
    
    if not generated_code or not test_code:
        return {
            'passed': False,
            'error': 'Missing generated code or test code',
            'output': '',
            'stdout': '',
            'stderr': 'Missing generated code or test code',
            'execution_time': None,
            'return_code': None
        }
    
    # Combine prompt (function signature) and generated code (function body) first
    indented_code = _indent_code_body(generated_code, indent_level=4)
    complete_code = prompt + '\n' + indented_code
    
    # Basic security check: prevent obviously dangerous code
    # Even though Docker provides isolation, we still check for common dangerous patterns
    dangerous_patterns = [
        '__import__', 'eval(', 'exec(', 'compile(',
        'open(', 'file(', '__builtins__', '__globals__',
        'os.system', 'subprocess.', 'shutil.', 'sys.exit'
    ]
    combined_code = (complete_code + '\n\n' + test_code).lower()
    for pattern in dangerous_patterns:
        if pattern in combined_code:
            # Allow if it's in a string (commented or string literal)
            if f'"{pattern}"' not in combined_code and f"'{pattern}'" not in combined_code:
                return {
                    'passed': False,
                    'error': f'Potentially dangerous code detected: {pattern}',
                    'output': '',
                    'stdout': '',
                    'stderr': f'Potentially dangerous code detected: {pattern}',
                    'execution_time': None,
                    'return_code': None
                }
    
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
    
    test_code_lower = test_code_with_alias.lower()
    has_check_def = 'def check(' in test_code_lower
    check_call_count = test_code_with_alias.count('check(')
    
    if has_check_def and check_call_count == 1:
        # check() is defined but not called - add the call
        test_code_with_alias += "\ncheck(candidate)\n"
    
    # Create a temporary Python file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            # Write the complete code with test
            f.write(test_code_with_alias)
            temp_file = f.name
        
        # Run the test file in Docker with timing
        start_time = time.time()
        try:
            result = _run_python_file_in_docker(
                host_py_path=temp_file,
                timeout=timeout,
                image=image
            )
            execution_time = time.time() - start_time
            return_code = result.returncode
            
        except subprocess.TimeoutExpired as e:
            execution_time = timeout
            return {
                'passed': False,
                'error': f'Test execution timed out after {timeout} seconds',
                'output': f'Timeout after {timeout} seconds',
                'stdout': getattr(e, 'stdout', '') or '',
                'stderr': getattr(e, 'stderr', '') or f'Timeout after {timeout} seconds',
                'execution_time': execution_time,
                'return_code': None
            }
        
        passed = return_code == 0
        output = result.stdout + result.stderr
        
        # Extract error message - prioritize stderr, but include stdout if stderr is empty
        error_msg = result.stderr.strip() if result.stderr.strip() else (result.stdout.strip() if not passed else '')
        
        return {
            'passed': passed,
            'error': error_msg,
            'output': output,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time,
            'return_code': return_code
        }
        
    except Exception as e:
        return {
            'passed': False,
            'error': f'Error running tests in Docker: {str(e)}',
            'output': '',
            'stdout': '',
            'stderr': f'Error running tests in Docker: {str(e)}',
            'execution_time': None,
            'return_code': None
        }
    finally:
        # Clean up temporary file
        if temp_file:
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                # File might have been deleted already or doesn't exist
                pass
