"""
Test script for format_code_completion_prompt function
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import bench_code_completion
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_code_completion import format_code_completion_prompt

def test_basic_usage():
    """Test basic usage with correct variables"""
    print("Test 1: Basic usage with correct variables")
 
    
    variables = {
        "example_prompt": "def add(a, b):\n    return a + b"
    }
    
    result = format_code_completion_prompt(variables)
    
    # Verify the result contains the expected content
    assert "def add(a, b):" in result, "Code should be in result"
    assert "return a + b" in result, "Code should be in result"
    assert "code completion assistant" in result.lower(), "Template should be present"
    
    print("Result:")
    print(result)
    print("\n[PASSED] - Result verified to contain expected content\n")


def test_missing_variable():
    """Test error when required variable is missing"""
    print("Test 2: Missing required variable (should error)")
   
    
    variables = {
        "wrong_key": "some value"
    }
    
    try:
        result = format_code_completion_prompt(variables)
        print("[FAILED] - Should have raised ValueError")
    except ValueError as e:
        print("[PASSED] - Correctly raised ValueError:")
        print(f"   {e}\n")


def test_extra_variables():
    """Test that extra variables are ignored"""
    print("Test 3: Extra variables (should be ignored)")
    
    variables = {
        "example_prompt": "def multiply(x, y):\n    return x * y",
        "extra_var": "this should be ignored",
        "another_extra": "also ignored"
    }
    
    result = format_code_completion_prompt(variables)
    
    # Verify the result contains the expected content
    assert "def multiply(x, y):" in result, "Code should be in result"
    assert "return x * y" in result, "Code should be in result"
    # Verify extra variables are NOT in the result (they should be ignored)
    assert "this should be ignored" not in result, "Extra variables should not appear in result"
    assert "also ignored" not in result, "Extra variables should not appear in result"
    
    print("Result:")
    print(result)
    print("\n[PASSED] - Extra variables correctly ignored\n")


def test_custom_template():
    """Test with custom template file"""
    print("Test 4: Custom template file")
    
    # Create a custom template for testing
    custom_template = """Task: {task_type}
Code to complete:
{code}
Instructions: {instructions}"""
    
    # Use parent directory for prompts folder
    prompts_dir = Path(__file__).parent.parent / "prompts"
    with open(prompts_dir / "custom_test.txt", "w", encoding="utf-8") as f:
        f.write(custom_template)
    
    variables = {
        "task_type": "code completion",
        "code": "def subtract(a, b):",
        "instructions": "Complete the function"
    }
    
    result = format_code_completion_prompt(variables, template_file="custom_test.txt")
    
    # Actually verify the result is correct
    assert "code completion" in result, "task_type should be in result"
    assert "def subtract(a, b):" in result, "code should be in result"
    assert "Complete the function" in result, "instructions should be in result"
    assert "Task:" in result, "Template structure should be preserved"
    assert "Code to complete:" in result, "Template structure should be preserved"
    
    print("Result:")
    print(result)
    print("\n[PASSED] - Result verified to contain all expected values\n")


def test_file_not_found():
    """Test error when template file doesn't exist"""
    print("Test 5: Template file not found (should error)")
    
    variables = {"example_prompt": "test"}
    
    try:
        result = format_code_completion_prompt(variables, template_file="nonexistent.txt")
        print("[FAILED] - Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print("[PASSED] - Correctly raised FileNotFoundError:")
        print(f"   {e}\n")


if __name__ == "__main__":
    print("\nTesting format_code_completion_prompt function\n")
    
    try:
        test_basic_usage()
        test_missing_variable()
        test_extra_variables()
        test_custom_template()
        test_file_not_found()
        
        print("[SUCCESS] All tests completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
