"""Tests for the math problem solver implementation."""

import pytest
from math_solver import MathProblem, MathSolution, MathResult
from math_prompter import MathPrompter
from math_executor import execute_math_code

def test_execute_math_code():
    """Test safe code execution."""
    # Test valid code
    code = "x = 5 + 3\nprint(x)"
    result, error = execute_math_code(code)
    assert result == "8"
    assert error is None

    # Test unsafe code
    unsafe_code = "import os\nos.system('ls')"
    result, error = execute_math_code(unsafe_code)
    assert result is None
    assert "unsafe operations" in error

def test_math_prompter():
    """Test MathPrompter functionality."""
    prompter = MathPrompter(model_name="gpt-4o-mini")

    # Test string input
    result = prompter.solve("What is 5 + 3?")
    assert isinstance(result, MathResult)
    assert result.question == "What is 5 + 3?"
    assert not result.error

    # Test dict input with expected answer
    problem = {
        "question": "What is 10 * 5?",
        "expected_answer": 50
    }
    result = prompter.solve(problem)
    assert isinstance(result, MathResult)
    assert result.is_correct is not None

    # Test batch processing
    problems = [
        "What is 2 + 2?",
        "What is 3 * 4?",
        "What is 15 / 3?"
    ]
    results = prompter.solve(problems)
    assert len(results) == 3

if __name__ == "__main__":
    pytest.main([__file__])
