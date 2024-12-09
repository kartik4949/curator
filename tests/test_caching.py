from datasets import Dataset
from pathlib import Path

from bespokelabs.curator import Prompter


def test_same_value_caching(tmp_path):
    """Test that using the same value multiple times uses cache."""
    values = []

    # Test with same value multiple times
    for _ in range(3):

        def prompt_func():
            return f"Say '1'. Do not explain."

        prompter = Prompter(
            prompt_func=prompt_func,
            model_name="gpt-4o-mini",
        )
        result = prompter(working_dir=str(tmp_path))
        values.append(result.to_pandas().iloc[0]["response"])

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
    assert values == ["1", "1", "1"], "Same value should produce same results"


def test_different_values_caching(tmp_path):
    """Test that using different values creates different cache entries."""
    values = []

    # Test with different values
    for x in [1, 2, 3]:

        def prompt_func():
            return f"Say '{x}'. Do not explain."

        prompter = Prompter(
            prompt_func=prompt_func,
            model_name="gpt-4o-mini",
        )
        result = prompter(working_dir=str(tmp_path))
        values.append(result.to_pandas().iloc[0]["response"])

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 3, f"Expected 3 cache directories but found {len(cache_dirs)}"
    assert values == ["1", "2", "3"], "Different values should produce different results"


def test_same_dataset_caching(tmp_path):
    """Test that using the same dataset multiple times uses cache."""
    dataset = Dataset.from_list([{"instruction": "Say '1'. Do not explain."}])
    prompter = Prompter(
        prompt_func=lambda x: x["instruction"],
        model_name="gpt-4o-mini",
    )

    result = prompter(dataset=dataset, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    result = prompter(dataset=dataset, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"


def test_different_dataset_caching(tmp_path):
    """Test that using different datasets creates different cache entries."""
    dataset1 = Dataset.from_list([{"instruction": "Say '1'. Do not explain."}])
    dataset2 = Dataset.from_list([{"instruction": "Say '2'. Do not explain."}])
    prompter = Prompter(
        prompt_func=lambda x: x["instruction"],
        model_name="gpt-4o-mini",
    )

    result = prompter(dataset=dataset1, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    result = prompter(dataset=dataset2, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "2"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


def test_nested_call_caching(tmp_path):
    """Test that changing a nested upstream function invalidates the cache."""

    def value_generator():
        return 1

    def prompt_func():
        return f"Say '{value_generator()}'. Do not explain."

    prompter = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
    )
    result = prompter(working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    def value_generator():
        return 2

    result = prompter(working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "2"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


def test_function_hash_file_independence():
    """Test that identical functions in different files produce the same hash."""
    from bespokelabs.curator.prompter.prompter import _get_function_hash
    import logging
    import sys
    from pathlib import Path
    import tempfile
    import os

    # Set up logging to write to a file
    debug_log = Path(tempfile.gettempdir()) / "function_debug.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        filename=str(debug_log),
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    def dump_function_details(func, prefix):
        """Helper to dump all function details."""
        logger.debug(f"\n{prefix} details:")
        # Basic attributes
        logger.debug(f"  __name__: {func.__name__}")
        logger.debug(f"  __module__: {func.__module__}")
        logger.debug(f"  __qualname__: {func.__qualname__}")

        # Code object details
        code = func.__code__
        logger.debug(f"  __code__.co_filename: {code.co_filename}")
        logger.debug(f"  __code__.co_name: {code.co_name}")
        logger.debug(f"  __code__.co_firstlineno: {code.co_firstlineno}")
        logger.debug(f"  __code__.co_consts: {code.co_consts}")
        logger.debug(f"  __code__.co_names: {code.co_names}")
        logger.debug(f"  __code__.co_varnames: {code.co_varnames}")
        logger.debug(f"  __code__.co_code: {code.co_code.hex()}")
        logger.debug(f"  __code__.co_flags: {code.co_flags}")
        logger.debug(f"  __code__.co_stacksize: {code.co_stacksize}")
        logger.debug(f"  __code__.co_freevars: {code.co_freevars}")
        logger.debug(f"  __code__.co_cellvars: {code.co_cellvars}")

        # Function context
        logger.debug(f"  __globals__ keys: {sorted(func.__globals__.keys())}")
        logger.debug(f"  __closure__: {func.__closure__}")
        logger.debug(f"  __defaults__: {func.__defaults__}")
        logger.debug(f"  __kwdefaults__: {func.__kwdefaults__}")

    def create_function(name, tmp_path):
        # Create a temporary file with a function definition
        path = tmp_path / f"{name}.py"
        with open(path, "w") as f:
            f.write("""
def test_func():
    x = 42  # Add a constant
    y = "Hello"  # Add a string constant
    z = [1, 2, 3]  # Add a list constant
    return f"{y}, {x}! {z}"  # Use all constants
""")

        # Import the function from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.test_func

    # Create two identical functions in different files
    with tempfile.TemporaryDirectory() as tmp_dir:
        func1 = create_function("module1", Path(tmp_dir))
        func2 = create_function("module2", Path(tmp_dir))

        # Dump detailed information about both functions
        dump_function_details(func1, "Function 1")
        dump_function_details(func2, "Function 2")

        # Both should produce the same hash
        hash1 = _get_function_hash(func1)
        hash2 = _get_function_hash(func2)
        logger.debug(f"\nHash comparison:")
        logger.debug(f"  hash1: {hash1}")
        logger.debug(f"  hash2: {hash2}")

        # Print the location of the debug log
        print(f"\nDebug log written to: {debug_log}")

        # Read and print the debug log
        with open(debug_log, 'r') as f:
            print("\nDebug log contents:")
            print(f.read())

        assert hash1 == hash2, "Identical functions should produce the same hash"


def test_function_hash_closure_handling():
    """Test that functions with closure variables are handled correctly."""
    from bespokelabs.curator.prompter.prompter import _get_function_hash

    def create_function():
        outer_value = "outer"
        inner_value = "inner"
        def func():
            return f"{outer_value} {inner_value}"
        return func

    # Create two instances of the same function with closures
    func1 = create_function()
    func2 = create_function()

    # Both should produce the same hash
    hash1 = _get_function_hash(func1)
    hash2 = _get_function_hash(func2)
    assert hash1 == hash2, "Identical functions with closures should produce the same hash"

    # Verify that different closure values produce different hashes
    def create_function_different_closure():
        outer_value = "different"
        inner_value = "values"
        def func():
            return f"{outer_value} {inner_value}"
        return func

    func3 = create_function_different_closure()
    hash3 = _get_function_hash(func3)
    assert hash1 != hash3, "Functions with different closure values should produce different hashes"
