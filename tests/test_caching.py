from datasets import Dataset

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


def test_file_path_independence(tmp_path):
    """Test that identical functions in different files produce the same hash."""
    def create_function(name):
        # Create a temporary file with a function definition
        path = tmp_path / f"{name}.py"
        with open(path, "w") as f:
            f.write("""
def test_func():
    return "Hello, World!"
""")

        # Import the function from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.test_func

    # Create two identical functions in different files
    func1 = create_function("module1")
    func2 = create_function("module2")

    # Create prompters with these functions
    prompter1 = Prompter(
        prompt_func=func1,
        model_name="gpt-4o-mini",
    )
    prompter2 = Prompter(
        prompt_func=func2,
        model_name="gpt-4o-mini",
    )

    # Both should use the same cache
    result1 = prompter1(working_dir=str(tmp_path))
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db and our temporary python files
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db" and not d.name.endswith(".py")]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
    assert result1.to_pandas().iloc[0]["response"] == result2.to_pandas().iloc[0]["response"]


def test_closure_variable_handling(tmp_path):
    """Test that functions with closure variables are handled correctly."""
    outer_value = "outer"

    def create_function():
        inner_value = "inner"
        def func():
            return f"Say '{outer_value} {inner_value}'. Do not explain."
        return func

    # Create two instances of the same function with closures
    func1 = create_function()
    func2 = create_function()

    prompter1 = Prompter(
        prompt_func=func1,
        model_name="gpt-4o-mini",
    )
    prompter2 = Prompter(
        prompt_func=func2,
        model_name="gpt-4o-mini",
    )

    # Both should use the same cache since they're identical
    result1 = prompter1(working_dir=str(tmp_path))
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
    assert result1.to_pandas().iloc[0]["response"] == result2.to_pandas().iloc[0]["response"]
    assert "outer inner" in result1.to_pandas().iloc[0]["response"]
