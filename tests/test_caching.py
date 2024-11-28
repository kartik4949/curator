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
