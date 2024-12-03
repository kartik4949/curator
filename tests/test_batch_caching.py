import os
from datasets import Dataset
import pytest

from bespokelabs.curator import Prompter
from bespokelabs.curator.request_processor.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.openai_online_request_processor import (
    OpenAIOnlineRequestProcessor,
)

def test_cache_with_api_key_changes(tmp_path):
    """Test that changing the API key handles batch reuse correctly."""
    api_key_1 = os.environ["openai_key_3"]
    api_key_2 = os.environ["openai_key_4"]

    def prompt_func():
        return "Say '1'. Do not explain."

    # Create minimal dataset with just 2 items to minimize API usage
    dataset = Dataset.from_dict({"text": ["item1", "item2"]})

    # Create first prompter with initial API key
    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",  # Using cheaper model for testing
        batch=True,
        batch_size=1,  # Minimal batch size
        api_key=api_key_1,
    )

    result1 = prompter1(dataset=dataset, working_dir=str(tmp_path))

    # Verify first run created response file
    first_cache_dir = next(d for d in tmp_path.glob("*") if d.is_dir())
    response_files = list(first_cache_dir.glob("responses_*.jsonl"))
    assert len(response_files) == 1, "Expected one completed batch in first run"
    assert "responses_0.jsonl" in {f.name for f in response_files}, "Batch 0 should be completed"

    # Create second prompter with different API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
        api_key=api_key_2,
    )
    result2 = prompter2(dataset=dataset, working_dir=str(tmp_path))

    # Verify second run created new cache directory
    second_cache_dir = next(d for d in tmp_path.glob("*") if d != first_cache_dir and d.is_dir())
    assert second_cache_dir.exists(), "Second run should create new cache directory with different API key"


def test_cache_with_same_api_key(tmp_path):
    """Test that using the same API key reuses the cache."""
    api_key = os.environ["openai_key_3"]

    def prompt_func():
        return "Say '1'. Do not explain."

    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
        api_key=api_key,
    )
    result1 = prompter1(working_dir=str(tmp_path))

    # Create a new prompter with the same API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
        api_key=api_key,
    )
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"


def test_cache_ignores_api_key_in_non_batch_mode(tmp_path):
    """Test that API key changes don't affect cache in non-batch mode."""
    api_key_1 = os.environ["openai_key_3"]
    api_key_2 = os.environ["openai_key_4"]

    def prompt_func():
        return "Say '1'. Do not explain."

    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
        api_key=api_key_1,
    )
    result1 = prompter1(working_dir=str(tmp_path))

    # Create second prompter with different API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
        api_key=api_key_2,
    )
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
