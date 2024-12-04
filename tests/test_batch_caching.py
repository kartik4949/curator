import os
import logging
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    api_key_1 = os.environ["openai_key_3"]
    api_key_2 = os.environ["openai_key_4"]

    def prompt_func():
        return "Say '1'. Do not explain."

    # Create minimal dataset with just 2 items to minimize API usage
    dataset = Dataset.from_dict({"text": ["item1", "item2"]})

    # Create first request processor with initial API key
    request_processor1 = OpenAIBatchRequestProcessor(
        model="gpt-4o-mini",  # Using cost-efficient model
        batch_size=1,  # Minimal batch size
        api_key=api_key_1,
    )

    # Create first prompter
    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
    )
    prompter1._request_processor = request_processor1  # Set the request processor with API key
    logger.info("Starting first batch processing run with API key 1")
    result1 = prompter1(dataset=dataset, working_dir=str(tmp_path))
    logger.info("Completed first batch processing run")

    # Verify first run created response files
    cache_dir = next(d for d in tmp_path.glob("*") if d.is_dir())
    response_files = list(cache_dir.glob("responses_*.jsonl"))
    assert len(response_files) == 2, "Expected two completed batches in first run"
    assert "responses_0.jsonl" in {f.name for f in response_files}, "Batch 0 should be completed"
    assert "responses_1.jsonl" in {f.name for f in response_files}, "Batch 1 should be completed"

    # Create second request processor with different API key
    request_processor2 = OpenAIBatchRequestProcessor(
        model="gpt-4o-mini",
        batch_size=1,
        api_key=api_key_2,
    )

    # Create second prompter
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
    )
    prompter2._request_processor = request_processor2  # Set the request processor with API key
    logger.info("Starting second batch processing run with API key 2")
    result2 = prompter2(dataset=dataset, working_dir=str(tmp_path))
    logger.info("Completed second batch processing run")

    # Verify second run reused the same cache directory and didn't resubmit completed batches
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, "Should reuse the same cache directory across API keys"

    # Verify no new response files were created (reusing existing ones)
    response_files_after = list(cache_dir.glob("responses_*.jsonl"))
    assert (
        len(response_files_after) == 2
    ), "Should not create new response files for completed batches"


def test_cache_with_same_api_key(tmp_path):
    """Test that using the same API key reuses the cache."""
    api_key = os.environ["openai_key_3"]

    def prompt_func():
        return "Say '1'. Do not explain."

    # Create request processor
    request_processor = OpenAIBatchRequestProcessor(
        model="gpt-4o-mini",
        batch_size=1,
        api_key=api_key,
    )

    # Create first prompter
    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
    )
    prompter1._request_processor = request_processor
    result1 = prompter1(working_dir=str(tmp_path))

    # Create second prompter with same request processor
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=1,
    )
    prompter2._request_processor = request_processor
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

    # Create first request processor
    request_processor1 = OpenAIOnlineRequestProcessor(
        model="gpt-4o-mini",
        api_key=api_key_1,
    )

    # Create first prompter
    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
    )
    prompter1._request_processor = request_processor1
    result1 = prompter1(working_dir=str(tmp_path))

    # Create second request processor with different API key
    request_processor2 = OpenAIOnlineRequestProcessor(
        model="gpt-4o-mini",
        api_key=api_key_2,
    )

    # Create second prompter
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
    )
    prompter2._request_processor = request_processor2
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
