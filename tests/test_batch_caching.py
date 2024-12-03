import os
from datasets import Dataset
import pytest

from bespokelabs.curator import Prompter
from bespokelabs.curator.request_processor.openai_batch_request_processor import OpenAIBatchRequestProcessor
from bespokelabs.curator.request_processor.openai_online_request_processor import OpenAIOnlineRequestProcessor
from tests.mock_request_processor import MockRequestProcessor


@pytest.fixture
def mock_request_processor(monkeypatch):
    """Fixture to provide a mock request processor."""
    def mock_new(cls, *args, **kwargs):
        return MockRequestProcessor(
            model=kwargs.get('model'),
            batch_size=kwargs.get('batch_size', 1),
            api_key=kwargs.get('api_key', 'test_key_1'),
            batch_mode=True
        )
    monkeypatch.setattr(OpenAIBatchRequestProcessor, '__new__', mock_new)
    monkeypatch.setattr(OpenAIOnlineRequestProcessor, '__new__', mock_new)


def test_cache_with_api_key_changes(tmp_path, mock_request_processor):
    """Test that changing the API key alters the cache fingerprint."""
    def prompt_func():
        return "Say '1'. Do not explain."

    # Create first prompter with initial API key
    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=10,
    )
    result1 = prompter1(working_dir=str(tmp_path))

    # Create second prompter with different API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=10,
    )
    # Change the API key in the mock processor
    prompter2._request_processor.api_key = "test_key_2"
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 2, f"Expected 2 cache directories but found {len(cache_dirs)}"


def test_cache_with_same_api_key(tmp_path, mock_request_processor):
    """Test that using the same API key reuses the cache."""
    def prompt_func():
        return "Say '1'. Do not explain."

    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=10,
    )
    result1 = prompter1(working_dir=str(tmp_path))

    # Create a new prompter with the same API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=True,
        batch_size=10,
    )
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"


def test_cache_ignores_api_key_in_non_batch_mode(tmp_path, mock_request_processor):
    """Test that API key changes don't affect cache in non-batch mode."""
    def prompt_func():
        return "Say '1'. Do not explain."

    prompter1 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
    )
    result1 = prompter1(working_dir=str(tmp_path))

    # Create second prompter with different API key
    prompter2 = Prompter(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
        batch=False,
    )
    # Change the API key in the mock processor
    prompter2._request_processor.api_key = "test_key_2"
    result2 = prompter2(working_dir=str(tmp_path))

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.is_dir()]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
