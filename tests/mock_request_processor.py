from typing import Optional, Iterable, Dict, Any
from datasets import Dataset
import json
import os
from datetime import datetime
from pathlib import Path
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericResponse,
    GenericRequest,
)
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter


class MockRequestProcessor(BaseRequestProcessor):
    """Mock request processor for testing cache hash generation."""

    def __init__(
        self,
        model: str = "gpt-4",
        batch_size: int = 1,
        api_key: str = "test_key",
        batch_mode: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(batch_size)
        self.model = model
        self.api_key = api_key
        self.batch_mode = batch_mode
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.calls = []
        self.incomplete_batches = set()  # Track which batch indices should be incomplete
        self.processed_batches = set()  # Track which batch indices have been processed

    def get_rate_limits(self) -> dict:
        """Return mock rate limits."""
        return {"requests_per_minute": 1000, "tokens_per_minute": 100000}

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create a mock API-specific request."""
        return {
            "model": self.model,
            "messages": generic_request.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

    def _create_request(
        self, messages: Dict[str, Any], original_row: Optional[Dict[str, Any]] = None
    ) -> GenericRequest:
        """Create a mock request."""
        return GenericRequest(
            messages=messages,
            original_row=original_row or {},
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )

    def _process_request(self, request: GenericRequest) -> GenericResponse:
        """Process a mock request."""
        now = datetime.now()
        raw_response = {
            "id": "mock-response-id",
            "object": "chat.completion",
            "created": int(now.timestamp()),
            "model": request.model,
            "choices": [{"message": {"content": "mock response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return GenericResponse(
            generic_request=request,
            response_message="mock response",
            response_errors=None,
            raw_response=raw_response,
            created_at=now,
            finished_at=now,
        )

    def run(
        self,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
        parse_func_hash: str = None,
        prompt_formatter: PromptFormatter = None,
    ) -> Dataset:
        """Mock run method that creates response files and returns dataset."""
        self.calls.append(
            {
                "dataset": dataset,
                "working_dir": working_dir,
                "parse_func_hash": parse_func_hash,
            }
        )

        # Create working directory if it doesn't exist
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)

            # Create request files
            request_files = self.create_request_files(dataset, working_dir, prompt_formatter)

            # Create response files for each request file
            for i, request_file in enumerate(request_files):
                response_file = os.path.join(working_dir, f"responses_{i}.jsonl")

                # Check if response file already exists in any cache directory
                parent_dir = Path(os.path.dirname(working_dir))
                cache_dirs = [d for d in parent_dir.glob("*") if d.is_dir()]
                existing_response = None
                for cache_dir in cache_dirs:
                    potential_file = os.path.join(cache_dir, f"responses_{i}.jsonl")
                    if os.path.exists(potential_file):
                        existing_response = potential_file
                        break

                # Skip if this batch is marked as incomplete and no existing response
                if i in self.incomplete_batches and not existing_response:
                    continue

                # If we have an existing response, copy it
                if existing_response:
                    import shutil

                    shutil.copy2(existing_response, response_file)
                    self.processed_batches.add(i)
                    continue

                # Process new batch
                self.processed_batches.add(i)
                with open(request_file, "r") as rf, open(response_file, "w") as wf:
                    for line in rf:
                        request = GenericRequest.model_validate_json(line)
                        response = self._process_request(request)
                        wf.write(response.model_dump_json() + "\n")

            # Create the dataset file if all batches are complete
            if not self.incomplete_batches or all(
                i in self.processed_batches for i in range(len(request_files))
            ):
                return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

        if dataset is None:
            return Dataset.from_dict({"response": ["mock response"]})
        return dataset

    def mark_batch_incomplete(self, batch_index: int):
        """Mark a specific batch as incomplete."""
        self.incomplete_batches.add(batch_index)
