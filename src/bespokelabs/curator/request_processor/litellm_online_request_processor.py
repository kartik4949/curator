import json
import logging
import os
import time
import asyncio
import resource
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

import aiohttp
import litellm
import instructor
from tqdm import tqdm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import TokenUsage

import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class StatusTracker:
    """Tracks the status of all requests."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0
    pbar: Optional[tqdm] = None
    
    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors - Rate Limit: {self.num_rate_limit_errors}, "
            f"API: {self.num_api_errors}, Other: {self.num_other_errors}"
        )

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata."""
    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    token_consumption: int  # Estimated token usage
    attempts_left: int
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        client: Any,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Calls the LiteLLM API and saves results."""
        try:
            if self.generic_request.response_format:
                response, completion_obj = await client.chat.completions.create_with_completion(
                    **self.api_specific_request,
                    response_model=self.prompt_formatter.response_format,
                    timeout=60.0
                )
                response_message = response.model_dump() if hasattr(response, 'model_dump') else response
            else:
                completion_obj = await litellm.acompletion(**self.api_specific_request, timeout=60.0)
                response_message = completion_obj.choices[0].message.content

            # Extract token usage
            usage = completion_obj.usage if hasattr(completion_obj, 'usage') else {}
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

            # Calculate cost using litellm
            cost = litellm.completion_cost(
                completion_response=completion_obj.model_dump()
            ) if hasattr(completion_obj, 'model_dump') else 0

            # Create and save response
            generic_response = GenericResponse(
                response_message=response_message,
                response_errors=None,
                raw_request=self.api_specific_request,
                raw_response=completion_obj.model_dump() if hasattr(completion_obj, 'model_dump') else completion_obj,
                generic_request=self.generic_request,
                created_at=self.created_at,
                finished_at=datetime.now(),
                token_usage=token_usage,
                response_cost=cost
            )
            
            await self._append_generic_response(generic_response, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            if status_tracker.pbar:
                status_tracker.pbar.update(1)

        except Exception as e:
            is_rate_limit = any(phrase in str(e).lower() for phrase in ['rate limit', 'throttle', 'too many requests'])
            
            if is_rate_limit:
                status_tracker.num_rate_limit_errors += 1
                status_tracker.time_of_last_rate_limit_error = time.time()
            else:
                status_tracker.num_api_errors += 1
            
            logger.warning(f"Request {self.task_id} failed with error: {str(e)}")
            self.result.append(e)
            
            if self.attempts_left > 0:
                self.attempts_left -= 1
                retry_queue.put_nowait(self)
            else:
                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=[str(e) for e in self.result],
                    raw_request=self.api_specific_request,
                    raw_response=None,
                    generic_request=self.generic_request,
                    created_at=self.created_at,
                    finished_at=datetime.now(),
                )
                await self._append_generic_response(generic_response, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                if status_tracker.pbar:
                    status_tracker.pbar.update(1)

    async def _append_generic_response(self, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file."""
        json_string = json.dumps(data.model_dump(), default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")

class LiteLLMOnlineRequestProcessor(BaseRequestProcessor):
    """Rate-limited request processor for LiteLLM with structured outputs via instructor."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(batch_size=None)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.client = instructor.from_litellm(litellm.acompletion)

    def get_rate_limits(self) -> dict:
        """Get rate limits from LiteLLM."""
        logger.info(f"Getting rate limits for model: {self.model}")
        
        completion = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
        )
        
        headers = completion._hidden_params.get('additional_headers', {})
        
        rpm = int(headers.get('x-ratelimit-limit-requests', 3000))
        tpm = int(headers.get('x-ratelimit-limit-tokens', 150_000))
        print(f"Rate limits - Requests/min: {rpm}, Tokens/min: {tpm}")
        print()
        
        logger.info(f"Rate limits - Requests/min: {rpm}, Tokens/min: {tpm}")
        
        return {
            "max_requests_per_minute": rpm,
            "max_tokens_per_minute": tpm
        }

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create a LiteLLM-specific request."""
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        return request

    async def process_requests_from_file(
        self,
        generic_requests_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
    ) -> None:
        """Process requests with rate limiting."""
        # Increase file descriptor limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 10000), hard))

        # Get rate limits
        rate_limits = self.get_rate_limits()
        rpm = rate_limits["max_requests_per_minute"]
        tpm = rate_limits["max_tokens_per_minute"]

        # Initialize trackers and queues
        status_tracker = StatusTracker()
        queue_of_requests_to_retry = asyncio.Queue()
        available_request_capacity = rpm
        available_token_capacity = tpm
        last_update_time = time.time()
        
        # Track completed requests
        completed_request_ids = set()
        if os.path.exists(save_filepath) and resume:
            async with aiofiles.open(save_filepath, "r") as f:
                async for line in f:
                    response = GenericResponse.model_validate_json(line)
                    if not response.response_errors:
                        completed_request_ids.add(response.generic_request.original_row_idx)

        # Count total requests for progress bar
        total_requests = sum(1 for _ in open(generic_requests_filepath))
        status_tracker.pbar = tqdm(total=total_requests, desc="Processing LiteLLM requests")

        # Process requests
        connector = aiohttp.TCPConnector(limit=10 * rpm)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with aiofiles.open(generic_requests_filepath) as file:
                async for line in file:
                    generic_request = GenericRequest.model_validate_json(line)
                    
                    if resume and generic_request.original_row_idx in completed_request_ids:
                        status_tracker.num_tasks_already_completed += 1
                        status_tracker.pbar.update(1)
                        continue

                    while True:
                        # Update rate limit capacities
                        current_time = time.time()
                        seconds_since_update = current_time - last_update_time
                        available_request_capacity = min(
                            available_request_capacity + rpm * seconds_since_update / 60.0,
                            rpm
                        )
                        available_token_capacity = min(
                            available_token_capacity + tpm * seconds_since_update / 60.0,
                            tpm
                        )
                        last_update_time = current_time

                        # Check for rate limit cooldown
                        seconds_since_rate_limit = current_time - status_tracker.time_of_last_rate_limit_error
                        if seconds_since_rate_limit < 15:  # 15 second cooldown
                            await asyncio.sleep(15 - seconds_since_rate_limit)
                            continue

                        # Process request if capacity available
                        api_request = APIRequest(
                            task_id=status_tracker.num_tasks_started,
                            generic_request=generic_request,
                            api_specific_request=self.create_api_specific_request(generic_request),
                            token_consumption=1000,  # Estimate, could implement token counting
                            attempts_left=max_attempts,
                            prompt_formatter=self.prompt_formatter
                        )

                        if (available_request_capacity >= 1 and 
                            available_token_capacity >= api_request.token_consumption):
                            
                            available_request_capacity -= 1
                            available_token_capacity -= api_request.token_consumption
                            
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            
                            asyncio.create_task(
                                api_request.call_api(
                                    session=session,
                                    client=self.client,
                                    retry_queue=queue_of_requests_to_retry,
                                    save_filepath=save_filepath,
                                    status_tracker=status_tracker,
                                )
                            )
                            break
                        
                        await asyncio.sleep(0.001)  # Small sleep to prevent CPU spinning

            # Process retry queue
            while not queue_of_requests_to_retry.empty():
                request = await queue_of_requests_to_retry.get()
                
                while True:
                    current_time = time.time()
                    seconds_since_update = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity + rpm * seconds_since_update / 60.0,
                        rpm
                    )
                    available_token_capacity = min(
                        available_token_capacity + tpm * seconds_since_update / 60.0,
                        tpm
                    )
                    last_update_time = current_time

                    if (available_request_capacity >= 1 and 
                        available_token_capacity >= request.token_consumption):
                        
                        available_request_capacity -= 1
                        available_token_capacity -= request.token_consumption
                        
                        await request.call_api(
                            session=session,
                            client=self.client,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                        break
                    
                    await asyncio.sleep(0.001)

        # Cleanup
        status_tracker.pbar.close()
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")
        
        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} "
                f"requests failed. Errors logged to {save_filepath}"
            )
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. "
                "Consider running at a lower rate."
            )

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using LiteLLM with rate limiting."""
        self.prompt_formatter = prompt_formatter
        generic_requests_files = self.create_request_files(
            dataset, working_dir, prompt_formatter
        )
        
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl"
            for i in range(len(generic_requests_files))
        ]

        # Process each request file
        for request_file, response_file in zip(
            generic_requests_files, generic_responses_files
        ):
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_requests_filepath=request_file,
                    save_filepath=response_file,
                    max_attempts=5,
                    resume=True,
                )
            )

        # Create the final dataset from responses
        return self.create_dataset_files(
            working_dir=working_dir,
            parse_func_hash=parse_func_hash,
            prompt_formatter=prompt_formatter
        )

    async def process_requests_from_file(
        self,
        generic_requests_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
    ) -> None:
        """Process requests with rate limiting."""
        # Get rate limits
        rate_limits = self.get_rate_limits()
        rpm = rate_limits["max_requests_per_minute"]
        tpm = rate_limits["max_tokens_per_minute"]

        # Initialize trackers and queues
        status_tracker = StatusTracker()
        queue_of_requests_to_retry = asyncio.Queue()
        available_request_capacity = rpm
        available_token_capacity = tpm
        last_update_time = time.time()
        
        # Track completed requests
        completed_request_ids = set()
        if os.path.exists(save_filepath) and resume:
            async with aiofiles.open(save_filepath, "r") as f:
                async for line in f:
                    response = GenericResponse.model_validate_json(line)
                    if not response.response_errors:
                        completed_request_ids.add(response.generic_request.original_row_idx)
                        status_tracker.num_tasks_already_completed += 1

        # Count total requests for progress bar
        total_requests = sum(1 for _ in open(generic_requests_filepath))
        status_tracker.pbar = tqdm(
            total=total_requests,
            initial=status_tracker.num_tasks_already_completed,
            desc="Processing LiteLLM requests"
        )

        # Process requests
        connector = aiohttp.TCPConnector(limit=10 * rpm)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create a list to track all tasks
            tasks = []
            
            async with aiofiles.open(generic_requests_filepath) as file:
                # Process each request in the file
                async for line in file:
                    generic_request = GenericRequest.model_validate_json(line)
                    
                    # Skip completed requests if resuming
                    if resume and generic_request.original_row_idx in completed_request_ids:
                        continue

                    while True:
                        # Update rate limit capacities
                        current_time = time.time()
                        seconds_since_update = current_time - last_update_time
                        available_request_capacity = min(
                            available_request_capacity + rpm * seconds_since_update / 60.0,
                            rpm
                        )
                        available_token_capacity = min(
                            available_token_capacity + tpm * seconds_since_update / 60.0,
                            tpm
                        )
                        last_update_time = current_time

                        # Check for rate limit cooldown
                        seconds_since_rate_limit = current_time - status_tracker.time_of_last_rate_limit_error
                        if seconds_since_rate_limit < 15:  # 15 second cooldown
                            await asyncio.sleep(15 - seconds_since_rate_limit)
                            continue

                        # Create and process request if capacity available
                        token_estimate = self._estimate_tokens(generic_request)
                        if (available_request_capacity >= 1 and 
                            available_token_capacity >= token_estimate):
                            
                            # Update capacities
                            available_request_capacity -= 1
                            available_token_capacity -= token_estimate
                            
                            # Create request
                            api_request = APIRequest(
                                task_id=status_tracker.num_tasks_started,
                                generic_request=generic_request,
                                api_specific_request=self.create_api_specific_request(generic_request),
                                token_consumption=token_estimate,
                                attempts_left=max_attempts,
                                prompt_formatter=self.prompt_formatter
                            )
                            
                            # Create and track task
                            task = asyncio.create_task(
                                api_request.call_api(
                                    session=session,
                                    client=self.client,
                                    retry_queue=queue_of_requests_to_retry,
                                    save_filepath=save_filepath,
                                    status_tracker=status_tracker,
                                )
                            )
                            tasks.append(task)
                            
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            break
                        
                        await asyncio.sleep(0.001)  # Prevent CPU spinning

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

            # Process retry queue
            while not queue_of_requests_to_retry.empty():
                request = await queue_of_requests_to_retry.get()
                
                while True:
                    current_time = time.time()
                    seconds_since_update = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity + rpm * seconds_since_update / 60.0,
                        rpm
                    )
                    available_token_capacity = min(
                        available_token_capacity + tpm * seconds_since_update / 60.0,
                        tpm
                    )
                    last_update_time = current_time

                    if (available_request_capacity >= 1 and 
                        available_token_capacity >= request.token_consumption):
                        
                        available_request_capacity -= 1
                        available_token_capacity -= request.token_consumption
                        
                        await request.call_api(
                            session=session,
                            client=self.client,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                        break
                    
                    await asyncio.sleep(0.001)

        # Cleanup and logging
        status_tracker.pbar.close()
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")
        
        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} "
                f"requests failed. Errors logged to {save_filepath}"
            )
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. "
                "Consider running at a lower rate."
            )

    def _estimate_tokens(self, generic_request: GenericRequest) -> int:
        """Estimate token count for a request. Could be improved with tiktoken."""
        # Simple estimation for now - could implement more accurate counting
        total_chars = sum(len(msg["content"]) for msg in generic_request.messages)
        return total_chars // 4  # Rough estimate of 4 chars per token