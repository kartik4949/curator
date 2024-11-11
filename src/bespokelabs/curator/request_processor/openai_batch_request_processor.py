import asyncio
import json
import logging
import os
from typing import Callable, Dict, Optional, TypeVar
from openai import AsyncOpenAI
import aiofiles
from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
)
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from tqdm import tqdm

T = TypeVar("T")
logger = logging.getLogger(__name__)


class OpenAIBatchRequestProcessor(BaseRequestProcessor):
    def __init__(
        self,
        batch_size: int = 1000,
        model: str = "gpt-4o-mini",
        check_interval: int = 10,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        url: str = "https://api.openai.com/v1/chat/completions",
    ):
        super().__init__(batch_size)
        self.url: str = url
        self.api_key: str = api_key
        self.check_interval: int = check_interval

    def get_rate_limits(self) -> dict:
        """
        Function to get rate limits for a given annotator. Not available via response headers, so
        the following is based on tier 5 limits on Nov 6th, 2024.

        These rate limits vary per model
        and are determined by your organization's usage tier. View the following:
        https://platform.openai.com/docs/guides/rate-limits/usage-tiers
        https://platform.openai.com/settings/organization/limits

        Args:
            model (str): The model for which to get the rate limits.
            request_url (str): The request URL for which to get the rate limits.

        Returns:
            tuple[int, int]: A tuple containing the maximum number of requests and tokens per minute.
        """
        model_tpd = {
            "gpt-3.5-turbo": 5_000_000_000,
            "gpt-3.5-turbo-0125": 5_000_000_000,
            "gpt-3.5-turbo-1106": 5_000_000_000,
            "gpt-3.5-turbo-16k": 5_000_000_000,
            "gpt-3.5-turbo-instruct": 200_000,
            "gpt-3.5-turbo-instruct-0914": 200_000,
            "gpt-4": 150_000_000,
            "gpt-4-0613": 150_000_000,
            "gpt-4-turbo": 300_000_000,
            "gpt-4o": 10_000_000_000,
            "gpt-4o-mini": 15_000_000_000,
        }

        if self.model not in model_tpd:
            tpd = 1_000_000_000
        else:
            tpd = model_tpd[self.model]

        logger.info(
            f"Automatically set max_tokens_per_day to {tpd}, model: {self.model} "
        )

        rate_limits = {"max_tokens_per_day": tpd}

        return rate_limits

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """
        Creates a API-specific request body from a generic request body.

        Using the api_parallel_processor, we can store whatever we want in the metadata. We will store both the row and the index.
        This is so we can later construct the new dataset row.

        Returns:
            dict: API specific request body
        """
        # NOTE(Ryan): We can have a shared place that creates the body (since it is the same for both online and batch).
        body = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }
        if generic_request.response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    # TODO(ryan): not sure if this should be something else.
                    # TODO(ryan): also not sure if we should use strict: True
                    "name": "output_schema",
                    "schema": generic_request.response_format.model_json_schema(),
                },
            }
        if generic_request.n > 1:
            body["n"] = generic_request.n

        request = {
            "custom_id": str(generic_request.row_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return request

    def get_generic_responses(
        self, response: Dict, prompt_formatter: PromptFormatter, dataset: Dataset
    ) -> list[GenericResponse]:
        """
        Parses a API-specific response into a generic response body.
        Does error handling on the response.
        If there is an error, return None.

        IMPORTANT: In the generic response body you need to provide either the original dataset row OR the index of the row in the original dataset.

        Args:
            response: API-specific response

        Returns:
            list[dict]: List of generic response bodies with an extra field "metadata" which contains the original dataset row or the index of the row in the original dataset
        """
        request_id = response["id"]
        status_code = response["response"]["status_code"]

        # TODO(Ryan): Add error handling. This should handle error files from BatchAPI.
        if status_code != 200:
            logger.warning(
                f"Request {request_id} failed with status code {status_code}"
            )
            return None

        # NOTE(Ryan): can we actually parse the response into a an OpenAI ChatCompletions object? Easier to access fields?
        # TODO(Ryan): if you add token tokens to generic response, we can parse that here too, similar to my comment above we can do that in the shared place.
        generic_responses = []
        for choice in response["response"]["body"]["choices"]:
            content = choice["message"]["content"]
            row_idx = int(response["custom_id"])

            if prompt_formatter.response_format:
                content = json.loads(content)

            # NOTE(Ryan): So dicts that have objects that are not JSON serializable will be converted to strings.

            if dataset is None:
                dataset_row = dict()
            else:
                dataset_row = dataset[row_idx]

            generic_responses.append(
                GenericResponse(
                    response=content,
                    row_idx=row_idx,
                    row=dataset_row,
                    raw_response=response,
                )
            )
        return generic_responses

    async def asubmit_batch(self, batch_file: str) -> dict:
        async with aiofiles.open(batch_file, "rb") as file:
            file_content = await file.read()
            batch_file_upload = await self.async_client.files.create(
                file=file_content, purpose="batch"
            )

        logger.info(f"File uploaded: {batch_file_upload}")

        batch_object = await self.async_client.batches.create(
            input_file_id=batch_file_upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "request_file_name": batch_file
            },  # for easily mapping back later, NOTE(Ryan): can convert to the int or UUID later
        )
        logger.info(f"Batch request submitted, received batch object: {batch_object}")

        return batch_object

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)

        Returns:
            Dataset: Completed dataset
        """
        requests_files = self.create_request_files(
            dataset, working_dir, prompt_formatter
        )
        batch_objects_file = f"{working_dir}/batch_objects.jsonl"

        # TODO(Ryan): we should have an easy way to cancel all batches in batch_objects.jsonl if the user realized they made a mistake
        if os.path.exists(batch_objects_file):
            logger.warning(
                f"Batch objects file already exists, skipping batch submission and resuming: {batch_objects_file}"
            )
        else:
            # upload requests files and submit batches
            self.async_client = AsyncOpenAI()

            # asyncio gather preserves order
            async def submit_all_batches():
                tasks = [
                    self.asubmit_batch(requests_files[i])
                    for i in range(len(requests_files))
                ]
                return await asyncio.gather(*tasks)

            batch_objects = asyncio.run(submit_all_batches())

            with open(batch_objects_file, "w") as f:
                # NOTE(Ryan): we can also store the request_file_name in this object here, instead of in the metadata during batch submission. Can find a nice abstraction across other batch APIs (e.g. claude)
                for obj in batch_objects:
                    f.write(json.dumps(obj.model_dump()) + "\n")
            logger.info(f"Batch objects written to {batch_objects_file}")

        # TODO(Ryan): Actually do accounting for tokens, so rate limits enforced locally.
        # NOTE(Ryan): Although this isn't really practical since the limits are for an entire day and an entire organization. Maybe skip this and just recognize what a rate limit error for batching looks like (need to try this on a low tier account).
        # rate_limits = self.get_rate_limits()
        # tpd = rate_limits["max_tokens_per_day"]
        # token_encoding_name = get_token_encoding_name(self.model)

        # TODO(Ryan): based on the files that are downloaded, update completed_ids. If any are errors, try to resubmit (depending on error type).
        # TODO(Ryan): This creates responses_0.jsonl, responses_1.jsonl, etc. errors named same way? or errors_0.jsonl, errors_1.jsonl?
        # TODO(Ryan): retries, resubmits on lagging batches - need to study this a little closer
        # TODO(Ryan): likely can add some logic for smarter check_interval based on batch size and if the batch has started or not, fine to do a dumb ping for now
        batch_watcher = BatchWatcher(working_dir, check_interval=self.check_interval)

        asyncio.run(
            batch_watcher.watch(prompt_formatter, self.get_generic_responses, dataset)
        )

        dataset = self.create_dataset_files(dataset, working_dir, prompt_formatter)
        return dataset


class BatchWatcher:
    def __init__(self, working_dir: str, check_interval) -> None:
        """Initialize BatchWatcher with batch objects file and check interval.

        Args:
            batch_objects_file (str): Path to the batch objects JSON file.
            check_interval (int): Time interval (in seconds) to check batch status.
        """
        self.client = AsyncOpenAI()
        with open(f"{working_dir}/batch_objects.jsonl", "r") as f:
            self.batch_objects = [json.loads(line) for line in f]
        self.batch_ids = [obj["id"] for obj in self.batch_objects]
        self.batch_id_to_request_file_name = {
            obj["id"]: obj["metadata"]["request_file_name"]
            for obj in self.batch_objects
        }
        self.batches = []
        self.check_interval = check_interval
        self.working_dir = working_dir

    async def check_batch_status(self, batch_id: str) -> tuple[str, str]:
        """Check the status of a batch by its ID.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            tuple[str, str]: The batch ID and its status.
        """
        batch = await self.client.batches.retrieve(batch_id)
        logger.info(
            f"Batch {batch_id} status: {batch.status} requests: {batch.request_counts.completed}/{batch.request_counts.failed}/{batch.request_counts.total} completed/failed/total"
        )
        return batch_id, batch

    async def watch(
        self,
        prompt_formatter: PromptFormatter,
        get_generic_responses: Callable[[Dict], list[GenericResponse]],
        dataset: Dataset,
    ) -> None:
        """Monitor the status of batches until all are completed (includes successfully, failed, expired or cancelled)."""

        total_requests = 1 if dataset is None else len(dataset)

        completed_batches = {}
        pbar = tqdm(
            total=total_requests,
            desc="Completed OpenAI requests in batches",
            unit="request",
        )

        while len(completed_batches) < len(self.batch_ids):
            pbar.n = 0
            status_tasks = []
            for batch_id in self.batch_ids:
                if batch_id not in completed_batches:
                    status_tasks.append(self.check_batch_status(batch_id))
                else:
                    pbar.n = (
                        pbar.n
                        + completed_batches[batch_id].request_counts.completed
                        + completed_batches[batch_id].request_counts.failed
                    )

            batches = await asyncio.gather(*status_tasks)
            newly_completed_batches = []
            for batch_id, batch in batches:
                if batch.status in ["completed", "failed", "expired", "cancelled"]:
                    logger.info(
                        f"Batch {batch_id} processing finished with status: {batch.status}"
                    )
                    completed_batches[batch_id] = batch
                    newly_completed_batches.append(batch)

                pbar.n = (
                    pbar.n
                    + batch.request_counts.completed
                    + batch.request_counts.failed
                )

            pbar.refresh()

            # NOTE(Ryan): Now downloading after each check, instead of waiting until all are completed
            tasks = [
                self.download_batch_result_file(
                    batch, prompt_formatter, get_generic_responses, dataset
                )
                for batch in newly_completed_batches
            ]
            await asyncio.gather(*tasks)

            if len(completed_batches) < len(self.batch_ids):
                logger.info(
                    f"Batches fully finished: {len(completed_batches)}/{len(self.batch_ids)} Requests completed: {pbar.n}/{total_requests}"
                )
                logger.info(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        pbar.close()
        self.batches = completed_batches.values()

    async def download_batch_result_file(
        self,
        batch,
        prompt_formatter: PromptFormatter,
        get_generic_responses: Callable[[Dict], list[GenericResponse]],
        dataset: Dataset,
    ) -> str:
        """Download the result of a completed batch to file.

        Args:
            batch: The batch object to download results from.

        Returns:
            str: Path to the downloaded result file.
        """
        if batch.status == "completed" and batch.output_file_id:
            file_content = await self.client.files.content(batch.output_file_id)
        elif batch.status == "failed" and batch.error_file_id:
            file_content = await self.client.files.content(batch.error_file_id)
        elif batch.status == "cancelled" or batch.status == "expired":
            logger.warning(f"Batch {batch.id} was cancelled or expired")
            return None

        # NOTE(Ryan): This is so the naming is consistent with the request file naming
        request_file_idx = (
            self.batch_id_to_request_file_name[batch.id].split("/")[-1].split("_", 1)[1]
        )
        output_path = f"{self.working_dir}/responses_{request_file_idx}"
        with open(output_path, "w") as f:
            for raw_response in file_content.text.splitlines():
                # TODO(Ryan): We should abstract this out
                generic_responses = get_generic_responses(
                    json.loads(raw_response), prompt_formatter, dataset
                )
                for generic_response in generic_responses:
                    f.write(
                        json.dumps(generic_response.model_dump(), default=str) + "\n"
                    )
        return output_path
