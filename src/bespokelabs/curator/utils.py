import asyncio
import json
import logging
import typing as t
import uuid

from datasets import Dataset, load_dataset

from bespokelabs.curator import constants
from bespokelabs.curator.client import Client, _SessionStatus
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

logger = logging.getLogger(__name__)


def push_to_viewer(dataset: Dataset | str, hf_params: t.Optional[t.Dict] = None, chunk_size: int = 100):
    """Push a dataset to the Curator Viewer.

    Args:
        dataset (Dataset | str): The dataset to push to the Curator Viewer.
        hf_params: (dict): Huggingface parameters for load dataset.
        chunk_size (int): The size of the chunks to push the dataset in.

    Returns:
        str: The URL to view the data
    """
    if isinstance(dataset, str):
        logger.info(f"Downloading dataset {dataset} from huggingface")
        hf_params = {} or hf_params
        dataset = load_dataset(dataset, **hf_params)

    client = Client(hosted=True)
    uid = str(uuid.uuid4())
    metadata = {
        "run_hash": uid,
        "dataset_hash": uid,
        "prompt_func": "N/A",
        "model_name": "*",
        "response_format": "N/A",
        "batch_mode": False,
        "status": _SessionStatus.STARTED,
    }

    session_id = client.create_session(metadata, verbose=False)
    if not client.session:
        logger.error("Failed to create session.")
        raise Exception("Failed to create session.")

    view_url = f"{constants.PUBLIC_CURATOR_VIEWER_DATASET_URL}/{session_id}"
    logger.info(f"View your data live at: {view_url}")
    num_shards = (len(dataset) // chunk_size) + 1

    async def send_responses():
        async def send_row(idx, row):
            response_data = {"parsed_response_message": [row]}
            response_data_json = json.dumps(response_data)
            await client.stream_response(response_data_json, idx)

        async def process_shard(shard_idx):
            shard = dataset.shard(num_shards=num_shards, index=shard_idx)
            tasks = [send_row(idx, row) for idx, row in enumerate(shard, start=shard_idx * len(shard))]
            await asyncio.gather(*tasks)

        for shard_idx in range(num_shards):
            await process_shard(shard_idx)

        await client.session_completed()

    run_in_event_loop(send_responses())
    return view_url
