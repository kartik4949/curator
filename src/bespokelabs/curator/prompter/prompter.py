"""Curator: Bespoke Labs Synthetic Data Generation Library."""

import inspect
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar, Union
import types

import dill
from datasets import Dataset
from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.openai_online_request_processor import (
    OpenAIOnlineRequestProcessor,
)
from bespokelabs.curator.request_processor.litellm_online_request_processor import (
    LiteLLMOnlineRequestProcessor,
)

_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
T = TypeVar("T")

logger = logging.getLogger(__name__)


class Prompter:
    """Interface for prompting LLMs."""

    @staticmethod
    def _determine_backend(
        model_name: str, response_format: Optional[Type[BaseModel]] = None
    ) -> str:
        """Determine which backend to use based on model name and response format.

        Args:
            model_name (str): Name of the model
            response_format (Optional[Type[BaseModel]]): Response format if specified

        Returns:
            str: Backend to use ("openai" or "litellm")
        """
        model_name = model_name.lower()

        # GPT-4o models with response format should use OpenAI
        if (
            response_format
            and OpenAIOnlineRequestProcessor(model_name).check_structured_output_support()
        ):
            logger.info(f"Requesting structured output from {model_name}, using OpenAI backend")
            return "openai"

        # GPT models and O1 models without response format should use OpenAI
        if not response_format and any(x in model_name for x in ["gpt-", "o1-preview", "o1-mini"]):
            logger.info(f"Requesting text output from {model_name}, using OpenAI backend")
            return "openai"

        # Default to LiteLLM for all other cases
        logger.info(
            f"Requesting {f'structured' if response_format else 'text'} output from {model_name}, using LiteLLM backend"
        )
        return "litellm"

    def __init__(
        self,
        model_name: str,
        prompt_func: Callable[[Union[Dict[str, Any], BaseModel]], Dict[str, str]],
        parse_func: Optional[
            Callable[
                [
                    Union[Dict[str, Any], BaseModel],
                    Union[Dict[str, Any], BaseModel],
                ],
                T,
            ]
        ] = None,
        response_format: Optional[Type[BaseModel]] = None,
        backend: Optional[str] = None,
        batch: bool = False,
        batch_size: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        delete_successful_batch_files: bool = True,
        delete_failed_batch_files: bool = False,  # To allow users to debug failed batches
    ):
        """Initialize a Prompter.

        Args:
            model_name (str): The name of the LLM to use
            prompt_func (Callable[[Dict[str, Any]], Union[str, List[Dict[str, Any]]]]): A function that takes a single row
                and returns either a string (assumed to be a user prompt) or messages list
            parse_func (Callable[[Dict[str, Any], Any], T]): A function that takes the input row and
                response object and returns the parsed output
            response_format (Optional[Type[BaseModel]]): A Pydantic model specifying the
                response format from the LLM.
            backend (Optional[str]): The backend to use ("openai" or "litellm"). If None, will be auto-determined
            batch (bool): Whether to use batch processing
            batch_size (Optional[int]): The size of the batch to use, only used if batch is True
            temperature (Optional[float]): The temperature to use for the LLM, only used if batch is False
            top_p (Optional[float]): The top_p to use for the LLM, only used if batch is False
            presence_penalty (Optional[float]): The presence_penalty to use for the LLM, only used if batch is False
            frequency_penalty (Optional[float]): The frequency_penalty to use for the LLM, only used if batch is False
        """
        prompt_sig = inspect.signature(prompt_func)
        if len(prompt_sig.parameters) > 1:
            raise ValueError(
                f"prompt_func must take one argument or less, got {len(prompt_sig.parameters)}"
            )

        if parse_func is not None:
            parse_sig = inspect.signature(parse_func)
            if len(parse_sig.parameters) != 2:
                raise ValueError(
                    f"parse_func must take exactly 2 arguments, got {len(parse_sig.parameters)}"
                )

        self.prompt_formatter = PromptFormatter(
            model_name, prompt_func, parse_func, response_format
        )
        self.batch_mode = batch

        # Auto-determine backend if not specified
        # Use provided backend or auto-determine based on model and format
        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._determine_backend(model_name, response_format)

        # Select request processor based on backend
        if self.backend == "openai":
            if batch:
                if batch_size is None:
                    batch_size = 1_000
                    logger.info(
                        f"batch=True but no batch_size provided, using default batch_size of {batch_size:,}"
                    )
                self._request_processor = OpenAIBatchRequestProcessor(
                    model=model_name,
                    batch_size=batch_size,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    delete_successful_batch_files=delete_successful_batch_files,
                    delete_failed_batch_files=delete_failed_batch_files,
                )
            else:
                if batch_size is not None:
                    logger.warning(
                        f"Prompter argument `batch_size` {batch_size} is ignored because `batch` is False"
                    )
                self._request_processor = OpenAIOnlineRequestProcessor(
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
        elif self.backend == "litellm":
            if batch:
                logger.warning(
                    "Batch mode is not supported with LiteLLM backend, ignoring batch=True"
                )
            self._request_processor = LiteLLMOnlineRequestProcessor(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def __call__(self, dataset: Optional[Iterable] = None, working_dir: str = None) -> Dataset:
        """
        Run completions on a dataset.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.
        """
        return self._completions(self._request_processor, dataset, working_dir)

    def _completions(
        self,
        request_processor: BaseRequestProcessor,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
    ) -> Dataset:
        """
        Apply structured completions in parallel to a dataset using specified model and
        prompts.

        Args:
            dataset (Iterable): A dataset consisting of a list of items to apply completions
            prompter (Prompter): A Prompter that contains the logic for formatting each
                item in the dataset
            working_dir (str): The working directory to save the requests.jsonl, responses.jsonl, and dataset.arrow files.

        Returns:
            Iterable: A list of structured outputs from the completions
        """
        # NOTE(Ryan): We convert from iterable to Dataset because Dataset has random access via row_idx
        if not isinstance(dataset, Dataset) and dataset is not None:
            dataset = Dataset.from_generator(dataset)

        if self is None:
            raise ValueError("Prompter must be provided")

        if working_dir is None:
            curator_cache_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        else:
            curator_cache_dir = working_dir

        dataset_hash = dataset._fingerprint if dataset is not None else xxh64("").hexdigest()

        prompt_func_hash = _get_function_hash(self.prompt_formatter.prompt_func)

        # Used to name the dataset .arrow file, but not the cache directory name
        # Modifying `parse_func` creates a new dataset file from cached responses
        parse_func_hash = _get_function_hash(self.prompt_formatter.parse_func)

        fingerprint_str = "_".join(
            [
                str(dataset_hash),
                str(prompt_func_hash),
                str(self.prompt_formatter.model_name),
                str(
                    self.prompt_formatter.response_format.schema_json()
                    if self.prompt_formatter.response_format
                    else "text"
                ),
                str(self.batch_mode),
                str(self.backend),
            ]
        )

        fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
        logger.debug(f"Curator Cache Fingerprint: {fingerprint}")

        metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
        metadata_db = MetadataDB(metadata_db_path)

        # Get the source code of the prompt function
        prompt_func_source = _get_function_source(self.prompt_formatter.prompt_func)
        if self.prompt_formatter.parse_func is not None:
            parse_func_source = _get_function_source(self.prompt_formatter.parse_func)
        else:
            parse_func_source = ""

        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "prompt_func": prompt_func_source,
            "parse_func": parse_func_source,
            "model_name": self.prompt_formatter.model_name,
            "response_format": (
                self.prompt_formatter.response_format.schema_json()
                if self.prompt_formatter.response_format
                else "text"
            ),
            "run_hash": fingerprint,
            "batch_mode": self.batch_mode,
        }
        metadata_db.store_metadata(metadata_dict)

        dataset = request_processor.run(
            dataset=dataset,
            working_dir=os.path.join(curator_cache_dir, fingerprint),
            parse_func_hash=parse_func_hash,
            prompt_formatter=self.prompt_formatter,
        )

        return dataset


class PathIndependentPickler(dill.Pickler):
    """A custom pickler that ensures consistent function serialization across different file paths."""

    def save_code(self, obj):
        """Override save_code to standardize the file path and line numbers."""
        # Create a copy of the code object with standardized filename and line number
        # Standardize constants by converting tuples to lists and sorting where possible
        standardized_consts = []
        for const in obj.co_consts:
            if isinstance(const, (tuple, list, set, frozenset)):
                # Convert to sorted list for consistent ordering
                standardized_consts.append(sorted(const))
            elif isinstance(const, types.CodeType):
                # Recursively standardize nested code objects
                file = BytesIO()
                pickler = PathIndependentPickler(file, recurse=True)
                pickler.save_code(const)
                standardized_consts.append(file.getvalue())
            else:
                standardized_consts.append(const)

        code = types.CodeType(
            obj.co_argcount,
            obj.co_posonlyargcount,
            obj.co_kwonlyargcount,
            obj.co_nlocals,
            obj.co_stacksize,
            obj.co_flags,  # Keep flags as-is to preserve function type information
            obj.co_code,  # Keep raw bytecode
            tuple(standardized_consts),  # Use standardized constants
            tuple(sorted(obj.co_names)),  # Sort names for consistent ordering
            tuple(sorted(obj.co_varnames)),  # Sort varnames for consistent ordering
            "<standardized>",  # Standardize filename
            obj.co_name,
            1,  # Standardize line number
            obj.co_lnotab,
            tuple(sorted(obj.co_freevars)),  # Sort freevars for consistent ordering
            tuple(sorted(obj.co_cellvars)),  # Sort cellvars for consistent ordering
        )
        # Use dill's save_reduce to properly handle the code object
        self.save_reduce(
            types.CodeType,
            args=(
                code.co_argcount,
                code.co_posonlyargcount,
                code.co_kwonlyargcount,
                code.co_nlocals,
                code.co_stacksize,
                code.co_flags,
                code.co_code,
                tuple(standardized_consts),  # Use standardized constants
                tuple(sorted(code.co_names)),  # Sort names for consistent ordering
                tuple(sorted(code.co_varnames)),  # Sort varnames for consistent ordering
                "<standardized>",
                code.co_name,
                1,
                code.co_lnotab,
                tuple(sorted(code.co_freevars)),  # Sort freevars for consistent ordering
                tuple(sorted(code.co_cellvars)),  # Sort cellvars for consistent ordering
            ),
            obj=obj,
        )


def _get_function_source(func) -> str:
    """Get the source code of a function."""
    if func is None:
        return ""
    try:
        return inspect.getsource(func)
    except (TypeError, OSError):
        return ""


def _get_function_hash(func) -> str:
    """Get a consistent hash of a function's essential components."""
    if func is None:
        return xxh64("").hexdigest()

    file = BytesIO()
    pickler = PathIndependentPickler(file, recurse=True)
    pickler.dump(func)
    return xxh64(file.getvalue()).hexdigest()
