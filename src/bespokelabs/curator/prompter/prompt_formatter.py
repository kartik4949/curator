import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from bespokelabs.curator.request_processor.generic_request import GenericRequest

T = TypeVar("T")


class PromptFormatter:
    model_name: str
    prompt_func: Callable[[Union[Dict[str, Any], BaseModel]], Dict[str, str]]
    parse_func: Optional[
        Callable[
            [Union[Dict[str, Any], BaseModel], Union[Dict[str, Any], BaseModel]], T
        ]
    ] = None
    response_format: Optional[Type[BaseModel]] = None
    n: int = 1

    def __init__(
        self,
        model_name: str,
        prompt_func: Callable[[Union[Dict[str, Any], BaseModel]], Dict[str, str]],
        parse_func: Optional[
            Callable[
                [Union[Dict[str, Any], BaseModel], Union[Dict[str, Any], BaseModel]], T
            ]
        ] = None,
        response_format: Optional[Type[BaseModel]] = None,
        n: int = 1,
    ):
        self.model_name = model_name
        self.prompt_func = prompt_func
        self.parse_func = parse_func
        self.response_format = response_format
        self.n = n

    def get_generic_request(
        self, row: Dict[str, Any] | BaseModel, idx: int
    ) -> GenericRequest:
        """Format the request object based off Prompter attributes."""
        sig = inspect.signature(self.prompt_func)
        if len(sig.parameters) == 0:
            prompts = self.prompt_func()
        elif len(sig.parameters) == 1:
            prompts = self.prompt_func(row)
        else:
            raise ValueError(
                f"Prompting function {self.prompt_func} must have 0 or 1 arguments."
            )

        if isinstance(prompts, str):
            messages = [{"role": "user", "content": prompts}]
        else:
            # TODO(Ryan): Add validation here
            messages = prompts

        # Convert BaseModel to dict for serialization
        if isinstance(row, BaseModel):
            row = row.model_dump()

        return GenericRequest(
            model=self.model_name,
            messages=messages,
            row=row,
            row_idx=idx,
            metadata=prompts,
            response_format=self.response_format,
            n=self.n,
        )
