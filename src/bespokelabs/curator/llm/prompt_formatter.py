import dataclasses
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from bespokelabs.curator.request_processor.generic_request import GenericRequest

T = TypeVar("T")
_DictOrBaseModel = Union[Dict[str, Any], BaseModel]


def _validate_messages(messages: list[dict]) -> None:
    """Validates that messages conform to the expected chat format.

    Args:
        messages: A list of message dictionaries to validate.

    Raises:
        ValueError: If messages don't meet the required format:
            - Must be a list of dictionaries
            - Each message must have 'role' and 'content' keys
            - Role must be one of: 'system', 'user', 'assistant'
    """
    valid_roles = {"system", "user", "assistant"}

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(
                "In the return value (a list) of the prompt_func, each "
                "message must be a dictionary"
            )

        if "role" not in msg or "content" not in msg:
            raise ValueError(
                "In the return value (a list) of the prompt_func, each "
                "message must contain 'role' and 'content' keys"
            )

        if msg["role"] not in valid_roles:
            raise ValueError(
                f"In the return value (a list) of the prompt_func, "
                f"each message role must be one of: {', '.join(sorted(valid_roles))}"
            )


@dataclasses.dataclass
class PromptFormatter:
    model_name: str
    prompt_func: Callable[[_DictOrBaseModel], Dict[str, str]]
    parse_func: Optional[Callable[[_DictOrBaseModel, _DictOrBaseModel], T]] = None
    response_format: Optional[Type[BaseModel]] = None

    def create_generic_request(self, row: _DictOrBaseModel, idx: int) -> GenericRequest:
        """Format the request object based off of `LLM` attributes."""
        sig = inspect.signature(self.prompt_func)
        if len(sig.parameters) == 0:
            prompts = self.prompt_func()
        elif len(sig.parameters) == 1:
            prompts = self.prompt_func(row)
        else:
            raise ValueError(f"Prompting function {self.prompt_func} must have 0 or 1 arguments.")

        if isinstance(prompts, str):
            messages = [{"role": "user", "content": prompts}]
        elif isinstance(prompts, list):
            _validate_messages(prompts)
            messages = prompts
        else:
            raise ValueError("The return value of the prompt_func must be a list of dictionaries.")

        # Convert BaseModel to dict for serialization
        if isinstance(row, BaseModel):
            row = row.model_dump()

        return GenericRequest(
            model=self.model_name,
            messages=messages,
            original_row=row,
            original_row_idx=idx,
            response_format=(
                self.response_format.model_json_schema() if self.response_format else None
            ),
        )
