from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class GenericRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    row: Dict[str, Any] | str
    row_idx: int
    response_format: Optional[Type[BaseModel]] = None
    n: int = 1
