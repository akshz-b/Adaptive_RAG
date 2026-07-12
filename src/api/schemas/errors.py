from typing import Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """
    Standard API error response model
    """

    detail: str
    error_code: Optional[str] = None
