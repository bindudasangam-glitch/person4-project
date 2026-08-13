from pydantic import BaseModel


class Claim(BaseModel):
    """
    Represents a single extracted claim.
    """

    id: int

    text: str

    source: str | None = None