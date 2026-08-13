from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request model for user query.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User question submitted to the system."
    )