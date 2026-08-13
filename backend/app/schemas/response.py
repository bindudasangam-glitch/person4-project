from pydantic import BaseModel

from app.schemas.claim import Claim
from app.schemas.score import HallucinationScore


class AnalysisResponse(BaseModel):
    """
    Final API response.
    """

    claims: list[Claim]

    score: HallucinationScore

    verdict: str