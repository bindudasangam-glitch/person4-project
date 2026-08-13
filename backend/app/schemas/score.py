from pydantic import BaseModel


class HallucinationScore(BaseModel):
    """
    Hallucination scoring result.
    """

    trust_score: float

    hallucination_probability: float

    reliability_score: float