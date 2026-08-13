"""
Self-Correction Agent
=========================

Rewrites an LLM response to remove, replace, or flag claims that Person
1's :class:`~app.services.hallucination_detector.HallucinationDetector`
found to be unsupported or contradicted, using only text already present
in the evidence Person 1 and Person 2 retrieved -- never inventing new
factual content.

Inputs
------
* **Person 1 hallucination analysis** -- the batch of
  :class:`~app.models.claim_model.ClaimModel` objects (post-detection,
  so each carries its final ``verification_status``) together with the
  parallel list of
  :class:`~app.services.hallucination_detector.ClaimDetectionOutcome`
  produced by ``HallucinationDetector.detect()``.
* **Person 2 evidence retrieval results** -- an optional
  ``{claim_id: EvidenceBundle}`` mapping of richer, embedding-based
  evidence from :class:`~app.retrieval.retriever.Retriever`. When
  supplied, it is preferred over the (possibly lexical-overlap-only)
  evidence already embedded in each ``ClaimDetectionOutcome``, since it
  carries full source attribution and similarity scoring.

Correction policy
------------------
For each claim, based on its ``verification_status``:

* ``SUPPORTED``               -> left untouched (``CorrectionAction.NONE``).
* ``CONTRADICTED``            -> replaced with the best available verified
  evidence text (``CorrectionAction.REPLACED``), or, only if no evidence
  text is available at all, removed outright (``CorrectionAction.REMOVED``)
  -- never replaced with fabricated text.
* ``INSUFFICIENT_EVIDENCE``   -> left untouched but flagged as uncertain
  (``CorrectionAction.FLAGGED``); nothing is invented to fill the gap.
* ``UNVERIFIED``              -> treated the same as ``INSUFFICIENT_EVIDENCE``
  (defensive: detection should always resolve this, but the agent must
  never silently drop or fabricate for an unexpected status).

The agent performs surgical, sentence-level substring edits against the
original response text (a claim's ``text`` is the verbatim sentence span
``ClaimExtractor`` pulled it from), rather than regenerating the whole
response, so every part of the response the agent did not decide to
change is guaranteed to be byte-for-byte identical to the original --
this is what "preserve facts that were already correct" and "never
invent new information" mean in code, not just in intent.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from app.agents.exceptions import SelfCorrectionError
from app.core.logging import logger
from app.models.claim_model import ClaimModel, VerificationStatus
from app.models.workflow_models import (
    ClaimCorrection,
    CorrectionAction,
    CorrectionResult,
    WorkflowModelValidationError,
)
from app.services.hallucination_detector import ClaimDetectionOutcome

if TYPE_CHECKING:
    from app.models.evidence import EvidenceBundle

__all__ = ["SelfCorrectionAgent"]

_FALLBACK_EMPTY_RESPONSE_TEXT = (
    "[All statements in this response were removed due to lack of supporting evidence.]"
)


class SelfCorrectionAgent:
    """
    Produces a corrected version of an LLM response using only claims and
    evidence already validated by Person 1 and Person 2 -- no external
    LLM call, no fabricated text.

    Args:
        action_confidence_weights: Per-``CorrectionAction`` confidence
            contribution used to compute the overall
            ``CorrectionResult.correction_confidence`` (see
            :meth:`_compute_confidence`). Injected so callers/tests can
            tune or replace the default, transparent, rule-based weights
            without subclassing. Defaults to
            :attr:`DEFAULT_ACTION_CONFIDENCE_WEIGHTS`.

    Design notes
    ------------
    This agent deliberately performs no generative rewriting: every
    character that ends up in ``corrected_response`` either (a) already
    existed in the original response, or (b) already existed in a piece
    of evidence retrieved by Person 1/Person 2. This is what makes
    "never invent new information" an enforceable property of the code
    rather than a hope about an LLM's behavior.
    """

    #: Default per-action confidence weight, in [0, 1], used by
    #: :meth:`_compute_confidence`. Higher means the agent trusts that
    #: kind of correction more. ``REPLACED``/``REWORDED`` are the most
    #: trusted since they substitute in verified evidence text;
    #: ``FLAGGED`` is the least trusted since the response's wording is
    #: left unchanged despite being unconfirmed.
    DEFAULT_ACTION_CONFIDENCE_WEIGHTS: dict[CorrectionAction, float] = {
        CorrectionAction.NONE: 1.0,
        CorrectionAction.REPLACED: 0.9,
        CorrectionAction.REWORDED: 0.85,
        CorrectionAction.REMOVED: 0.75,
        CorrectionAction.FLAGGED: 0.5,
    }

    def __init__(self, action_confidence_weights: dict[CorrectionAction, float] | None = None) -> None:
        weights = action_confidence_weights or dict(self.DEFAULT_ACTION_CONFIDENCE_WEIGHTS)
        missing = set(CorrectionAction) - set(weights)
        if missing:
            raise SelfCorrectionError(
                f"action_confidence_weights is missing entries for: {sorted(a.value for a in missing)}."
            )
        for action, weight in weights.items():
            if not 0.0 <= weight <= 1.0:
                raise SelfCorrectionError(
                    f"action_confidence_weights[{action.value}] must be within [0.0, 1.0], got {weight}."
                )
        self._action_confidence_weights = weights

    def correct(
        self,
        response_text: str,
        claims: list[ClaimModel],
        detection_outcomes: list[ClaimDetectionOutcome],
        evidence_bundles: dict[int, "EvidenceBundle"] | None = None,
    ) -> CorrectionResult:
        """
        Generate a corrected response, rewriting only unsupported or
        contradicted claims and leaving everything else untouched.

        Args:
            response_text: The original, full LLM response text.
            claims: Post-detection claims from Person 1's
                ``ClaimExtractor``/``HallucinationDetector`` (each already
                carrying its final ``verification_status``).
            detection_outcomes: The parallel
                ``ClaimDetectionOutcome`` list from
                ``HallucinationDetector.detect()``, used as the fallback
                source of evidence text when ``evidence_bundles`` doesn't
                cover a given claim.
            evidence_bundles: Optional ``{claim_id: EvidenceBundle}``
                mapping of Person 2 evidence retrieval results, preferred
                over ``detection_outcomes``' embedded evidence when present.

        Returns:
            A :class:`~app.models.workflow_models.CorrectionResult`
            describing what (if anything) was changed.

        Raises:
            SelfCorrectionError: If ``response_text`` is empty, ``claims``
                is ``None``, or correction fails unexpectedly for a claim.
        """
        if response_text is None or not response_text.strip():
            raise SelfCorrectionError("Cannot correct an empty response_text.")
        if claims is None:
            raise SelfCorrectionError("claims must not be None.")

        outcome_by_claim_id = {outcome.claim_id: outcome for outcome in (detection_outcomes or [])}
        evidence_bundles = evidence_bundles or {}

        if not claims:
            logger.info("SelfCorrectionAgent: no claims supplied; returning the response unchanged.")
            return CorrectionResult(
                original_response=response_text,
                corrected_response=response_text,
                was_corrected=False,
                correction_confidence=1.0,
                corrections=(),
                explanation="No claims were extracted from the response; nothing to correct.",
            )

        running_text = response_text
        corrections: list[ClaimCorrection] = []

        for claim in sorted(claims, key=lambda c: c.id):
            outcome = outcome_by_claim_id.get(claim.id)

            try:
                action, corrected_text, reason = self._decide_action(claim, outcome, evidence_bundles)
            except SelfCorrectionError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize to domain error
                logger.exception("SelfCorrectionAgent failed to decide an action for claim %d.", claim.id)
                raise SelfCorrectionError(
                    f"Failed to decide a correction action for claim {claim.id}."
                ) from exc

            if action in (CorrectionAction.REPLACED, CorrectionAction.REWORDED, CorrectionAction.REMOVED):
                new_text, applied = self._apply_correction_to_text(
                    running_text, claim.text, action, corrected_text
                )
                if applied:
                    running_text = new_text
                else:
                    logger.warning(
                        "SelfCorrectionAgent could not locate claim %d's exact text in the "
                        "response for a safe substitution; downgrading to FLAGGED instead of "
                        "risking an unsafe or partial edit.",
                        claim.id,
                    )
                    action = CorrectionAction.FLAGGED
                    corrected_text = None
                    reason = (
                        f"{reason} (Exact claim text could not be located in the response for "
                        "a safe, surgical edit, so the original wording was left untouched and "
                        "flagged instead.)"
                    )

            try:
                corrections.append(
                    ClaimCorrection(
                        claim_id=claim.id,
                        original_text=claim.text,
                        action=action,
                        corrected_text=corrected_text,
                        reason=reason or "",
                    )
                )
            except WorkflowModelValidationError as exc:
                logger.exception("SelfCorrectionAgent produced an invalid ClaimCorrection for claim %d.", claim.id)
                raise SelfCorrectionError(f"Invalid correction produced for claim {claim.id}.") from exc

        if not running_text.strip():
            logger.warning(
                "SelfCorrectionAgent removed all content from the response; substituting a "
                "transparent placeholder rather than returning an empty response."
            )
            running_text = _FALLBACK_EMPTY_RESPONSE_TEXT

        was_corrected = any(correction.action is not CorrectionAction.NONE for correction in corrections)
        confidence = self._compute_confidence(corrections)
        explanation = self._build_explanation(corrections)

        try:
            result = CorrectionResult(
                original_response=response_text,
                corrected_response=running_text,
                was_corrected=was_corrected,
                correction_confidence=confidence,
                corrections=tuple(corrections),
                explanation=explanation,
            )
        except WorkflowModelValidationError as exc:
            logger.exception("SelfCorrectionAgent produced an invalid CorrectionResult.")
            raise SelfCorrectionError("Failed to construct a valid CorrectionResult.") from exc

        logger.info(
            "SelfCorrectionAgent complete: %d claim(s) processed, was_corrected=%s, confidence=%.3f.",
            len(corrections),
            was_corrected,
            confidence,
        )
        return result

    # ------------------------------------------------------------------ #
    # Per-claim decision logic
    # ------------------------------------------------------------------ #
    def _decide_action(
        self,
        claim: ClaimModel,
        outcome: ClaimDetectionOutcome | None,
        evidence_bundles: dict[int, "EvidenceBundle"],
    ) -> tuple[CorrectionAction, str | None, str]:
        """
        Decide what (if anything) should be done about a single claim.

        Returns:
            A ``(action, corrected_text, reason)`` tuple. ``corrected_text``
            is non-``None`` only when ``action`` is ``REPLACED`` or ``REWORDED``.
        """
        status = claim.verification_status

        if status is VerificationStatus.SUPPORTED:
            return (
                CorrectionAction.NONE,
                None,
                "Claim is supported by retrieved evidence; no correction needed.",
            )

        if status is VerificationStatus.CONTRADICTED:
            evidence_text = self._select_best_evidence_text(claim.id, outcome, evidence_bundles)
            if evidence_text:
                return (
                    CorrectionAction.REPLACED,
                    evidence_text,
                    "Claim contradicted retrieved evidence; replaced with verified evidence text.",
                )
            return (
                CorrectionAction.REMOVED,
                None,
                "Claim contradicted retrieved evidence, but no usable evidence text was "
                "available to substitute; removed the unsupported statement instead of "
                "inventing a replacement.",
            )

        if status is VerificationStatus.INSUFFICIENT_EVIDENCE:
            return (
                CorrectionAction.FLAGGED,
                None,
                "No sufficient evidence was found to confirm or refute this claim; flagged "
                "as uncertain rather than invented or removed without cause.",
            )

        # VerificationStatus.UNVERIFIED (defensive: detection should always
        # resolve claims to one of the three statuses above, but this
        # agent must never silently drop or fabricate for an unexpected status).
        return (
            CorrectionAction.FLAGGED,
            None,
            "Claim verification status is unresolved; flagged as uncertain.",
        )

    @staticmethod
    def _select_best_evidence_text(
        claim_id: int,
        outcome: ClaimDetectionOutcome | None,
        evidence_bundles: dict[int, "EvidenceBundle"],
    ) -> str | None:
        """
        Pick the single best piece of verified evidence text available for
        a claim, preferring Person 2's richer, embedding-based
        ``EvidenceBundle`` (when supplied) over the evidence already
        embedded in Person 1's ``ClaimDetectionOutcome``.
        """
        bundle = evidence_bundles.get(claim_id)
        if bundle is not None and bundle.results:
            best_evidence = max(bundle.results, key=lambda evidence: evidence.similarity_score)
            if best_evidence.text and best_evidence.text.strip():
                return best_evidence.text.strip()

        if outcome is not None and outcome.evidence:
            best_passage = max(outcome.evidence, key=lambda passage: passage.relevance_score)
            if best_passage.text and best_passage.text.strip():
                return best_passage.text.strip()

        return None

    # ------------------------------------------------------------------ #
    # Text-level editing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_correction_to_text(
        running_text: str,
        claim_text: str,
        action: CorrectionAction,
        corrected_text: str | None,
    ) -> tuple[str, bool]:
        """
        Apply a single claim-level edit to the running response text.

        Args:
            running_text: The response text as edited so far.
            claim_text: The exact original sentence span to locate and edit.
            action: ``REPLACED``, ``REWORDED``, or ``REMOVED``.
            corrected_text: Replacement text, required for
                ``REPLACED``/``REWORDED``.

        Returns:
            A ``(new_text, applied)`` tuple. ``applied`` is ``False`` if
            ``claim_text`` could not be located verbatim in
            ``running_text`` (e.g. it was already consumed by a prior
            overlapping edit), signaling the caller to fall back to a
            non-text-editing action instead of risking an incorrect edit.
        """
        if claim_text not in running_text:
            return running_text, False

        if action in (CorrectionAction.REPLACED, CorrectionAction.REWORDED):
            assert corrected_text is not None  # enforced by ClaimCorrection validation upstream
            return running_text.replace(claim_text, corrected_text, 1), True

        if action is CorrectionAction.REMOVED:
            return SelfCorrectionAgent._remove_sentence(running_text, claim_text), True

        return running_text, True

    @staticmethod
    def _remove_sentence(text: str, sentence: str) -> str:
        """Remove one occurrence of ``sentence`` from ``text`` and tidy up leftover whitespace."""
        removed = text.replace(sentence, "", 1)
        removed = re.sub(r"[ \t]{2,}", " ", removed)
        removed = re.sub(r"\n{3,}", "\n\n", removed)
        removed = re.sub(r" +\n", "\n", removed)
        return removed.strip()

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _compute_confidence(self, corrections: list[ClaimCorrection]) -> float:
        """
        Compute the overall ``correction_confidence`` as the mean of each
        applied correction's configured action weight (see
        :attr:`DEFAULT_ACTION_CONFIDENCE_WEIGHTS`), rounded to 4 decimal places.

        A response with no claims requiring correction (everything
        ``NONE``) yields ``1.0``; a response where every claim had to be
        conservatively ``FLAGGED`` yields a low score, transparently
        signaling that little could actually be confirmed or fixed.
        """
        if not corrections:
            return 1.0
        total_weight = sum(self._action_confidence_weights[correction.action] for correction in corrections)
        return round(total_weight / len(corrections), 4)

    @staticmethod
    def _build_explanation(corrections: list[ClaimCorrection]) -> str:
        """Build a short, human-readable summary of what the correction pass did."""
        action_counts = Counter(correction.action for correction in corrections)
        kept = action_counts.get(CorrectionAction.NONE, 0)
        replaced = action_counts.get(CorrectionAction.REPLACED, 0) + action_counts.get(
            CorrectionAction.REWORDED, 0
        )
        removed = action_counts.get(CorrectionAction.REMOVED, 0)
        flagged = action_counts.get(CorrectionAction.FLAGGED, 0)

        parts = [f"{len(corrections)} claim(s) reviewed"]
        if kept:
            parts.append(f"{kept} already supported and left unchanged")
        if replaced:
            parts.append(f"{replaced} replaced with verified evidence")
        if removed:
            parts.append(f"{removed} removed for lacking any verifiable replacement")
        if flagged:
            parts.append(f"{flagged} flagged as uncertain due to insufficient evidence")
        return "; ".join(parts) + "."
