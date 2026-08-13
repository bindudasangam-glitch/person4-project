"""
Knowledge Drift Detection Agent
===================================

Detects whether the claims in a response have drifted from a previously
established baseline -- an earlier verification outcome recorded for
the same (or a textually equivalent) claim. Drift can signal either
stale source material (the evidence corpus changed since the baseline
was recorded) or an LLM response diverging from previously grounded
facts across repeated interactions on the same topic.

Design notes
------------
This project has no persistence layer for storing historical
verification results (Person 1 and Person 2 both operate on a single
request at a time), so this agent depends on a small, injectable
:class:`DriftBaselineStore` seam -- the same "protocol + swappable
default implementation" pattern already used by Person 1's
``HallucinationDetector`` for its ``EvidenceSource``. The default
implementation, :class:`InMemoryDriftBaselineStore`, is a simple
process-lifetime dictionary: sufficient to detect drift across multiple
requests within a single running server process, with the same explicit
"does not persist across restarts" trade-off already documented for
``InMemoryGraphFallback`` in Batch 1.

Recording a baseline is an explicit, opt-in operation
(:meth:`DriftDetectionAgent.record_baseline`), never a side effect of
detection itself -- mirroring
``KnowledgeGraphAgent.seed_supported_claims`` from Batch 3 -- so callers
(the LangGraph workflow, in a later batch) decide when a run's results
are trustworthy enough to become the new baseline for future comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from app.agents.exceptions import DriftDetectionError
from app.core.logging import logger
from app.models.claim_model import ClaimModel, VerificationStatus
from app.models.workflow_models import DriftedClaim, DriftReport, SeverityLevel, WorkflowModelValidationError
from app.services.hallucination_detector import ClaimDetectionOutcome

__all__ = [
    "HistoricalClaimRecord",
    "DriftBaselineStore",
    "InMemoryDriftBaselineStore",
    "DriftDetectionAgent",
]


@dataclass(frozen=True, slots=True)
class HistoricalClaimRecord:
    """
    A single previously-recorded verification outcome for a claim,
    stored by a :class:`DriftBaselineStore` and later compared against a
    new verification outcome for the "same" claim to detect drift.

    Attributes:
        claim_text: The claim's text as originally recorded.
        verification_status: The verification status recorded at that time.
        support_score: The claim's support/combined score at that time, in [0, 1].
        recorded_at: UTC timestamp of when this record was captured.
    """

    claim_text: str
    verification_status: VerificationStatus
    support_score: float
    recorded_at: datetime


@runtime_checkable
class DriftBaselineStore(Protocol):
    """
    Storage seam for historical claim verification records, injected into
    :class:`DriftDetectionAgent` so the baseline persistence mechanism can
    be swapped (in-memory for a single process, a database or cache in a
    future deployment) without changing any drift-detection logic.
    """

    def get_baseline(self, claim_key: str) -> HistoricalClaimRecord | None:
        """Return the most recent historical record for ``claim_key``, or ``None`` if there is none."""
        ...

    def record(self, claim_key: str, record: HistoricalClaimRecord) -> None:
        """Store (overwriting any prior record for) ``claim_key``."""
        ...


class InMemoryDriftBaselineStore:
    """
    Default :class:`DriftBaselineStore` implementation: a simple,
    process-lifetime, in-memory dictionary.

    Not persisted across restarts and not shared across worker
    processes -- an explicit, documented trade-off (matching Batch 1's
    ``InMemoryGraphFallback``) favoring "drift detection keeps working
    within a running process" over requiring an external datastore this
    project does not otherwise have.
    """

    def __init__(self) -> None:
        self._records: dict[str, HistoricalClaimRecord] = {}

    def get_baseline(self, claim_key: str) -> HistoricalClaimRecord | None:
        """Return the stored record for ``claim_key``, or ``None`` if none exists."""
        return self._records.get(claim_key)

    def record(self, claim_key: str, record: HistoricalClaimRecord) -> None:
        """Store (overwriting any prior record for) ``claim_key``."""
        self._records[claim_key] = record

    def clear(self) -> None:
        """Remove every stored record. Intended for tests."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


class DriftDetectionAgent:
    """
    Detects knowledge drift by comparing a batch of just-verified claims
    against a stored historical baseline for each claim.

    Args:
        baseline_store: Where historical verification records are read
            from and written to. Defaults to a fresh
            :class:`InMemoryDriftBaselineStore` when omitted (``None``).
            Injected so callers/tests can supply a store pre-seeded with
            specific baselines, or a future persistent implementation.
            An explicitly provided store is always used as-is -- including
            an empty one -- since presence is checked via ``is not None``
            rather than truthiness (an empty ``InMemoryDriftBaselineStore``
            is falsy, as it defines ``__len__``).
        per_claim_drift_threshold: Minimum per-claim drift score, in
            [0, 1], for a claim to be reported in
            ``DriftReport.drifted_claims``. Defaults to ``0.3``.
        severity_thresholds: ``(low_upper, medium_upper, high_upper)``
            cutoffs on the overall drift score used to classify
            ``DriftReport.drift_severity`` once drift has been detected.
            Must be strictly increasing and within ``(0, 1]``. Defaults
            to ``(0.35, 0.6, 0.85)``.

    Raises:
        DriftDetectionError: If any constructor argument is out of range.
    """

    #: Numeric "reliability" mapped to each verification status, used to
    #: measure how much a claim's status has drifted between the
    #: baseline and the current run. Mirrors
    #: ``ConfidenceScorer._SUPPORT_VALUES`` (Person 1) so both modules
    #: agree on what each status is "worth" on a comparable scale.
    _STATUS_RELIABILITY_VALUE: dict[VerificationStatus, float] = {
        VerificationStatus.SUPPORTED: 1.0,
        VerificationStatus.INSUFFICIENT_EVIDENCE: 0.5,
        VerificationStatus.UNVERIFIED: 0.5,
        VerificationStatus.CONTRADICTED: 0.0,
    }

    def __init__(
        self,
        baseline_store: DriftBaselineStore | None = None,
        per_claim_drift_threshold: float = 0.3,
        severity_thresholds: tuple[float, float, float] = (0.35, 0.6, 0.85),
    ) -> None:
        if not 0.0 <= per_claim_drift_threshold <= 1.0:
            raise DriftDetectionError(
                f"per_claim_drift_threshold must be within [0.0, 1.0], got {per_claim_drift_threshold}."
            )

        low, medium, high = severity_thresholds
        if not (0.0 < low < medium < high <= 1.0):
            raise DriftDetectionError(
                f"severity_thresholds must be strictly increasing within (0, 1], got {severity_thresholds}."
            )

        self._baseline_store: DriftBaselineStore = (
            baseline_store if baseline_store is not None else InMemoryDriftBaselineStore()
        )
        self._per_claim_drift_threshold = per_claim_drift_threshold
        self._severity_thresholds = severity_thresholds

    def detect_drift(
        self,
        claims: list[ClaimModel],
        detection_outcomes: list[ClaimDetectionOutcome],
    ) -> DriftReport:
        """
        Compare each claim's current verification outcome against its
        stored baseline (if any) and report any significant drift.

        Args:
            claims: Post-detection claims from Person 1 (each already
                carrying its final ``verification_status``).
            detection_outcomes: The parallel ``ClaimDetectionOutcome``
                list from ``HallucinationDetector.detect()``, used for
                each claim's current ``support_score``.

        Returns:
            A :class:`~app.models.workflow_models.DriftReport` describing
            any drift detected. Claims with no prior baseline are simply
            excluded from comparison (there being nothing yet to compare
            against), not treated as evidence of drift.

        Raises:
            DriftDetectionError: If ``claims`` is ``None``, the baseline
                store raises unexpectedly, or an invalid ``DriftReport``
                would otherwise be produced.
        """
        if claims is None:
            raise DriftDetectionError("claims must not be None.")

        if not claims:
            logger.info("DriftDetectionAgent: no claims supplied; returning a no-drift report.")
            return DriftReport(
                has_drift=False,
                drift_severity=SeverityLevel.NONE,
                overall_drift_score=0.0,
                explanation="No claims were supplied for drift detection.",
            )

        outcome_by_claim_id = {outcome.claim_id: outcome for outcome in (detection_outcomes or [])}

        drifted: list[DriftedClaim] = []
        per_claim_scores: list[float] = []
        compared_count = 0

        for claim in claims:
            outcome = outcome_by_claim_id.get(claim.id)
            current_support_score = outcome.support_score if outcome is not None else 0.0
            claim_key = self._normalize_claim_key(claim.text)

            try:
                baseline = self._baseline_store.get_baseline(claim_key)
            except Exception as exc:  # noqa: BLE001 - normalize to domain error
                logger.exception("DriftDetectionAgent failed to fetch a baseline for claim %d.", claim.id)
                raise DriftDetectionError(f"Failed to fetch a baseline for claim {claim.id}.") from exc

            if baseline is None:
                continue

            compared_count += 1
            claim_drift_score = self._compute_claim_drift_score(claim, current_support_score, baseline)
            per_claim_scores.append(claim_drift_score)

            if claim_drift_score >= self._per_claim_drift_threshold:
                drifted.append(
                    DriftedClaim(
                        claim_id=claim.id,
                        claim_text=claim.text,
                        drift_score=claim_drift_score,
                        reason=self._describe_drift(claim, baseline, current_support_score),
                    )
                )

        has_drift = len(drifted) > 0
        overall_drift_score = (
            round(sum(per_claim_scores) / len(per_claim_scores), 4) if per_claim_scores else 0.0
        )
        severity = self._classify_severity(has_drift, overall_drift_score)
        baseline_source = (
            f"{compared_count} prior claim record(s) from {type(self._baseline_store).__name__}"
            if compared_count
            else None
        )
        explanation = self._build_explanation(len(claims), compared_count, drifted)

        try:
            report = DriftReport(
                has_drift=has_drift,
                drift_severity=severity,
                overall_drift_score=overall_drift_score,
                drifted_claims=tuple(drifted),
                baseline_source=baseline_source,
                explanation=explanation,
            )
        except WorkflowModelValidationError as exc:
            logger.exception("DriftDetectionAgent produced an invalid DriftReport.")
            raise DriftDetectionError("Failed to construct a valid DriftReport.") from exc

        logger.info(
            "DriftDetectionAgent complete: compared=%d/%d claim(s), has_drift=%s, "
            "severity=%s, overall_score=%.3f.",
            compared_count,
            len(claims),
            has_drift,
            severity.value,
            overall_drift_score,
        )
        return report

    def record_baseline(
        self,
        claims: list[ClaimModel],
        detection_outcomes: list[ClaimDetectionOutcome],
    ) -> int:
        """
        Store the current verification outcome of each claim as its new
        baseline for future drift comparisons.

        This is an explicit, opt-in step -- it is never called
        automatically by :meth:`detect_drift` -- so callers decide when a
        run's results are trustworthy enough to become the reference
        point for detecting future drift.

        Args:
            claims: Claims whose current outcome should become the new baseline.
            detection_outcomes: The parallel ``ClaimDetectionOutcome``
                list supplying each claim's current ``support_score``.

        Returns:
            The number of claims whose baseline was successfully recorded.
        """
        if not claims:
            return 0

        outcome_by_claim_id = {outcome.claim_id: outcome for outcome in (detection_outcomes or [])}
        recorded = 0

        for claim in claims:
            outcome = outcome_by_claim_id.get(claim.id)
            support_score = outcome.support_score if outcome is not None else 0.0
            claim_key = self._normalize_claim_key(claim.text)
            record = HistoricalClaimRecord(
                claim_text=claim.text,
                verification_status=claim.verification_status,
                support_score=support_score,
                recorded_at=datetime.now(timezone.utc),
            )
            try:
                self._baseline_store.record(claim_key, record)
                recorded += 1
            except Exception:  # noqa: BLE001 - recording is best-effort, never fatal
                logger.exception("DriftDetectionAgent failed to record a baseline for claim %d.", claim.id)

        logger.info("DriftDetectionAgent recorded %d baseline record(s) out of %d claim(s).", recorded, len(claims))
        return recorded

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _compute_claim_drift_score(
        self,
        claim: ClaimModel,
        current_support_score: float,
        baseline: HistoricalClaimRecord,
    ) -> float:
        """
        Compute a single claim's drift score, in [0, 1], as a weighted
        blend of how much its verification status's reliability value
        changed (70% weight) and how much its raw support score changed
        (30% weight) since the baseline was recorded. Status changes
        dominate the score since they represent a qualitative shift
        (e.g. previously SUPPORTED, now CONTRADICTED), while score
        movement alone still contributes a smaller, secondary signal.
        """
        status_drift = abs(
            self._STATUS_RELIABILITY_VALUE[claim.verification_status]
            - self._STATUS_RELIABILITY_VALUE[baseline.verification_status]
        )
        score_drift = abs(current_support_score - baseline.support_score)
        return round(min(1.0, (0.7 * status_drift) + (0.3 * score_drift)), 4)

    def _classify_severity(self, has_drift: bool, overall_drift_score: float) -> SeverityLevel:
        """Map ``has_drift``/``overall_drift_score`` onto a coarse :class:`SeverityLevel`."""
        if not has_drift:
            return SeverityLevel.NONE

        low_cutoff, medium_cutoff, high_cutoff = self._severity_thresholds
        if overall_drift_score < low_cutoff:
            return SeverityLevel.LOW
        if overall_drift_score < medium_cutoff:
            return SeverityLevel.MEDIUM
        if overall_drift_score < high_cutoff:
            return SeverityLevel.HIGH
        return SeverityLevel.CRITICAL

    @staticmethod
    def _normalize_claim_key(text: str) -> str:
        """Normalize claim text into a stable lookup key for the baseline store."""
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _describe_drift(claim: ClaimModel, baseline: HistoricalClaimRecord, current_support_score: float) -> str:
        """Build a human-readable explanation of what changed for a single drifted claim."""
        if claim.verification_status != baseline.verification_status:
            return (
                f"Verification status changed from '{baseline.verification_status.value}' to "
                f"'{claim.verification_status.value}' since the prior baseline recorded on "
                f"{baseline.recorded_at.isoformat()}."
            )
        return (
            f"Support score shifted from {baseline.support_score:.2f} to "
            f"{current_support_score:.2f} since the prior baseline recorded on "
            f"{baseline.recorded_at.isoformat()}, despite an unchanged "
            f"'{claim.verification_status.value}' status."
        )

    @staticmethod
    def _build_explanation(total_claims: int, compared_count: int, drifted: list[DriftedClaim]) -> str:
        """Build a short, human-readable summary of the drift detection pass."""
        if compared_count == 0:
            return (
                f"{total_claims} claim(s) reviewed; no prior baseline was available for any "
                "of them, so drift could not be assessed."
            )
        parts = [f"{compared_count}/{total_claims} claim(s) had a prior baseline to compare against"]
        if drifted:
            parts.append(f"{len(drifted)} showed significant drift")
        else:
            parts.append("none showed significant drift")
        return "; ".join(parts) + "."
