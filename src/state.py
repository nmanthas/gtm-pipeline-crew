"""Structured Flow state.

CrewAI Flows carry a single typed state object between steps. Using
Pydantic models (rather than the default dict state) makes every
transition explicit, debuggable, and easy to persist.
"""

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class Opportunity(BaseModel):
    opportunity_id: str
    account_name: str
    opportunity_name: str
    amount: float
    stage: str
    probability: int
    close_date: str
    last_activity_date: str
    discount_pct: float
    has_exec_sponsor: bool
    cpq_approval_pending: bool
    days_since_activity: int = 0
    days_to_close: int = 0


class RiskAssessment(BaseModel):
    opportunity_id: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    risk_factors: list[str] = []


class PipelineState(BaseModel):
    """Single source of truth passed between every Flow step."""

    opportunities: list[Opportunity] = []
    risk_assessments: list[RiskAssessment] = []
    analysis_summary: str = ""
    recommendations: str = ""
    manager_brief: str = ""
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer_notes: str = ""

    @property
    def high_risk_deals(self) -> list[RiskAssessment]:
        return [r for r in self.risk_assessments if r.risk_level == RiskLevel.HIGH]

    @property
    def total_at_risk_amount(self) -> float:
        high_risk_ids = {r.opportunity_id for r in self.high_risk_deals}
        return sum(o.amount for o in self.opportunities if o.opportunity_id in high_risk_ids)
