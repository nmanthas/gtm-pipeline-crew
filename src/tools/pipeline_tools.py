"""Deterministic pipeline tools.

Design principle: anything that CAN be a deterministic rule SHOULD be a
deterministic rule. The LLM agents interpret and advise on top of these
scores — they don't compute them. That keeps risk scoring reproducible
and auditable, and keeps token spend down.

Swap `load_pipeline` for a simple-salesforce (or Salesforce MCP) query
to run against a live org — the rest of the flow is unchanged.
"""

import csv
from datetime import date, datetime
from pathlib import Path

from crewai.tools import tool

from src.state import Opportunity, RiskAssessment, RiskLevel

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "pipeline.csv"

# Rule thresholds — in a real org these come from RevOps policy, not code.
STALE_ACTIVITY_DAYS = 21
DISCOUNT_GUARDRAIL_PCT = 15
CLOSE_WINDOW_DAYS = 30
BIG_DEAL_AMOUNT = 250_000


def _days_between(earlier: str, later: date) -> int:
    return (later - datetime.strptime(earlier, "%Y-%m-%d").date()).days


def load_pipeline(as_of: date | None = None) -> list[Opportunity]:
    """Load opportunities from CSV and enrich with derived date fields."""
    as_of = as_of or date.today()
    opportunities: list[Opportunity] = []
    with open(DATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            opp = Opportunity(
                opportunity_id=row["opportunity_id"],
                account_name=row["account_name"],
                opportunity_name=row["opportunity_name"],
                amount=float(row["amount"]),
                stage=row["stage"],
                probability=int(row["probability"]),
                close_date=row["close_date"],
                last_activity_date=row["last_activity_date"],
                discount_pct=float(row["discount_pct"]),
                has_exec_sponsor=row["has_exec_sponsor"] == "True",
                cpq_approval_pending=row["cpq_approval_pending"] == "True",
            )
            opp.days_since_activity = _days_between(opp.last_activity_date, as_of)
            opp.days_to_close = -_days_between(opp.close_date, as_of)
            opportunities.append(opp)
    return opportunities


def score_opportunity(opp: Opportunity) -> RiskAssessment:
    """Rule-based risk scoring. Deterministic and explainable by design."""
    score = 0
    factors: list[str] = []

    if opp.days_since_activity > STALE_ACTIVITY_DAYS:
        score += 30
        factors.append(f"No activity in {opp.days_since_activity} days")

    if opp.discount_pct > DISCOUNT_GUARDRAIL_PCT:
        score += 20
        factors.append(f"Discount {opp.discount_pct}% exceeds {DISCOUNT_GUARDRAIL_PCT}% guardrail")

    if opp.cpq_approval_pending and opp.days_to_close <= CLOSE_WINDOW_DAYS:
        score += 20
        factors.append("CPQ approval pending inside close window")

    if opp.amount >= BIG_DEAL_AMOUNT and not opp.has_exec_sponsor:
        score += 20
        factors.append("Large deal with no executive sponsor")

    if opp.days_to_close <= CLOSE_WINDOW_DAYS and opp.probability < 60:
        score += 10
        factors.append("Closing soon with sub-60% probability")

    if score >= 50:
        level = RiskLevel.HIGH
    elif score >= 25:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return RiskAssessment(
        opportunity_id=opp.opportunity_id,
        risk_level=level,
        risk_score=min(score, 100),
        risk_factors=factors,
    )


@tool("Pipeline Snapshot")
def pipeline_snapshot_tool() -> str:
    """Return a formatted snapshot of all open opportunities with derived
    activity/close-date metrics. Use this to see the current pipeline."""
    lines = []
    for o in load_pipeline():
        lines.append(
            f"{o.opportunity_id} | {o.account_name} | ${o.amount:,.0f} | {o.stage} "
            f"| prob {o.probability}% | closes {o.close_date} ({o.days_to_close}d) "
            f"| last activity {o.days_since_activity}d ago | discount {o.discount_pct}% "
            f"| exec sponsor: {o.has_exec_sponsor} | CPQ pending: {o.cpq_approval_pending}"
        )
    return "\n".join(lines)


@tool("Risk Scores")
def risk_scores_tool() -> str:
    """Return the deterministic rule-based risk score and contributing risk
    factors for every open opportunity. Use this as the ground truth for
    which deals are at risk — do not re-score deals yourself."""
    lines = []
    for o in load_pipeline():
        a = score_opportunity(o)
        factors = "; ".join(a.risk_factors) or "none"
        lines.append(
            f"{a.opportunity_id} | {a.risk_level.value.upper()} | score {a.risk_score} | {factors}"
        )
    return "\n".join(lines)
