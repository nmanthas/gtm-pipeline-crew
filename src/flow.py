"""PipelineHealthFlow — the deterministic, event-driven backbone.

Flow (deterministic) decides WHAT runs WHEN.
Crews (LLM-driven) do the fuzzy reasoning INSIDE each step.

Steps:
    ingest_pipeline  -> run_analysis_crew -> triage (router)
        "escalate"   -> run_advisory_crew -> human_review -> publish
        "summarize"  -> log_healthy_pipeline

Run:
    python -m src.flow            # full run (requires ANTHROPIC_API_KEY)
    python -m src.flow --plot     # write flow_diagram.html only (no LLM calls)
"""

import sys
from pathlib import Path

from crewai.flow.flow import Flow, listen, router, start

from src.crews.advisory_crew import build_advisory_crew
from src.crews.analysis_crew import build_analysis_crew
from src.state import ApprovalStatus, PipelineState
from src.tools.pipeline_tools import load_pipeline, score_opportunity

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pipeline_brief.md"


class PipelineHealthFlow(Flow[PipelineState]):
    # ------------------------------------------------------------------
    @start()
    def ingest_pipeline(self):
        """Deterministic: load + score every opportunity. No LLM involved."""
        self.state.opportunities = load_pipeline()
        self.state.risk_assessments = [
            score_opportunity(o) for o in self.state.opportunities
        ]
        print(
            f"[flow] Ingested {len(self.state.opportunities)} opportunities — "
            f"{len(self.state.high_risk_deals)} high risk "
            f"(${self.state.total_at_risk_amount:,.0f} at risk)"
        )

    # ------------------------------------------------------------------
    @listen(ingest_pipeline)
    def run_analysis_crew(self):
        """LLM layer 1: interpret the deterministic scores."""
        crew = build_analysis_crew()
        result = crew.kickoff()
        # Task outputs, in order: [analysis, patterns]
        self.state.analysis_summary = str(result.tasks_output[0].raw)
        self.state.recommendations = str(result.tasks_output[1].raw)

    # ------------------------------------------------------------------
    @router(run_analysis_crew)
    def triage(self):
        """Deterministic routing on structured state — not on LLM text."""
        if self.state.high_risk_deals:
            return "escalate"
        return "summarize"

    # ------------------------------------------------------------------
    @listen("escalate")
    def run_advisory_crew(self):
        """LLM layer 2: interventions + manager brief."""
        crew = build_advisory_crew(
            analysis=self.state.analysis_summary,
            patterns=self.state.recommendations,
        )
        result = crew.kickoff()
        self.state.manager_brief = str(result.tasks_output[-1].raw)

    # ------------------------------------------------------------------
    @listen(run_advisory_crew)
    def human_review(self):
        """Gradual-autonomy gate: a human approves before anything ships.

        v1 starts at 100% human review. As trust builds, this gate can be
        relaxed (e.g. auto-approve briefs with < N deals) — but autonomy is
        earned, never assumed.
        """
        print("\n" + "=" * 70)
        print("DRAFT MANAGER BRIEF — awaiting human review")
        print("=" * 70)
        print(self.state.manager_brief)
        print("=" * 70)

        decision = input("Approve this brief? [y = approve / e = edit note / n = reject]: ").strip().lower()
        if decision == "y":
            self.state.approval_status = ApprovalStatus.APPROVED
        elif decision == "e":
            self.state.reviewer_notes = input("Reviewer note to append: ").strip()
            self.state.approval_status = ApprovalStatus.EDITED
        else:
            self.state.approval_status = ApprovalStatus.REJECTED

    # ------------------------------------------------------------------
    @listen(human_review)
    def publish(self):
        """Deterministic: persist the approved brief. Swap for Slack/Chatter."""
        if self.state.approval_status == ApprovalStatus.REJECTED:
            print("[flow] Brief rejected — nothing published.")
            return

        brief = self.state.manager_brief
        if self.state.reviewer_notes:
            brief += f"\n\n---\n*Reviewer note: {self.state.reviewer_notes}*"

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(brief)
        print(f"[flow] Brief published to {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    @listen("summarize")
    def log_healthy_pipeline(self):
        """Cheap path: no high-risk deals, no advisory crew, no brief."""
        print(
            f"[flow] Pipeline healthy — {len(self.state.opportunities)} deals, "
            "no high-risk escalations. Skipping advisory crew."
        )


def main():
    flow = PipelineHealthFlow()
    if "--plot" in sys.argv:
        flow.plot("flow_diagram")  # writes flow_diagram.html — no LLM calls
        print("Wrote flow_diagram.html")
        return
    flow.kickoff()


if __name__ == "__main__":
    main()
