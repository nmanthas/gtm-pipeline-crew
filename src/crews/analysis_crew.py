"""Analysis Crew — the fuzzy-reasoning layer on top of deterministic scores.

Two sequential agents:
  1. Deal Risk Analyst   — interprets the rule-based risk scores in context
  2. Win-Pattern Researcher — contrasts at-risk deals with healthy ones
"""

from crewai import Agent, Crew, Process, Task

from src.tools.pipeline_tools import pipeline_snapshot_tool, risk_scores_tool

DEFAULT_LLM = "anthropic/claude-sonnet-4-6"


def build_analysis_crew(llm: str = DEFAULT_LLM) -> Crew:
    risk_analyst = Agent(
        role="Deal Risk Analyst",
        goal=(
            "Interpret the deterministic risk scores for the current pipeline "
            "and explain, in business terms, why each HIGH-risk deal is at risk "
            "and what the aggregate exposure is."
        ),
        backstory=(
            "You are a RevOps analyst at a B2B security SaaS company. You trust "
            "the rule-based risk scores as ground truth and never re-score deals "
            "yourself — your value is interpretation, not arithmetic."
        ),
        tools=[pipeline_snapshot_tool, risk_scores_tool],
        llm=llm,
        verbose=True,
    )

    win_pattern_researcher = Agent(
        role="Win-Pattern Researcher",
        goal=(
            "Contrast the HIGH-risk deals against the healthiest deals in the "
            "same pipeline and identify the 2-3 structural differences that "
            "most plausibly explain the gap."
        ),
        backstory=(
            "You are a sales strategy researcher who studies what winning deals "
            "have in common — executive sponsorship, activity cadence, "
            "disciplined discounting — and uses those patterns to explain risk."
        ),
        tools=[pipeline_snapshot_tool, risk_scores_tool],
        llm=llm,
        verbose=True,
    )

    analyze_task = Task(
        description=(
            "Review the full pipeline snapshot and the deterministic risk scores. "
            "Produce an analysis covering: (1) every HIGH-risk deal with its risk "
            "factors restated in plain business language, (2) total dollar amount "
            "at risk, (3) any MEDIUM-risk deals trending toward high risk."
        ),
        expected_output=(
            "A structured analysis with one short paragraph per HIGH-risk deal, "
            "a total at-risk dollar figure, and a watchlist of medium-risk deals."
        ),
        agent=risk_analyst,
    )

    pattern_task = Task(
        description=(
            "Using the analysis above plus the pipeline snapshot, compare the "
            "HIGH-risk deals to the LOW-risk deals. Identify the 2-3 structural "
            "patterns that separate healthy deals from at-risk deals in this "
            "specific pipeline."
        ),
        expected_output=(
            "2-3 named patterns, each with one sentence of evidence citing "
            "specific opportunity IDs from both the healthy and at-risk groups."
        ),
        agent=win_pattern_researcher,
        context=[analyze_task],
    )

    return Crew(
        agents=[risk_analyst, win_pattern_researcher],
        tasks=[analyze_task, pattern_task],
        process=Process.sequential,
        verbose=True,
    )
