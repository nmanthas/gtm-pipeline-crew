"""Advisory Crew — turns analysis into an actionable, manager-ready brief.

Two sequential agents:
  1. Deal Desk Advisor — one concrete intervention per at-risk deal
  2. Comms Writer      — assembles the final manager brief in markdown
"""

from crewai import Agent, Crew, Process, Task

DEFAULT_LLM = "anthropic/claude-sonnet-4-6"


def build_advisory_crew(analysis: str, patterns: str, llm: str = DEFAULT_LLM) -> Crew:
    deal_desk_advisor = Agent(
        role="Deal Desk Advisor",
        goal=(
            "For each HIGH-risk deal, recommend exactly one primary intervention "
            "and one fallback, grounded in the stated risk factors."
        ),
        backstory=(
            "You are a senior deal desk strategist with deep CPQ and pricing "
            "governance experience. Your playbook includes: discount guardrail "
            "escalations, executive sponsor plays, CPQ approval fast-tracks, "
            "re-engagement sequences for stale deals, and close-date resets. "
            "You never recommend interventions that ignore pricing governance."
        ),
        llm=llm,
        verbose=True,
    )

    comms_writer = Agent(
        role="GTM Communications Writer",
        goal=(
            "Assemble a concise, skimmable manager brief in markdown that a "
            "sales VP can act on in under three minutes."
        ),
        backstory=(
            "You write for busy sales leadership. Every brief leads with the "
            "headline number, keeps one short block per deal, and ends with a "
            "clear list of asks. No filler, no hedging."
        ),
        llm=llm,
        verbose=True,
    )

    recommend_task = Task(
        description=(
            "Here is the pipeline risk analysis:\n\n"
            f"{analysis}\n\n"
            "And the win-pattern findings:\n\n"
            f"{patterns}\n\n"
            "For each HIGH-risk deal, produce: the deal ID and name, the primary "
            "intervention (one sentence, concrete, with an owner role), and a "
            "fallback intervention. Respect pricing governance — deals over the "
            "discount guardrail need an approval path, not a bigger discount."
        ),
        expected_output=(
            "A per-deal action plan: deal ID, primary intervention with owner, "
            "fallback intervention."
        ),
        agent=deal_desk_advisor,
    )

    brief_task = Task(
        description=(
            "Assemble the final manager brief in markdown with these sections: "
            "'# Pipeline Health Brief', '## Headline' (total at-risk dollars and "
            "deal count), '## At-Risk Deals & Recommended Actions' (one compact "
            "block per deal), '## Patterns Observed', and '## Asks for Leadership' "
            "(max 3 bullets)."
        ),
        expected_output="A complete markdown brief following the exact section structure.",
        agent=comms_writer,
        context=[recommend_task],
    )

    return Crew(
        agents=[deal_desk_advisor, comms_writer],
        tasks=[recommend_task, brief_task],
        process=Process.sequential,
        verbose=True,
    )
