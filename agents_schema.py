"""Pydantic schemas for RCA agents."""

from __future__ import annotations

from typing import Literal, Self, Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


class UserPromptStructuredOutput(BaseModel):
    """Structured interpretation of the user's message for downstream RCA agents."""

    title: str = Field(
        ...,
        description=(
            "Short session title inferred from the user's message; "
            "at most 5 words (normalized on validation)."
        ),
    )
    intent: Literal[
        "codebase_summarization",
        "troubleshooting",
        "analysis",
        "general_rca",
        "out_of_scope",
    ] = Field(
        ...,
        description="Primary classification of what the user is asking for.",
    )
    user_goal_summary: str = Field(
        ...,
        description="One or two sentences restating the user's goal in clear terms.",
    )
    repo_status: Literal["verified", "not_provided", "error"] = Field(
        ...,
        description=(
            "not_provided if verify_github_repo is not in the tool list or you did not run it; "
            "verified after a successful verify_github_repo call; "
            "error if you called verify_github_repo and it reported failure or was inaccessible."
        ),
    )
    arch_status: Literal["verified", "not_provided", "error"] = Field(
        ...,
        description=(
            "not_provided if verify_architecture is not in the tool list or you did not run it; "
            "verified after verify_architecture when every architecture entry succeeded; "
            "error if any architecture entry failed or the tool errored."
        ),
    )
    next_action: Literal["proceed", "exit"] = Field(
        default="proceed",
        description=(
            "exit if intent is out_of_scope or repo_status or arch_status is error; "
            "otherwise proceed (set by model validator)."
        ),
    )

    @field_validator("title")
    @classmethod
    def _title_at_most_five_words(cls, v: str) -> str:
        words = [w for w in (v or "").strip().split() if w]
        if not words:
            return "RCA session"
        return " ".join(words[:5])

    @model_validator(mode="after")
    def _sync_next_action_from_statuses(self) -> Self:
        """Force next_action from intent and verification outcomes."""
        derived: Literal["proceed", "exit"] = (
            "exit"
            if self.intent == "out_of_scope"
            or self.repo_status == "error"
            or self.arch_status == "error"
            else "proceed"
        )
        object.__setattr__(self, "next_action", derived)
        return self


class LogAnalysisStructuredOutput(BaseModel):
    """Structured result from the log analyser agent."""

    relevant_files: list[str] = Field(
        default_factory=list,
        description="Source files, paths, or modules mentioned or implied in the logs (deduplicated, best-effort).",
    )
    identified_issues: list[str] = Field(
        default_factory=list,
        description="Distinct problems or anomalies found (errors, warnings worth acting on, cascading failures).",
    )
    log_summary: str = Field(
        ...,
        description="Concise narrative of what happened across the log window (timeline, severity, impact).",
    )


class CodeAnalysisStructuredOutput(BaseModel):
    """Structured handoff from the code analyser agent to the meta analyser."""

    examined_files: list[str] = Field(
        default_factory=list,
        description=(
            "Repository paths the agent actually relied on (from GitHub MCP tools); "
            "deduplicated; empty only if no suitable files were found."
        ),
    )
    file_relevance: str = Field(
        ...,
        description="Why these paths matter for the orchestrator's task (e.g. log stack traces, module names, errors).",
    )
    code_insights: str = Field(
        ...,
        description=(
            "Grounded findings from reading those files: behavior, dependencies, defect or risk signals—"
            "only what tools returned; state clearly if nothing conclusive was found."
        ),
    )
    revision_notes: str = Field(
        default="",
        description=(
            "Notable commit or history context for key files (recent changes, regressions); "
            "empty string if history was not consulted or added no signal."
        ),
    )
    gaps: str = Field(
        default="",
        description=(
            "What could not be verified, tool or permission failures, or what another agent "
            "(e.g. logs, runtime, web) should still check."
        ),
    )


class WebResearchLinkItem(BaseModel):
    """One documentation or web page the research agent recommends for the RCA."""

    url: str = Field(
        ...,
        description="Full URL exactly as obtained from search or browsing tools (no fabricated links).",
    )
    title: str = Field(
        ...,
        description="Page or document title from the tool result or the page heading.",
    )
    extraction_reason: str = Field(
        ...,
        description="Why this link matters for the investigation (e.g. matches error, API, version, CVE, vendor guidance).",
    )


class WebResearchStructuredOutput(BaseModel):
    """Structured handoff from the web research agent to the meta analyser."""

    links: list[WebResearchLinkItem] = Field(
        default_factory=list,
        description=(
            "Relevant documentation and webpages: each entry is url, title, and why it was selected. "
            "Empty list only if tools returned nothing usable."
        ),
    )


class AnalysisPlanStep(BaseModel):
    """One planned hop in the RCA orchestrator: which specialist runs, with prompt and context."""

    agent_name: Literal["log_analyser", "code_analyser", "web_researcher"] = Field(
        ...,
        description="Target specialist agent identifier.",
    )
    prompt: str = Field(..., description="What that agent should focus on.")
    context: str = Field(
        ...,
        description="Background: user goal, prior findings, and constraints for this step.",
    )


class AnalyserGraphOutput(BaseModel):
    """Structured orchestrator output for each meta-analyser hop (plan + route or finish)."""

    reasoning: str = Field(..., description="Brief rationale for the plan or next route.")
    plan_steps: list[AnalysisPlanStep] = Field(
        default_factory=list,
        description="Ordered RCA plan; populate on the first turn (and when revising).",
    )
    next_action: Literal["log_analyser", "code_analyser", "web_researcher", "finish"] = Field(
        ...,
        description="Next specialist to invoke, or finish.",
    )
    prompt_for_sub_agent: str = Field(
        default="",
        description="Task for the next sub-agent when next_action is not finish.",
    )
    context_for_sub_agent: str = Field(
        default="",
        description="Handoff context for the next sub-agent when next_action is not finish.",
    )
    final_analysis: str = Field(
        default="",
        description="Full RCA narrative when next_action is finish.",
    )

    @model_validator(mode="after")
    def _validate_action(self) -> Self:
        if self.next_action == "finish":
            if not self.final_analysis.strip():
                raise ValueError("final_analysis is required when next_action is finish")
        else:
            if not self.prompt_for_sub_agent.strip():
                raise ValueError("prompt_for_sub_agent is required when delegating to a sub-agent")
        return self


class HypothesisStructuredOutput(BaseModel):
    """Structured root-cause hypothesis after integrated RCA analysis."""

    primary_hypothesis: str = Field(
        ...,
        description="Most likely explanation of the incident or failure mode, grounded in the prior analysis.",
    )
    supporting_points: list[str] = Field(
        default_factory=list,
        description="Evidence from logs, code, or research that supports the primary hypothesis.",
    )
    alternative_hypotheses: list[str] = Field(
        default_factory=list,
        description="Other plausible causes, briefly stated, if the evidence allows.",
    )
    confidence_rationale: str = Field(
        ...,
        description="What would increase confidence; what remains unverified or needs validation.",
    )


class RCAReportStructuredOutput(BaseModel):
    """Final RCA report for stakeholders: problem, solution, analysis, references, and code touchpoints."""

    title: str = Field(..., description="Short title for the report.")
    problem_statement: str = Field(
        ...,
        description="Clear statement of what went wrong, scope, and user or business impact.",
    )
    solution_summary: str = Field(
        ...,
        description="Recommended remediation, workarounds, or follow-up actions.",
    )
    analysis_summary: str = Field(
        ...,
        description="Concise synthesis of the investigation (may reference the hypothesis).",
    )
    hypothesis_summary: str = Field(
        ...,
        description="How the leading hypothesis fits the evidence (one cohesive paragraph or short bullets).",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Documentation URLs, tickets, vendor notes, articles, or citations used or suggested.",
    )
    relevant_code_files: list[str] = Field(
        default_factory=list,
        description="Repository paths, modules, or symbols that should be inspected or changed.",
    )



class Repository(BaseModel):
    ownerId: str
    name: str
    branch: Optional[str] = "main"

class Architecture(BaseModel):
    type: Literal["raw", "lambda"]
    value: str
    desc: str

class Payload(BaseModel):
    prompt: str
    sessionId: str
    repo: Optional[Repository] = None
    arch: Optional[List[Architecture]] = None