"""State and perception primitives for the staged safety-agent architecture.

Phase B deliberately keeps these components separate from the production
agent. Later phases can integrate them without changing their public types.
"""

from .models import (
    Fact,
    FactSource,
    FactStatus,
    PerceptionResult,
    RequestIntent,
    RequestPhase,
    RequestState,
)
from .adjudication import (
    ActionSpec,
    DecisionDraft,
    ValidationIssue,
    ValidationOutcome,
    ValidationRoute,
    adjudication_messages,
    extract_clause_ids,
    parse_decision_draft,
    validate_decision_draft,
)
from .investigation import (
    InvestigationState,
    InvestigationStatus,
    build_investigation,
)
from .execution import (
    CompletedToolStep,
    ExecutionQueue,
    ExecutionStatus,
    StepMutability,
    ToolStep,
    compile_execution_queue,
)
from .context import (
    RECENT_DIALOGUE_LIMIT,
    approximate_input_tokens,
    build_request_context,
)
from .perception import (
    PerceptionEngine,
    analyze_message,
    build_request_key,
    detect_pressure,
    extract_identifiers,
    extract_material_facts,
)
from .tool_state import (
    CompletedToolCall,
    FailedToolCall,
    PendingToolCall,
    ToolCallTracker,
    ToolCategory,
    ToolResultDisposition,
)
from .workflows import (
    INVESTIGATION_WORKFLOWS,
    available_workflow_steps,
    workflow_for,
)

__all__ = [
    "ActionSpec",
    "RECENT_DIALOGUE_LIMIT",
    "DecisionDraft",
    "ExecutionQueue",
    "ExecutionStatus",
    "Fact",
    "FactSource",
    "FactStatus",
    "FailedToolCall",
    "CompletedToolCall",
    "InvestigationState",
    "InvestigationStatus",
    "INVESTIGATION_WORKFLOWS",
    "PendingToolCall",
    "StepMutability",
    "ToolStep",
    "CompletedToolStep",
    "PerceptionEngine",
    "PerceptionResult",
    "RequestIntent",
    "RequestPhase",
    "RequestState",
    "ToolCallTracker",
    "ToolCategory",
    "ToolResultDisposition",
    "ValidationIssue",
    "ValidationOutcome",
    "ValidationRoute",
    "adjudication_messages",
    "analyze_message",
    "available_workflow_steps",
    "approximate_input_tokens",
    "build_investigation",
    "build_request_key",
    "build_request_context",
    "compile_execution_queue",
    "detect_pressure",
    "extract_identifiers",
    "extract_clause_ids",
    "extract_material_facts",
    "parse_decision_draft",
    "validate_decision_draft",
    "workflow_for",
]
