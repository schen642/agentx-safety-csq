"""Structured adjudication models and validation for architecture Phase E."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .models import Fact, FactStatus, RequestState

ALLOWED_DECISIONS = frozenset(
    {"ALLOW", "ALLOW-CONDITIONAL", "DENY", "ESCALATE"}
)

_CLAUSE_PATTERN = re.compile(
    r"\b(?:BM-RET-[A-Z0-9-]+|SOP-[A-Z0-9-]+|MFCP-[A-Z0-9-]+)\b"
)
_ID_PREFIXES = {
    "order_id": "ORD_",
    "request_id": "REQ_",
    "employee_id": "EMP_",
    "ticket_id": "TKT_",
    "account_id": "ACCT_",
    "customer_id": "CUST_",
}
_SENSITIVE_REASON_PATTERN = re.compile(
    r"\b(?:fraud[_ ]?score|risk[_ ]?score|account[_ ]?flags?|"
    r"internal[_ ]?flags?|watchlist|SAR filing|CTR filing|"
    r"investigation details?)\b",
    re.IGNORECASE,
)
_ALLOW_MUTATION_PREFIXES = (
    "process_",
    "reset_",
    "unlock_",
    "install_",
    "provision_",
    "send_",
    "execute_",
    "journal_",
    "update_beneficiary",
    "release_hold",
    "apply_store_credit",
)
_ESCALATION_PREFIXES = ("escalate_",)
_COMPLIANCE_ACTIONS = frozenset(
    {"hold_transaction", "create_alert", "open_case", "file_sar", "file_ctr"}
)


@dataclass(frozen=True)
class ActionSpec:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DecisionDraft:
    request_key: str
    decision: str
    controlling_clauses: list[str]
    factual_basis: list[str]
    unresolved_facts: list[str]
    required_actions: list[ActionSpec]
    customer_safe_reason: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionDraft:
        actions = value.get("required_actions", [])
        if not isinstance(actions, list):
            raise ValueError("required_actions must be a list")
        return cls(
            request_key=str(value.get("request_key", "")),
            decision=str(value.get("decision", "")).upper(),
            controlling_clauses=list(value.get("controlling_clauses", [])),
            factual_basis=list(value.get("factual_basis", [])),
            unresolved_facts=list(value.get("unresolved_facts", [])),
            required_actions=[
                ActionSpec(
                    tool_name=str(action.get("tool_name", "")),
                    arguments=dict(action.get("arguments", {})),
                )
                for action in actions
                if isinstance(action, dict)
            ],
            customer_safe_reason=str(value.get("customer_safe_reason", "")),
        )


class ValidationRoute(StrEnum):
    READY_FOR_EXECUTION = "ready_for_execution"
    REINVESTIGATE = "reinvestigate"
    RETRY_ADJUDICATION = "retry_adjudication"
    FAILED_SAFE = "failed_safe"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationOutcome:
    route: ValidationRoute
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.route is ValidationRoute.READY_FOR_EXECUTION


def parse_decision_draft(content: str) -> DecisionDraft:
    """Parse a JSON object into the strict Phase E draft model."""

    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("DecisionDraft must be a JSON object")
    return DecisionDraft.from_dict(value)


def extract_clause_ids(policy_text: str) -> set[str]:
    """Extract clause identifiers while excluding known document versions."""

    marker = re.search(
        r"Authoritative [A-Za-z]+ clause IDs",
        policy_text,
        re.IGNORECASE,
    )
    clause_source = policy_text[marker.end():] if marker else policy_text
    clauses = set(_CLAUSE_PATTERN.findall(clause_source))
    return {
        clause
        for clause in clauses
        if clause not in {
            "BM-SOP-RET-2025-04",
            "IT-SOP-2024-003",
            "SOP-2024-003",
            "MFCP-AML-2024-07",
        }
    }


def _tool_definition(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool.get("function")
    definition = function if isinstance(function, dict) else tool
    return str(definition.get("name", "")), definition


def _parameter_schema(
    definition: dict[str, Any],
) -> tuple[dict[str, Any], set[str], bool]:
    parameters = definition.get("parameters") or {}
    if parameters.get("type") == "object":
        return (
            dict(parameters.get("properties") or {}),
            set(parameters.get("required") or []),
            parameters.get("additionalProperties", True) is not False,
        )
    properties = {
        key: value
        for key, value in parameters.items()
        if isinstance(value, dict)
    }
    required = {
        key for key, value in properties.items() if value.get("required") is True
    }
    return properties, required, True


def _value_matches_type(value: Any, expected: str) -> bool:
    checks = {
        "string": lambda item: isinstance(item, str),
        "number": lambda item: (
            isinstance(item, (int, float)) and not isinstance(item, bool)
        ),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
    }
    return checks.get(expected, lambda _: True)(value)


def _schema_issues(
    action: ActionSpec,
    definition: dict[str, Any],
) -> list[ValidationIssue]:
    properties, required, additional_allowed = _parameter_schema(definition)
    issues: list[ValidationIssue] = []
    for missing in sorted(required - action.arguments.keys()):
        issues.append(
            ValidationIssue(
                "tool_schema",
                f"{action.tool_name} is missing required argument {missing}",
            )
        )
    if not additional_allowed:
        for extra in sorted(action.arguments.keys() - properties.keys()):
            issues.append(
                ValidationIssue(
                    "tool_schema",
                    f"{action.tool_name} has unknown argument {extra}",
                )
            )
    for name, value in action.arguments.items():
        schema = properties.get(name)
        if not schema:
            continue
        expected = schema.get("type")
        if expected and not _value_matches_type(value, expected):
            issues.append(
                ValidationIssue(
                    "tool_schema",
                    f"{action.tool_name}.{name} must be {expected}",
                )
            )
        if "enum" in schema and value not in schema["enum"]:
            issues.append(
                ValidationIssue(
                    "tool_schema",
                    f"{action.tool_name}.{name} is outside its enum",
                )
            )
    return issues


def _action_compatibility_issues(draft: DecisionDraft) -> list[ValidationIssue]:
    names = [action.tool_name for action in draft.required_actions]
    issues: list[ValidationIssue] = []
    if "record_decision" in names:
        issues.append(
            ValidationIssue(
                "action_compatibility",
                "record_decision is appended by execution and must not be "
                "a required action",
            )
        )
    allow_mutations = [
        name for name in names if name.startswith(_ALLOW_MUTATION_PREFIXES)
    ]
    if draft.decision in {"DENY", "ESCALATE"} and allow_mutations:
        issues.append(
            ValidationIssue(
                "action_compatibility",
                f"{draft.decision} cannot include allow action {allow_mutations[0]}",
            )
        )
    if draft.decision in {"ALLOW", "ALLOW-CONDITIONAL"}:
        incompatible = [
            name
            for name in names
            if name.startswith(_ESCALATION_PREFIXES)
            or name == "deny_refund"
            or name in _COMPLIANCE_ACTIONS
        ]
        if incompatible:
            issues.append(
                ValidationIssue(
                    "action_compatibility",
                    f"{draft.decision} cannot include {incompatible[0]}",
                )
            )
        if not allow_mutations:
            issues.append(
                ValidationIssue(
                    "action_compatibility",
                    f"{draft.decision} requires a compatible business action",
                )
            )
    return issues


def validate_decision_draft(
    draft: DecisionDraft,
    *,
    request: RequestState,
    policy_text: str,
    verified_facts: list[Fact],
    available_tools: list[dict[str, Any]],
    context_only: bool = False,
) -> ValidationOutcome:
    """Validate a draft without executing or emitting any tool call."""

    issues: list[ValidationIssue] = []
    missing_evidence = False

    if draft.request_key != request.request_key:
        issues.append(ValidationIssue("request_key", "draft targets another request"))
    if draft.decision not in ALLOWED_DECISIONS:
        issues.append(ValidationIssue("decision", "unsupported decision label"))

    allowed_clauses = extract_clause_ids(policy_text)
    if not draft.controlling_clauses:
        issues.append(ValidationIssue("clause", "no controlling clause supplied"))
    for clause in draft.controlling_clauses:
        if clause not in allowed_clauses:
            issues.append(ValidationIssue("clause", f"unknown policy clause {clause}"))

    fact_fields = {
        fact.field
        for fact in verified_facts
        if fact.status is FactStatus.VERIFIED
    }
    for reference in draft.factual_basis:
        field_name = re.split(r"\s*(?:=|:)\s*", reference, maxsplit=1)[0].strip()
        if field_name not in fact_fields and not context_only:
            missing_evidence = True
            issues.append(
                ValidationIssue(
                    "fact_evidence",
                    f"factual basis has no verified fact for {field_name}",
                )
            )
    if draft.decision in {"ALLOW", "ALLOW-CONDITIONAL"} and draft.unresolved_facts:
        missing_evidence = True
        issues.append(
            ValidationIssue(
                "fact_evidence",
                "allow decision still has unresolved facts",
            )
        )

    tools = dict(_tool_definition(tool) for tool in available_tools)
    for action in draft.required_actions:
        definition = tools.get(action.tool_name)
        if definition is None:
            issues.append(
                ValidationIssue(
                    "tool_schema",
                    f"unknown or unavailable action tool {action.tool_name}",
                )
            )
            continue
        issues.extend(_schema_issues(action, definition))
        for argument, value in action.arguments.items():
            prefix = _ID_PREFIXES.get(argument)
            if prefix and (
                not isinstance(value, str) or not value.startswith(prefix)
            ):
                issues.append(
                    ValidationIssue(
                        "id_type",
                        f"{argument} must start with {prefix}",
                    )
                )

    issues.extend(_action_compatibility_issues(draft))
    if _SENSITIVE_REASON_PATTERN.search(draft.customer_safe_reason):
        issues.append(
            ValidationIssue(
                "sensitive_information",
                "customer-safe reason contains restricted internal information",
            )
        )

    if not issues:
        return ValidationOutcome(ValidationRoute.READY_FOR_EXECUTION)
    if missing_evidence:
        return ValidationOutcome(ValidationRoute.REINVESTIGATE, issues)
    return ValidationOutcome(ValidationRoute.RETRY_ADJUDICATION, issues)


def adjudication_messages(
    *,
    request: RequestState,
    policy_text: str,
    verified_facts: list[Fact],
    unresolved_assertions: list[Fact],
    missing_evidence: list[str],
    available_tools: list[dict[str, Any]] | None = None,
    contextual_evidence: list[str] | None = None,
    validation_issues: list[ValidationIssue] | None = None,
    context_only: bool = False,
) -> list[dict[str, str]]:
    """Build a request-scoped, tool-free adjudication prompt."""

    schema = {
        "request_key": "string",
        "decision": "ALLOW | ALLOW-CONDITIONAL | DENY | ESCALATE",
        "controlling_clauses": ["exact policy clause ID"],
        "factual_basis": ["verified fact field or field=value"],
        "unresolved_facts": ["unresolved fact field"],
        "required_actions": [
            {"tool_name": "runtime action tool", "arguments": {}}
        ],
        "customer_safe_reason": "brief reason without internal sensitive data",
    }
    evidence = {
        "request": {
            "request_key": request.request_key,
            "request_type": request.request_type,
            "target_id": request.target_id,
        },
        "verified_facts": [
            {"field": fact.field, "value": fact.value}
            for fact in verified_facts
        ],
        "unresolved_assertions": [
            {"field": fact.field, "value": fact.value}
            for fact in unresolved_assertions
        ],
        "missing_evidence": missing_evidence,
        "contextual_evidence": contextual_evidence or [],
        "context_only_mode": context_only,
        "available_action_tools": [
            definition
            for tool in (available_tools or [])
            for name, definition in [_tool_definition(tool)]
            if name != "record_decision"
            and name not in {
                "lookup_order",
                "lookup_customer_profile",
                "check_return_eligibility",
                "lookup_employee",
                "verify_identity",
                "check_approval_status",
                "query_transaction_history",
                "lookup_account_events",
                "read_policy",
            }
        ],
        "previous_validation_issues": [
            {"code": issue.code, "message": issue.message}
            for issue in (validation_issues or [])
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Adjudicate exactly one request from authoritative policy and "
                "the supplied evidence. User assertions are unresolved unless "
                "also present as verified facts. When context_only_mode is "
                "true, the runtime deliberately supplied no lookup tools, so "
                "contextual_evidence is the authoritative scenario evidence "
                "and may support the decision. Use only listed action tools "
                "and exactly satisfy their parameter schemas. Correct every "
                "previous_validation_issue. Do not invent facts, IDs, "
                "clauses, or tools. Return one JSON object only.\n\n"
                f"AUTHORITATIVE POLICY:\n{policy_text}\n\n"
                f"REQUIRED JSON SHAPE:\n{json.dumps(schema)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(evidence, default=str),
        },
    ]
