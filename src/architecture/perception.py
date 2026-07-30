"""High-precision, rule-based perception for Pi-Bench-style messages.

This module classifies observable message content only. It never makes a
policy decision and never invokes or selects a tool.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass

from .models import Fact, FactSource, FactStatus, PerceptionResult, RequestIntent


EVENT_TYPES = frozenset(
    {
        "NEW_REQUEST",
        "REQUEST_UPDATE",
        "MATERIAL_NEW_FACT",
        "PRESSURE_ONLY",
        "CLARIFICATION",
        "CLOSING",
    }
)

_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("order_id", re.compile(r"\bORD_[A-Z0-9_]+\b", re.IGNORECASE)),
    ("request_id", re.compile(r"\bREQ_[A-Z0-9_]+\b", re.IGNORECASE)),
    ("employee_id", re.compile(r"\bEMP_[A-Z0-9_]+\b", re.IGNORECASE)),
    ("ticket_id", re.compile(r"\bTKT_[A-Z0-9_]+\b", re.IGNORECASE)),
    ("account_id", re.compile(r"\bACCT_[A-Z0-9_]+\b", re.IGNORECASE)),
    ("customer_id", re.compile(r"\bCUST_[A-Z0-9_]+\b", re.IGNORECASE)),
)

_PRESSURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "manager_threat",
        re.compile(
            r"\b(?:speak|talk|complain|escalate|take this)\b.{0,30}"
            r"\b(?:manager|supervisor|management)\b|"
            r"\b(?:get|find) (?:me )?(?:(?:a|the|your) )?"
            r"(?:manager|supervisor)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "complaint_threat",
        re.compile(
            r"\b(?:file|make|submit) (?:a )?complaint\b|"
            r"\b(?:lawyer|attorney|lawsuit|sue|chargeback|"
            r"better business bureau|social media)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "urgency",
        re.compile(
            r"\b(?:urgent|urgently|asap|immediately|right now|emergency|"
            r"time[- ]sensitive|before (?:the )?deadline)\b|"
            r"\b(?:in|within) \d+ (?:minutes?|hours?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "authority",
        re.compile(
            r"\b(?:vip|executive|c[- ]?suite|vice president|director|"
            r"senior leader|gold member|platinum member|loyal customer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exception_request",
        re.compile(
            r"\b(?:make|grant|give|need) (?:me |an? )?exception\b|"
            r"\boverride (?:the |this )?(?:rule|policy|process)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "emotional_appeal",
        re.compile(
            r"\b(?:desperate|devastated|overwhelmed|stressed out|"
            r"this is ruining|my family (?:needs|depends))\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class _OperationMatch:
    request_type: str
    operation: str
    position: int
    end: int
    compatible_ids: tuple[str, ...]


_OPERATION_PATTERNS: tuple[
    tuple[str, str, tuple[str, ...], re.Pattern[str]], ...
] = (
    (
        "refund",
        "refund",
        ("order_id",),
        re.compile(r"\b(?:refund|return(?:ing)?|money back)\b", re.IGNORECASE),
    ),
    (
        "exchange",
        "exchange",
        ("order_id",),
        re.compile(r"\b(?:exchange|replacement)\b", re.IGNORECASE),
    ),
    (
        "password_reset",
        "password_reset",
        ("employee_id", "ticket_id"),
        re.compile(
            r"\b(?:reset|change|forgot|recover)\b.{0,25}\bpassword\b|"
            r"\bpassword\b.{0,25}\b(?:reset|change|recovery)\b|"
            r"\b(?:(?:admin|privileged)\s+)?reset path\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_unlock",
        "account_unlock",
        ("employee_id", "account_id", "ticket_id"),
        re.compile(
            r"\b(?:unlock|unblock)\b.{0,20}\b(?:account|login|user)\b|"
            r"\b(?:account|login|user)\b.{0,20}\b(?:locked|blocked)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "software_install",
        "software_install",
        ("employee_id", "ticket_id"),
        re.compile(
            r"\b(?:install|installation|deploy)\b.{0,30}"
            r"\b(?:software|application|app|program|figma|slack|zoom)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "access_provisioning",
        "access_provisioning",
        ("employee_id", "ticket_id", "account_id"),
        re.compile(
            r"\b(?:grant|provide|request|need)\b.{0,25}"
            r"\b(?:access|permission|privilege|role)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wire_transfer",
        "wire_transfer",
        ("request_id", "account_id"),
        re.compile(
            r"\b(?:wires?|wire transfers?|transfer funds|send funds)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trade",
        "trade",
        ("request_id", "account_id"),
        re.compile(
            r"\b(?:place|execute|cancel|reverse|unwind)\b.{0,20}"
            r"\b(?:trade|order|transaction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hold_release",
        "hold_release",
        ("request_id", "account_id"),
        re.compile(
            r"\b(?:release|remove|lift|clear)\b.{0,20}\bhold\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sensitive_disclosure",
        "disclose",
        ("customer_id", "account_id"),
        re.compile(
            r"\b(?:show|tell|reveal|disclose|give)\b.{0,30}"
            r"\b(?:fraud score|risk score|internal score|account flags?|"
            r"fraud flags?|watchlist status)\b",
            re.IGNORECASE,
        ),
    ),
)

_MATERIAL_FACT_PATTERNS: tuple[
    tuple[str, object, re.Pattern[str]], ...
] = (
    ("item_condition", "defective", re.compile(r"\bdefective\b", re.IGNORECASE)),
    (
        "item_condition",
        "damaged",
        re.compile(r"\b(?:damaged|broken|cracked)\b", re.IGNORECASE),
    ),
    (
        "item_condition",
        "wrong_item",
        re.compile(r"\b(?:wrong|incorrect) item\b", re.IGNORECASE),
    ),
    (
        "item_final_sale",
        True,
        re.compile(r"\bfinal[- ]sale\b", re.IGNORECASE),
    ),
    (
        "approval_status",
        "approved",
        re.compile(
            r"\b(?:ticket|request|access|reset|install(?:ation)?)\b.{0,30}"
            r"\b(?:is |was |has been )?approved\b|"
            r"\bapproved\b.{0,30}\b(?:ticket|request|access|reset|install(?:ation)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "identity_information",
        "provided",
        re.compile(
            r"\b(?:date of birth|dob|last four|last 4|employee number|"
            r"verification code|security answer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "beneficiary_change",
        True,
        re.compile(
            r"\b(?:new|changed|updated|recently added)\b.{0,20}"
            r"\b(?:beneficiary|recipient|payee)\b|"
            r"\b(?:beneficiary|recipient|payee)\b.{0,20}"
            r"\b(?:new|changed|updated|recently added)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_privilege",
        "privileged",
        re.compile(
            r"\b(?:admin|administrator|privileged|root) account\b",
            re.IGNORECASE,
        ),
    ),
    (
        "device_activated",
        True,
        re.compile(
            r"\b(?:device|phone|laptop|product)\b.{0,20}"
            r"\b(?:activated|registered|used)\b",
            re.IGNORECASE,
        ),
    ),
)

_PIVOT_PATTERN = re.compile(
    r"\b(?:instead|switch to|change (?:it|this) to|different request|"
    r"alternate|alternative|rather than)\b",
    re.IGNORECASE,
)
_CLOSING_PATTERN = re.compile(
    r"^\s*(?:ok(?:ay)?|thanks?|thank you|got it|understood|goodbye|bye)"
    r"[\s.!]*$",
    re.IGNORECASE,
)


def _identifier_occurrences(text: str) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    for kind, pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            found.append((kind, match.group(0).upper(), match.start()))
    return sorted(found, key=lambda item: item[2])


def extract_identifiers(text: str) -> dict[str, str]:
    """Extract identifiers without discarding repeated identifiers of one kind."""

    result: dict[str, str] = {}
    counts: dict[str, int] = {}
    for kind, value, _ in _identifier_occurrences(text):
        counts[kind] = counts.get(kind, 0) + 1
        key = kind if counts[kind] == 1 else f"{kind}_{counts[kind]}"
        result[key] = value
    return result


def detect_pressure(text: str) -> list[str]:
    """Return canonical pressure labels in stable priority order."""

    return [
        pressure_type
        for pressure_type, pattern in _PRESSURE_PATTERNS
        if pattern.search(text)
    ]


def build_request_key(
    request_type: str,
    target_id: str | None,
    message: str = "",
) -> str:
    """Build a stable business-object key or a deterministic anonymous key."""

    normalized_type = re.sub(r"[^a-z0-9]+", "_", request_type.lower()).strip("_")
    if target_id:
        return f"{normalized_type}:{target_id.upper()}"
    normalized_message = " ".join(re.findall(r"[a-z0-9]+", message.lower()))
    digest = hashlib.sha256(normalized_message.encode("utf-8")).hexdigest()[:12]
    return f"{normalized_type}:anon_{digest}"


def _operation_matches(text: str) -> list[_OperationMatch]:
    matches: list[_OperationMatch] = []
    for request_type, operation, compatible_ids, pattern in _OPERATION_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                _OperationMatch(
                    request_type=request_type,
                    operation=operation,
                    position=match.start(),
                    end=match.end(),
                    compatible_ids=compatible_ids,
                )
            )
    return sorted(matches, key=lambda item: item.position)


def _targets_for_operation(
    operation: _OperationMatch,
    identifiers: list[tuple[str, str, int]],
    operation_count: int,
) -> list[str | None]:
    candidates = [
        (value, position)
        for kind, value, position in identifiers
        if kind in operation.compatible_ids
    ]
    if not candidates:
        return [None]
    if operation_count == 1:
        return [value for value, _ in candidates]
    nearest = min(candidates, key=lambda item: abs(item[1] - operation.end))
    return [nearest[0]]


def _requests_from_message(
    text: str,
    known_request_keys: Collection[str],
) -> list[RequestIntent]:
    identifiers = _identifier_occurrences(text)
    operations = _operation_matches(text)
    pivot_language = bool(_PIVOT_PATTERN.search(text))
    results: list[RequestIntent] = []
    seen: set[tuple[str, str | None]] = set()

    for operation in operations:
        targets = _targets_for_operation(operation, identifiers, len(operations))
        for target in targets:
            matching_known = [
                key
                for key in known_request_keys
                if key.split(":", 1)[0] == operation.request_type
            ]
            if target is None and len(matching_known) == 1 and not pivot_language:
                existing_key = matching_known[0]
                existing_target = existing_key.split(":", 1)[1]
                if not existing_target.startswith("anon_"):
                    target = existing_target
            identity = (operation.request_type, target)
            if identity in seen:
                continue
            seen.add(identity)
            request_key = build_request_key(operation.request_type, target, text)
            replacement = None
            if pivot_language and request_key not in known_request_keys:
                replacement = next(
                    (
                        key
                        for key in known_request_keys
                        if key.split(":", 1)[0] == operation.request_type
                    ),
                    None,
                )
            results.append(
                RequestIntent(
                    request_type=operation.request_type,
                    target_id=target,
                    requested_operation=operation.operation,
                    request_key=request_key,
                    is_pivot=replacement is not None,
                    replaces_request_key=replacement,
                )
            )
    return results


def _fact_candidates(text: str) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    seen: set[tuple[str, object]] = set()
    for field, value, pattern in _MATERIAL_FACT_PATTERNS:
        if pattern.search(text) and (field, value) not in seen:
            seen.add((field, value))
            candidates.append((field, value))
    return candidates


def extract_material_facts(
    text: str,
    request_key: str | None = None,
) -> list[Fact]:
    """Extract user assertions; none are promoted to verified facts."""

    return [
        Fact(
            field=field,
            value=value,
            source=FactSource.USER_ASSERTED,
            status=FactStatus.ASSERTED,
            request_key=request_key,
        )
        for field, value in _fact_candidates(text)
    ]


def _classify_event(
    text: str,
    requests: list[RequestIntent],
    facts: list[Fact],
    pressure: list[str],
    known_request_keys: set[str],
) -> str:
    if facts:
        return "MATERIAL_NEW_FACT"
    if requests:
        if any(request.request_key not in known_request_keys for request in requests):
            return "NEW_REQUEST"
        return "REQUEST_UPDATE"
    if pressure:
        return "PRESSURE_ONLY"
    if _CLOSING_PATTERN.match(text):
        return "CLOSING"
    return "CLARIFICATION"


def analyze_message(
    text: str,
    *,
    domain: str | None = None,
    known_request_keys: Collection[str] = (),
) -> PerceptionResult:
    """Analyze one message without changing agent, request, or tool state.

    ``domain`` is accepted for a stable future-facing API. Phase B rules are
    deliberately domain-neutral and do not make policy decisions.
    """

    del domain
    known = set(known_request_keys)
    requests = _requests_from_message(text, known)
    pressure = detect_pressure(text)

    default_key: str | None = None
    if len(requests) == 1:
        default_key = requests[0].request_key
    elif not requests and len(known) == 1:
        default_key = next(iter(known))
    facts = extract_material_facts(text, default_key)

    return PerceptionResult(
        requests=requests,
        event_type=_classify_event(text, requests, facts, pressure, known),
        pressure_types=pressure,
        asserted_facts=facts,
        identifiers=extract_identifiers(text),
    )


class PerceptionEngine:
    """Small injectable facade around the stateless perception function."""

    def analyze(
        self,
        text: str,
        *,
        domain: str | None = None,
        known_request_keys: Collection[str] = (),
    ) -> PerceptionResult:
        return analyze_message(
            text,
            domain=domain,
            known_request_keys=known_request_keys,
        )
