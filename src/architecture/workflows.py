"""Investigation workflow registry for architecture Phase D."""

from __future__ import annotations

from collections.abc import Collection

_RETAIL_LOOKUPS = (
    "lookup_order",
    "lookup_customer_profile",
    "check_return_eligibility",
)
_HELPDESK_FULL_LOOKUPS = (
    "lookup_employee",
    "verify_identity",
    "check_approval_status",
)
_HELPDESK_APPROVAL_LOOKUPS = (
    "lookup_employee",
    "check_approval_status",
)
_FINRA_LOOKUPS = (
    "lookup_customer_profile",
    "query_transaction_history",
    "lookup_account_events",
)


INVESTIGATION_WORKFLOWS: dict[tuple[str, str], tuple[str, ...]] = {
    ("retail", "refund"): _RETAIL_LOOKUPS,
    ("retail", "exchange"): _RETAIL_LOOKUPS,
    ("retail", "sensitive_disclosure"): _RETAIL_LOOKUPS,
    ("retail", "generic"): _RETAIL_LOOKUPS,
    ("helpdesk", "password_reset"): _HELPDESK_FULL_LOOKUPS,
    ("helpdesk", "account_unlock"): _HELPDESK_FULL_LOOKUPS,
    ("helpdesk", "software_install"): _HELPDESK_APPROVAL_LOOKUPS,
    ("helpdesk", "access_provisioning"): _HELPDESK_APPROVAL_LOOKUPS,
    ("helpdesk", "sensitive_disclosure"): _HELPDESK_FULL_LOOKUPS,
    ("helpdesk", "generic"): _HELPDESK_FULL_LOOKUPS,
    ("finra", "wire_transfer"): _FINRA_LOOKUPS,
    ("finra", "trade"): _FINRA_LOOKUPS,
    ("finra", "hold_release"): _FINRA_LOOKUPS,
    ("finra", "sensitive_disclosure"): _FINRA_LOOKUPS,
    ("finra", "generic"): _FINRA_LOOKUPS,
}

KNOWN_LOOKUP_TOOLS = frozenset(
    tool
    for workflow in INVESTIGATION_WORKFLOWS.values()
    for tool in workflow
)


def workflow_for(domain: str, request_type: str) -> tuple[str, ...]:
    """Return the configured ordered lookup sequence, if supported."""

    normalized_domain = domain.lower()
    return INVESTIGATION_WORKFLOWS.get(
        (normalized_domain, request_type),
        INVESTIGATION_WORKFLOWS.get((normalized_domain, "generic"), ()),
    )


def available_workflow_steps(
    domain: str,
    request_type: str,
    available_tool_names: Collection[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split required lookups into executable and missing ordered steps."""

    required = workflow_for(domain, request_type)
    available = set(available_tool_names)
    return (
        tuple(step for step in required if step in available),
        tuple(step for step in required if step not in available),
    )
