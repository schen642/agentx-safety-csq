"""Single fail-closed registry for runtime tool classification."""

from __future__ import annotations

from enum import StrEnum


class RegisteredToolKind(StrEnum):
    LOOKUP = "lookup"
    ACTION = "action"
    DECISION = "decision"


LOOKUP_TOOLS = frozenset(
    {
        "check_approval_status",
        "check_return_eligibility",
        "lookup_account_events",
        "lookup_certificate_deposits",
        "lookup_customer_profile",
        "lookup_employee",
        "lookup_order",
        "lookup_related_account_activity",
        "lookup_security_info",
        "query_transaction_history",
        "read_policy",
        "verify_identity",
    }
)

ACTION_TOOLS = frozenset(
    {
        "apply_store_credit",
        "create_access_request",
        "create_alert",
        "deny_refund",
        "dismiss_alert",
        "escalate_to_compliance",
        "escalate_to_it_security",
        "escalate_to_manager",
        "escalate_to_tier2",
        "execute_trade",
        "file_ctr",
        "file_sar",
        "flag_account",
        "hold_transaction",
        "install_software",
        "journal_security_positions",
        "log_ticket",
        "open_case",
        "process_annuity_cancellation",
        "process_cash_deposit",
        "process_exchange",
        "process_refund",
        "process_wire_transfer",
        "provision_vpn_access",
        "release_hold",
        "reset_password",
        "unlock_account",
        "update_beneficiary_designation",
    }
)

REGISTERED_TOOLS = LOOKUP_TOOLS | ACTION_TOOLS | {"record_decision"}

IRREVERSIBLE_TOOLS = frozenset(
    {
        "execute_trade",
        "install_software",
        "journal_security_positions",
        "process_annuity_cancellation",
        "process_cash_deposit",
        "process_exchange",
        "process_refund",
        "process_wire_transfer",
        "provision_vpn_access",
        "release_hold",
        "reset_password",
        "unlock_account",
        "update_beneficiary_designation",
    }
)

DOMAIN_TOOL_PRIORITIES: dict[str, dict[str, int]] = {
    "retail": {
        "flag_account": 10,
        "deny_refund": 20,
        "process_refund": 20,
        "process_exchange": 20,
        "apply_store_credit": 20,
        "escalate_to_manager": 30,
    },
    "helpdesk": {
        "unlock_account": 10,
        "reset_password": 20,
        "provision_vpn_access": 20,
        "install_software": 20,
        "create_access_request": 20,
        "log_ticket": 30,
        "escalate_to_tier2": 40,
        "escalate_to_it_security": 40,
    },
    "finra": {
        "hold_transaction": 10,
        "process_wire_transfer": 10,
        "process_cash_deposit": 10,
        "execute_trade": 10,
        "journal_security_positions": 10,
        "process_annuity_cancellation": 10,
        "update_beneficiary_designation": 10,
        "create_alert": 20,
        "open_case": 30,
        "file_ctr": 35,
        "file_sar": 35,
        "escalate_to_compliance": 40,
    },
}


def registered_tool_kind(name: str) -> RegisteredToolKind | None:
    if name == "record_decision":
        return RegisteredToolKind.DECISION
    if name in LOOKUP_TOOLS:
        return RegisteredToolKind.LOOKUP
    if name in ACTION_TOOLS:
        return RegisteredToolKind.ACTION
    return None
