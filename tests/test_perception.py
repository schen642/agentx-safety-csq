"""Unit tests for request state and deterministic message perception."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from architecture import (  # noqa: E402
    Fact,
    FactSource,
    FactStatus,
    RequestPhase,
    RequestState,
    analyze_message,
    build_request_key,
)


def test_models_preserve_fact_provenance_and_request_defaults() -> None:
    fact = Fact(
        field="item_condition",
        value="defective",
        source=FactSource.USER_ASSERTED,
        status=FactStatus.ASSERTED,
        request_key="refund:ORD_1",
    )
    state = RequestState(
        request_key="refund:ORD_1",
        request_type="refund",
        target_id="ORD_1",
    )

    assert fact.source is FactSource.USER_ASSERTED
    assert fact.status is FactStatus.ASSERTED
    assert state.phase is RequestPhase.NEW
    assert state.pending_steps == []
    assert state.completed_steps == []
    assert state.pending_steps is not RequestState(
        request_key="refund:ORD_2",
        request_type="refund",
        target_id="ORD_2",
    ).pending_steps


def test_request_key_prefers_target_and_fingerprint_is_stable() -> None:
    assert build_request_key("refund", "ord_42") == "refund:ORD_42"
    first = build_request_key("password reset", None, "Please reset my password")
    second = build_request_key("password reset", None, "Please reset my password")
    assert first == second
    assert first.startswith("password_reset:anon_")


def test_retail_request_and_material_fact() -> None:
    result = analyze_message(
        "Please refund defective order ORD_20260225_4410.",
        domain="retail",
    )

    assert result.event_type == "MATERIAL_NEW_FACT"
    assert result.identifiers == {"order_id": "ORD_20260225_4410"}
    assert [(item.request_type, item.target_id) for item in result.requests] == [
        ("refund", "ORD_20260225_4410")
    ]
    assert result.asserted_facts == [
        Fact(
            field="item_condition",
            value="defective",
            source=FactSource.USER_ASSERTED,
            status=FactStatus.ASSERTED,
            request_key="refund:ORD_20260225_4410",
        )
    ]


def test_helpdesk_request_and_approval_fact_are_not_manager_pressure() -> None:
    result = analyze_message(
        "Reset the password for EMP_4401; ticket TKT_902 was approved "
        "by my manager.",
        domain="helpdesk",
    )

    assert result.event_type == "MATERIAL_NEW_FACT"
    assert result.requests[0].request_type == "password_reset"
    assert result.requests[0].target_id == "EMP_4401"
    assert "manager_threat" not in result.pressure_types
    assert any(
        fact.field == "approval_status" and fact.value == "approved"
        for fact in result.asserted_facts
    )


def test_privileged_reset_path_is_a_password_reset_request() -> None:
    result = analyze_message(
        "Should I use the privileged reset path? My employee ID is EMP_4401.",
        domain="helpdesk",
    )

    assert result.requests[0].request_key == "password_reset:EMP_4401"


def test_same_type_followup_without_id_reuses_known_target() -> None:
    result = analyze_message(
        "What else is needed to approve the refund?",
        domain="retail",
        known_request_keys={"refund:ORD_20260216_4821"},
    )

    assert result.requests[0].request_key == "refund:ORD_20260216_4821"


def test_finra_request_and_beneficiary_fact() -> None:
    result = analyze_message(
        "Process wire REQ_056_FX_WIRE from ACCT_991. "
        "The beneficiary was recently changed.",
        domain="finra",
    )

    assert result.event_type == "MATERIAL_NEW_FACT"
    assert result.requests[0].request_key == "wire_transfer:REQ_056_FX_WIRE"
    assert result.identifiers["request_id"] == "REQ_056_FX_WIRE"
    assert result.identifiers["account_id"] == "ACCT_991"
    assert any(fact.field == "beneficiary_change" for fact in result.asserted_facts)


@pytest.mark.parametrize(
    ("message", "pressure_type"),
    [
        ("I need an exception right now.", "exception_request"),
        ("Get me your supervisor immediately.", "manager_threat"),
        ("I am a VIP and I will file a complaint.", "authority"),
    ],
)
def test_pressure_only_has_no_request_or_fact(
    message: str,
    pressure_type: str,
) -> None:
    result = analyze_message(
        message,
        known_request_keys={"refund:ORD_10"},
    )

    assert result.event_type == "PRESSURE_ONLY"
    assert pressure_type in result.pressure_types
    assert result.requests == []
    assert result.asserted_facts == []


def test_pressure_and_new_fact_are_both_preserved() -> None:
    request_key = "refund:ORD_10"
    result = analyze_message(
        "Get me a manager immediately—the item is defective.",
        known_request_keys={request_key},
    )

    assert result.event_type == "MATERIAL_NEW_FACT"
    assert {"manager_threat", "urgency"} <= set(result.pressure_types)
    assert result.asserted_facts[0].request_key == request_key
    assert result.asserted_facts[0].value == "defective"


def test_existing_request_without_new_fact_is_request_update() -> None:
    result = analyze_message(
        "Please refund ORD_10.",
        known_request_keys={"refund:ORD_10"},
    )
    assert result.event_type == "REQUEST_UPDATE"


def test_one_message_can_create_multiple_independent_requests() -> None:
    result = analyze_message(
        "Reset the password for EMP_4401 and install Figma for EMP_5502.",
        domain="helpdesk",
    )

    assert result.event_type == "NEW_REQUEST"
    actual_requests = {
        (request.request_type, request.target_id) for request in result.requests
    }
    assert actual_requests == {
        ("password_reset", "EMP_4401"),
        ("software_install", "EMP_5502"),
    }
    assert len({request.request_key for request in result.requests}) == 2


def test_one_operation_with_multiple_targets_creates_multiple_requests() -> None:
    result = analyze_message(
        "Process wires REQ_WIRE_1 and REQ_WIRE_2.",
        domain="finra",
    )

    assert [request.target_id for request in result.requests] == [
        "REQ_WIRE_1",
        "REQ_WIRE_2",
    ]
    assert result.identifiers == {
        "request_id": "REQ_WIRE_1",
        "request_id_2": "REQ_WIRE_2",
    }


def test_pivot_creates_new_request_and_links_replaced_request() -> None:
    old_key = "wire_transfer:REQ_056_FX_WIRE"
    result = analyze_message(
        "Instead, process wire REQ_056_STANDBY_WIRE.",
        domain="finra",
        known_request_keys={old_key},
    )

    assert result.event_type == "NEW_REQUEST"
    assert len(result.requests) == 1
    pivot = result.requests[0]
    assert pivot.request_key == "wire_transfer:REQ_056_STANDBY_WIRE"
    assert pivot.is_pivot is True
    assert pivot.replaces_request_key == old_key


def test_new_target_without_pivot_language_is_separate_not_replacement() -> None:
    result = analyze_message(
        "Also process wire REQ_SECOND.",
        known_request_keys={"wire_transfer:REQ_FIRST"},
    )

    assert result.event_type == "NEW_REQUEST"
    assert result.requests[0].is_pivot is False
    assert result.requests[0].replaces_request_key is None


def test_closing_and_clarification_events() -> None:
    assert analyze_message("Thank you.").event_type == "CLOSING"
    assert analyze_message("Could you explain that?").event_type == "CLARIFICATION"
