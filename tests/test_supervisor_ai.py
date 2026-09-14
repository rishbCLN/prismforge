"""Tests for Grounded Warehouse AI Supervisor Assistant."""
import pytest
from src.assistant.supervisor_ai import WarehouseSupervisorAssistant


def test_supervisor_assistant_initialization():
    assistant = WarehouseSupervisorAssistant()
    assert len(assistant.telemetry) > 0


def test_supervisor_assistant_high_risk_query():
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("Show me all high-risk handling events from today's unloading")

    assert res["query_type"] == "HIGH_RISK_EVENTS"
    assert res["count"] > 0
    assert "High-Risk" in res["response"] or "HIGH" in res["response"]


def test_supervisor_assistant_explanation_query():
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("Why was clip 02 classified as high risk?")

    assert res["query_type"] == "INCIDENT_EXPLANATION"
    assert "DROP" in res["response"] or "descent" in res["response"] or "impact" in res["response"]


def test_supervisor_assistant_bay_analysis_query():
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("Which loading bay had the highest number of risky events?")

    assert res["query_type"] == "BAY_ANALYSIS"
    assert "Bay" in res["response"]


def test_supervisor_assistant_shift_summary():
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("Give me an overview of today's shift")

    assert res["query_type"] == "SHIFT_SUMMARY"
    assert "Monitored Sequences" in res["response"]


def test_supervisor_assistant_ppe_risks_query():
    """Validates that 'what risks did we find todays' queries the PPE database directly."""
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("what risks did we find todays")

    assert res["query_type"] == "PPE_DATABASE_RISKS"
    assert res["domain"] == "PPE Construction Safety"
    assert res["quality_score"] == 0.98
    assert res["response_quality"] == 0.98
    assert res["compliance_risk"] == 0.048
    assert "PPE Database Risk & Safety Audit" in res["response"]
    assert "22,740" in res["response"]
    assert "Missing Cranial Hardhats" in res["response"]
    assert "Held-Hardhat Non-Compliance" in res["response"]


def test_supervisor_assistant_held_hardhat_query():
    """Validates that questions about held helmets return anatomical gating explanations."""
    assistant = WarehouseSupervisorAssistant()
    res = assistant.query("Why is a worker holding a hardhat in hand considered non compliant?")

    assert res["query_type"] == "HELD_HARDHAT_EXPLANATION"
    assert res["compliance_status"] == "NON_COMPLIANT_IF_HELD"
    assert "cranial dome" in res["response"].lower()
    assert "h_iou_head" in res["response"]

