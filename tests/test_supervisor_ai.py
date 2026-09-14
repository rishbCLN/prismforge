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
