"""Tests for Safe PRISM Observability Client & Credit Safeguards."""
import pytest
from unittest.mock import patch, MagicMock
from src.prism.client import PrismClient, CreditConfirmationRequiredError


def test_prism_client_headers():
    client = PrismClient(api_key="pt-sk-testkey123", project_id="proj-abc", org_id="org-xyz")
    headers = client._get_headers()
    assert headers["X-PRISMtrace-Key"] == "pt-sk-testkey123"
    assert "Authorization" not in headers


def test_guarded_paid_endpoints_block_by_default():
    client = PrismClient(api_key="pt-sk-testkey123")

    # 1. Analyze clusters (5 credits) must raise error
    with pytest.raises(CreditConfirmationRequiredError) as exc1:
        client.analyze_clusters(confirm_spend=False)
    assert "costs 5 PRISM credit(s)" in str(exc1.value)

    # 2. Recommend fix (2 credits) must raise error
    with pytest.raises(CreditConfirmationRequiredError) as exc2:
        client.recommend_fix(trace_id="tr-123", confirm_spend=False)
    assert "costs 2 PRISM credit(s)" in str(exc2.value)

    # 3. Backfill analyses (1 credit/trace) must raise error
    with pytest.raises(CreditConfirmationRequiredError) as exc3:
        client.backfill_trace_analyses(confirm_spend=False)
    assert "costs 1 PRISM credit(s)" in str(exc3.value)

    # 4. Generate fix (5 credits) must raise error
    with pytest.raises(CreditConfirmationRequiredError) as exc4:
        client.generate_fix(cluster_id="cl-456", confirm_spend=False)
    assert "costs 5 PRISM credit(s)" in str(exc4.value)


@patch("urllib.request.urlopen")
def test_free_endpoints_execute_without_blocking(mock_urlopen):
    # Mock successful server response
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"live_connected": true, "overall": "connected"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    client = PrismClient(api_key="pt-sk-testkey123")
    res = client.get_setup_doctor()

    assert res["live_connected"] is True
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_confirmed_paid_endpoint_executes_when_approved(mock_urlopen):
    # Mock successful server response
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"analysis_id": "rca-999", "status": "running"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    client = PrismClient(api_key="pt-sk-testkey123")
    # Explicitly confirm spend
    res = client.analyze_clusters(confirm_spend=True)

    assert res["analysis_id"] == "rca-999"
    assert mock_urlopen.called
