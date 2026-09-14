"""PRISM Observability Client & Safety Layer.
Provides authenticated access to the PRISM Cloud API with strict built-in
safeguards against unintended credit consumption.
"""
import os
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Any, Optional


class CreditConfirmationRequiredError(Exception):
    """Raised when a credit-spending PRISM endpoint is invoked without explicit confirmation."""
    def __init__(self, action: str, cost_credits: int, endpoint: str):
        self.action = action
        self.cost_credits = cost_credits
        self.endpoint = endpoint
        super().__init__(
            f"Safety Guard: Action '{action}' costs {cost_credits} PRISM credit(s) at {endpoint}. "
            "To execute, you must explicitly pass `confirm_spend=True` after receiving user approval."
        )


class PrismClient:
    """Safe, idiomatic Python client for the PRISM Observability API."""

    DEFAULT_HOST = "https://prism-api-prod.up.railway.app"
    DEFAULT_PROJECT_ID = "e430ddf1-9e0f-425b-a438-bbbc3977f140"
    DEFAULT_ORG_ID = "afa090bb-98bc-48e4-8c9d-009028e16f20"

    def __init__(
        self,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        org_id: Optional[str] = None,
        api_key: Optional[str] = None,
        env_file: Optional[str] = ".env"
    ):
        # Load from .env file if present and values not already in env
        if env_file and os.path.exists(env_file):
            self._load_env_file(env_file)

        self.host = (host or os.environ.get("PRISMTRACE_HOST") or self.DEFAULT_HOST).rstrip("/")
        self.project_id = project_id or os.environ.get("PRISMTRACE_PROJECT_ID") or self.DEFAULT_PROJECT_ID
        self.org_id = org_id or os.environ.get("PRISMTRACE_ORG_ID") or self.DEFAULT_ORG_ID
        self.api_key = api_key or os.environ.get("PRISMTRACE_API_KEY", "")

    @staticmethod
    def _load_env_file(filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PRISMForge-Python-Client/1.0"
        }
        if self.api_key:
            headers["X-PRISMtrace-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.host}{path}"
        if params:
            query_str = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query_str:
                url = f"{url}?{query_str}"

        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._get_headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_text = resp.read().decode("utf-8")
                return json.loads(resp_text) if resp_text else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = {"raw_error": err_body}
            if e.code == 402:
                raise RuntimeError(f"PRISM Credit Depletion (402 Payment Required): {err_json}") from e
            raise RuntimeError(f"PRISM API error [{e.code}]: {err_json}") from e
        except Exception as e:
            raise RuntimeError(f"PRISM network request failed: {e}") from e

    # --------------------------------------------------------------------------
    # Free Reading Endpoints ($0 / 0 Credits)
    # --------------------------------------------------------------------------

    def get_setup_doctor(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Checks connection status and reports if live traces are arriving."""
        pid = project_id or self.project_id
        return self._request("GET", "/api/setup-doctor", params={"project_id": pid})

    def list_traces(
        self,
        status: str = "failed",
        page: int = 1,
        page_size: int = 20,
        score_min: Optional[float] = None,
        score_max: Optional[float] = None,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Lists traces. `status` is required: 'failed', 'error', 'blocked', 'flagged', or 'success'."""
        valid_statuses = {"failed", "error", "blocked", "flagged", "success", "broken", "failing", "failure"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

        params = {
            "project_id": self.project_id,
            "status": status,
            "page": page,
            "page_size": page_size,
            "score_min": score_min,
            "score_max": score_max,
            "agent_name": agent_name
        }
        return self._request("GET", "/api/traces", params=params)

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Fetches full payload, inputs, outputs, and latency for a single trace."""
        return self._request("GET", f"/api/traces/{trace_id}")

    def get_spans(self, trace_id: str) -> Dict[str, Any]:
        """Retrieves the hierarchical span tree for tool calls and nested model calls."""
        return self._request("GET", f"/api/spans/{trace_id}")

    def get_scores_summary(self) -> Dict[str, Any]:
        """Fetches aggregate evaluation and model scores."""
        return self._request("GET", "/api/scores/summary", params={"project_id": self.project_id})

    def get_metrics_summary(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Fetches volume, latency, and throughput metrics."""
        params = {"project_id": self.project_id}
        if period:
            params["period"] = period
        return self._request("GET", "/api/metrics/summary", params=params)

    def get_intelligence(self, days: int = 7, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetches SQL aggregations on intent resolution, risk trends, and coverage."""
        params = {"project_id": self.project_id, "days": days, "agent_name": agent_name}
        return self._request("GET", "/api/intelligence", params=params)

    def list_clusters(self, status: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Lists existing Root Cause Analysis failure clusters computed for this project."""
        params = {"project_id": self.project_id, "status": status, "limit": limit}
        return self._request("GET", "/api/intelligence/clusters", params=params)

    def get_credit_balance(self) -> Dict[str, Any]:
        """Checks remaining PRISM credit balance for the organization."""
        return self._request("GET", "/api/credits/balance", params={"org_id": self.org_id})

    def get_credit_ledger(self) -> Dict[str, Any]:
        """Inspects past credit expenditures and timestamped ledger entries."""
        return self._request("GET", "/api/credits/ledger", params={"org_id": self.org_id})

    def check_active_backfill(self) -> Dict[str, Any]:
        """Checks whether a trace analysis backfill job is currently running."""
        return self._request("GET", "/api/backfill/trace-analyses/active", params={"project_id": self.project_id})

    # --------------------------------------------------------------------------
    # Guarded Paid Endpoints (Require Explicit `confirm_spend=True`)
    # --------------------------------------------------------------------------

    def analyze_clusters(self, confirm_spend: bool = False) -> Dict[str, Any]:
        """Runs heavy Root-Cause Analysis over project failures. COSTS 5 CREDITS."""
        if not confirm_spend:
            raise CreditConfirmationRequiredError(
                action="Run Root-Cause Analysis (Cluster Analysis)",
                cost_credits=5,
                endpoint="POST /api/intelligence/clusters/analyze"
            )
        return self._request("POST", "/api/intelligence/clusters/analyze", body={"project_id": self.project_id})

    def recommend_fix(self, trace_id: str, user_id: str = "supervisor", confirm_spend: bool = False) -> Dict[str, Any]:
        """Generates AI fix recommendations for one trace findings. COSTS 2 CREDITS."""
        if not confirm_spend:
            raise CreditConfirmationRequiredError(
                action="Generate Fix Recommendations",
                cost_credits=2,
                endpoint="POST /api/remediation/recommend"
            )
        body = {"project_id": self.project_id, "trace_id": trace_id, "user_id": user_id}
        return self._request("POST", "/api/remediation/recommend", body=body)

    def backfill_trace_analyses(self, mode: str = "classify", max_credits: Optional[int] = None, confirm_spend: bool = False) -> Dict[str, Any]:
        """Re-runs analysis pipeline over unanalyzed traces. COSTS 1 CREDIT PER TRACE."""
        if not confirm_spend:
            raise CreditConfirmationRequiredError(
                action="Backfill Trace Analysis",
                cost_credits=max_credits or 1,
                endpoint="POST /api/backfill/trace-analyses"
            )
        body: Dict[str, Any] = {"project_id": self.project_id, "mode": mode}
        if max_credits is not None:
            body["max_credits"] = max_credits
        return self._request("POST", "/api/backfill/trace-analyses", body=body)

    def generate_fix(self, cluster_id: str, confirm_spend: bool = False) -> Dict[str, Any]:
        """Attempts an automated code fix / PR generation for an RCA cluster. COSTS 5 CREDITS."""
        if not confirm_spend:
            raise CreditConfirmationRequiredError(
                action="Automated Code Fix Generation",
                cost_credits=5,
                endpoint="POST /api/rca/remediation/generate-fix"
            )
        body = {"project_id": self.project_id, "cluster_id": cluster_id}
        return self._request("POST", "/api/rca/remediation/generate-fix", body=body)
