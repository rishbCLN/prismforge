# PRISMForge — Standing Instructions for Coding Agents

This repository houses the **PRISMForge / HazardMesh / DamageMesh** physical safety and material handling intelligence platform.

All coding agents, automated workflows, and contributors operating in this repository must strictly adhere to the following standing instructions.

---

## 1. PRISM Tracing & Observability Standing Rules

PRISM is the live observability and evaluation platform for AI agents and models in this system.

### Connection & Authentication
- Tracing host: `https://prism-api-prod.up.railway.app`
- Default Project ID: `e430ddf1-9e0f-425b-a438-bbbc3977f140`
- Org ID: `afa090bb-98bc-48e4-8c9d-009028e16f20`
- Header format: All requests MUST authenticate with **`X-PRISMtrace-Key: <api_key>`**.
  - **Never** use `Authorization: Bearer` with the API key (that is for dashboard user sessions only).
  - **Never** commit API keys or hardcode secrets into source files. Load them from `os.environ` or the untracked `.env` file. Keep `.env.example` committed with template names only.

### Free Reading Operations (Cost: $0 / 0 Credits)
Agents should exhaust these endpoints before considering any actions:
- `GET /api/setup-doctor?project_id=...` — Health check: confirms live connectivity and last arrival time.
- `GET /api/traces?project_id=...&status=failed` — Lists failed/flagged traces (`status` is required: `failed`, `error`, `blocked`, `flagged`, `success`).
- `GET /api/traces/{trace_id}` — Inspects full trace payload, prompts, responses, and latency.
- `GET /api/spans/{trace_id}` — Span tree for tool calls and nested steps.
- `GET /api/scores/summary?project_id=...` — Evaluation scores across all runs.
- `GET /api/metrics/summary?project_id=...&period=...` — Throughput, latency, and cost summaries.
- `GET /api/intelligence?project_id=...` — Coverage, risk trends, and failure pattern aggregations (pure SQL, free).
- `GET /api/intelligence/clusters?project_id=...` — Existing Root Cause Analysis (RCA) clusters.
- `GET /api/credits/balance?org_id=...` — Check remaining allowance before any paid operation.

---

## 2. Paid Endpoints & Credit Spending Safeguards

PRISM meters heavy AI operations against a monthly allowance. Agents must follow these **STRICT SPENDING RULES**:

| Action | Endpoint | Cost |
| :--- | :--- | :--- |
| **Cluster RCA Analysis** | `POST /api/intelligence/clusters/analyze` | 5 credits |
| **Fix Recommendations** | `POST /api/remediation/recommend` | 2 credits |
| **Trace Backfill** | `POST /api/backfill/trace-analyses` | 1 credit per unanalyzed trace |
| **AI Narrative Briefing** | `GET /api/intelligence/narrative` | 1 credit / day |
| **Automated Code Fix** | `POST /api/rca/remediation/generate-fix` | 5 credits |

### Mandatory Spending Protocol
1. **Never call a paid endpoint without user confirmation.** You must state the exact credit cost (e.g., *"Running RCA analysis costs 5 credits"*) and receive explicit approval first.
2. **Check the balance first:** Query `GET /api/credits/balance?org_id=$PRISMTRACE_ORG_ID` before committing to any paid plan.
3. **Read before re-running:** `GET /api/intelligence/clusters` already contains past analyses. Never run a duplicate analysis if the data already exists.
4. **Handle 402 gracefully:** A `402 Payment Required` means out of credits, not a code defect. Report it cleanly and stop.
5. **No Hallucinated Data:** Never report a metric, latency, or status without fetching it from a live endpoint.

---

## 3. Media & Artifact Hygiene
- Never commit raw `.mp4`, `.avi`, `.jpg`, `.png` datasets to git. Keep them under `data/` and protected by `.gitignore`.
- Documentation charts and web UI static assets may be committed under `reports/` and `web/static/`.
- All Python implementations must maintain 100% test pass rate with `pytest tests/ -v`.
