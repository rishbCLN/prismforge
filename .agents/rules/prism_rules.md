# PRISM Tracing & Observability Rules

## Core Directives
1. **Never stage or commit `.env` or API credentials.** Use environment variables `PRISMTRACE_API_KEY`, `PRISMTRACE_PROJECT_ID`, `PRISMTRACE_ORG_ID`, and `PRISMTRACE_HOST`.
2. **Authenticate with Header:** Always use `X-PRISMtrace-Key: <key>`. Do not use `Authorization: Bearer`.
3. **Credit Safeguards (Strict):**
   - Free operations (`GET /api/setup-doctor`, `GET /api/traces`, `GET /api/intelligence`, `GET /api/metrics/summary`, `GET /api/credits/balance`) may be called freely.
   - Paid operations (`POST /api/intelligence/clusters/analyze` [5 credits], `POST /api/remediation/recommend` [2 credits], `POST /api/backfill/trace-analyses` [1 credit/trace], `POST /api/rca/remediation/generate-fix` [5 credits]) **MUST NEVER BE CALLED** without first stating the exact credit cost and receiving explicit user confirmation.
4. **Read Before Running:** Always check existing records (`GET /api/intelligence/clusters` or `GET /api/backfill/trace-analyses/active`) before triggering new background jobs.
5. **Verified Telemetry:** Never invent numbers or report that an endpoint succeeded without executing and verifying the HTTP response.
