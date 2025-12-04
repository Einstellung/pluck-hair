# Backend Todo

- **Streaming/Redis health**: add a dedicated health probe (e.g., `/api/health/stream`) that checks Redis availability and whether `pluck:frames` is receiving recent frames; expose last frame timestamp/length and fail fast in strict mode.
- **Startup strict mode**: optional flag/config to treat Redis/frame publisher init failures as fatal (exit with clear ERROR) instead of warning-only; default could remain permissive but should be opt-in for critical deployments.
- **Operational visibility**: surface streaming status (enabled/disabled, last frame age, Redis errors) prominently in logs and/or a status endpoint to avoid key warnings being buried in debug noise.
