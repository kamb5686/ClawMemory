---
name: seva-service
description: "Ensures SEVA assist layer service is reachable; emits guidance if not"
metadata: { "openclaw": { "emoji": "🧠", "events": ["gateway:startup"] } }
---

# SEVA Service Hook

On gateway startup, this hook checks whether the local SEVA service is reachable at:

- `http://127.0.0.1:18790/status`

If unreachable, it logs a short message explaining how to install/start it.

This hook is intentionally lightweight and does not block startup.
