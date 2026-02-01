# ClawMemory SEVA Plugin

Adds a deterministic `/seva` command family to OpenClaw.

## Install

```bash
openclaw plugins install -l ./plugins/openclaw-seva
openclaw plugins enable clawmemory-seva
openclaw gateway restart
```

## Commands

- `/seva status`
- `/seva doctor`
- `/seva mode list`
- `/seva mode lite|standard|pro`
- `/seva recall [k] <query>`
- `/seva verify <claim>`
- `/seva verify --provider sympy <claim>`
- `/seva verify --provider wolfram <claim>`
- `/seva verify --all <claim>`

## Plugin config

- `baseUrl` (default `http://127.0.0.1:18790`)
- `timeoutMs`
- `defaultRecallK`
- `startupCheck`

