# ClawMemory SEVA OpenClaw Plugin

This plugin adds **real chat commands** (auto-reply, no LLM required) for controlling the local **SEVA** service that ships with this repo.

Commands (chat):

- `/seva status`
- `/seva on` / `/seva off`
- `/seva mode` (show current)
- `/seva mode <name>` (set)
- `/seva mode list` (best-effort list)
- `/seva recall [k] <query>`
- `/seva verify <claim>`
- `/seva doctor`

The plugin talks to the SEVA HTTP API at `http://127.0.0.1:18790` by default.

## Install (local dev / from git checkout)

From a clone of `kamb5686/ClawMemory`:

```bash
cd ClawMemory
openclaw plugins install -l ./plugins/openclaw-seva
openclaw plugins enable clawmemory-seva
openclaw gateway restart
```

## Configure (optional)

In `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "clawmemory-seva": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:18790",
          "timeoutMs": 2500,
          "defaultRecallK": 5,
          "startupCheck": true
        }
      }
    }
  }
}
```

Restart the Gateway after config changes.

## Start the SEVA service

This plugin only talks to the service; it does not start it.

- Linux (systemd): see `service/systemd/openclaw-seva.service`
- macOS (launchd): see `service/macos/com.openclaw.seva.plist`

If you’re unsure, run:

- `/seva doctor`

and then ensure something is listening on TCP port `18790`.
