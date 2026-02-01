# ClawMemory

ClawMemory contains the local **SEVA** service + OpenClaw integrations.

## OpenClaw plugin (recommended)

This repo ships an **OpenClaw plugin** that provides real chat commands:

- `/seva status | mode | recall | verify | on | off | doctor`

Plugin path:

- `plugins/openclaw-seva`

Install (local path):

```bash
git clone https://github.com/kamb5686/ClawMemory.git
cd ClawMemory

openclaw plugins install -l ./plugins/openclaw-seva
openclaw plugins enable clawmemory-seva
openclaw gateway restart
```

See [`plugins/openclaw-seva/README.md`](plugins/openclaw-seva/README.md) for full details.

## Existing hook pack

This repo also includes a legacy hook pack at `hooks/seva-service` that checks whether
SEVA is reachable on Gateway startup. The plugin includes a similar startup check,
so the hook pack is optional.
