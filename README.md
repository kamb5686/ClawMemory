# ClawMemory

ClawMemory is a **SEVA assist layer** for OpenClaw:

- **Memory**: episodic + semantic (ChromaDB)
- **Verification**: Wikipedia/Wikidata now; Wolfram|Alpha optional
- **Modes**: `lite`, `standard`, `pro` (quality presets)
- **Plugin**: adds real OpenClaw commands (`/seva ...`) that bypass the LLM for deterministic control

## Quickstart (Linux/macOS)

```bash
git clone https://github.com/kamb5686/ClawMemory
cd ClawMemory
bash ./scripts/install.sh
openclaw gateway restart
```

## Usage (in chat)

Once the plugin is installed:

- `/seva status`
- `/seva mode` (show current)
- `/seva mode lite|standard|pro`
- `/seva recall 8 what was that plan`
- `/seva verify Barack Obama was born in 1961`
- `/seva doctor`

## Modes

Modes are defined in `seva/presets.json`.

- **lite**: fast + small embedding model
- **standard**: balanced
- **pro**: strongest embedding model

Set a mode:

```bash
./scripts/seva.sh mode standard
```

Or in chat:

```text
/seva mode standard
```

## Verification providers

### Wikipedia/Wikidata (default)
Works well for entity/date-style claims.

### Wolfram|Alpha (optional)
Enable by setting:

- `verification.wolfram.enabled=true`
- `verification.wolfram.appid=<YOUR_APPID>`

You can set it via API:

```bash
curl -sS -X POST http://127.0.0.1:18790/config-set \
  -H 'content-type: application/json' \
  -d '{"set":["verification.wolfram.enabled=true","verification.wolfram.appid=YOUR_APPID"]}'
```

Or via environment:

- `WOLFRAM_APPID=...`

## Notes

- ClawMemory does **not** replace OpenClaw’s LLM; it augments it.
- Store-all semantic memory can grow; monitor `~/.openclaw/seva/data/`.
