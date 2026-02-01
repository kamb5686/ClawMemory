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
- `/seva memory status`
- `/seva memory prune --dry-run --max 5000 --policy oldest`
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
Enable by setting an AppID. If `verification.wolfram.enabled=true` but no AppID is configured, SEVA will **auto-disable** Wolfram on next config refresh.

You can set it in chat:

- `/seva wolfram set <YOUR_APPID>`
- `/seva wolfram status`

Or by config keys:

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

## Stage 2: Retention + temporal decay/reinforcement

Semantic memory can be bounded and made time-aware via config:

- `memory.retention.max_items` (0 = unlimited)
- `memory.retention.prune_policy` (`oldest` | `least_reinforced`)
- `memory.temporal.decay_rate` (per-day exponential decay)
- `memory.temporal.reinforcement_boost` (score bonus per reinforcement)

SEVA exposes helper endpoints/commands:

- `/seva memory status`
- `/seva memory prune [--dry-run] [--max N] [--policy oldest|least_reinforced]`

Notes on behavior:

- When semantic memory is enabled and `store_all=true`, new items get `timestamp`, `last_accessed`, and `reinforcement=0` metadata.
- During `/recall`, top semantic hits are reinforced (metadata updated) and relevance is adjusted by decay + reinforcement.
- If `memory.retention.max_items>0`, SEVA prunes automatically after storing.

## Notes

- ClawMemory does **not** replace OpenClaw’s LLM; it augments it.
- Store-all semantic memory can grow; monitor `~/.openclaw/seva/data/`.

Run basic tests:

```bash
python3 -m unittest discover -s seva/tests
```

## Stage 1 Verification

Providers are configurable via `verification.providers` (ordered).

- `wikipedia` (Wikidata/Wikipedia)
- `sympy` (local math checks)
- `wolfram` (Wolfram|Alpha evidence; requires AppID)

