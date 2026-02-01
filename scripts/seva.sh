#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${OPENCLAW_SEVA_URL:-http://127.0.0.1:18790}"
cmd="${1:-}"
shift || true

case "$cmd" in
  status)
    curl -sS "$BASE_URL/status"; echo ;;
  mode)
    mode="${1:-}"; [ -n "$mode" ] || { echo "usage: seva.sh mode <lite|standard|pro>" >&2; exit 1; }
    curl -sS -X POST "$BASE_URL/mode-set" -H 'content-type: application/json' -d "{\"mode\":\"$mode\"}"; echo ;;
  recall)
    q="${1:-}"; [ -n "$q" ] || { echo "usage: seva.sh recall <query> [k]" >&2; exit 1; }
    k="${2:-5}"
    curl -sS -X POST "$BASE_URL/recall" -H 'content-type: application/json' -d "{\"query\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$q"),\"k\":$k}"; echo ;;
  verify)
    c="${1:-}"; [ -n "$c" ] || { echo "usage: seva.sh verify <claim>" >&2; exit 1; }
    curl -sS -X POST "$BASE_URL/verify" -H 'content-type: application/json' -d "{\"claim\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$c")}"; echo ;;
  *)
    echo "usage: seva.sh <status|mode|recall|verify>" >&2
    exit 1
    ;;
esac
