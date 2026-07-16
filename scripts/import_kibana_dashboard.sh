#!/usr/bin/env bash
# Import the ARES Ground Control Kibana dashboard.
# Usage:  bash scripts/import_kibana_dashboard.sh [kibana_url]
# Default kibana_url: http://localhost:5601

set -euo pipefail

KIBANA_URL="${1:-http://localhost:5601}"
DASHBOARD_FILE="$(dirname "$0")/../kibana/mars_scout_dashboard.ndjson"

echo "Waiting for Kibana at ${KIBANA_URL}…"
for i in $(seq 1 24); do
  if curl -sf "${KIBANA_URL}/api/status" | grep -q '"level":"available"'; then
    echo "Kibana ready."
    break
  fi
  [ "$i" -eq 24 ] && { echo "Kibana not ready after 2 min — aborting."; exit 1; }
  sleep 5
done

echo "Importing dashboard…"
curl -sf -X POST \
  "${KIBANA_URL}/api/saved_objects/_import?overwrite=true" \
  -H "kbn-xsrf: true" \
  -H "Content-Type: multipart/form-data" \
  --form "file=@${DASHBOARD_FILE}" \
  | python3 -m json.tool

echo ""
echo "Done. Open: ${KIBANA_URL}/app/dashboards"
