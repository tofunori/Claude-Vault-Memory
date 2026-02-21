#!/bin/bash
set -euo pipefail
cd /volume1/Services/mcp/memory
set -a
source .env
set +a
exec /volume1/Services/mcp/ragdoc/ragdoc-env-new/bin/python3 src/server.py
