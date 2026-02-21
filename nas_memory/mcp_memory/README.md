# memory MCP bridge (Codex/OpenClaw/Claude)

This is a thin MCP server that forwards tool calls to `memory-api`.

## Tools

- `memory_retrieve(query, session_id, agent="codex", top_k=5)`
- `memory_enqueue_event(event_type, session_id, payload, agent="codex")`
- `memory_note_add(session_id, note_id, note_content, agent="codex")`
- `memory_health()`

## NAS deploy

```bash
ssh rorqual "mkdir -p /volume1/Services/mcp/memory/src"
scp nas_memory/mcp_memory/src/server.py rorqual:/volume1/Services/mcp/memory/src/server.py
scp nas_memory/mcp_memory/run-mcp.sh rorqual:/volume1/Services/mcp/memory/run-mcp.sh
ssh rorqual "chmod +x /volume1/Services/mcp/memory/run-mcp.sh"
ssh rorqual "cp /volume1/Services/memory/.env /volume1/Services/mcp/memory/.env"
```

## Codex config

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.memory]
type = "stdio"
command = "ssh"
args = ["rorqual", "bash '/volume1/Services/mcp/memory/run-mcp.sh'"]
```
