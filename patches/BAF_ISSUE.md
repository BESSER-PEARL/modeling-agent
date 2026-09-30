## Summary

`WebSocketPlatform` silently drops agent replies when two WebSocket connections share the same session key. On disconnect, the connection cleanup deletes the `_connections` slot **unconditionally**, even when that slot is now owned by a *different, still-open* connection. The agent then has no live connection to send replies through, so messages are dropped with no error.

## Affected code

`baf/platforms/websocket/websocket_platform.py` (current `main`, commit `0e19286`):

```python
# registration — both connections sharing a key overwrite the same slot
session_key = header_session or query_session or str(conn.id)
self._connections[str(session_key)] = conn
session = self._agent.get_or_create_session(session_key, self, username, session_name)
...
finally:
    if session:
        session_id = str(session.id)
        if session_id in self._connections:
            del self._connections[session_id]   # <-- deletes whatever is in the slot,
                                                 #     even a different live connection
```

## Root cause

`_connections` is keyed by session key, but more than one connection can resolve to the same key (e.g. a client that opens two sockets with the same `user_id`/`session_id`, or a reconnect that briefly overlaps the old socket). The sequence:

1. `connA` opens → `_connections[key] = connA`
2. `connB` opens with the same key → `_connections[key] = connB` (connA orphaned, socket still open)
3. `connA`'s handler ends → `finally` runs `del _connections[key]` → **deletes connB's live slot**
4. Agent calls `_send(session.id, ...)` → `session_id not in _connections` → reply silently dropped

It's a classic check-then-act / lost-update race on a shared map: the cleanup assumes the slot still belongs to the connection that's closing.

## Reproduction

- A browser frontend that opens two WebSocket connections in the same tab (common when two UI components each hold a client) and passes a stable `user_id` (or `session_id`) so both resolve to one session key.
- Connect both, let one close, then send a message on the other → the agent processes the message (visible in logs) but the reply never reaches the client.

## Impact

Agent appears "dead": it receives and processes messages but replies vanish. No exception is raised, so it's hard to diagnose. Affects any deployment where a client may hold more than one connection under a single session key — increasingly likely now that `main` supports explicit `user_id`/`session_id` keying.

## Suggested fix

Only delete the slot if this connection still owns it (compare-and-delete), and re-claim the slot when a connection sends a message so replies route to the connection the user is actually using:

```python
# on USER_MESSAGE: point the reply slot at the sender (the demonstrably-live conn)
elif payload.action == PayloadAction.USER_MESSAGE.value:
    self._connections[str(session_key)] = conn
    ...

# on disconnect: only evict if we still own the slot
finally:
    if session:
        session_id = str(session.id)
        if session_id in self._connections and self._connections[session_id] is conn:
            del self._connections[session_id]
```

The ownership guard (`is conn`) is the essential part; the re-claim on `USER_MESSAGE` makes reply routing deterministic when several connections share a key. Happy to open a PR if useful.

_Found while running the BESSER Web Modeling Editor's modeling agent (BAF 4.3.2); confirmed the same cleanup logic is present on current `main`._
