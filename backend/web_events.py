"""SSE event framing for the web/Android streaming API.

Web clients receive named JSON events so a chat UI can render optimistically::

    event: user_message
    data: {"id": "msg_...", "role": "user", ...}

    event: delta
    data: {"content": "Hello"}

    event: done
    data: {"message": {...assistant message...}, "conversation_id": "conv_..."}

    event: error
    data: {"message": "..."}
"""

from __future__ import annotations

import json
from typing import Any


def web_message_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def web_error_event(message: str) -> bytes:
    return web_message_event("error", {"message": message})
