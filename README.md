# ArenaHub

**ArenaHub** is a unified AI gateway that sits between your applications and the
**official Arena API**. It turns Arena models into a private, self-hosted
platform that speaks the wire formats your tools already understand.

ArenaHub can serve as:

1. A **ChatGPT-style web application backend** (conversations, streaming,
   model picker, uploads, edit/regenerate)
2. An **Android application backend** (JSON REST + SSE)
3. An **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`)
4. An **Anthropic-compatible API** (`/v1/messages`)
5. A **coding-agent gateway** for VS Code / Claude Code-style clients
   (long contexts, tool calls, streaming)
6. A **unified model router** over the official Arena API (caching + aliases)

> ArenaHub talks **only** to the official Arena API
> (`https://api.preview.arena.ai`). It does **not** scrape arena.ai, automate
> the website, reverse-engineer browser traffic, or bypass authentication or
> rate limits. All Arena authentication stays server-side.

```
  OpenAI SDK  ─┐
  Anthropic SDK┼─►  /v1/chat/completions   ┐
  Claude Code ─┤   /v1/messages            ├─► ArenaHub ──Bearer ARENA_API_KEY──► Official Arena API
  Web app     ─┤   /api/...                │      (127.0.0.1 by default)
  Android app ─┘   SSE streaming           ┘
```

---

## Table of contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Interfaces](#interfaces)
  - [OpenAI-compatible API](#1-openai-compatible-api)
  - [Anthropic-compatible API](#2-anthropic-compatible-api)
  - [Web / Android REST API](#3-web--android-rest-api)
  - [CLI](#4-cli)
- [Model router & aliases](#model-router--aliases)
- [Coding-agent setup](#coding-agent-setup)
- [Android client contract](#android-client-contract)
- [Security](#security)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Testing](#testing)
- [Roadmap](#roadmap)

## Quick start

```bash
# Linux / macOS
git clone https://github.com/<you>/ArenaHub.git && cd ArenaHub
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # set ARENA_API_KEY and ARENAHUB_API_KEY
arenahub serve                  # or: python -m backend.main
```

```powershell
# Windows (PowerShell)
git clone https://github.com/<you>/ArenaHub.git; cd ArenaHub
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # set ARENA_API_KEY and ARENAHUB_API_KEY
arenahub serve
```

Get an Arena API key from your account settings at **https://arena.ai**.

## Architecture

```
arena-hub/
├── backend/
│   ├── main.py             # FastAPI app factory, middleware, error handlers
│   ├── arena_client.py     # Async client for the official Arena API
│   ├── model_router.py     # Model catalogue cache + alias resolution
│   ├── routes.py           # OpenAI-compatible /v1/* endpoints
│   ├── routes_anthropic.py # Anthropic-compatible /v1/messages
│   ├── routes_api.py       # Web/Android /api/* (conversations, files)
│   ├── anthropic.py        # Anthropic <-> OpenAI translation + SSE events
│   ├── models.py           # OpenAI-shaped pydantic schemas + normalisation
│   ├── schemas.py          # Web/Android REST schemas
│   ├── db.py               # SQLite conversation repository (Postgres-ready)
│   ├── middleware.py       # Auth, request IDs, rate limiting, body size
│   ├── config.py           # Env/.env settings
│   └── errors.py           # Structured exception hierarchy
├── cli/main.py             # Typer + Rich CLI
└── tests/                  # pytest suite (mocked upstream)
```

Request flow: a client authenticates to ArenaHub with the **ArenaHub gateway
key** (`ARENAHUB_API_KEY`). ArenaHub authenticates to Arena with the **Arena
key** (`ARENA_API_KEY`), which never leaves the server.

## Interfaces

All endpoints except `/health` require the ArenaHub key, sent as
`Authorization: Bearer <ARENAHUB_API_KEY>`, `X-API-Key: <key>`, or Anthropic
style `X-Api-Key: <key>`.

### 1. OpenAI-compatible API

#### `GET /v1/models`

Returns real Arena models plus ArenaHub aliases. Alias entries carry
`"owned_by": "arenahub"` and `"alias_for": "<real model id>"`.

```bash
curl -H "Authorization: Bearer $ARENAHUB_API_KEY" http://127.0.0.1:8000/v1/models
```

```json
{ "object": "list", "data": [
  { "id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic" },
  { "id": "arena/claude-sonnet", "object": "model", "owned_by": "arenahub",
    "alias_for": "claude-sonnet-4-6" } ] }
```

#### `POST /v1/chat/completions`

Streaming and non-streaming, tools/function calling, aliases, and the
`X-Arena-Model` header (for clients that cannot set the body model).

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $ARENAHUB_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"arena/claude","stream":true,
       "messages":[{"role":"user","content":"Hello"}]}'
```

OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=ARENAHUB_API_KEY)
r = client.chat.completions.create(
    model="arena/claude",
    messages=[{"role": "user", "content": "Hello"}],
    tools=[{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
)
```

### 2. Anthropic-compatible API

#### `POST /v1/messages`

Supports system prompts, user/assistant messages, streaming
(`message_start`/`content_block_delta`/`message_delta`/`message_stop`),
`max_tokens`, `temperature`, tools (`tool_use`) and tool results
(`tool_result`). `max_tokens` is required (Anthropic semantics).

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "x-api-key: $ARENAHUB_API_KEY" -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"arena/claude-sonnet","max_tokens":1024,
       "system":"You are concise.",
       "messages":[{"role":"user","content":"Write a haiku about SQL."}]}'
```

Anthropic SDK:

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8000", auth_token=ARENAHUB_API_KEY)
with client.messages.stream(
    model="arena/claude-sonnet", max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

Tool call response (non-streaming) returns Anthropic content blocks:

```json
{ "type": "message", "role": "assistant", "model": "arena/claude-sonnet",
  "content": [ { "type": "tool_use", "id": "toolu_...",
                 "name": "get_weather", "input": {"city": "Paris"} } ],
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 9, "output_tokens": 4} }
```

Unsupported Anthropic features (e.g. extended thinking blocks, unknown content
block types) return a clear `invalid_request_error` envelope — ArenaHub never
pretends a feature works if it cannot be translated to the Arena API.

### 3. Web / Android REST API

JSON over HTTP; chat streams as **Server-Sent Events** (works in browsers via
`fetch`/`EventSource` and natively on Android via `OkHttp` `EventSource`).

| Method | Path                                            | Purpose                                            |
| ------ | ----------------------------------------------- | -------------------------------------------------- |
| GET    | `/health`                                       | Liveness (no key)                                  |
| GET    | `/api/models`                                   | Model catalogue for the model selector (incl. aliases) |
| POST   | `/api/conversations`                            | Create a conversation `{title?, model?, metadata?}`|
| GET    | `/api/conversations?limit=&offset=`             | Sidebar list (newest first)                        |
| GET    | `/api/conversations/{id}`                       | Full conversation with messages                    |
| PATCH  | `/api/conversations/{id}`                       | Rename: `{"title": "..."}`                         |
| DELETE | `/api/conversations/{id}`                       | Delete (204)                                       |
| POST   | `/api/conversations/{id}/messages`              | Send a message; assistant streams back (`stream:true` default) |
| POST   | `/api/conversations/{id}/messages/{msgId}/edit` | Edit a user message and regenerate                 |
| POST   | `/api/conversations/{id}/regenerate`            | Regenerate the last assistant reply                |
| POST   | `/api/files`                                    | Multipart upload; returns `{id, filename, ...}`    |

Send a message (streaming):

```bash
curl -N http://127.0.0.1:8000/api/conversations/$CID/messages \
  -H "Authorization: Bearer $ARENAHUB_API_KEY" -H "Content-Type: application/json" \
  -d '{"content":"Explain SQLite indexes","stream":true}'
```

SSE event sequence:

```
event: user_message   data: {"id":"msg_...","role":"user","content":"..."}
event: delta          data: {"content":"SQLite indexes ..."}
event: delta          data: {"content":"..."}
event: done           data: {"message":{...assistant...},"conversation_id":"conv_..."}
event: error          data: {"message":"..."}      # on failure
```

`content` may be a plain string or OpenAI-style content blocks (text /
image_url / tool results), supporting coding-agent and multimodal clients.
Markdown and code blocks are delivered as text; the frontend renders them.
Image understanding depends on the selected Arena model and surfaces a
compatibility error if unavailable.

### 4. CLI

```bash
arenahub                 # interactive menu (Chat / Models / Settings / Exit)
arenahub chat -m arena/claude-sonnet
arenahub models          # real models + aliases (provider/alias column)
arenahub health          # config + Arena connectivity/auth check
arenahub serve           # start the gateway
```

In-chat commands: `/model [id]`, `/models`, `/clear`, `/new`, `/help`, `/exit`.

## Model router & aliases

- The catalogue is fetched dynamically from Arena and cached for
  `ARENA_MODEL_CACHE_TTL` seconds (default 300).
- **Built-in aliases** resolve against the live catalogue, so they keep
  working as Arena updates its model list:

  | Alias                | Resolves to                            |
  | -------------------- | -------------------------------------- |
  | `arena/claude`       | balanced Claude (prefers Sonnet)       |
  | `arena/claude-sonnet`| latest available Claude Sonnet         |
  | `arena/claude-opus`  | latest available Claude Opus           |
  | `arena/gpt`          | latest available GPT                   |
  | `arena/gemini`       | latest available Gemini (prefers Pro)  |

- Add your own with `ARENA_MODEL_ALIASES="fast=gpt-4o,work=claude-sonnet-4-6"`.
- Aliases work everywhere (`/v1/chat/completions`, `/v1/messages`, `/api`, CLI).
  The upstream call uses the resolved Arena id; responses echo the alias you
  asked for. Unresolvable `arena/...` aliases return a clear model error.

## Coding-agent setup

ArenaHub supports long conversations, tool/function calls, structured tool
results, streaming events, and model selection via header/env.

- **Claude Code / Anthropic tools**: point the client's base URL at
  `http://127.0.0.1:8000` and use the ArenaHub key; models can be an
  Anthropic-style name or an `arena/...` alias.
- **OpenAI-based agents (VS Code "OpenAI compatible", Continue, etc.)**:
  base URL `http://127.0.0.1:8000/v1`, key = `ARENAHUB_API_KEY`.
- **Model via header**: send `X-Arena-Model: arena/claude` to force the model
  regardless of the request body.
- **Model via env**: set `ARENA_DEFAULT_MODEL=arena/claude` server-side.
- Large contexts are sent through as-is (respect Arena model limits); the
  server enforces a configurable request body cap.

## Android client contract

- Pure JSON REST under `/api/*`; authenticate with `Authorization: Bearer
  <ARENAHUB_API_KEY>` on every call.
- Chat streaming is SSE: `POST /api/conversations/{id}/messages`
  (`Content-Type: application/json`) returns `text/event-stream` with the
  named events above. Use OkHttp + an `EventSource` library (or consume the
  chunked response). The same events drive the web client.
- Every response includes an `X-Request-ID` header (you may send your own)
  for tracing/support.
- Offline/queue: SQLite stores conversations/messages server-side; a client
  can re-fetch `GET /api/conversations/{id}` to resume state.
- Files: `POST /api/files` (multipart) returns a `file_id`; reference it from
  message content blocks.

## Security

- **Two independent credentials**: `ARENA_API_KEY` (upstream, server-side
  only — never returned to clients) and `ARENAHUB_API_KEY` (clients).
- Authentication middleware protects every non-`/health` route; keys are
  compared with `hmac.compare_digest`.
- Secrets are never printed (except the ephemeral gateway key in the local
  startup banner) and Authorization headers are never logged.
- Binds to `127.0.0.1` by default; CORS restricted to loopback origins unless
  `ARENAHUB_ALLOW_ORIGINS` is set.
- Request body size cap (`ARENAHUB_MAX_REQUEST_BYTES`, default 2 MiB JSON) and
  a swappable in-memory rate limiter (`ARENAHUB_RATE_LIMIT_PER_MINUTE`).
- Every request gets an `X-Request-ID`; error envelopes match the calling
  surface (OpenAI vs Anthropic vs web) and never leak upstream internals.

## Deployment

Defaults are local/development (`127.0.0.1`). For public deployment:

- **Bind & secrets**: set `ARENAHUB_HOST=0.0.0.0` (only behind a proxy),
  provide a strong stable `ARENAHUB_API_KEY`, and inject all secrets via your
  platform's secret manager (never commit `.env`).
- **HTTPS / reverse proxy**: terminate TLS at nginx/Caddy/Traefik or your load
  balancer and proxy to the loopback port; enable HTTP/2 for SSE. Example
  nginx: `proxy_pass http://127.0.0.1:8000;`, `proxy_buffering off;`
  (SSE), and forward `Authorization`.
- **PostgreSQL**: implement the `ConversationRepository` protocol in
  `backend/db.py` (SQLAlchemy/asyncpg) and pass it to `create_app(...,
  repository=...)`; no route changes are needed.
- **Rate limiting**: replace `middleware.RateLimiter` with a Redis-backed
  limiter for multi-process/multi-node deployments.
- **Docs**: set `ARENAHUB_ENABLE_DOCS=true` to expose `/docs` (keep disabled
  publicly or protect it).
- Run under a process manager (systemd, Docker, uvicorn workers); SSE works
  across workers since state is in the database.

## Configuration

| Variable                        | Default                       | Meaning                                   |
| ------------------------------- | ----------------------------- | ----------------------------------------- |
| `ARENA_API_KEY`                 | — (required for API calls)    | Official Arena API key (server-side).     |
| `ARENA_BASE_URL`                | `https://api.preview.arena.ai`| Official Arena API base URL.              |
| `ARENA_DEFAULT_MODEL`           | —                             | Default model/alias.                      |
| `ARENA_TIMEOUT`                 | `120`                         | Upstream request timeout (s).             |
| `ARENA_MODEL_CACHE_TTL`         | `300`                         | Model catalogue cache TTL (s).            |
| `ARENA_MODEL_ALIASES`           | —                             | Extra `alias=id` pairs, comma-separated.  |
| `ARENAHUB_HOST` / `ARENAHUB_PORT` | `127.0.0.1` / `8000`        | Bind address.                             |
| `ARENAHUB_API_KEY`              | random per start              | Key clients must present.                 |
| `ARENAHUB_MAX_REQUEST_BYTES`    | `2097152`                     | JSON body size cap.                       |
| `ARENAHUB_RATE_LIMIT_PER_MINUTE`| `120`                         | Per-IP requests/minute (0 disables).      |
| `ARENAHUB_ALLOW_ORIGINS`        | loopback only                 | Extra CORS origins.                       |
| `ARENAHUB_ENABLE_DOCS`          | `false`                       | Expose FastAPI `/docs`.                   |
| `ARENAHUB_DB_PATH` / `ARENAHUB_UPLOAD_DIR` | `~/.arenahub/...`   | Storage locations.                        |

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite uses mocked HTTP transports (no real Arena calls) and covers: the
OpenAI endpoint, the Anthropic endpoint, streaming (OpenAI SSE and Anthropic
events), aliases, model routing/caching, conversation CRUD, edit/regenerate,
file upload, authentication, tool calls, invalid requests, upstream errors,
request IDs, body-size/rate limits, and Arena-key isolation.

## Roadmap

Text gateway and coding-agent compatibility come first. Not implemented yet
(but designed for): image generation/editing, audio/video, multiple Arena
keys with rotation, usage metering, a web dashboard frontend, WebSocket
streaming, and MCP server support.
