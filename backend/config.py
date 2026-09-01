"""Configuration loading for ArenaHub.

Values are read from environment variables (optionally populated from a
``.env`` file via python-dotenv). The Arena API key is only ever read from the
environment — it is never hardcoded, printed, or logged.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ArenaConfigError

DEFAULT_BASE_URL = "https://api.preview.arena.ai"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_HISTORY_DIR = Path.home() / ".arenahub"
DEFAULT_MODEL_CACHE_TTL = 300.0
DEFAULT_REQUEST_MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB JSON bodies
DEFAULT_RATE_LIMIT_PER_MINUTE = 120

USER_AGENT = "ArenaHub/0.2.0 (+https://arena.ai)"


def _env_str(name: str, default: str | None = None) -> str | None:
    """Read a trimmed string env var, treating empty strings as unset."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ArenaConfigError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ArenaConfigError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(slots=True)
class Settings:
    """Runtime settings for the Arena client, CLI, gateway and web API."""

    arena_api_key: str | None
    arena_base_url: str = DEFAULT_BASE_URL
    arena_default_model: str | None = None
    arena_timeout: float = DEFAULT_TIMEOUT_SECONDS
    gateway_host: str = DEFAULT_HOST
    gateway_port: int = DEFAULT_PORT
    gateway_api_key: str | None = None
    history_dir: Path = field(default_factory=lambda: DEFAULT_HISTORY_DIR)
    # Model router
    model_cache_ttl: float = DEFAULT_MODEL_CACHE_TTL
    model_aliases: dict[str, str] = field(default_factory=dict)
    # Platform API
    database_path: Path | None = None
    upload_dir: Path | None = None
    # Security middleware
    request_max_body_bytes: int = DEFAULT_REQUEST_MAX_BODY_BYTES
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    enable_docs: bool = False
    allow_origins: tuple[str, ...] = ()

    # -- paths -------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.database_path or (self.history_dir / "arenahub.db")

    @property
    def uploads_path(self) -> Path:
        return self.upload_dir or (self.history_dir / "uploads")

    # -- Arena API key -----------------------------------------------------
    def require_arena_api_key(self) -> str:
        """Return the API key or raise a clear configuration error."""
        if not self.arena_api_key:
            raise ArenaConfigError(
                "ARENA_API_KEY is not set. Copy .env.example to .env and set your Arena API "
                "key, or export ARENA_API_KEY in your shell."
            )
        return self.arena_api_key

    @property
    def arena_api_key_configured(self) -> bool:
        return bool(self.arena_api_key)

    def masked_arena_key(self) -> str:
        """Return a non-revealing representation for UI display."""
        return "configured" if self.arena_api_key else "not set"

    # -- Gateway key -------------------------------------------------------
    def ensure_gateway_api_key(self) -> str:
        """Return the local gateway key, generating an ephemeral one if unset.

        The generated key exists only for the lifetime of the server process
        and is shown once in the local server console (never logged elsewhere).
        """
        if not self.gateway_api_key:
            self.gateway_api_key = secrets.token_urlsafe(32)
        return self.gateway_api_key


def _parse_aliases(raw: str | None) -> dict[str, str]:
    """Parse ``ARENA_MODEL_ALIASES`` in ``alias=id,alias2=id2`` form."""
    aliases: dict[str, str] = {}
    if not raw:
        return aliases
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ArenaConfigError(
                f"ARENA_MODEL_ALIASES entry {pair!r} must look like 'alias=model-id'."
            )
        alias, target = pair.split("=", 1)
        alias, target = alias.strip(), target.strip()
        if alias and target:
            aliases[alias] = target
    return aliases


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment and a local ``.env`` file."""
    # Load .env from the current working directory if present; existing env
    # vars always take precedence (do not override them).
    load_dotenv(override=False)

    history_dir = Path(_env_str("ARENAHUB_HISTORY_DIR") or DEFAULT_HISTORY_DIR)

    return Settings(
        arena_api_key=_env_str("ARENA_API_KEY"),
        arena_base_url=_env_str("ARENA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        arena_default_model=_env_str("ARENA_DEFAULT_MODEL"),
        arena_timeout=_env_float("ARENA_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
        gateway_host=_env_str("ARENAHUB_HOST", DEFAULT_HOST) or DEFAULT_HOST,
        gateway_port=_env_int("ARENAHUB_PORT", DEFAULT_PORT),
        gateway_api_key=_env_str("ARENAHUB_API_KEY"),
        history_dir=history_dir,
        model_cache_ttl=_env_float("ARENA_MODEL_CACHE_TTL", DEFAULT_MODEL_CACHE_TTL),
        model_aliases=_parse_aliases(_env_str("ARENA_MODEL_ALIASES")),
        database_path=Path(p) if (p := _env_str("ARENAHUB_DB_PATH")) else None,
        upload_dir=Path(p) if (p := _env_str("ARENAHUB_UPLOAD_DIR")) else None,
        request_max_body_bytes=_env_int(
            "ARENAHUB_MAX_REQUEST_BYTES", DEFAULT_REQUEST_MAX_BODY_BYTES
        ),
        rate_limit_per_minute=_env_int(
            "ARENAHUB_RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE
        ),
        enable_docs=_env_bool("ARENAHUB_ENABLE_DOCS", False),
        allow_origins=tuple(
            origin.strip()
            for origin in _env_str("ARENAHUB_ALLOW_ORIGINS", "").split(",")
            if origin.strip()
        ),
    )
