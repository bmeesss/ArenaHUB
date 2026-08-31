"""Conversation persistence.

Local development uses SQLite behind a clean :class:`ConversationRepository`
*protocol* so a PostgreSQL implementation (SQLAlchemy/SQLModel/asyncpg) can be
dropped in later by satisfying the same interface in
:meth:`backend.main.create_app` — no route code changes required.

Schema
------
``conversations``
    id, title, model, created_at, updated_at, metadata (JSON)
``messages``
    id, conversation_id, role, content (JSON), model, created_at,
    parent_message_id (for edit/regenerate branching)

All blocking sqlite calls are wrapped with :func:`asyncio.to_thread` so the
event loop is never stalled. Initialization is idempotent and lazy, so the
repository works both under the FastAPI lifespan and in tests/mounts that
skip startup.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> float:
    return time.time()


@dataclass(slots=True)
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: Any  # JSON-serialisable: string or list of content blocks
    model: str | None = None
    created_at: float = 0.0
    parent_message_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "created_at": self.created_at,
            "parent_message_id": self.parent_message_id,
        }


@dataclass(slots=True)
class ConversationRecord:
    id: str
    title: str
    model: str
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata or {},
        }


@runtime_checkable
class ConversationRepository(Protocol):
    """Interface a PostgreSQL repository must implement."""

    async def initialize(self) -> None: ...
    async def create_conversation(
        self, *, title: str, model: str, metadata: dict[str, Any] | None = None
    ) -> ConversationRecord: ...
    async def list_conversations(self, *, limit: int = 100, offset: int = 0) -> list[ConversationRecord]: ...
    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None: ...
    async def rename_conversation(self, conversation_id: str, title: str) -> ConversationRecord | None: ...
    async def delete_conversation(self, conversation_id: str) -> bool: ...
    async def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: Any,
        model: str | None = None,
        parent_message_id: str | None = None,
    ) -> MessageRecord: ...
    async def list_messages(self, conversation_id: str) -> list[MessageRecord]: ...
    async def get_message(self, message_id: str) -> MessageRecord | None: ...
    async def truncate_messages_after(self, message_id: str) -> int: ...
    async def touch_conversation(self, conversation_id: str) -> None: ...


class SqliteConversationRepository:
    """Blocking SQLite calls run in a worker thread to stay async-safe."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    # -- connection / init -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def initialize(self) -> None:
        def _init() -> None:
            self._connection = self._connect()
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id                  TEXT PRIMARY KEY,
                    conversation_id     TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role                TEXT NOT NULL,
                    content             TEXT NOT NULL,
                    model               TEXT,
                    created_at          REAL NOT NULL,
                    parent_message_id   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                """
            )
            self._connection.commit()

        await asyncio.to_thread(_init)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if not self._initialized:
                await self.initialize()

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:  # pragma: no cover - initialize() always called
            self._connection = self._connect()
        return self._connection

    # -- conversations -----------------------------------------------------

    async def create_conversation(
        self, *, title: str, model: str, metadata: dict[str, Any] | None = None
    ) -> ConversationRecord:
        await self._ensure_initialized()

        def _create() -> ConversationRecord:
            now = _now()
            record = ConversationRecord(
                id=_new_id("conv"),
                title=title or "New conversation",
                model=model,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            conn = self._conn()
            conn.execute(
                "INSERT INTO conversations (id, title, model, created_at, updated_at, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (record.id, record.title, record.model, record.created_at,
                 record.updated_at, json.dumps(record.metadata)),
            )
            conn.commit()
            return record

        async with self._lock:
            return await asyncio.to_thread(_create)

    async def list_conversations(self, *, limit: int = 100, offset: int = 0) -> list[ConversationRecord]:
        await self._ensure_initialized()

        def _list() -> list[ConversationRecord]:
            rows = self._conn().execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_conversation(row) for row in rows]

        return await asyncio.to_thread(_list)

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        await self._ensure_initialized()

        def _get() -> ConversationRecord | None:
            row = self._conn().execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return self._row_to_conversation(row) if row else None

        return await asyncio.to_thread(_get)

    async def rename_conversation(self, conversation_id: str, title: str) -> ConversationRecord | None:
        await self._ensure_initialized()

        def _rename() -> ConversationRecord | None:
            conn = self._conn()
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conversation_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return self._row_to_conversation(row) if row else None

        async with self._lock:
            return await asyncio.to_thread(_rename)

    async def delete_conversation(self, conversation_id: str) -> bool:
        await self._ensure_initialized()

        def _delete() -> bool:
            conn = self._conn()
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cur.rowcount > 0

        async with self._lock:
            return await asyncio.to_thread(_delete)

    async def touch_conversation(self, conversation_id: str) -> None:
        await self._ensure_initialized()

        def _touch() -> None:
            conn = self._conn()
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )
            conn.commit()

        async with self._lock:
            await asyncio.to_thread(_touch)

    # -- messages ----------------------------------------------------------

    async def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: Any,
        model: str | None = None,
        parent_message_id: str | None = None,
    ) -> MessageRecord:
        await self._ensure_initialized()

        def _add() -> MessageRecord:
            record = MessageRecord(
                id=_new_id("msg"),
                conversation_id=conversation_id,
                role=role,
                content=content,
                model=model,
                created_at=_now(),
                parent_message_id=parent_message_id,
            )
            conn = self._conn()
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, model, created_at,"
                " parent_message_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.id, record.conversation_id, record.role, json.dumps(record.content),
                 record.model, record.created_at, record.parent_message_id),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (record.created_at, conversation_id),
            )
            conn.commit()
            return record

        async with self._lock:
            return await asyncio.to_thread(_add)

    async def list_messages(self, conversation_id: str) -> list[MessageRecord]:
        await self._ensure_initialized()

        def _list() -> list[MessageRecord]:
            rows = self._conn().execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC",
                (conversation_id,),
            ).fetchall()
            return [self._row_to_message(row) for row in rows]

        return await asyncio.to_thread(_list)

    async def get_message(self, message_id: str) -> MessageRecord | None:
        await self._ensure_initialized()

        def _get() -> MessageRecord | None:
            row = self._conn().execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            return self._row_to_message(row) if row else None

        return await asyncio.to_thread(_get)

    async def truncate_messages_after(self, message_id: str) -> int:
        """Delete ``message_id`` and everything after it in its conversation.

        Used by edit/regenerate: branch from an earlier user message by
        dropping the old turn and everything that followed.
        """
        await self._ensure_initialized()

        def _truncate() -> int:
            conn = self._conn()
            row = conn.execute(
                "SELECT conversation_id, created_at FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if row is None:
                return 0
            cur = conn.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND (created_at > ?"
                " OR (created_at = ? AND id = ?))",
                (row["conversation_id"], row["created_at"], row["created_at"], message_id),
            )
            conn.commit()
            return cur.rowcount

        async with self._lock:
            return await asyncio.to_thread(_truncate)

    # -- row mapping -------------------------------------------------------

    @staticmethod
    def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=json.loads(row["content"]),
            model=row["model"],
            created_at=row["created_at"],
            parent_message_id=row["parent_message_id"],
        )


def build_repository(settings: Any) -> ConversationRepository:
    """Factory — swap this for a PostgreSQL repository in production."""
    return SqliteConversationRepository(settings.db_path)
