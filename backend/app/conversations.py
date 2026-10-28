from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4


class ConversationNotFound(Exception):
    pass


class RequestIdConflict(Exception):
    pass


class ConversationStore:
    """SQLite-backed, isolated conversation storage."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_through_sequence INTEGER NOT NULL DEFAULT 0,
                    next_sequence INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    response_json TEXT,
                    UNIQUE (conversation_id, sequence),
                    UNIQUE (conversation_id, request_id, role)
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_sequence
                    ON messages(conversation_id, sequence);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self) -> dict[str, str]:
        conversation_id = str(uuid4())
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return {"conversation_id": conversation_id, "created_at": now}

    def context(self, conversation_id: str) -> dict:
        with self._connection() as connection:
            conversation = connection.execute(
                "SELECT summary, summary_through_sequence FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            messages = connection.execute(
                """
                SELECT id, sequence, role, content, created_at
                FROM messages WHERE conversation_id = ? ORDER BY sequence
                """,
                (conversation_id,),
            ).fetchall()
        return {
            "summary": conversation["summary"],
            "summary_through_sequence": conversation["summary_through_sequence"],
            "messages": [dict(message) for message in messages],
        }

    def update_summary(
        self,
        conversation_id: str,
        expected_sequence: int,
        summary: str,
        through_sequence: int,
    ) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET summary = ?, summary_through_sequence = ?, updated_at = ?
                WHERE id = ? AND summary_through_sequence = ?
                """,
                (summary, through_sequence, self._now(), conversation_id, expected_sequence),
            )
            if cursor.rowcount == 0:
                exists = connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if exists is None:
                    raise ConversationNotFound(conversation_id)
                return False
            return True

    def cached_response(
        self,
        conversation_id: str,
        request_id: str,
        user_content: str,
    ) -> dict | None:
        with self._connection() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            row = connection.execute(
                """
                SELECT assistant.response_json, user.content AS user_content
                FROM messages AS assistant
                JOIN messages AS user
                  ON user.conversation_id = assistant.conversation_id
                 AND user.request_id = assistant.request_id
                 AND user.role = 'user'
                WHERE assistant.conversation_id = ?
                  AND assistant.request_id = ?
                  AND assistant.role = 'assistant'
                """,
                (conversation_id, request_id),
            ).fetchone()
        if row and row["user_content"] != user_content:
            raise RequestIdConflict(request_id)
        return json.loads(row["response_json"]) if row and row["response_json"] else None

    def save_turn(
        self,
        conversation_id: str,
        request_id: str,
        user_content: str,
        response: dict,
    ) -> dict:
        """Atomically persist a complete turn, returning the canonical response."""
        now = self._now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            conversation = connection.execute(
                "SELECT next_sequence FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            cached = connection.execute(
                """
                SELECT assistant.response_json, user.content AS user_content
                FROM messages AS assistant
                JOIN messages AS user
                  ON user.conversation_id = assistant.conversation_id
                 AND user.request_id = assistant.request_id
                 AND user.role = 'user'
                WHERE assistant.conversation_id = ?
                  AND assistant.request_id = ?
                  AND assistant.role = 'assistant'
                """,
                (conversation_id, request_id),
            ).fetchone()
            if cached:
                if cached["user_content"] != user_content:
                    raise RequestIdConflict(request_id)
                return json.loads(cached["response_json"])

            sequence = conversation["next_sequence"]
            response_json = json.dumps(response, ensure_ascii=False)
            connection.execute(
                """
                INSERT INTO messages
                    (id, conversation_id, sequence, role, content, created_at, request_id)
                VALUES (?, ?, ?, 'user', ?, ?, ?)
                """,
                (str(uuid4()), conversation_id, sequence, user_content, now, request_id),
            )
            connection.execute(
                """
                INSERT INTO messages
                    (id, conversation_id, sequence, role, content, created_at, request_id, response_json)
                VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    sequence + 1,
                    response["response"],
                    now,
                    request_id,
                    response_json,
                ),
            )
            connection.execute(
                "UPDATE conversations SET next_sequence = ?, updated_at = ? WHERE id = ?",
                (sequence + 2, now, conversation_id),
            )
        return response

    def delete(self, conversation_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)
