import datetime as dt
import hashlib
import json
import secrets
import sqlite3
from typing import Any, Optional


class DatabaseHandler:
    def __init__(self, path: str):
        self.path = path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self._enable_foreign_keys()

    def _enable_foreign_keys(self) -> None:
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.connection.commit()

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.cursor.execute(query, params)

    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        cur = self.cursor.execute(query, params)
        return cur.fetchone()

    def fetchall(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        cur = self.cursor.execute(query, params)
        return cur.fetchall()

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.close()

    @staticmethod
    def _utc_now_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()


class CacheDatabase(DatabaseHandler):
    """
    Stores request/response cache entries to reduce repeated model runs.
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.init_schema()

    def init_schema(self) -> None:
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                request_hash TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.commit()

    @staticmethod
    def _stable_json(data: Any) -> str:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def hash_request(cls, data: Any) -> str:
        payload = cls._stable_json(data)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_entry(self, request_hash: str) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            """
            SELECT request_hash, request_json, response_json, created_at, updated_at, hit_count
            FROM cache_entries
            WHERE request_hash = ?
            """,
            (request_hash,),
        )
        if not row:
            return None
        return {
            "request_hash": row["request_hash"],
            "request": json.loads(row["request_json"]),
            "response": json.loads(row["response_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "hit_count": row["hit_count"],
        }

    def set_entry(self, request_data: Any, response_data: Any) -> str:
        request_hash = self.hash_request(request_data)
        now = self._utc_now_iso()
        self.execute(
            """
            INSERT INTO cache_entries (request_hash, request_json, response_json, created_at, updated_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(request_hash) DO UPDATE SET
                response_json = excluded.response_json,
                updated_at = excluded.updated_at
            """,
            (
                request_hash,
                self._stable_json(request_data),
                self._stable_json(response_data),
                now,
                now,
            ),
        )
        self.commit()
        return request_hash

    def increment_hit(self, request_hash: str) -> None:
        self.execute(
            """
            UPDATE cache_entries
            SET hit_count = hit_count + 1, updated_at = ?
            WHERE request_hash = ?
            """,
            (self._utc_now_iso(), request_hash),
        )
        self.commit()

    def get_or_push(self, data: Any, response_data: Any = None) -> tuple[Optional[dict[str, Any]], bool, str]:
        """
        Returns (response, cache_hit, request_hash)

        - If request exists: returns cached response and cache_hit=True
        - If request does not exist and response_data is provided: stores and returns response_data with cache_hit=False
        - If request does not exist and response_data is None: returns (None, False, request_hash)
        """
        request_hash = self.hash_request(data)
        entry = self.get_entry(request_hash)

        if entry is not None:
            self.increment_hit(request_hash)
            return entry["response"], True, request_hash

        if response_data is None:
            return None, False, request_hash

        self.set_entry(data, response_data)
        return response_data, False, request_hash


class UserDatabase(DatabaseHandler):
    """
    Handles users, API tokens, and links between users/tokens and cache entries.
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.init_schema()

    def init_schema(self) -> None:
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT NOT NULL UNIQUE DEFAULT '',
                password_hash TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        for col, definition in [("email", "TEXT NOT NULL DEFAULT ''"),
                                ("password_hash", "TEXT NOT NULL DEFAULT ''")]:
            try:
                self.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                self.commit()
            except Exception:
                pass

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_tokens (
                token_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                token_prefix TEXT NOT NULL,
                label      TEXT,
                scopes_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                expires_at TEXT,
                revoked_at TEXT,
                last_used_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_cache_links (
                link_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                request_hash TEXT NOT NULL,
                token_id     INTEGER,
                created_at   TEXT NOT NULL,
                UNIQUE(user_id, request_hash, token_id),
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY(request_hash) REFERENCES cache_entries(request_hash) ON DELETE CASCADE,
                FOREIGN KEY(token_id) REFERENCES user_tokens(token_id) ON DELETE SET NULL
            )
            """
        )
        self.commit()

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def create_user(self, username: str, email: str, password_hash: str) -> dict[str, Any]:
        now = self._utc_now_iso()
        self.execute(
            """
            INSERT INTO users (username, email, password_hash, created_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (username, email.lower(), password_hash, now),
        )
        self.commit()
        row = self.fetchone(
            "SELECT user_id, username, email, created_at, is_active FROM users WHERE username = ?",
            (username,),
        )
        return dict(row) if row else {}

    def get_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT user_id, username, created_at, is_active FROM users WHERE user_id = ?",
            (user_id,),
        )
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT user_id, username, created_at, is_active FROM users WHERE username = ?",
            (username,),
        )
        return dict(row) if row else None
    
    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT user_id, username, email, password_hash, created_at, is_active FROM users WHERE email = ?",
            (email.lower(),),
        )
        return dict(row) if row else None

    def get_user_by_identity(self, identity: str) -> Optional[dict[str, Any]]:
        """Look up a user by username OR email, returning password_hash for verification."""
        row = self.fetchone(
            """
            SELECT user_id, username, email, password_hash, created_at, is_active
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identity, identity.lower()),
        )
        return dict(row) if row else None

    def create_token(
        self,
        user_id: int,
        label: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        raw_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(raw_token)
        token_prefix = raw_token[:8]
        now = self._utc_now_iso()

        expires_at = None
        if ttl_seconds is not None:
            expires_at = (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=ttl_seconds)
            ).isoformat()

        self.execute(
            """
            INSERT INTO user_tokens (
                user_id, token_hash, token_prefix, label, scopes_json, created_at, expires_at, revoked_at, last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                user_id,
                token_hash,
                token_prefix,
                label,
                json.dumps(scopes or [], ensure_ascii=True),
                now,
                expires_at,
            ),
        )
        token_id = self.cursor.lastrowid
        self.commit()

        return {
            "token_id": token_id,
            "token": raw_token,
            "token_prefix": token_prefix,
            "expires_at": expires_at,
        }

    def validate_token(self, raw_token: str, required_scope: Optional[str] = None) -> Optional[dict[str, Any]]:
        token_hash = self._hash_token(raw_token)
        now = self._utc_now_iso()

        row = self.fetchone(
            """
            SELECT token_id, user_id, token_prefix, label, scopes_json, created_at, expires_at, revoked_at, last_used_at
            FROM user_tokens
            WHERE token_hash = ?
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (token_hash, now),
        )
        if row is None:
            return None

        scopes = json.loads(row["scopes_json"] or "[]")
        if required_scope is not None and required_scope not in scopes:
            return None

        self.execute(
            "UPDATE user_tokens SET last_used_at = ? WHERE token_id = ?",
            (now, row["token_id"]),
        )
        self.commit()

        user = self.get_user_by_id(row["user_id"])
        if not user or int(user.get("is_active", 0)) != 1:
            return None

        return {
            "token_id": row["token_id"],
            "user_id": row["user_id"],
            "user": user,
            "token_prefix": row["token_prefix"],
            "label": row["label"],
            "scopes": scopes,
            "expires_at": row["expires_at"],
        }

    def revoke_token_by_id(self, token_id: int) -> bool:
        now = self._utc_now_iso()
        self.execute(
            "UPDATE user_tokens SET revoked_at = ? WHERE token_id = ? AND revoked_at IS NULL",
            (now, token_id),
        )
        changed = self.cursor.rowcount > 0
        self.commit()
        return changed

    def revoke_token_value(self, raw_token: str) -> bool:
        now = self._utc_now_iso()
        token_hash = self._hash_token(raw_token)
        self.execute(
            "UPDATE user_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (now, token_hash),
        )
        changed = self.cursor.rowcount > 0
        self.commit()
        return changed

    def link_cached_request(self, user_id: int, request_hash: str, token_id: Optional[int] = None) -> None:
        self.execute(
            """
            INSERT INTO user_cache_links (user_id, request_hash, token_id, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, request_hash, token_id) DO NOTHING
            """,
            (user_id, request_hash, token_id, self._utc_now_iso()),
        )
        self.commit()

    def list_user_cached_requests(self, user_id: int, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT
                l.link_id,
                l.created_at AS linked_at,
                l.token_id,
                c.request_hash,
                c.request_json,
                c.response_json,
                c.created_at,
                c.updated_at,
                c.hit_count
            FROM user_cache_links l
            JOIN cache_entries c ON c.request_hash = l.request_hash
            WHERE l.user_id = ?
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "link_id": row["link_id"],
                    "linked_at": row["linked_at"],
                    "token_id": row["token_id"],
                    "request_hash": row["request_hash"],
                    "request": json.loads(row["request_json"]),
                    "response": json.loads(row["response_json"]),
                    "cache_created_at": row["created_at"],
                    "cache_updated_at": row["updated_at"],
                    "cache_hit_count": row["hit_count"],
                }
            )
        return out