from __future__ import annotations

import builtins
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from h3ltx.errors import RuntimeFault

TERMINAL = {"succeeded", "failed", "cancelled"}
TRANSITIONS = {
    "queued": {"preparing", "cancelled"},
    "preparing": {"running", "failed", "cancelled"},
    "running": {"finalizing", "failed", "cancelled"},
    "finalizing": {"succeeded", "failed", "cancelled"},
}


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS jobs (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
          type TEXT NOT NULL, state TEXT NOT NULL, request TEXT NOT NULL,
          resolved TEXT NOT NULL, fingerprint TEXT NOT NULL, client_key TEXT UNIQUE,
          priority INTEGER NOT NULL, tags TEXT NOT NULL,
          created REAL NOT NULL, updated REAL NOT NULL,
          progress REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT 'queued',
          error TEXT, result TEXT, gpu INTEGER, parent TEXT REFERENCES jobs(id), run_dir TEXT);
        CREATE INDEX IF NOT EXISTS queue ON jobs(state, priority DESC, sequence);
        CREATE TABLE IF NOT EXISTS events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT REFERENCES jobs(id),
          timestamp REAL NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS completions (
          cursor INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE REFERENCES jobs(id));
        CREATE TABLE IF NOT EXISTS measurements (
          job_id TEXT PRIMARY KEY REFERENCES jobs(id), data TEXT NOT NULL);
        """)

    def close(self) -> None:
        self.db.close()

    def recover(self) -> builtins.list[dict[str, Any]]:
        interrupted = self.list(states=["preparing", "running", "finalizing"], limit=1000000)
        for job in interrupted:
            self.transition(
                job["id"],
                "failed",
                error={
                    "code": "interrupted_by_restart",
                    "message": "Worker stopped during execution",
                },
            )
        return interrupted

    @staticmethod
    def decode(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("request", "resolved", "tags", "error", "result"):
            if data.get(key) is not None:
                data[key] = json.loads(data[key])
        return data

    def submit(
        self, request: dict[str, Any], resolved: dict[str, Any], parent: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        key = request.get("client_request_id")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if key:
                    old = self.db.execute(
                        "SELECT * FROM jobs WHERE client_key=?", (key,)
                    ).fetchone()
                    if old:
                        if old["fingerprint"] != fingerprint:
                            raise RuntimeFault(
                                "idempotency_conflict", "Key reused with different request", 409
                            )
                        self.db.execute("COMMIT")
                        return self.decode(old), False
                identifier, now = uuid.uuid4().hex, time.time()
                self.db.execute(
                    """INSERT INTO jobs
                (id,type,state,request,resolved,fingerprint,client_key,priority,tags,created,updated,parent)
                VALUES(?,?,'queued',?,?,?,?,?,?,?,?,?)""",
                    (
                        identifier,
                        request["type"],
                        json.dumps(request),
                        json.dumps(resolved),
                        fingerprint,
                        key,
                        request.get("priority", 0),
                        json.dumps(request.get("tags", [])),
                        now,
                        now,
                        parent,
                    ),
                )
                self.event(identifier, {"state": "queued"})
                self.db.execute("COMMIT")
            except BaseException:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise
        return self.get(identifier), True

    def get(self, identifier: str) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise RuntimeFault("job_not_found", "Job not found", 404)
        return self.decode(row)

    def list(
        self,
        *,
        states: builtins.list[str] | None = None,
        kind: str | None = None,
        tag: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> builtins.list[dict[str, Any]]:
        clauses: builtins.list[str] = []
        args: builtins.list[Any] = []
        if states:
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            args.extend(states)
        if kind:
            clauses.append("type=?")
            args.append(kind)
        if since is not None:
            clauses.append("created>=?")
            args.append(since)
        if tag:
            clauses.append("EXISTS (SELECT 1 FROM json_each(jobs.tags) WHERE value=?)")
            args.append(tag)
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        with self.lock:
            rows = self.db.execute(
                sql + " ORDER BY priority DESC, sequence LIMIT ?", [*args, limit]
            ).fetchall()
        return [self.decode(row) for row in rows]

    def claim(self, gpu: int, eligible: builtins.list[str] | None = None) -> dict[str, Any] | None:
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                jobs = self.list(states=["queued"], limit=100000)
                job = next((j for j in jobs if eligible is None or j["id"] in eligible), None)
                if job:
                    self.db.execute(
                        "UPDATE jobs SET state='preparing',stage='preparing',gpu=?,updated=? "
                        "WHERE id=?",
                        (gpu, time.time(), job["id"]),
                    )
                    self.event(job["id"], {"state": "preparing", "gpu": gpu})
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return self.get(job["id"]) if job else None

    def transition(self, identifier: str, state: str, **fields: Any) -> None:
        with self.lock:
            previous = self.get(identifier)["state"]
            if state not in TRANSITIONS.get(previous, set()):
                raise RuntimeFault("invalid_transition", f"Cannot move {previous} to {state}", 409)
            allowed = {"error", "result", "progress", "stage", "run_dir", "resolved"}
            if fields.keys() - allowed:
                raise ValueError("Unsupported job field")
            updates: dict[str, Any] = {
                "state": state,
                "stage": state,
                "updated": time.time(),
                **fields,
            }
            if state in TERMINAL:
                updates["progress"] = (
                    100 if state == "succeeded" else self.get(identifier)["progress"]
                )
            for key in ("error", "result", "resolved"):
                if key in updates:
                    updates[key] = json.dumps(updates[key])
            self.db.execute(
                "UPDATE jobs SET " + ",".join(f"{key}=?" for key in updates) + " WHERE id=?",
                [*updates.values(), identifier],
            )
            self.event(identifier, {"state": state, **fields})
            if state in TERMINAL:
                self.db.execute(
                    "INSERT OR IGNORE INTO completions(job_id) VALUES(?)", (identifier,)
                )

    def progress(self, identifier: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE jobs SET progress=?,stage=?,updated=? WHERE id=? AND state NOT "
                "IN ('succeeded','failed','cancelled')",
                (data.get("progress", 0), data.get("stage", "running"), time.time(), identifier),
            )
            self.event(identifier, data)

    def event(self, identifier: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO events(job_id,timestamp,data) VALUES(?,?,?)",
                (identifier, time.time(), json.dumps(data)),
            )

    def events(self, identifier: str, after: int = 0) -> builtins.list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM events WHERE job_id=? AND sequence>? ORDER BY sequence",
                (identifier, after),
            ).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def export(self, cursor: int, limit: int = 100) -> dict[str, Any]:
        with self.lock:
            rows = self.db.execute(
                "SELECT cursor,job_id FROM completions WHERE cursor>? ORDER BY cursor LIMIT ?",
                (cursor, limit),
            ).fetchall()
        return {
            "runs": [
                {
                    "id": row["job_id"],
                    "cursor": row["cursor"],
                    "bundle_url": f"/v1/jobs/{row['job_id']}/bundle",
                }
                for row in rows
            ],
            "cursor": rows[-1]["cursor"] if rows else cursor,
        }

    def measure(self, identifier: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO measurements VALUES(?,?)", (identifier, json.dumps(data))
            )

    def measurements(self, since: float | None = None) -> builtins.list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT j.*,m.data AS measured FROM jobs j "
                "LEFT JOIN measurements m ON m.job_id=j.id "
                "WHERE j.created>=? AND j.state IN ('succeeded','failed','cancelled')",
                (since or 0,),
            ).fetchall()
        result = []
        for row in rows:
            if row["measured"]:
                result.append(json.loads(row["measured"]))
                continue
            resolved = json.loads(row["resolved"])
            params = resolved.get("params", {})
            result.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "state": row["state"],
                    "profile": resolved.get("profile"),
                    "engine": params.get("engine"),
                    "resolution": f"{params.get('width')}x{params.get('height')}",
                    "steps": params.get("steps"),
                    "wall_seconds": 0 if row["gpu"] is None else None,
                    "timings_incomplete": row["gpu"] is not None,
                    "stages": {},
                    "output_seconds": 0,
                    "cost_estimate_usd": 0,
                }
            )
        return result
