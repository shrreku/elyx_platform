import os
import sqlite3
from typing import List, Dict, Optional


DB_PATH = os.getenv("ELYX_DB_PATH", os.path.join("data", "elyx.db"))


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS suggestions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                agent TEXT,
                title TEXT,
                details TEXT,
                category TEXT,
                status TEXT,
                created_at TEXT,
                conversation_id TEXT,
                message_index INTEGER,
                message_timestamp TEXT,
                source TEXT,
                origin TEXT,
                source_message TEXT,
                context_json TEXT
            );
            """
        )
        # Ensure columns exist for upgrades
        cols = {r[1] for r in cur.execute("PRAGMA table_info(suggestions)").fetchall()}
        def ensure(col: str, ddl: str):
            if col not in cols:
                cur.execute(f"ALTER TABLE suggestions ADD COLUMN {ddl}")
        ensure("conversation_id", "conversation_id TEXT")
        ensure("message_index", "message_index INTEGER")
        ensure("message_timestamp", "message_timestamp TEXT")
        ensure("source", "source TEXT")
        ensure("origin", "origin TEXT")
        ensure("source_message", "source_message TEXT")
        ensure("context_json", "context_json TEXT")
        con.commit()

        # User profiles
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        con.commit()
        # Issues table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS issues (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                details TEXT,
                category TEXT,
                severity TEXT,
                conversation_id TEXT,
                message_index INTEGER,
                message_timestamp TEXT,
                created_at TEXT
            );
            """
        )
        # Ensure columns exist for upgrades (older DBs may miss these)
        issue_cols = {r[1] for r in cur.execute("PRAGMA table_info(issues)").fetchall()}
        def ensure_issue(col: str, ddl: str):
            if col not in issue_cols:
                cur.execute(f"ALTER TABLE issues ADD COLUMN {ddl}")
        ensure_issue("status", "status TEXT")
        ensure_issue("progress_percent", "progress_percent INTEGER")
        ensure_issue("last_reviewed_at", "last_reviewed_at TEXT")
        ensure_issue("priority", "priority TEXT")
        ensure_issue("time_window", "time_window TEXT")
        ensure_issue("resolve_trigger_reference", "resolve_trigger_reference TEXT")
        ensure_issue("triggered_by", "triggered_by TEXT")
        con.commit()
        # Episodes and related tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                trigger_type TEXT,
                trigger_description TEXT,
                trigger_timestamp TEXT,
                status TEXT,
                priority INTEGER,
                member_state_before TEXT,
                member_state_after TEXT,
                confidence REAL,
                created_at TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episode_friction (
                id TEXT PRIMARY KEY,
                episode_id TEXT,
                category TEXT,
                description TEXT,
                severity INTEGER,
                resolution_action TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episode_interventions (
                id TEXT PRIMARY KEY,
                episode_id TEXT,
                action TEXT,
                responsible_agent TEXT,
                timestamp TEXT,
                outcome TEXT
            );
            """
        )
        # Decisions and evidence
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                timestamp TEXT,
                responsible_agent TEXT,
                rationale TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_evidence (
                id TEXT PRIMARY KEY,
                decision_id TEXT,
                evidence_type TEXT,
                source TEXT,
                data_json TEXT,
                timestamp TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_messages (
                id TEXT PRIMARY KEY,
                decision_id TEXT,
                message_id TEXT,
                message_index INTEGER,
                message_timestamp TEXT
            );
            """
        )
        # Member Timeline State
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS member_timeline_state (
                user_id TEXT PRIMARY KEY,
                state_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        con.commit()
        # Experiment Ledger
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_ledger (
                id TEXT PRIMARY KEY,
                template TEXT,
                hypothesis TEXT,
                protocol_json TEXT,
                duration TEXT,
                member_id TEXT,
                status TEXT,
                outcome TEXT,
                success INTEGER,
                created_at TEXT
            );
            """
        )
        # Ensure new columns exist for upgrades
        exp_cols = {r[1] for r in cur.execute("PRAGMA table_info(experiment_ledger)").fetchall()}
        def ensure_exp(col: str, ddl: str):
            if col not in exp_cols:
                cur.execute(f"ALTER TABLE experiment_ledger ADD COLUMN {ddl}")
        ensure_exp("intervention", "intervention TEXT")
        ensure_exp("decision", "decision TEXT")
        ensure_exp("plan_section_updated", "plan_section_updated TEXT")
        con.commit()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_measurements (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                name TEXT,
                value REAL,
                ts TEXT,
                raw_json TEXT
            );
            """
        )
        con.commit()


def suggestions_list() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM suggestions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def suggestions_add_many(items: List[Dict]):
    if not items:
        return
    with _conn() as con:
        cur = con.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO suggestions (
                id, user_id, agent, title, details, category, status, created_at,
                conversation_id, message_index, message_timestamp, source, origin, source_message, context_json
            )
            VALUES (
                :id, :user_id, :agent, :title, :details, :category, :status, :created_at,
                :conversation_id, :message_index, :message_timestamp, :source, :origin, :source_message, :context_json
            )
            """,
            items,
        )
        con.commit()


def issues_add_many(items: List[Dict]):
    if not items:
        return
    # Normalize items to ensure new fields exist
    from datetime import datetime
    now_iso = datetime.now().isoformat()
    normalized: List[Dict] = []
    for it in items:
        norm = {
            "id": it.get("id"),
            "user_id": it.get("user_id"),
            "title": it.get("title"),
            "details": it.get("details"),
            "category": it.get("category"),
            "severity": it.get("severity"),
            "status": it.get("status", "open"),
            "progress_percent": it.get("progress_percent", 0),
            "last_reviewed_at": it.get("last_reviewed_at", now_iso),
            "priority": it.get("priority"),
            "time_window": it.get("time_window"),
            "resolve_trigger_reference": it.get("resolve_trigger_reference"),
            "triggered_by": it.get("triggered_by"),
            "conversation_id": it.get("conversation_id"),
            "message_index": it.get("message_index"),
            "message_timestamp": it.get("message_timestamp"),
            "created_at": it.get("created_at", now_iso),
        }
        normalized.append(norm)
    with _conn() as con:
        cur = con.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO issues (
                id, user_id, title, details, category, severity, status, progress_percent, last_reviewed_at,
                priority, time_window, resolve_trigger_reference, triggered_by,
                conversation_id, message_index, message_timestamp, created_at
            ) VALUES (
                :id, :user_id, :title, :details, :category, :severity, :status, :progress_percent, :last_reviewed_at,
                :priority, :time_window, :resolve_trigger_reference, :triggered_by,
                :conversation_id, :message_index, :message_timestamp, :created_at
            )
            """,
            normalized,
        )
        con.commit()


def issues_list() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM issues ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def issues_update_progress(item_id: str, status: str, progress_percent: int) -> bool:
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "UPDATE issues SET status=?, progress_percent=?, last_reviewed_at=datetime('now') WHERE id=?",
            (status, progress_percent, item_id),
        )
        con.commit()
        return cur.rowcount > 0


def issues_update(item_id: str, fields: Dict) -> bool:
    """Update allowed issue fields generically."""
    allowed = {
        "title",
        "details",
        "category",
        "severity",
        "status",
        "progress_percent",
        "priority",
        "time_window",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    sets = ", ".join(f"{k}=?" for k in updates.keys())
    values = list(updates.values())
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"UPDATE issues SET {sets}, last_reviewed_at=datetime('now') WHERE id=?",
            (*values, item_id),
        )
        con.commit()
        return cur.rowcount > 0


def issues_close_by_text(text: str, reference: str | None = None, triggered_by: str | None = None) -> int:
    """Mark issues as resolved if their title/details are contradicted by a resolution text.

    Heuristic: if text contains phrases like "feels fine now", "no more", "resolved", and the issue title words appear,
    set status='resolved', progress_percent=100.
    """
    lower = text.lower()
    # Broader improvement phrases
    improvement_markers = [
        "feels fine",
        "feel fine",
        "feels alright",
        "feel alright",
        "feels okay",
        "feel okay",
        "feels good",
        "feel good",
        "feels better",
        "feel better",
        "no more",
        "no longer",
        "resolved",
        "better now",
        "getting better",
        "improved",
        "improving",
        "okay now",
        "alright now",
        "good now",
        "back to normal",
        "gone",
        "pain reduced",
        "less pain",
        "subsided",
        "cleared up",
        "all good",
        "much better",
    ]
    if not any(k in lower for k in improvement_markers):
        return 0
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, title, details, category FROM issues WHERE status IS NULL OR status!='resolved'"
        ).fetchall()
        to_close = []
        for r in rows:
            title = (r["title"] or "").lower()
            details = (r["details"] or "").lower()
            category = (r["category"] or "").lower()
            # simple token overlap from title + details
            tokens = [t for t in (title + " " + details).split() if len(t) > 3]
            score = sum(1 for t in tokens if t in lower)
            # category hinting
            cat_hints = {
                "physio": ["back", "knee", "shoulder", "leg", "mobility", "pain"],
                "medical": ["headache", "migraine", "fever", "viral", "stomach", "glucose"],
                "performance": ["sleep", "hrv", "recovery", "fatigue"],
                "nutrition": ["stomach", "bloating", "meal"],
            }
            if category in cat_hints and any(h in lower for h in cat_hints[category]):
                score += 2
            if score >= 1:
                to_close.append(r["id"])
        if not to_close:
            return 0
        cur = con.cursor()
        if reference is None:
            reference = ""
        if triggered_by is None:
            triggered_by = "user"
        cur.executemany(
            "UPDATE issues SET status='resolved', progress_percent=100, last_reviewed_at=datetime('now'), resolve_trigger_reference=?, triggered_by=? WHERE id=?",
            [(reference, triggered_by, i) for i in to_close],
        )
        con.commit()
        return len(to_close)


def suggestions_update_status(item_id: str, status: str) -> bool:
    with _conn() as con:
        cur = con.cursor()
        cur.execute("UPDATE suggestions SET status=? WHERE id=?", (status, item_id))
        con.commit()
        return cur.rowcount > 0


def issues_update_priority_time(item_id: str, priority: str, time_window: str) -> bool:
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "UPDATE issues SET priority=?, time_window=?, last_reviewed_at=datetime('now') WHERE id=?",
            (priority, time_window, item_id),
        )
        con.commit()
        return cur.rowcount > 0


# Episodes CRUD
def episodes_list() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM episodes ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def episodes_add(item: Dict):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO episodes (
                id, user_id, title, trigger_type, trigger_description, trigger_timestamp, status, priority,
                member_state_before, member_state_after, confidence, created_at
            ) VALUES (
                :id, :user_id, :title, :trigger_type, :trigger_description, :trigger_timestamp, :status, :priority,
                :member_state_before, :member_state_after, :confidence, :created_at
            )
            """,
            item,
        )
        con.commit()


def episodes_update_status(episode_id: str, status: str) -> bool:
    with _conn() as con:
        cur = con.cursor()
        cur.execute("UPDATE episodes SET status=? WHERE id=?", (status, episode_id))
        con.commit()
        return cur.rowcount > 0


def episode_add_intervention(item: Dict):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO episode_interventions (
                id, episode_id, action, responsible_agent, timestamp, outcome
            ) VALUES (
                :id, :episode_id, :action, :responsible_agent, :timestamp, :outcome
            )
            """,
            item,
        )
        con.commit()


def episode_list_interventions(episode_id: str) -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM episode_interventions WHERE episode_id=? ORDER BY timestamp ASC",
            (episode_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# Decisions CRUD
def decisions_add(item: Dict, evidence: List[Dict], messages: List[Dict]):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO decisions (
                id, type, content, timestamp, responsible_agent, rationale
            ) VALUES (
                :id, :type, :content, :timestamp, :responsible_agent, :rationale
            )
            """,
            item,
        )
        if evidence:
            cur.executemany(
                """
                INSERT OR IGNORE INTO decision_evidence (
                    id, decision_id, evidence_type, source, data_json, timestamp
                ) VALUES (
                    :id, :decision_id, :evidence_type, :source, :data_json, :timestamp
                )
                """,
                evidence,
            )
        if messages:
            cur.executemany(
                """
                INSERT OR IGNORE INTO decision_messages (
                    id, decision_id, message_id, message_index, message_timestamp
                ) VALUES (
                    :id, :decision_id, :message_id, :message_index, :message_timestamp
                )
                """,
                messages,
            )
        con.commit()


def decisions_list() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM decisions ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]


def decisions_get_with_why(decision_id: str) -> Dict:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        decision = con.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if not decision:
            return {}
        evidence = con.execute("SELECT * FROM decision_evidence WHERE decision_id=?", (decision_id,)).fetchall()
        messages = con.execute("SELECT * FROM decision_messages WHERE decision_id=?", (decision_id,)).fetchall()
        return {
            "decision": dict(decision),
            "evidence": [dict(r) for r in evidence],
            "messages": [dict(r) for r in messages],
        }


# Experiment Ledger CRUD
def experiment_ledger_add(item: Dict):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO experiment_ledger (
                id, template, hypothesis, protocol_json, duration, member_id, status, outcome, success, created_at,
                intervention, decision, plan_section_updated
            ) VALUES (
                :id, :template, :hypothesis, :protocol_json, :duration, :member_id, :status, :outcome, :success, :created_at,
                :intervention, :decision, :plan_section_updated
            )
            """,
            item,
        )
        con.commit()


def experiment_ledger_list() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM experiment_ledger ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def experiment_ledger_add_measurement(item: Dict):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO experiment_measurements (
                id, experiment_id, name, value, ts, raw_json
            ) VALUES (
                :id, :experiment_id, :name, :value, :ts, :raw_json
            )
            """,
            item,
        )
        con.commit()


def experiment_ledger_results() -> List[Dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM experiment_ledger WHERE success=1 OR status='completed' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# Member Timeline State CRUD
def member_timeline_state_get(user_id: str) -> Dict:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM member_timeline_state WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return {}
        import json as _json
        data = dict(row)
        if data.get("state_json"):
            try:
                data["state"] = _json.loads(data["state_json"])  # type: ignore[assignment]
            except Exception:
                data["state"] = None
        return data


def member_timeline_state_set(user_id: str, state: Dict) -> bool:
    import json as _json
    now = __import__("datetime").datetime.now().isoformat()
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO member_timeline_state (user_id, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at
            """,
            (user_id, _json.dumps(state), now, now),
        )
        con.commit()
        return cur.rowcount > 0


# User profile CRUD
def user_profile_get(user_id: str) -> Dict:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return {}
        import json as _json
        data = dict(row)
        if data.get("profile_json"):
            try:
                data["profile"] = _json.loads(data["profile_json"])  # type: ignore[assignment]
            except Exception:
                data["profile"] = None
        return data


def user_profile_set(user_id: str, profile: Dict) -> bool:
    import json as _json
    now = __import__("datetime").datetime.now().isoformat()
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO user_profiles (user_id, profile_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at
            """,
            (user_id, _json.dumps(profile), now, now),
        )
        con.commit()
        return cur.rowcount > 0

