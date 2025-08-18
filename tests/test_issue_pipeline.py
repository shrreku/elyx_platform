import os
import json
import uuid
from fastapi.testclient import TestClient
import pytest

from backend.main import app, router, issue_extractor, issues_add_many, issues_list, issues_close_by_text

client = TestClient(app)


def clear_issues_db():
    # Remove all issues
    from data.db import _conn
    con = _conn()
    cur = con.cursor()
    cur.execute("DELETE FROM issues")
    con.commit()
    con.close()


def test_issue_extractor_detects_issue(monkeypatch):
    # Mock the underlying LLM response for IssueExtractor to return a single issue
    sample_raw = json.dumps({
        "issues": [
            {
                "title": "Test headache",
                "details": "Mild headache after coffee",
                "category": "medical",
                "severity": "low"
            }
        ],
        "improvements": []
    })

    def fake_call(msgs):
        return sample_raw

    monkeypatch.setattr(issue_extractor.agent, "call_openrouter", fake_call)

    res = issue_extractor.extract_all("I have a headache after coffee", None)
    issues = res.get("issues", [])
    assert isinstance(issues, list)
    assert len(issues) == 1
    assert issues[0]["title"] == "Test headache"
    assert issues[0]["category"] == "medical"


def test_improvement_triggers_auto_close(monkeypatch):
    # Prepare DB with one open issue that should be closed by improvement message
    clear_issues_db()
    existing = [
        {
            "id": str(uuid.uuid4()),
            "user_id": "rohan",
            "title": "Lower back pain after deadlift",
            "details": "Soreness and stiffness in lower back after session",
            "category": "physio",
            "severity": "medium"
        }
    ]
    issues_add_many(existing)
    # Sanity check inserted
    items = issues_list()
    assert len(items) == 1

    # Monkeypatch router.extract_key_info to indicate an improvement
    def fake_extract_key_info(msg, ctx):
        return {"is_improvement": True, "improvement": {"title": "Back pain improved", "details": "Back pain much better now"}}
    monkeypatch.setattr(router, "extract_key_info", fake_extract_key_info)

    # Send chat containing improvement markers so issues_close_by_text will trigger
    msg = "My lower back feels much better now, the soreness is gone."
    r = client.post("/chat", json={"sender": "Rohan", "message": msg, "context": None, "use_crewai": False})
    assert r.status_code == 200

    # Verify the issue was auto-closed (status resolved)
    rows = issues_list()
    assert len(rows) == 1
    assert rows[0].get("status") == "resolved" or rows[0].get("progress_percent") == 100


def test_fallback_extractor_creates_issue(monkeypatch):
    # Clear DB
    clear_issues_db()

    # Make router.extract_key_info return empty (no explicit issue)
    monkeypatch.setattr(router, "extract_key_info", lambda m, c: {})

    # Mock issue_extractor.extract_all to return an issue
    def fake_extract_all(msg, ctx):
        return {
            "issues": [
                {"title": "Mock GI upset", "details": "Bloating after meal", "category": "nutrition", "severity": "low"}
            ],
            "improvements": []
        }
    monkeypatch.setattr(issue_extractor, "extract_all", fake_extract_all)

    r = client.post("/chat", json={"sender": "Rohan", "message": "I had bloating after lunch", "context": None, "use_crewai": False})
    assert r.status_code == 200

    rows = issues_list()
    assert any("Mock GI upset" in it.get("title", "") for it in rows)
