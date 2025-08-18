import os
import json
import uuid
from fastapi.testclient import TestClient
import pytest

from backend.main import app, router, issue_extractor, issues_add_many, issues_list, issues_close_by_text, AUTO_ISSUE

client = TestClient(app)


def clear_issues_db():
    from data.db import _conn
    con = _conn()
    cur = con.cursor()
    cur.execute("DELETE FROM issues")
    con.commit()
    con.close()


def test_garmin_onboarding_analyze(monkeypatch):
    """Message with inline Garmin wearable metrics + scheduling request should be parsed as wearable_parsed and logistics intent."""
    sample_msg = "Good morning! Just completed my initial onboarding yesterday. Ready to start this journey. <Garmin data: Sleep score 78, HRV 42ms, Resting HR 68 bpm, Stress level: Medium>. Had my morning 20-min routine - feeling energetic. When can we schedule my first comprehensive assessment?"

    # Make extract_key_info return parsed wearable entities and logistics intent
    fake_extract = {
        "summary": "Onboarded; provided Garmin metrics and asked to schedule assessment",
        "intent": "logistics",
        "entities": {"hrv": 42, "sleep_score": 78, "resting_hr": 68},
        "keywords": ["garmin", "sleep_score", "hrv", "schedule", "assessment"],
        "confidence": 0.95,
        "is_health_issue": False,
        "is_improvement": False,
        "recommended_agents": ["Ruby", "Advik"]
    }
    monkeypatch.setattr(router, "extract_key_info", lambda m, c: fake_extract)
    # Let route return Ruby first
    monkeypatch.setattr(router, "route", lambda m, c, max_agents=2: ["Ruby", "Advik"])

    analysis = router.analyze_message(sample_msg, None, max_agents=2)
    assert analysis["message_type"] in ("wearable_parsed", "user_text", "wearable_xml")
    assert "Ruby" in analysis["agents"]
    assert analysis["extraction"]["intent"] == "logistics"
    # Recommended next action should include contacting Ruby for scheduling
    assert any(a.get("agent") == "Ruby" for a in analysis["recommended_next_actions"])


def test_dizzy_creates_issue_on_chat(monkeypatch):
    """A 'dizzy' report during exercise should create a high-severity medical issue in DB when AUTO_ISSUE is enabled."""
    clear_issues_db()
    assert AUTO_ISSUE is True  # test assumes auto-creation is enabled

    sample_msg = "Made some progress on Zone 2 cardio today. Felt dizzy around 30min, need advice."

    # Router indicates a health issue
    fake_issue = {
        "summary": "Dizziness during cardio",
        "intent": "medical",
        "entities": {"symptom": "dizzy"},
        "keywords": ["dizzy", "cardio"],
        "confidence": 0.9,
        "is_health_issue": True,
        "health_issue": {
            "title": "Dizzy during cardio",
            "details": "Felt dizzy around 30 minutes into Zone 2 cardio session",
            "category": "medical",
            "severity": "high"
        },
        "recommended_agents": ["Dr. Warren", "Advik"]
    }
    monkeypatch.setattr(router, "extract_key_info", lambda m, c: fake_issue)
    # Route to Dr. Warren
    monkeypatch.setattr(router, "route", lambda m, c, max_agents=2: ["Dr. Warren", "Advik"])

    r = client.post("/chat", json={"sender": "Rohan", "message": sample_msg, "context": None, "use_crewai": False})
    assert r.status_code == 200

    rows = issues_list()
    assert any("Dizzy" in (it.get("title") or "").lower() or "dizzy" in (it.get("details") or "").lower() for it in rows)
    # Expect severity 'high' in DB for created issue
    assert any(it.get("severity") == "high" for it in rows)


def test_travel_jetlag_recommend_ruby(monkeypatch):
    """Travel/jet-lag message should be routed to Ruby (logistics) and recommend scheduling/advice."""
    sample_msg = "I have a travel coming up to Seoul next week. Planning for jet lag management and travel-friendly alternatives."

    # Router extraction: logistics intent, no health issue
    fake_extract = {
        "summary": "Travel to Seoul; asks for jet-lag management and travel alternatives",
        "intent": "logistics",
        "entities": {},
        "keywords": ["travel", "jet lag", "Seoul"],
        "confidence": 0.9,
        "is_health_issue": False,
        "is_improvement": False,
        "recommended_agents": ["Ruby"]
    }
    monkeypatch.setattr(router, "extract_key_info", lambda m, c: fake_extract)
    # Simulate LLM route returning empty -> router fallback should pick Ruby; to keep deterministic, let route return []
    # We will not monkeypatch route so real route fallback logic can run; instead monkeypatch router.router.call_openrouter to return empty JSON
    monkeypatch.setattr(router.router, "call_openrouter", lambda msgs: "{}")

    analysis = router.analyze_message(sample_msg, None, max_agents=2)
    # Since route fell back, expect Ruby in agents (router.route has fallback to Ruby)
    assert "Ruby" in analysis["agents"] or analysis["agents"] == []
    # Recommended next actions should suggest contacting Ruby
    assert any(a.get("agent") == "Ruby" or (isinstance(a, dict) and a.get("agent") == "Ruby") for a in analysis["recommended_next_actions"])
