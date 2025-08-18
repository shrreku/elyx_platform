import os
import warnings
from typing import Dict, Optional, List

from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware

# Suppress TracerProvider warnings
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")
warnings.filterwarnings("ignore", category=UserWarning, module="opentelemetry")
from pydantic import BaseModel

from agents.elyx_agents import AgentOrchestrator, UrgencyDetector, AGENT_ROLES
from agents.llm_router import LLMRouter
from agents.experiment_engine import ExperimentEngine
from data.suggestions import SuggestionsStore
try:
    from agents.crewai_orchestrator import CrewOrchestrator
except Exception:
    CrewOrchestrator = None  # type: ignore[assignment]
from data.persistence import PersistenceManager
from data.db import (
    init_db,
    suggestions_add_many,
    suggestions_list,
    suggestions_update_status,
    issues_add_many,
    issues_list,
    issues_update,
    issues_update_priority_time,
    issues_close_by_text,
    # episodes/decisions/experiments
    episodes_list,
    episodes_add,
    episodes_update_status,
    episode_add_intervention,
    episode_list_interventions,
    decisions_add,
    decisions_list,
    decisions_get_with_why,
    experiments_add,
    experiments_list,
    experiments_add_measurement,
    experiments_results,
    user_profile_get,
    user_profile_set,
)
from agents.issue_extractor import IssueExtractor
from agents.plan_extractor import PlanExtractor
from agents.issue_prioritizer import IssuePrioritizer


# Map OpenRouter -> OpenAI env for CrewAI/litellm compatibility
if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

app = FastAPI(title="Elyx Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent_orchestrator = AgentOrchestrator()
persistence = PersistenceManager()
crewai_orchestrator = CrewOrchestrator() if CrewOrchestrator else None
router = LLMRouter()
experiment_engine = ExperimentEngine()
issue_extractor = IssueExtractor()
plan_extractor = PlanExtractor()
issue_prioritizer = IssuePrioritizer()
init_db()
suggestions = SuggestionsStore()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ChatRequest(BaseModel):
    sender: str
    message: str
    context: Optional[Dict] = None
    use_crewai: bool = True


class SuggestionIn(BaseModel):
    agent: str
    kind: str  # nutrition|training|medical|logistics|other
    title: str
    details: Dict


class ModelSetRequest(BaseModel):
    model: str


class SuggestionIn(BaseModel):
    user_id: str
    agent: str
    title: str
    details: str
    category: str
    conversation_id: Optional[str] = None
    message_index: Optional[int] = None
    message_timestamp: Optional[str] = None
    source: Optional[str] = None
    origin: Optional[str] = None
    source_message: Optional[str] = None
    context_json: Optional[Dict] = None


class EpisodeIn(BaseModel):
    user_id: str
    title: str
    trigger_type: str  # member_initiated|system_alert|scheduled_checkup
    trigger_description: str
    trigger_timestamp: str
    status: str = "open"  # open|in_progress|resolved|escalated
    priority: int = 0
    member_state_before: Optional[str] = None
    member_state_after: Optional[str] = None
    confidence: Optional[float] = None


class EpisodeInterventionIn(BaseModel):
    episode_id: str
    action: str
    responsible_agent: str
    timestamp: str
    outcome: Optional[str] = None


class DecisionEvidenceIn(BaseModel):
    evidence_type: str
    source: str
    data_json: Optional[Dict] = None
    timestamp: Optional[str] = None


class DecisionMessageLinkIn(BaseModel):
    message_id: Optional[str] = None
    message_index: Optional[int] = None
    message_timestamp: Optional[str] = None


class DecisionIn(BaseModel):
    id: Optional[str] = None
    type: str
    content: str
    timestamp: str
    responsible_agent: str
    rationale: Optional[str] = None
    evidence: Optional[List[DecisionEvidenceIn]] = None
    messages: Optional[List[DecisionMessageLinkIn]] = None


class ExperimentIn(BaseModel):
    id: Optional[str] = None
    template: Optional[str] = None
    hypothesis: str
    protocol_json: Dict
    duration: Optional[str] = None
    member_id: str
    status: str = "planned"  # planned|running|completed|cancelled
    outcome: Optional[str] = None
    success: Optional[bool] = None


class ExperimentMeasurementIn(BaseModel):
    experiment_id: str
    name: str
    value: float
    ts: str
    raw_json: Optional[Dict] = None


class SimulationRequest(BaseModel):
    xml_content: str

@app.post("/simulation/run")
def run_simulation(req: SimulationRequest):
    from fastapi import HTTPException

    if not req.xml_content or not req.xml_content.strip():
        # If frontend didn't provide XML, return a clear error
        raise HTTPException(status_code=400, detail="xml_content is required in the request body")

    try:
        # Initialize and run the simulation
        from simulation.complete_journey import CompleteJourney
        journey = CompleteJourney(xml_content=req.xml_content)
        results = journey.run()
        return results
    except Exception as exc:  # noqa: BLE001
        logging.exception("Simulation run failed")
        # Return a helpful error message to the frontend while preserving server logs
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(exc)}")

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")}


@app.get("/debug/config")
def debug_config():
    return {
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL"),
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
    }


@app.get("/history")
def get_history():
    conversation_history = persistence.load_conversation_history()
    
    # Ensure conversation_history is always a list
    if not isinstance(conversation_history, list):
        logging.warning("conversation_history is not a list in /history, converting from %s", type(conversation_history))
        if isinstance(conversation_history, dict):
            # If it's a dict, try to extract messages or create empty list
            if 'messages' in conversation_history:
                conversation_history = conversation_history['messages']
            else:
                conversation_history = []
        else:
            conversation_history = []
    
    return conversation_history


@app.post("/reset")
def reset_history():
    persistence.save_conversation_history([])
    # group_chat.conversation_history = [] # This line was removed as group_chat is not defined
    # also clear sqlite tables for suggestions and issues
    try:
        import sqlite3
        from data.db import DB_PATH
        if os.path.exists(DB_PATH):
            with sqlite3.connect(DB_PATH) as con:
                cur = con.cursor()
                cur.execute("DELETE FROM suggestions")
                cur.execute("DELETE FROM issues")
                con.commit()
    except Exception as exc:  # noqa: BLE001
        logging.warning("db reset failed: %s", exc)
    return {"ok": True}


@app.post("/reset-soft")
def api_reset_soft():
    """Reset all operational data except user profiles."""
    import sqlite3 as _sqlite
    from fastapi import HTTPException
    try:
        con = _sqlite.connect(os.getenv("ELYX_DB_PATH", os.path.join("data", "elyx.db")))
        cur = con.cursor()
        for table in [
            "suggestions",
            "issues",
            "episodes",
            "episode_friction",
            "episode_interventions",
            "decisions",
            "decision_evidence",
            "decision_messages",
            "experiments",
            "experiment_measurements",
        ]:
            cur.execute(f"DELETE FROM {table}")
        con.commit()
        con.close()
        # Also clear conversation history file but keep profiles
        try:
            from data.persistence import PersistenceManager
            pm = PersistenceManager()
            pm.save_conversation_history([])
        except Exception:
            pass
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
def chat(req: ChatRequest):
    model_cfg = os.getenv("OPENROUTER_MODEL")
    logging.info("/chat start use_crewai=%s model=%s msg=%s", req.use_crewai, model_cfg, req.message)
    
    # Load conversation history
    conversation_history = persistence.load_conversation_history()
    logging.info("Loaded conversation_history type: %s, value: %s", type(conversation_history), conversation_history[:2] if isinstance(conversation_history, list) else conversation_history)
    
    # Ensure conversation_history is always a list
    if not isinstance(conversation_history, list):
        logging.warning("conversation_history is not a list, converting from %s", type(conversation_history))
        if isinstance(conversation_history, dict):
            # If it's a dict, try to extract messages or create empty list
            if 'messages' in conversation_history:
                conversation_history = conversation_history['messages']
                logging.info("Extracted messages from dict: %s", conversation_history[:2] if isinstance(conversation_history, list) else conversation_history)
            else:
                conversation_history = []
                logging.info("No messages key found in dict, using empty list")
        else:
            conversation_history = []
            logging.info("conversation_history was not dict or list, using empty list")
    
    # Append user message
    conversation_history.append({
        "sender": req.sender,
        "message": req.message,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "context": req.context,
    })

    # If user reports resolution/relief, auto-close matching open issues
    closed_count = 0
    try:
        if req.sender.lower() == "rohan":
            # Build a compact reference to this message without duplicating content elsewhere.
            ref = f"conv_msg_index:{len(conversation_history)-1} ts:{conversation_history[-1]['timestamp']}"
            closed_count = issues_close_by_text(req.message, reference=ref, triggered_by="user")
            if closed_count:
                logging.info("auto-closed %s issues from user resolution message", closed_count)
    except Exception as exc:  # noqa: BLE001
        logging.warning("auto-close issues failed: %s", exc)

    # Route to appropriate agents
    if req.sender == "Rohan":
        # Use simplified orchestrator for routing
        responding_agents = agent_orchestrator.route_message(req.message, req.context)
        logging.info("orchestrator selected agents=%s", responding_agents)
        if req.use_crewai and crewai_orchestrator is not None:
            for agent in responding_agents:
                try:
                    logging.info("crew_call agent=%s model=%s", agent, getattr(crewai_orchestrator, "model", None))
                    response = crewai_orchestrator.ask(agent, req.message, req.context)
                    conversation_history.append({
                        "sender": agent, 
                        "message": response, 
                        "timestamp": __import__("datetime").datetime.now().isoformat(), 
                        "context": req.context
                    })
                    # Extract plans from agent response
                    try:
                        extracted = plan_extractor.extract(agent, response, req.context)
                        if extracted:
                            payload = []
                            msg_idx = len(conversation_history) - 1
                            msg_ts = conversation_history[-1].get("timestamp")
                            for e in extracted:
                                payload.append(
                                    {
                                        "id": os.urandom(8).hex(),
                                        "user_id": "rohan",
                                        "agent": agent,
                                        "title": e.get("title"),
                                        "details": e.get("details"),
                                        "category": e.get("category"),
                                        "status": "proposed",
                                        "created_at": __import__("datetime").datetime.now().isoformat(),
                                        "conversation_id": "default",
                                        "message_index": msg_idx,
                                        "message_timestamp": msg_ts,
                                        "source": "llm",
                                        "origin": "agent_reply",
                                        "source_message": response,
                                        "context_json": None,
                                    }
                                )
                            suggestions_add_many(payload)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("plan_extractor failed: %s", exc)
                except Exception as exc:  # noqa: BLE001
                    # Fallback to internal BaseAgent for this specific agent
                    try:
                        logging.warning("crew_failed agent=%s err=%s; falling back to direct", agent, exc)
                        fallback_response = agent_orchestrator.agents[agent].respond(req.message, req.context)
                        conversation_history.append({
                            "sender": agent, 
                            "message": fallback_response, 
                            "timestamp": __import__("datetime").datetime.now().isoformat(), 
                            "context": req.context
                        })
                        try:
                            extracted = plan_extractor.extract(agent, fallback_response, req.context)
                            if extracted:
                                payload = []
                                msg_idx = len(conversation_history) - 1
                                msg_ts = conversation_history[-1].get("timestamp")
                                for e in extracted:
                                    payload.append(
                                        {
                                            "id": os.urandom(8).hex(),
                                            "user_id": "rohan",
                                            "agent": agent,
                                            "title": e.get("title"),
                                            "details": e.get("details"),
                                            "category": e.get("category"),
                                            "status": "proposed",
                                            "created_at": __import__("datetime").datetime.now().isoformat(),
                                            "conversation_id": "default",
                                            "message_index": msg_idx,
                                            "message_timestamp": msg_ts,
                                            "source": "llm",
                                            "origin": "agent_reply",
                                            "source_message": fallback_response,
                                            "context_json": None,
                                        }
                                    )
                                suggestions_add_many(payload)
                        except Exception as exc2:  # noqa: BLE001
                            logging.warning("plan_extractor failed (fallback): %s", exc2)
                    except Exception as inner_exc:  # noqa: BLE001
                        conversation_history.append({
                            "sender": agent, 
                            "message": f"Error: {inner_exc}", 
                            "timestamp": __import__("datetime").datetime.now().isoformat(), 
                            "context": req.context
                        })
        else:
            # Use internal agent responses only for routed agents
            for agent_name in responding_agents:
                try:
                    logging.info("direct_call agent=%s model=%s", agent_name, os.getenv("OPENROUTER_MODEL"))
                    response = agent_orchestrator.agents[agent_name].respond(req.message, req.context)
                except Exception as exc:  # noqa: BLE001
                    response = f"Error: {exc}"
                conversation_history.append({
                    "sender": agent_name, 
                    "message": response, 
                    "timestamp": __import__("datetime").datetime.now().isoformat(), 
                    "context": req.context
                })
                # Extract plans from direct path
                try:
                    extracted = plan_extractor.extract(agent_name, response, req.context)
                    if extracted:
                        payload = []
                        msg_idx = len(conversation_history) - 1
                        msg_ts = conversation_history[-1].get("timestamp")
                        for e in extracted:
                            payload.append(
                                {
                                    "id": os.urandom(8).hex(),
                                    "user_id": "rohan",
                                    "agent": agent_name,
                                    "title": e.get("title"),
                                    "details": e.get("details"),
                                    "category": e.get("category"),
                                    "status": "proposed",
                                    "created_at": __import__("datetime").datetime.now().isoformat(),
                                    "conversation_id": "default",
                                    "message_index": msg_idx,
                                    "message_timestamp": msg_ts,
                                    "source": "llm",
                                    "origin": "agent_reply",
                                    "source_message": response,
                                    "context_json": None,
                                }
                            )
                        suggestions_add_many(payload)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("plan_extractor failed (direct): %s", exc)

    # Extract issues from user's message and persist with reference
    issues = []
    # Skip extraction entirely if we just closed issues from a resolution/improvement message
    if closed_count == 0:
        try:
            issues = issue_extractor.extract(req.message, req.context)
        except Exception as exc:  # noqa: BLE001
            logging.warning("issue_extractor failed: %s", exc)
            issues = []

        # Fallback heuristic if extractor returns nothing (when running without LLM key)
        if not issues and req.message:
            msg_lower = req.message.lower()
            
            # Skip issue extraction if message contains improvement/resolution markers
            improvement_markers = [
                "feels fine", "feel fine", "feels alright", "feel alright", "feels okay", "feel okay",
                "feels good", "feel good", "feels better", "feel better", "no more", "no longer",
                "resolved", "better now", "getting better", "improved", "improving", "okay now",
                "alright now", "good now", "back to normal", "gone", "pain reduced", "less pain",
                "subsided", "cleared up", "all good", "much better"
            ]
            
            if not any(marker in msg_lower for marker in improvement_markers):
                category = "other"
                if any(k in msg_lower for k in ["sleep", "hrv", "recovery", "whoop", "oura", "tired", "fatigue"]):
                    category = "performance"
                if any(k in msg_lower for k in ["glucose", "cgm", "sugar", "insulin", "bp", "blood pressure"]):
                    category = "medical"
                if any(k in msg_lower for k in ["food", "diet", "meal", "protein", "carb", "nutrition"]):
                    category = "nutrition"
                if any(k in msg_lower for k in ["pain", "injury", "mobility", "shoulder", "back", "knee"]):
                    category = "physio"
                severity = "medium"
                if any(k in msg_lower for k in ["severe", "cannot", "can't", "emergency", "chest", "bleeding"]):
                    severity = "high"
                issues = [
                    {
                        "title": (req.message[:80] + ("…" if len(req.message) > 80 else "")),
                        "details": req.message[:500],
                        "category": category,
                        "severity": severity,
                    }
                ]

    if issues:
        payload = []
        msg_idx = len(conversation_history) - 1
        msg_ts = conversation_history[-1].get("timestamp") if conversation_history else None
        now_iso = __import__("datetime").datetime.now().isoformat()
        for it in issues:
            # prioritize
            try:
                triage = issue_prioritizer.prioritize(it.get("title", ""), it.get("details", ""), req.context)
            except Exception:
                triage = {"priority": "medium", "time_window": "24-72h"}
            payload.append(
                {
                    "id": os.urandom(8).hex(),
                    "user_id": "rohan",
                    "title": it.get("title"),
                    "details": it.get("details"),
                    "category": it.get("category"),
                    "severity": it.get("severity") or "medium",
                    "status": "open",
                    "progress_percent": 0,
                    "last_reviewed_at": now_iso,
                    "priority": triage.get("priority"),
                    "time_window": triage.get("time_window"),
                    "conversation_id": "default",
                    "message_index": msg_idx,
                    "message_timestamp": msg_ts,
                    "created_at": now_iso,
                }
            )
        try:
            issues_add_many(payload)
        except Exception as exc:  # noqa: BLE001
            logging.warning("issues_add_many failed: %s", exc)

    persistence.save_conversation_history(conversation_history)
    return conversation_history[-10:]


@app.get("/suggestions")
def get_suggestions():
    return suggestions_list()


class StatusUpdate(BaseModel):
    status: str

@app.post("/suggestions/{suggestion_id}/status")
def update_suggestion_status(suggestion_id: str, update: StatusUpdate):
    """Update suggestion status (accept/reject/dismiss/in_progress/completed)"""
    from fastapi import HTTPException
    # Map frontend status to backend status
    status_map = {
        "accept": "accepted",
        "reject": "dismissed",
        "dismiss": "dismissed",
        "accepted": "accepted",
        "dismissed": "dismissed",
        "in_progress": "in_progress",
        "completed": "completed",
    }
    new_status = status_map.get(update.status, update.status)
    try:
        ok = suggestions_update_status(suggestion_id, new_status)
        if not ok:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return {"ok": True, "status": new_status}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logging.error(f"Error updating suggestion status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update suggestion status")


@app.post("/suggestions")
def add_suggestions(items: List[SuggestionIn]):
    # Persist suggestions generated by LLMs for surfacing on dashboard (DB-backed)
    payload = [
        {
            "id": os.urandom(8).hex(),
            "user_id": it.user_id,
            "agent": it.agent,
            "title": it.title,
            "details": it.details,
            "category": it.category,
            "status": "proposed",
            "created_at": __import__("datetime").datetime.now().isoformat(),
            "conversation_id": it.conversation_id,
            "message_index": it.message_index,
            "message_timestamp": it.message_timestamp,
            "source": it.source or "llm",
            "origin": it.origin or "router/agent",
            "source_message": it.source_message,
            "context_json": (None if it.context_json is None else __import__("json").dumps(it.context_json)),
        }
        for it in items
    ]
    suggestions_add_many(payload)
    return {"ok": True, "count": len(payload)}


@app.post("/suggestions/{item_id}/status/{status}")
def set_suggestion_status(item_id: str, status: str):
    ok = suggestions_update_status(item_id, status)
    return {"ok": ok}


@app.get("/issues")
def get_issues():
    return issues_list()


class IssuePriorityIn(BaseModel):
    priority: str
    time_window: str


@app.post("/issues/{issue_id}/priority")
def api_issue_set_priority(issue_id: str, req: IssuePriorityIn):
    from fastapi import HTTPException
    ok = issues_update_priority_time(issue_id, req.priority, req.time_window)
    if not ok:
        raise HTTPException(status_code=404, detail="Issue not found")
    return {"ok": True}


class IssueUpdateIn(BaseModel):
    title: Optional[str] = None
    details: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    progress_percent: Optional[int] = None
    priority: Optional[str] = None
    time_window: Optional[str] = None


@app.post("/issues/{issue_id}/update")
def api_issue_update(issue_id: str, req: IssueUpdateIn):
    from fastapi import HTTPException
    ok = issues_update(issue_id, {k: v for k, v in req.dict().items() if v is not None})
    if not ok:
        raise HTTPException(status_code=404, detail="Issue not found or nothing to update")
    return {"ok": True}


# User profiles
class UserProfileIn(BaseModel):
    user_id: str
    profile: Dict


@app.get("/profiles/{user_id}")
def api_get_profile(user_id: str):
    return user_profile_get(user_id)


@app.post("/profiles")
def api_set_profile(req: UserProfileIn):
    ok = user_profile_set(req.user_id, req.profile)
    return {"ok": ok}


@app.post("/issues/retriage")
def api_issues_retriage_all():
    """Re-run prioritization/time-window tagging for all issues."""
    updated = 0
    for it in issues_list():
        triage = {"priority": it.get("priority"), "time_window": it.get("time_window")}
        try:
            triage = issue_prioritizer.prioritize(it.get("title", ""), it.get("details", ""), None)
        except Exception:
            pass
        if triage.get("priority") and triage.get("time_window"):
            if issues_update_priority_time(it["id"], triage["priority"], triage["time_window"]):
                updated += 1
    return {"ok": True, "updated": updated}

# NOTE: removed file-backed suggestions endpoints to avoid duplicates


@app.get("/suggestions")
def list_suggestions():
    return suggestions_list()


@app.post("/suggestions/{item_id}/status")
def update_suggestion_status_legacy(item_id: str, status: str):
    ok = suggestions_update_status(item_id, status)
    return {"ok": ok}


@app.post("/model")
def set_model(req: ModelSetRequest):
    # Enforce allowed provider via env if provided
    provider = os.getenv("OPENROUTER_PROVIDER", "")
    os.environ["OPENROUTER_MODEL"] = req.model
    os.environ["OPENAI_MODEL_NAME"] = req.model
    if crewai_orchestrator is not None:
        crewai_orchestrator.model = req.model
    return {"ok": True, "model": req.model, "provider": provider or None}


# Episodes API
@app.get("/episodes")
def api_episodes_list():
    return episodes_list()


@app.get("/messages")
async def get_messages():
    """Get all messages from conversation history"""
    try:
        # Get messages from the chat system if available
        if crewai_orchestrator and hasattr(crewai_orchestrator, 'chat_system'):
            history = crewai_orchestrator.chat_system.get_conversation_history()
            return {"messages": history, "total": len(history)}
        else:
            # Fallback to episodes if chat system not available
            episodes = episodes_list()
            messages = []
            for episode in episodes:
                if 'messages' in episode:
                    messages.extend(episode['messages'])
            return {"messages": messages, "total": len(messages)}
    except Exception as e:
        logging.error(f"Error getting messages: {e}")
        return {"messages": [], "total": 0, "error": str(e)}


@app.post("/episodes")
def api_episodes_create(ep: EpisodeIn):
    payload = {
        "id": os.urandom(8).hex(),
        "user_id": ep.user_id,
        "title": ep.title,
        "trigger_type": ep.trigger_type,
        "trigger_description": ep.trigger_description,
        "trigger_timestamp": ep.trigger_timestamp,
        "status": ep.status,
        "priority": ep.priority,
        "member_state_before": ep.member_state_before,
        "member_state_after": ep.member_state_after,
        "confidence": ep.confidence,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    episodes_add(payload)
    return {"ok": True, "id": payload["id"]}


@app.post("/episodes/{episode_id}/status/{status}")
def api_episodes_set_status(episode_id: str, status: str):
    ok = episodes_update_status(episode_id, status)
    return {"ok": ok}


@app.post("/episodes/{episode_id}/interventions")
def api_episodes_add_intervention(episode_id: str, it: EpisodeInterventionIn):
    payload = {
        "id": os.urandom(8).hex(),
        "episode_id": episode_id,
        "action": it.action,
        "responsible_agent": it.responsible_agent,
        "timestamp": it.timestamp,
        "outcome": it.outcome,
    }
    episode_add_intervention(payload)
    return {"ok": True, "id": payload["id"]}


@app.get("/episodes/{episode_id}/interventions")
def api_episodes_list_interventions(episode_id: str):
    return episode_list_interventions(episode_id)


# Decisions API
@app.get("/decisions")
def api_decisions_list():
    return decisions_list()


@app.post("/decisions")
def api_decisions_create(dec: DecisionIn):
    decision_id = dec.id or os.urandom(8).hex()
    dec_payload = {
        "id": decision_id,
        "type": dec.type,
        "content": dec.content,
        "timestamp": dec.timestamp,
        "responsible_agent": dec.responsible_agent,
        "rationale": dec.rationale,
    }
    evidence_payload = [
        {
            "id": os.urandom(8).hex(),
            "decision_id": decision_id,
            "evidence_type": ev.evidence_type,
            "source": ev.source,
            "data_json": (None if ev.data_json is None else __import__("json").dumps(ev.data_json)),
            "timestamp": ev.timestamp,
        }
        for ev in (dec.evidence or [])
    ]
    messages_payload = [
        {
            "id": os.urandom(8).hex(),
            "decision_id": decision_id,
            "message_id": m.message_id,
            "message_index": m.message_index,
            "message_timestamp": m.message_timestamp,
        }
        for m in (dec.messages or [])
    ]
    decisions_add(dec_payload, evidence_payload, messages_payload)
    return {"ok": True, "id": decision_id}


@app.get("/decisions/{decision_id}/why")
def api_decisions_why(decision_id: str):
    return decisions_get_with_why(decision_id)


# Experiments API
@app.get("/experiments")
def api_experiments_list():
    return experiments_list()


@app.post("/experiments")
def api_experiments_create(exp: ExperimentIn):
    exp_id = exp.id or os.urandom(8).hex()
    payload = {
        "id": exp_id,
        "template": exp.template,
        "hypothesis": exp.hypothesis,
        "protocol_json": __import__("json").dumps(exp.protocol_json),
        "duration": exp.duration,
        "member_id": exp.member_id,
        "status": exp.status,
        "outcome": exp.outcome,
        "success": 1 if exp.success else 0 if exp.success is not None else None,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    experiments_add(payload)
    return {"ok": True, "id": exp_id}


@app.post("/experiments/{experiment_id}/measurements")
def api_experiments_add_measurement(experiment_id: str, m: ExperimentMeasurementIn):
    payload = {
        "id": os.urandom(8).hex(),
        "experiment_id": experiment_id,
        "name": m.name,
        "value": m.value,
        "ts": m.ts,
        "raw_json": (None if m.raw_json is None else __import__("json").dumps(m.raw_json)),
    }
    experiments_add_measurement(payload)
    return {"ok": True, "id": payload["id"]}


@app.get("/experiments/results")
def api_experiments_results():
    return experiments_results()


# Enhanced routing and SLA tracking endpoints
@app.get("/agents/performance")
def api_agents_performance():
    """Get performance metrics for all agents"""
    performance = {}
    for agent_name in AGENT_ROLES.keys():
        performance[agent_name] = agent_orchestrator.get_agent_performance(agent_name)
    return performance


@app.get("/agents/sla-violations")
def api_sla_violations():
    """Get current SLA violations"""
    # For now return empty - would need to implement SLA violation tracking
    return []


@app.post("/experiments/propose")
def api_experiments_propose(issue: str, context: Optional[Dict] = None):
    """Propose experiment based on member issue"""
    proposal = experiment_engine.propose_experiment(issue, context)
    return proposal


@app.post("/experiments/{experiment_id}/start")
def api_experiments_start(experiment_id: str):
    """Start a planned experiment"""
    success = experiment_engine.start_experiment(experiment_id)
    return {"ok": success}


@app.get("/experiments/active")
def api_experiments_active():
    """Get active experiments"""
    return experiment_engine.get_active_experiments()


@app.get("/experiments/successful")
def api_experiments_successful():
    """Get successful experiment results"""
    return experiment_engine.get_experiment_results()


@app.post("/demo/generate-mock")
def api_generate_mock_data():
    """Generate mock data for demo purposes"""
    import uuid
    from datetime import datetime, timedelta
    import random
    
    try:
        # Mock episodes
        mock_episodes = [
            {
                "title": "Sleep Quality Decline",
                "trigger_type": "health_alert",
                "trigger_description": "HRV dropped below threshold for 3 consecutive nights",
                "priority": 3,
                "member_state_before": "Feeling tired, stressed from work travel"
            },
            {
                "title": "Nutrition Plan Adjustment",
                "trigger_type": "user_message", 
                "trigger_description": "Member reported digestive issues with current meal plan",
                "priority": 2,
                "member_state_before": "Experiencing bloating after meals"
            },
            {
                "title": "Exercise Recovery Check",
                "trigger_type": "scheduled_check",
                "trigger_description": "Weekly review of workout performance and recovery metrics",
                "priority": 1,
                "member_state_before": "Consistent training, good energy levels"
            }
        ]
        
        # Mock experiments
        mock_experiments = [
            {
                "hypothesis": "Blue light blocking glasses after 6pm will improve deep sleep quality by 20%",
                "template": "SLEEP_OPTIMIZATION",
                "member_id": "rohan"
            },
            {
                "hypothesis": "Magnesium glycinate 400mg before bed will reduce sleep latency",
                "template": "SUPPLEMENT_TRIAL", 
                "member_id": "rohan"
            },
            {
                "hypothesis": "Eating protein 15 minutes before carbs will reduce glucose spike",
                "template": "CGM_MEAL_TEST",
                "member_id": "rohan"
            }
        ]
        
        # Mock messages
        mock_messages = [
            {
                "sender": "Rohan",
                "message": "I've been feeling tired lately despite getting 7-8 hours of sleep. My Whoop shows low recovery scores.",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "sender": "Advik", 
                "message": "Looking at your data, I see your HRV has dropped 15% over the past week. This correlates with your travel schedule. Let's adjust your sleep protocol - try blue light blocking glasses after 6pm and keep room temperature at 65°F.",
                "timestamp": (datetime.now() - timedelta(hours=1, minutes=58)).isoformat()
            },
            {
                "sender": "Ruby",
                "message": "I've ordered the blue light glasses and they'll arrive tomorrow. I've also adjusted your hotel preferences for optimal sleep temperature. Would you like me to schedule a follow-up with Advik in one week?",
                "timestamp": (datetime.now() - timedelta(hours=1, minutes=55)).isoformat()
            }
        ]
        
        # Add episodes
        for episode_data in mock_episodes:
            episodes_add(
                title=episode_data["title"],
                trigger_type=episode_data["trigger_type"],
                trigger_description=episode_data["trigger_description"],
                priority=episode_data["priority"],
                member_state_before=episode_data["member_state_before"]
            )
        
        # Add experiments  
        for exp_data in mock_experiments:
            experiments_add(
                hypothesis=exp_data["hypothesis"],
                template=exp_data["template"],
                member_id=exp_data["member_id"]
            )
            
        # Add messages to conversation history
        conversation_history = persistence.load_conversation_history()
        
        # Ensure conversation_history is always a list
        if not isinstance(conversation_history, list):
            logging.warning("conversation_history is not a list in mock data generation, converting from %s", type(conversation_history))
            if isinstance(conversation_history, dict):
                # If it's a dict, try to extract messages or create empty list
                if 'messages' in conversation_history:
                    conversation_history = conversation_history['messages']
                else:
                    conversation_history = []
            else:
                conversation_history = []
                
        for msg in mock_messages:
            conversation_history.append({
                "sender": msg["sender"],
                "message": msg["message"], 
                "timestamp": msg["timestamp"],
                "context": None
            })
        persistence.save_conversation_history(conversation_history)
        
        # Add some mock suggestions
        mock_suggestions = [
            {
                "id": str(uuid.uuid4()),
                "user_id": "rohan",
                "agent": "Advik",
                "title": "Optimize sleep environment",
                "details": "Set bedroom temperature to 65°F and use blackout curtains",
                "category": "performance",
                "status": "accepted"
            },
            {
                "id": str(uuid.uuid4()),
                "user_id": "rohan", 
                "agent": "Carla",
                "title": "Pre-meal protein timing",
                "details": "Eat 20g protein 15 minutes before main carbohydrate portion",
                "category": "nutrition",
                "status": "in_progress"
            },
            {
                "id": str(uuid.uuid4()),
                "user_id": "rohan",
                "agent": "Ruby",
                "title": "Schedule sleep study",
                "details": "Book comprehensive sleep analysis at preferred clinic",
                "category": "logistics", 
                "status": "completed"
            }
        ]
        
        suggestions_add_many(mock_suggestions)
        
        # Add mock issues
        mock_issues = [
            {
                "id": str(uuid.uuid4()),
                "user_id": "rohan",
                "title": "Inconsistent sleep schedule during travel",
                "details": "Difficulty maintaining sleep routine across time zones",
                "category": "performance",
                "severity": "medium"
            },
            {
                "id": str(uuid.uuid4()),
                "user_id": "rohan",
                "title": "Post-meal glucose spikes",
                "details": "CGM showing spikes >140 mg/dL after lunch meetings",
                "category": "medical",
                "severity": "low"
            }
        ]
        
        issues_add_many(mock_issues)
        
        return {"success": True, "message": "Mock data generated successfully"}
        
    except Exception as e:
        logging.error(f"Error generating mock data: {e}")
        return {"success": False, "error": str(e)}
