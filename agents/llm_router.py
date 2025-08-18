from __future__ import annotations

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from .base_agent import BaseAgent
from .elyx_agents import (
    RubyAgent,
    DrWarrenAgent,
    AdvikAgent,
    CarlaAgent,
    RachelAgent,
    NeelAgent,
)


logger = logging.getLogger(__name__)


class LLMRouter:
    """LLM-based router and lightweight orchestrator.

    Responsibilities added:
    - Extract concise key information from the user's text (summary, intent, entities, keywords).
    - Decide which agents should be involved (keeps existing `route` behavior for compatibility).
    - Build per-agent prompts that include the extracted key info and explicit instructions:
        * Agents should only reply within their domain of expertise.
        * If an agent receives a request outside its domain it should reply with a standard token so
          the router can forward to the appropriate agent instead of returning out-of-domain content.
    - Provide an `orchestrate` method that returns tailored messages and execution strategies for each selected agent.

    Backwards compatibility:
    - `route(message, context, max_agents)` still returns a list of agent names (unchanged).
    - New APIs: `extract_key_info`, `build_agent_messages`, `orchestrate`, `analyze_message`.
    """

    def __init__(self):
        # Build registry with agent role/system prompts for routing/context
        self.agent_descriptions: Dict[str, str] = {
            "Ruby": RubyAgent().system_prompt,
            "Dr. Warren": DrWarrenAgent().system_prompt,
            "Advik": AdvikAgent().system_prompt,
            "Carla": CarlaAgent().system_prompt,
            "Rachel": RachelAgent().system_prompt,
            "Neel": NeelAgent().system_prompt,
        }

        # Concrete agent instances for execution (used by execute_strategy)
        self.agent_instances: Dict[str, BaseAgent] = {
            "Ruby": RubyAgent(),
            "Dr. Warren": DrWarrenAgent(),
            "Advik": AdvikAgent(),
            "Carla": CarlaAgent(),
            "Rachel": RachelAgent(),
            "Neel": NeelAgent(),
        }

        # Router agent (used for extraction/routing decisions)
        self.router = BaseAgent(
            name="Router",
            role="Orchestrator",
            system_prompt=(
                """
You are an expert router/orchestrator for a multi-agent healthcare team.
Your job is twofold: (1) extract concise structured information from the user's message
(summary, top intent, important entities/measurements), and (2) decide which agent(s)
should handle the user's request.

Guidelines & rules:
- Prefer the single most relevant specialist for domain questions (medical/physio/nutrition/performance).
- If the message clearly indicates physiotherapy / movement / pain (keywords: "pain", "injury", "back", "knees", "mobility", "rehab"), RECOMMEND the physiotherapist ("Rachel") as the primary agent and DO NOT recommend the nutritionist ("Carla") for that message.
- If the message clearly indicates nutrition/metabolic questions (keywords: "cgm", "glucose", "meal", "diet", "food", "supplement"), RECOMMEND "Carla".
- Avoid recommending agents outside the domain signaled by the text. If uncertain, return a conservative list (or Ruby for coordination).
- Return structured JSON only.

Return JSON only for extraction tasks, for example:
{"summary": "...", "intent": "nutrition|medical|logistics|performance|physio|other", "entities": {"glucose": "150", "sleep_hours": "6"}, "keywords": ["cgm", "spike"], "confidence": 0.8}
"""
            ),
        )

    def _build_route_prompt(self, message: str, context: Optional[Dict] = None) -> List[Dict[str, str]]:
        descriptions = "\n".join([f"- {k}: {v.splitlines()[0][:200]}" for k, v in self.agent_descriptions.items()])
        route_instructions = (
            f"Agents (brief descriptions):\n{descriptions}\n\n"
            f"User Message: {message}\n"
            f"Context: {json.dumps(context) if context else '{}'}\n\n"
            "Return STRICT JSON only in the shape: {\"agents\": [\"Name\", ...]} where names match the team list."
        )
        return [
            {"role": "system", "content": self.router.system_prompt},
            {"role": "user", "content": route_instructions},
        ]

    def _extract_json(self, text: str) -> Dict:
        # Try direct JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to find first JSON object in text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                logger.debug("Failed to parse JSON from text (len=%d)", len(text))
                return {}
        return {}

    _cache: Dict[Tuple[str, str], List[str]] = {}
    _extract_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def extract_key_info(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Use the router LLM to extract summary/intent/entities and potential health-issue or improvement info.

        Extended result includes:
        - summary, intent, entities, keywords, confidence
        - is_health_issue: bool
        - health_issue: {title, details, category, severity} or empty dict
        - is_improvement: bool
        - improvement: {title, details, related_issue_title?} or empty dict
        - recommended_agents: optional list (LLM suggestion of agents, maximum 2-3 agents)
        """
        cache_key = (message.strip(), json.dumps(context or {}, sort_keys=True))
        if cache_key in self._extract_cache:
            logger.debug("extract_key_info: cache hit")
            return self._extract_cache[cache_key]

        prompt = [
            {"role": "system", "content": self.router.system_prompt},
            {
                "role": "user",
                "content": (
                    "Extract the following from the user message in strict JSON:\n"
                    '{"summary":"...","intent":"...","entities":{...},"keywords":[...],"confidence":0-1,'
                    '"is_health_issue": true|false,'
                    '"health_issue": {"title":"...","details":"...","category":"medical|physio|nutrition|performance|other","severity":"low|medium|high"},'
                    '"is_improvement": true|false,'
                    '"improvement": {"title":"...","details":"...","related_issue_title":"optional"},'
                    '"recommended_agents":["Ruby","Carla", ...]'
                    '}\n\n'
                    f"User Message: {message}\nContext: {json.dumps(context) if context else '{}'}\n\n"
                    "If you cannot identify entities/health_issue/improvement, return empty objects/false accordingly. Return JSON only."
                ),
            },
        ]
        logger.debug("extract_key_info: calling router LLM for message '%s' (len=%d)", (message or "")[:120], len(message or ""))
        raw = self.router.call_openrouter(prompt)
        logger.debug("extract_key_info: raw response length=%d", len(raw) if isinstance(raw, str) else 0)
        data = self._extract_json(raw)
        if not isinstance(data, dict):
            data = {}

        # Normalize fields with safe fallbacks
        summary = data.get("summary", "") if isinstance(data.get("summary", ""), str) else ""
        intent = data.get("intent", "other")
        entities = data.get("entities", {}) if isinstance(data.get("entities", {}), dict) else {}
        keywords = data.get("keywords", []) if isinstance(data.get("keywords", []), list) else []
        confidence = float(data.get("confidence", 0) or 0)
        is_health_issue = bool(data.get("is_health_issue")) if "is_health_issue" in data else False
        health_issue = data.get("health_issue", {}) if isinstance(data.get("health_issue", {}), dict) else {}
        is_improvement = bool(data.get("is_improvement")) if "is_improvement" in data else False
        improvement = data.get("improvement", {}) if isinstance(data.get("improvement", {}), dict) else {}
        recommended_agents = data.get("recommended_agents", []) if isinstance(data.get("recommended_agents", []), list) else []

        result = {
            "summary": summary,
            "intent": intent,
            "entities": entities,
            "keywords": keywords,
            "confidence": confidence,
            "is_health_issue": is_health_issue,
            "health_issue": health_issue,
            "is_improvement": is_improvement,
            "improvement": improvement,
            "recommended_agents": recommended_agents,
        }

        # Enforce mutual exclusivity: a message cannot be both an 'issue' and an 'improvement'.
        # Prefer marking as an issue (is_health_issue) in case of conflict, since issues require triage.
        if result.get("is_health_issue") and result.get("is_improvement"):
            logger.warning(
                "extract_key_info: both is_health_issue and is_improvement are true — resolving conflict by preferring is_health_issue"
            )
            result["is_improvement"] = False

        logger.debug("extract_key_info: result=%s", {k: v for k, v in result.items() if k not in ("entities","health_issue","improvement")})

        # Cache small LRU
        if len(self._extract_cache) > 128:
            self._extract_cache.pop(next(iter(self._extract_cache)))
        self._extract_cache[cache_key] = result
        return result

    def analyze_message(self, message: str, context: Optional[Dict] = None, max_agents: int = 2) -> Dict[str, Any]:
        """Higher-level analysis combining key info, routing, message-type detection, and agent-role details.

        Enhanced return schema (recommended):
        {
          "agents": ["Name", ...],                       # selected agents in priority order
          "agents_roles": {agent: {...}},                # metadata for each agent
          "selected_agents_with_reasons": {agent: "why"},# why each agent was selected
          "message_type": "user_text" | "wearable_xml" | "report_xml" | "wearable_parsed" | "report_parsed" | "unknown",
          "xml_root": "root_tag" | None,
          "message_origin": "user" | "device" | "system" | "agent",
          "extraction": { ... },                         # output of extract_key_info(...)
          "is_health_issue": bool,
          "health_issue": {standardized issue schema or {}},
          "recommended_next_actions": [ { "type": "contact_agent"|"schedule_appointment"|"self_care", "agent": "Name"|None, "details": "..." }, ...]
        }
        """
        key_info = self.extract_key_info(message, context)
        logger.debug("analyze_message: key_info.intent=%s keywords=%s", key_info.get("intent"), key_info.get("keywords"))

        # Decide agents (legacy + heuristic promotion)
        selected = self.route(message, context, max_agents=max_agents)
        logger.debug("analyze_message: initial selected agents=%s", selected)

        # Build agent role metadata
        agents_roles: Dict[str, Dict[str, Any]] = {}
        try:
            from .elyx_agents import AGENT_ROLES as _AGENTS_META
            for a in selected:
                meta = _AGENTS_META.get(a)
                if meta:
                    agents_roles[a] = {
                        "responsibilities": getattr(meta, "responsibilities", []),
                        "response_style": getattr(meta, "response_style", ""),
                        "sla_target_hours": getattr(meta, "sla_target_hours", None),
                        "escalation_threshold_hours": getattr(meta, "escalation_threshold_hours", None),
                    }
                else:
                    agents_roles[a] = {"responsibilities": [], "response_style": "", "sla_target_hours": None, "escalation_threshold_hours": None}
        except Exception as exc:
            logger.debug("analyze_message: error loading AGENT_ROLES: %s", exc)
            for a in selected:
                agents_roles[a] = {"responsibilities": [], "response_style": "", "sla_target_hours": None, "escalation_threshold_hours": None}

        # Give reasons for selection where possible (LLM recommended_agents or heuristics)
        selected_agents_with_reasons: Dict[str, str] = {}
        recs = key_info.get("recommended_agents") or []
        for a in selected:
            if a in recs:
                selected_agents_with_reasons[a] = "recommended_by_extractor"
            else:
                # fallback: map based on detected intent/entities
                intent = (key_info.get("intent") or "").lower()
                if (intent == "physio" and a == "Rachel") or (intent == "medical" and a == "Dr. Warren") or (intent == "nutrition" and a == "Carla") or (intent == "performance" and a == "Advik"):
                    selected_agents_with_reasons[a] = "intent_match"
                else:
                    # responsibility keyword match
                    roles = agents_roles.get(a, {}).get("responsibilities", [])
                    if any(k.lower() in " ".join(roles).lower() for k in key_info.get("keywords", []) or []):
                        selected_agents_with_reasons[a] = "keyword_role_match"
                    else:
                        selected_agents_with_reasons[a] = "fallback"

        logger.debug("analyze_message: selected_agents_with_reasons=%s", selected_agents_with_reasons)

        # Detect XML / message type heuristics
        msg_strip = (message or "").strip()
        message_type = "user_text"
        xml_root = None
        if msg_strip.startswith("<"):
            m = re.match(r"<\s*([a-zA-Z0-9_\-:]+)", msg_strip)
            if m:
                xml_root = m.group(1).lower()
                inner = msg_strip.lower()
                if "wear" in xml_root or "sensor" in xml_root or "wearable" in xml_root or any(k in inner for k in ["hrv", "whoop", "sleep"]):
                    message_type = "wearable_xml"
                elif "report" in xml_root or "analysis" in xml_root or any(k in inner for k in ["lab", "result", "glucose", "blood"]):
                    message_type = "report_xml"
                else:
                    message_type = "unknown"
        else:
            # Not raw XML — check parsed entities for signals
            ent_keys = " ".join(key.lower() for key in (key_info.get("entities") or {}).keys())
            if any(k in ent_keys for k in ["hrv", "whoop", "oura", "sleep", "steps"]):
                message_type = "wearable_parsed"
            if any(k in ent_keys for k in ["lab", "glucose", "cgm", "result", "test"]):
                message_type = "report_parsed"

        # message_origin detection (simple heuristic)
        if msg_strip.startswith("<"):
            message_origin = "device"
        else:
            # context may indicate system/agent origin otherwise assume user
            if context and isinstance(context, dict) and context.get("source") in ("system", "agent"):
                message_origin = context.get("source")
            else:
                message_origin = "user"

        # Standardize health issue schema if present
        is_health_issue = bool(key_info.get("is_health_issue", False))
        raw_issue = key_info.get("health_issue") or {}
        health_issue: Dict[str, Any] = {}
        if is_health_issue and isinstance(raw_issue, dict):
            # Normalize fields
            title = raw_issue.get("title") or raw_issue.get("summary") or ""
            details = raw_issue.get("details") or raw_issue.get("description") or ""
            category = raw_issue.get("category") or "other"
            severity = raw_issue.get("severity") or "medium"
            detected_entities = key_info.get("entities") or {}
            suggested_owner = None
            # Map category to owner
            if category == "physio":
                suggested_owner = "Rachel"
            elif category == "medical":
                suggested_owner = "Dr. Warren"
            elif category == "nutrition":
                suggested_owner = "Carla"
            elif category == "performance":
                suggested_owner = "Advik"
            else:
                suggested_owner = selected[0] if selected else "Ruby"

            health_issue = {
                "id": None,  # not persisted yet; caller may create an id
                "title": title,
                "details": details,
                "category": category,
                "severity": severity,
                "detected_entities": detected_entities,
                "confidence": float(key_info.get("confidence", 0) or 0),
                "suggested_owner": suggested_owner,
                "recommended_followups": [
                    {"type": "contact_agent", "agent": suggested_owner, "details": f"Review issue and advise next steps ({category})"},
                    {"type": "track_issue", "agent": "system", "details": "Create issue entry in DB and monitor progress"}
                ],
            }

        # Recommended next actions (lightweight heuristics + extractor hints)
        recommended_next_actions: List[Dict[str, Any]] = []
        if is_health_issue:
            recommended_next_actions.append({"type": "create_issue", "details": "Persist issue in database", "priority": "high" if health_issue.get("severity") == "high" else "medium"})
            recommended_next_actions.append({"type": "contact_agent", "agent": health_issue.get("suggested_owner"), "details": "Assign and triage"})
        else:
            # Non-issue recommendations based on intent
            intent = (key_info.get("intent") or "").lower()
            if intent == "logistics":
                recommended_next_actions.append({"type": "contact_agent", "agent": "Ruby", "details": "Assist with scheduling/coordination"})
            elif intent == "nutrition":
                recommended_next_actions.append({"type": "contact_agent", "agent": "Carla", "details": "Provide nutrition guidance"})
            elif intent == "performance":
                recommended_next_actions.append({"type": "contact_agent", "agent": "Advik", "details": "Analyze wearable data"})
            elif intent == "physio":
                recommended_next_actions.append({"type": "contact_agent", "agent": "Rachel", "details": "Provide movement/rehab guidance"})
            else:
                # fallback: suggest Ruby for coordination
                recommended_next_actions.append({"type": "contact_agent", "agent": "Ruby", "details": "Clarify the request and route to specialist"})

        logger.debug("analyze_message: message_type=%s message_origin=%s is_health_issue=%s", message_type, message_origin, is_health_issue)

        return {
            "agents": selected,
            "agents_roles": agents_roles,
            "selected_agents_with_reasons": selected_agents_with_reasons,
            "message_type": message_type,
            "xml_root": xml_root,
            "message_origin": message_origin,
            "extraction": key_info,
            "is_health_issue": is_health_issue,
            "health_issue": health_issue,
            "recommended_next_actions": recommended_next_actions,
        }

    def route(self, message: str, context: Optional[Dict] = None, max_agents: int = 2) -> List[str]:
        """Deprecated compatibility wrapper.

        Agent selection is now performed by extract_key_info (recommended_agents) and by orchestrate.
        This function simply returns canonical agent names based on extractor recommendations.
        Falls back to 'Ruby' when no recommendation is present.
        """
        key_info = self.extract_key_info(message, context)
        raw_recs = key_info.get("recommended_agents") or []
        name_map = {k.lower(): k for k in self.agent_descriptions.keys()}
        selected: List[str] = []
        for r in raw_recs:
            if isinstance(r, str):
                canon = name_map.get(r.lower())
                if canon and canon not in selected:
                    selected.append(canon)
        if not selected:
            selected = ["Ruby"]
        return selected[:max_agents]

    def build_agent_messages(self, agent_name: str, key_info: Dict[str, Any], original_message: str, context: Optional[Dict] = None, primary_agent: Optional[str] = None) -> List[Dict[str, str]]:
        """Construct messages to send to the specified agent.

        Adds:
        - The agent's system prompt (from agent_descriptions).
        - A short instruction forcing the agent to stay in-domain and a standard token if out-of-scope.
        - The extracted key info and the original user message.

        Special behavior:
        - If this agent is 'Ruby' and the request intent is not logistics (or heuristic indicates a specialist),
          Ruby must *begin* its reply with 'HANDOFF:<AgentName>' where <AgentName> is the primary specialist (e.g. Rachel)
          and may optionally append a 1-line coordination comment. This ensures Ruby does not provide domain content
          outside its remit and acts as the router/concierge as intended.
        """
        system_prompt = self.agent_descriptions.get(agent_name, "")

        # Default domain instruction for all agents
        domain_instruction = (
            "You must ONLY answer within your domain/responsibilities. "
            "If the user's request is outside your domain, do NOT improvise or provide domain-specific advice. "
            "Reply with the token 'OUT_OF_SCOPE' (single token) if you cannot answer and optionally add one short coordinating sentence. "
            "When the orchestrator instructs you to respond post-routing, provide domain-specific, actionable content only. "
            "Keep replies concise (<200 words) unless asked for deeper detail."
        )

        # Ruby acts as concierge/orchestrator. Ruby clarifies and coordinates but must not self-handoff.
        intent = (key_info.get("intent") or "").lower()
        ruby_instruction = ""
        if agent_name == "Ruby":
            ruby_instruction = (
                "You are the Concierge/Orchestrator. If clarification is required, ask a single concise clarifying question. "
                "Do NOT provide domain-specific clinical advice. Coordinate specialists and defer domain responses to them; "
                "do not perform self-handoffs. Follow the orchestrator's execution plan when instructed."
            )

        # If there's a primary agent and this is not the primary, instruct non-primary agents to wait for explicit handoff
        prim_note = ""
        if primary_agent and agent_name != primary_agent and agent_name != "Ruby":
            prim_note = (
                f"Primary agent for this request is '{primary_agent}'. Do not respond fully unless the primary agent "
                "explicitly hands off to you. If the primary agent asks you to respond, proceed with domain-specific content."
            )

        user_payload = {
            "original_message": original_message,
            "summary": key_info.get("summary"),
            "intent": key_info.get("intent"),
            "entities": key_info.get("entities"),
            "keywords": key_info.get("keywords"),
            "confidence": key_info.get("confidence"),
            "context": context or {},
        }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": domain_instruction},
        ]
        if ruby_instruction:
            messages.append({"role": "system", "content": ruby_instruction})
        if prim_note:
            messages.append({"role": "system", "content": prim_note})

        messages.append({"role": "user", "content": f"Key Info: {json.dumps(user_payload)}\n\nUser Message: {original_message}"})
        logger.debug("build_agent_messages: agent=%s primary=%s messages_count=%d", agent_name, primary_agent, len(messages))
        return messages

    def orchestrate(self, message: str, context: Optional[Dict] = None, max_agents: int = 2) -> List[Dict[str, Any]]:
        """Primary orchestration entry.

        Returns a list (in priority order) of dicts:
        [
            {
                "agent": "Ruby",
                "messages": [ ... ],            # messages to send to this agent now (system/user)
                "strategy": { ... }             # execution strategy for this agent
            },
            ...
        ]

        Typical usage:
        1. router.orchestrate(...) -> get messages for each agent
        2. call agent.respond using the constructed messages (or BaseAgent.call_openrouter)
        3. if an agent returns HANDOFF:<AgentName>, forward the relevant message to that agent
        """
        # Decide agents: extractor recommendations are authoritative.
        key_info = self.extract_key_info(message, context)

        # Normalize extractor recommended agent names to canonical registry names
        raw_recs = key_info.get("recommended_agents") or []
        name_map = {k.lower(): k for k in self.agent_descriptions.keys()}
        recs: List[str] = []
        for r in raw_recs:
            if isinstance(r, str):
                canon = name_map.get(r.lower())
                if canon and canon not in recs:
                    recs.append(canon)

        # Use extractor recommendations if present; otherwise fallback to Ruby (concierge/orchestrator)
        selected = recs[:] if recs else ["Ruby"]
        # keep a merged variable for backward-compatible logging/flows
        merged: List[str] = selected[:]
        # Ensure we have at least one fallback
        if not merged:
            merged = ["Ruby"]

        # Strong domain override: if message clearly indicates physio, force Rachel as primary and remove Carla.
        msg_lower_for_override = (message or "").lower()
        physio_hints_local = ["pain", "injury", "back", "knee", "shoulder", "mobility", "rehab", "twist", "sprain", "swelling"]
        if any(h in msg_lower_for_override for h in physio_hints_local):
            if "Rachel" in merged:
                merged = ["Rachel"] + [a for a in merged if a != "Rachel" and a != "Carla"]
            else:
                merged = ["Rachel"] + [a for a in merged if a != "Carla"]
            logger.debug("orchestrate: strong physio hints detected; forcing Rachel primary and removing Carla; merged=%s", merged)

        # Trim to max_agents
        selected = merged[:max_agents]

        # Heuristic re-ordering to ensure domain specialists are primary when intent/keywords indicate so.
        intent = (key_info.get("intent") or "").lower()
        intent_map = {
            "physio": "Rachel",
            "medical": "Dr. Warren",
            "nutrition": "Carla",
            "performance": "Advik",
            "logistics": "Ruby",
        }
        specialist = intent_map.get(intent)

        msg_lower = (message or "").lower()
        physio_hints = ["pain", "injury", "back", "knee", "shoulder", "mobility", "rehab"]
        medical_hints = ["fever", "bleeding", "chest", "diagnosis", "lab", "blood", "test", "result", "prescription"]
        nutrition_hints = ["cgm", "glucose", "meal", "food", "diet", "protein", "carb", "supplement"]
        performance_hints = ["whoop", "oura", "hrv", "sleep", "recovery", "workout", "exercise"]

        if any(h in msg_lower for h in physio_hints):
            specialist = "Rachel"
        elif any(h in msg_lower for h in medical_hints):
            specialist = "Dr. Warren"
        elif any(h in msg_lower for h in nutrition_hints):
            specialist = "Carla"
        elif any(h in msg_lower for h in performance_hints):
            specialist = "Advik"

        confidence = float(key_info.get("confidence", 0) or 0)
        if specialist and specialist in self.agent_descriptions:
            override = False
            if not recs:
                override = True
            else:
                if confidence < 0.85:
                    override = True
                else:
                    strong_hint = any(h in msg_lower for h in physio_hints + medical_hints + nutrition_hints + performance_hints)
                    if strong_hint:
                        override = True

            if override:
                if specialist in selected:
                    selected = [specialist] + [a for a in selected if a != specialist]
                    logger.debug("orchestrate: promoting specialist %s to primary (override rules applied, confidence=%s, recs=%s)", specialist, confidence, recs)
                else:
                    selected = [specialist] + [a for a in selected if a != specialist]
                    logger.debug("orchestrate: prepending specialist %s to selection (override rules applied, confidence=%s, recs=%s)", specialist, confidence, recs)
            else:
                logger.debug("orchestrate: keeping extractor recommendations (recs=%s) despite specialist=%s (confidence=%s)", recs, specialist, confidence)

        try:
            if specialist == "Rachel" and "Carla" in selected:
                selected = [s for s in selected if s != "Carla"]
                if "Rachel" not in selected:
                    selected = ["Rachel"] + selected
                logger.debug("orchestrate: removed Carla from selection because physio specialist Rachel was detected")
            if specialist == "Carla" and "Rachel" in selected:
                selected = [s for s in selected if s != "Rachel"]
                if "Carla" not in selected:
                    selected = ["Carla"] + selected
                logger.debug("orchestrate: removed Rachel from selection because nutrition specialist Carla was detected")
        except Exception:
            pass

        orchestrated: List[Dict[str, Any]] = []
        primary = selected[0] if selected else None
        logger.debug("orchestrate: final selected=%s primary=%s (recs=%s routed=%s)", selected, primary, recs, merged)

        for agent in selected:
            msgs = self.build_agent_messages(agent, key_info, message, context, primary_agent=primary)

            # Determine strategy for this agent
            confidence = float(key_info.get("confidence", 0) or 0)
            summary = (key_info.get("summary") or "").strip()
            entities = key_info.get("entities") or {}

            # Default: agents can respond immediately
            strategy: Dict[str, Any] = {"wait_for_user": False, "initiator": "orchestrator", "reason": "immediate_response"}

            # Strategy logic: ensure proper sequencing of agent responses
            if agent == primary:
                # Primary agent always responds first
                strategy = {"wait_for_user": False, "initiator": "orchestrator", "reason": "primary_agent"}
            elif agent == "Ruby" and primary != "Ruby":
                # Ruby coordinates but doesn't respond if not primary (unless low confidence)
                if confidence < 0.7 or (not summary and not entities):
                    strategy = {
                        "wait_for_user": False,
                        "initiator": "Ruby",
                        "action": "ask_clarifying_question",
                        "clarify_prompt": "Quick clarifying question: please state the main symptom or timeframe in 1-2 short phrases."
                    }
                else:
                    strategy = {"wait_for_user": True, "initiator": "primary", "reason": "ruby_waits_for_primary"}
            else:
                # Non-primary specialists wait for primary agent or Ruby to coordinate
                strategy = {
                    "wait_for_user": True,
                    "initiator": primary or "Ruby",
                    "wait_for_event": "primary_response_or_handoff",
                    "reason": "waiting_for_primary_or_coordination"
                }

            orchestrated.append(
                {
                    "agent": agent,
                    "messages": msgs,
                    "strategy": strategy,
                }
            )
            logger.debug("orchestrate: appended agent=%s strategy=%s messages_len=%d", agent, strategy, len(msgs))

        return orchestrated

    def execute_plan(self, orchestrated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute an orchestrated plan.

        Calls agent instances according to each agent's strategy. If an agent's strategy
        indicates waiting (strategy['wait_for_user'] is True) the agent will be skipped
        and returned with result=None. Returned list items have keys:
        - agent: agent name
        - messages: messages sent
        - strategy: execution strategy
        - result: agent response string or None
        """
        results: List[Dict[str, Any]] = []
        for item in orchestrated:
            agent_name = item.get("agent")
            messages = item.get("messages", []) or []
            strategy = item.get("strategy", {}) or {}

            # If instructed to wait, skip execution for now
            if strategy.get("wait_for_user"):
                logger.debug("execute_plan: skipping agent=%s (waiting for user/handoff)", agent_name)
                results.append({"agent": agent_name, "messages": messages, "strategy": strategy, "result": None})
                continue

            agent_instance = self.agent_instances.get(agent_name)
            if not agent_instance:
                logger.debug("execute_plan: no instance registered for agent=%s", agent_name)
                results.append({"agent": agent_name, "messages": messages, "strategy": strategy, "result": None})
                continue

            try:
                resp = agent_instance.call_openrouter(messages)
            except Exception as exc:
                logger.exception("execute_plan: agent %s call failed: %s", agent_name, exc)
                resp = None

            results.append({"agent": agent_name, "messages": messages, "strategy": strategy, "result": resp})

        return results
