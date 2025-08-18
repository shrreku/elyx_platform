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
        * If an agent receives a request outside its domain it should reply with a standard handoff token so
          the router can forward to the appropriate agent instead of returning out-of-domain content.
    - Provide an `orchestrate` method that returns tailored messages for each selected agent as well as
      handoff prompts the router can use to call downstream agents when needed.

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
        - recommended_agents: optional list (LLM suggestion of agents)
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
        """Legacy routing - returns agent names. Uses router LLM to pick relevant agents."""
        cache_key = (message.strip(), json.dumps(context or {}, sort_keys=True))
        if cache_key in self._cache:
            logger.debug("route: cache hit")
            return self._cache[cache_key][:max_agents]

        msgs = self._build_route_prompt(message, context)
        logger.debug("route: calling router LLM for message (len=%d)", len(message or ""))
        raw = self.router.call_openrouter(msgs)
        logger.debug("route: raw router response len=%d", len(raw) if isinstance(raw, str) else 0)
        data = self._extract_json(raw)
        agents = data.get("agents") if isinstance(data, dict) else None
        result: List[str]
        if isinstance(agents, list):
            valid = [a for a in agents if a in self.agent_descriptions]
            seen = set()
            ordered: List[str] = []
            for a in valid:
                if a not in seen:
                    seen.add(a)
                    ordered.append(a)
            result = ordered[:max_agents]
        else:
            # Conservative fallback: no agent selected
            result = []

        logger.debug("route: selected agents=%s", result)

        # If the router selected no agents, route to Ruby by default
        if not result:
            logger.debug("route: no agents selected by LLM; falling back to Ruby")
            result = ["Ruby"][:max_agents]

        # Tiny LRU: cap cache size to 128 entries
        if len(self._cache) > 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result

    def build_agent_messages(self, agent_name: str, key_info: Dict[str, Any], original_message: str, context: Optional[Dict] = None, primary_agent: Optional[str] = None) -> List[Dict[str, str]]:
        """Construct messages to send to the specified agent.

        Adds:
        - The agent's system prompt (from agent_descriptions).
        - A short instruction forcing the agent to stay in-domain and a standard handoff token if out-of-scope.
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
            "If the user's request is outside your domain, do NOT improvise. Reply with a HANDOFF token to indicate forwarding. "
            "If you are handed a request to respond after an explicit handoff, provide domain-specific, actionable content only. "
            "Keep replies concise (<200 words) unless asked for deeper detail."
        )

        # If Ruby is present but not the primary specialist, instruct Ruby to handoff explicitly
        ruby_handoff_instruction = ""
        intent = (key_info.get("intent") or "").lower()
        if agent_name == "Ruby" and primary_agent and primary_agent != "Ruby":
            ruby_handoff_instruction = (
                f"BEGIN YOUR REPLY with the token 'HANDOFF:{primary_agent}' followed by an optional 1-line note like "
                f"'Forwarding to {primary_agent} for domain-specific response.' Do NOT provide domain-specific medical/physio/nutrition advice."
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
        if ruby_handoff_instruction:
            messages.append({"role": "system", "content": ruby_handoff_instruction})
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
                "handoff_message": [ ... ]     # optional: messages to send to this agent when a handoff occurs
            },
            ...
        ]

        Typical usage:
        1. router.orchestrate(...) -> get messages for each agent
        2. call agent.respond using the constructed messages (or BaseAgent.call_openrouter)
        3. if an agent returns HANDOFF:<AgentName>, forward the relevant handoff_message to that agent
        """
        # Decide agents (keep legacy route selection)
        selected = self.route(message, context, max_agents=max_agents)
        key_info = self.extract_key_info(message, context)

        # Heuristic re-ordering to ensure domain specialists are primary when intent/keywords indicate so.
        # This prevents Ruby (concierge) from being primary and answering domain-specific questions.
        intent = (key_info.get("intent") or "").lower()
        intent_map = {
            "physio": "Rachel",
            "medical": "Dr. Warren",
            "nutrition": "Carla",
            "performance": "Advik",
            "logistics": "Ruby",
        }
        specialist = intent_map.get(intent)

        # Keyword hints override intent when present (strong signals for physio/medical/nutrition)
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

        # If we found a specialist and it's in the selected list, promote it to primary
        if specialist and specialist in selected:
            logger.debug("orchestrate: promoting specialist %s to primary", specialist)
            selected = [specialist] + [a for a in selected if a != specialist]

        orchestrated: List[Dict[str, Any]] = []
        primary = selected[0] if selected else None
        logger.debug("orchestrate: final selected=%s primary=%s", selected, primary)

        for agent in selected:
            # Build messages for immediate call
            msgs = self.build_agent_messages(agent, key_info, message, context, primary_agent=primary)
            # Build an explicit handoff prompt for this agent to use if asked to produce domain content after a forward
            handoff_prompt = [
                {"role": "system", "content": self.agent_descriptions.get(agent, "")},
                {"role": "system", "content": "You were asked to respond after an explicit handoff. Provide domain-specific, actionable content only."},
                {"role": "user", "content": f"Key Info: {json.dumps(key_info)}\n\nUser Message (for your domain): {message}"},
            ]
            orchestrated.append(
                {
                    "agent": agent,
                    "messages": msgs,
                    "handoff_message": handoff_prompt,
                }
            )
            logger.debug("orchestrate: appended agent=%s messages_len=%d handoff_len=%d", agent, len(msgs), len(handoff_prompt))

        return orchestrated
