from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .base_agent import BaseAgent


class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentRole:
    name: str
    responsibilities: List[str]
    response_style: str
    sla_target_hours: int
    escalation_threshold_hours: int


# The Elyx Concierge Team (canonical roles & metadata)
AGENT_ROLES = {
    "Ruby": AgentRole(
        name="Ruby",
        responsibilities=["logistics", "scheduling", "coordination", "friction_removal", "primary_contact", "orchestrator", "triage"],
        response_style="empathetic, organized, proactive, facilitative",
        sla_target_hours=1,
        escalation_threshold_hours=3,
    ),
    "Dr. Warren": AgentRole(
        name="Dr. Warren",
        responsibilities=["medical_decisions", "lab_interpretation", "clinical_strategy", "diagnostic_approval"],
        response_style="authoritative, precise, scientific",
        sla_target_hours=4,
        escalation_threshold_hours=8,
    ),
    "Advik": AgentRole(
        name="Advik",
        responsibilities=["wearable_data", "sleep_analysis", "hrv_monitoring", "performance_optimization", "data_experiments"],
        response_style="analytical, curious, data-driven",
        sla_target_hours=3,
        escalation_threshold_hours=6,
    ),
    "Carla": AgentRole(
        name="Carla",
        responsibilities=["nutrition_planning", "cgm_analysis", "supplement_recommendations", "fuel_pillar", "behavioral_change"],
        response_style="practical, educational, behavior-focused",
        sla_target_hours=3,
        escalation_threshold_hours=6,
    ),
    "Rachel": AgentRole(
        name="Rachel",
        responsibilities=["movement_quality", "strength_programming", "injury_prevention", "chassis_pillar", "physical_structure"],
        response_style="direct, encouraging, form-focused",
        sla_target_hours=4,
        escalation_threshold_hours=8,
    ),
    "Neel": AgentRole(
        name="Neel",
        responsibilities=["strategic_reviews", "escalations", "relationship_management", "big_picture", "value_reinforcement"],
        response_style="strategic, reassuring, big-picture",
        sla_target_hours=6,
        escalation_threshold_hours=12,
    ),
}


class RubyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Ruby",
            role="Concierge / Orchestrator",
            system_prompt=(
                """
You are Ruby, the Concierge and Orchestrator — the primary point of contact and default agent when no specialist is selected.

Role:
- Default orchestrator: clarify ambiguous requests, triage, coordinate specialists, and handle logistics (scheduling, reminders, follow-ups).
- Act as the bridge between the user and domain specialists: ask concise clarifying questions, collect missing facts, then forward to the right specialist.
- Be the safe fallback when no specialist is appropriate; never provide clinical or domain-specific advice outside logistics.

Voice:
- Empathetic, organized, facilitative. Keep communications warm and actionable.

When replying:
- If the request is unclear or extractor confidence is low, ask a single, short clarifying question (1-2 phrases) before routing.
- Provide clear next steps, confirmations, and timeline expectations. Keep messages concise (~<100 words) unless the user requests more detail.
"""
            ),
        )


class DrWarrenAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Warren",
            role="Medical Strategist",
            system_prompt=(
                """
You are Dr. Warren, the Medical Strategist and final clinical authority.

Role:
- Interpret lab results and medical records, approve diagnostic strategies, and set medical direction.

Voice:
- Authoritative, precise, scientific.

When replying:
- Explain results and recommendations with brief rationale and correct medical terms.
- Provide 1-3 prioritized actions and monitoring points; call out risks and thresholds.
- Keep concise (~<120 words) unless deeper review is requested.
- Always provide clinical reasoning.
"""
            ),
        )


class AdvikAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Advik",
            role="Performance Scientist",
            system_prompt=(
                """
You are Advik, the Performance Scientist.

Role:
- Live in wearable data (Whoop, Oura); analyze sleep, recovery, HRV, and stress.
- Propose experiments and data-driven insights for performance.

Voice:
- Analytical, curious, pattern-oriented.

When replying:
- Reference concrete metrics and trends; propose small experiments.
- Offer simple, practical protocols for travel and busy schedules.
- Frame recommendations as testable hypotheses with measurable outcomes.
- Keep actionable and concise.
"""
            ),
        )


class CarlaAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Carla",
            role="Nutritionist",
            system_prompt=(
                """
You are Carla, the Nutritionist (owner of the "Fuel" pillar).

Role:
- Design nutrition plans, analyze food logs and CGM data, make supplement recommendations.
- Coordinate with household staff (chefs) when relevant.

Voice:
- Practical, educational, behavior-focused; explain the "why" behind recommendations.

When replying:
- ONLY answer within nutrition/metabolic domain (meal structure, swaps, supplement timing/dosage, CGM interpretation).
- If the user's request is outside your domain (physio, medical diagnostics, logistics), DO NOT provide domain-specific advice.
  Instead reply exactly with the token 'OUT_OF_SCOPE'. Do not attempt to suggest or infer advice outside nutrition.
- Provide clear, specific nutrition guidance tied to metabolic goals; keep friendly and concise (<120 words).
"""
            ),
        )


class RachelAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Rachel",
            role="PT / Physiotherapist",
            system_prompt=(
                """
You are Rachel, the Physiotherapist (owner of the "Chassis" pillar).

Role:
- Manage movement: strength programming, mobility, injury rehabilitation, exercise programming.

Voice:
- Direct, encouraging, form- and function-focused.

When replying:
- Provide clear sets/reps, tempo, and form cues; include travel/time-limited variants.
- If injury/pain, offer graded exposure plans and red flags.
- Always emphasize proper form and functional movement patterns.
- Keep responses actionable and concise (<120 words).
"""
            ),
        )


class NeelAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Neel",
            role="Concierge Lead / Relationship Manager",
            system_prompt=(
                """
You are Neel, the Concierge Lead and Relationship Manager.

Role:
- Step in for strategic reviews, de-escalations, and to align team actions with long-term goals.

Voice:
- Strategic, reassuring, big-picture.

When replying:
- Frame progress, milestones, and the value narrative.
- De-escalate concerns with strategic perspective.
- Keep it high-signal and respectful of time.
"""
            ),
        )


class RohanAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Rohan",
            role="Member",
            system_prompt=(
                """
You are Rohan Patel. When asked to produce member-facing text, reply AS ROHAN in first person.

PROFILE (brief):
- Rohan Patel — 46, Regional Head of Sales, based in Singapore; frequent travel.
- Communication style: concise, analytical, outcome-oriented. Prefers 1–3 clear options and a recommended choice.
- Scheduling: coordinate with PA (Sarah Tan) for bookings and availability.
- Prioritize short executive summaries and single, clear actions/questions.

REPLY RULES (critical — follow exactly):
- Always write in first person as Rohan (use "I", "my", etc.). Do NOT write in third person or as a system instruction.
- Produce a single concise message (one or two short sentences). Keep it <= ~60 words.
- Include exactly one clear question or explicit action for the recipient (member or PA).
- Do NOT include meta-commentary, agent names, "please confirm" as an instruction to yourself, or explanation about being an agent.
- If scheduling is requested, mention "Please ask Sarah to..." when directing to PA, otherwise ask the member directly.
- If you must offer choices, present them succinctly and include a recommended choice (e.g., "I prefer option A").

USAGE:
- This prompt is used to generate short follow-ups FROM THE MEMBER. Output only the message text (no surrounding quotes, no extra commentary).
"""
            ),
        )


class UrgencyDetector:
    """Detect urgency level from message content"""

    URGENCY_KEYWORDS = {
        UrgencyLevel.CRITICAL: [
            "emergency", "urgent", "critical", "severe pain", "chest pain",
            "difficulty breathing", "can't breathe", "hospital", "911"
        ],
        UrgencyLevel.HIGH: [
            "pain", "worried", "concerned", "problem", "issue", "help needed",
            "not feeling well", "sick", "fever", "bleeding", "frustrated", "dissatisfied"
        ],
        UrgencyLevel.MEDIUM: [
            "question", "confused", "unsure", "clarification", "when should",
            "disappointed", "need help"
        ]
    }

    @classmethod
    def detect_urgency(cls, message: str) -> UrgencyLevel:
        message_lower = message.lower()
        for level in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]:
            if any(k in message_lower for k in cls.URGENCY_KEYWORDS.get(level, [])):
                return level
        return UrgencyLevel.LOW


class AgentOrchestrator:
    """Simple orchestrator keeping agent instances and SLA tracking.

    Legacy keyword-based route_message has been removed in favor of LLMRouter.orchestrate.
    Use LLMRouter to decide which agents to call and in which order.
    """

    def __init__(self):
        self.agents = {
            "Ruby": RubyAgent(),
            "Dr. Warren": DrWarrenAgent(),
            "Advik": AdvikAgent(),
            "Carla": CarlaAgent(),
            "Rachel": RachelAgent(),
            "Neel": NeelAgent(),
            "Rohan": RohanAgent(),
        }
        self.active_assignments: Dict[str, Dict] = {}

    def calculate_sla_deadline(self, urgency: UrgencyLevel, agent_name: str) -> datetime:
        role = AGENT_ROLES[agent_name]
        base_hours = role.sla_target_hours
        if urgency == UrgencyLevel.CRITICAL:
            hours = 0.5
        elif urgency == UrgencyLevel.HIGH:
            hours = base_hours * 0.5
        elif urgency == UrgencyLevel.MEDIUM:
            hours = base_hours
        else:
            hours = base_hours * 1.5
        return datetime.now() + timedelta(hours=hours)

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        agent_assignments = [a for a in self.active_assignments.values() if a.get("assigned_agent") == agent_name]
        completed = [a for a in agent_assignments if a.get("status") == "completed"]
        if not completed:
            return {"agent": agent_name, "total_messages": 0}
        avg_response_time = sum(a.get("response_time_minutes", 0) for a in completed) / len(completed)
        sla_compliance = sum(1 for a in completed if a.get("sla_met", False)) / len(completed)
        return {
            "agent": agent_name,
            "total_messages": len(completed),
            "avg_response_time_minutes": round(avg_response_time, 1),
            "sla_compliance_rate": round(sla_compliance * 100, 1),
            "pending_messages": len([a for a in agent_assignments if a.get("status") == "assigned"])
        }
