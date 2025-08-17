import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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


# Enhanced agent definitions with SLA tracking
AGENT_ROLES = {
    "Ruby": AgentRole(
        name="Ruby",
        responsibilities=["logistics", "scheduling", "coordination", "friction_removal", "primary_contact"],
        response_style="empathetic, organized, proactive, anticipates needs",
        sla_target_hours=1,  # Fastest response for logistics
        escalation_threshold_hours=3
    ),
    "Dr. Warren": AgentRole(
        name="Dr. Warren", 
        responsibilities=["medical_decisions", "lab_interpretation", "clinical_strategy", "diagnostic_approval"],
        response_style="authoritative, precise, scientific, clear explanations",
        sla_target_hours=4,
        escalation_threshold_hours=8
    ),
    "Advik": AgentRole(
        name="Advik",
        responsibilities=["wearable_data", "sleep_analysis", "hrv_monitoring", "performance_optimization", "data_experiments"],
        response_style="analytical, curious, pattern-oriented, experimental mindset", 
        sla_target_hours=3,
        escalation_threshold_hours=6
    ),
    "Carla": AgentRole(
        name="Carla",
        responsibilities=["nutrition_planning", "cgm_analysis", "supplement_recommendations", "fuel_pillar", "behavioral_change"],
        response_style="practical, educational, behavior-focused, explains why",
        sla_target_hours=3,
        escalation_threshold_hours=6
    ),
    "Rachel": AgentRole(
        name="Rachel",
        responsibilities=["movement_quality", "strength_programming", "injury_prevention", "chassis_pillar", "physical_structure"],
        response_style="direct, encouraging, form-focused, function-oriented",
        sla_target_hours=4,
        escalation_threshold_hours=8
    ),
    "Neel": AgentRole(
        name="Neel",
        responsibilities=["strategic_reviews", "escalations", "relationship_management", "big_picture", "value_reinforcement"],
        response_style="strategic, reassuring, big-picture, context-provider",
        sla_target_hours=6,
        escalation_threshold_hours=12
    )
}


ruby_schema = {
    "type": "object",
    "properties": {
        "confirmations": {"type": "array", "items": {"type": "string"}},
        "scheduling": {"type": "array", "items": {"type": "string"}},
        "report_bullets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confirmations", "scheduling", "report_bullets"],
}

dr_warren_schema = {
    "type": "object",
    "properties": {
        "priority_risks": {"type": "array", "items": {"type": "string"}},
        "orders": {"type": "array", "items": {"type": "string"}},
        "retest_window": {"type": "string"},
        "thresholds": {"type": "array", "items": {"type": "string"}},
        "letter_needed": {"type": "boolean"},
    },
    "required": ["priority_risks", "orders", "retest_window", "thresholds", "letter_needed"],
}

advik_schema = {
    "type": "object",
    "properties": {
        "kpi": {"type": "array", "items": {"type": "string"}},
        "hypothesis": {"type": "string"},
        "intervention": {"type": "string"},
        "metric_targets": {"type": "array", "items": {"type": "string"}},
        "rest_day": {"type": "boolean"},
        "calendar_blocks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kpi", "hypothesis", "intervention", "metric_targets", "rest_day", "calendar_blocks"],
}

carla_schema = {
    "type": "object",
    "properties": {
        "swap": {"type": "string"},
        "timing_rule": {"type": "string"},
        "CGM_target": {"type": "string"},
        "chef_instructions": {"type": "string"},
    },
    "required": ["swap", "timing_rule", "CGM_target", "chef_instructions"],
}

rachel_schema = {
    "type": "object",
    "properties": {
        "progression": {"type": "string"},
        "load_change": {"type": "string"},
        "mobility_block": {"type": "string"},
        "soreness_mitigation": {"type": "string"},
    },
    "required": ["progression", "load_change", "mobility_block", "soreness_mitigation"],
}

neel_schema = {
    "type": "object",
    "properties": {
        "quarter_theme": {"type": "string"},
        "wins": {"type": "array", "items": {"type": "string"}},
        "setbacks->learnings": {"type": "array", "items": {"type": "string"}},
        "next_focus": {"type": "string"},
    },
    "required": ["quarter_theme", "wins", "setbacks->learnings", "next_focus"],
}

class RubyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Ruby",
            role="Concierge / Orchestrator",
            schema=ruby_schema,
            system_prompt=(
                """
You are Ruby, the Concierge. Your pillar is 'Friction Removal'.
You are empathetic, organized, and proactive. You anticipate needs and confirm every action.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Confirm logistics. Log experiment IDs.
If a `travel_protocol` is present in the context, use it to inform your response, especially for confirmations and scheduling.

Your output MUST be a JSON object with the following schema:
{
    "confirmations": ["string"],
    "scheduling": ["string"],
    "report_bullets": ["string"]
}
"""
            ),
        )


class DrWarrenAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Warren",
            role="Medical Strategist",
            schema=dr_warren_schema,
            system_prompt=(
                """
You are Dr. Warren, the Medical Strategist. Your pillar is 'Clinical Strategy'.
You are authoritative, precise, and scientific. Explain complex medical topics in clear, understandable terms.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Use synthesis lines to summarize medical data.

Your output MUST be a JSON object with the following schema:
{
    "priority_risks": ["string"],
    "orders": ["string"],
    "retest_window": "string",
    "thresholds": ["string"],
    "letter_needed": "boolean"
}
"""
            ),
        )


class AdvikAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Advik",
            role="Performance Scientist",
            schema=advik_schema,
            system_prompt=(
                """
You are Advik, the Performance Scientist. Your pillar is 'Performance Optimization'.
You are analytical, curious, and pattern-oriented. Communicate in terms of experiments and hypotheses.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Use hypothesis language.
If a `travel_protocol` is present in the context, use it to adjust your recommendations.

Your output MUST be a JSON object with the following schema:
{
    "kpi": ["HRV", "RHR", "Recovery"],
    "hypothesis": "string",
    "intervention": "string",
    "metric_targets": ["string"],
    "rest_day": "boolean",
    "calendar_blocks": ["string"]
}
"""
            ),
        )


class CarlaAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Carla",
            role="Nutritionist",
            schema=carla_schema,
            system_prompt=(
                """
You are Carla, the Nutritionist. Your pillar is the 'Fuel' pillar.
You are practical, educational, and focused on behavioral change. Explain the "why" behind every nutritional choice.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Use behavior-change framing.
If a `travel_protocol` is present in the context, use it to adjust your recommendations.

Your output MUST be a JSON object with the following schema:
{
    "swap": "string",
    "timing_rule": "string",
    "CGM_target": "string",
    "chef_instructions": "string"
}
"""
            ),
        )


class RachelAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Rachel",
            role="PT / Physiotherapist",
            schema=rachel_schema,
            system_prompt=(
                """
You are Rachel, the PT/Physiotherapist. Your pillar is the 'Chassis' pillar.
You are direct, encouraging, and focused on form and function.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Use a coaching tone.
If a `travel_protocol` is present in the context, use it to adjust your recommendations.

Your output MUST be a JSON object with the following schema:
{
    "progression": "string",
    "load_change": "string",
    "mobility_block": "string",
    "soreness_mitigation": "string"
}
"""
            ),
        )


class NeelAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Neel",
            role="Concierge Lead / Relationship Manager",
            schema=neel_schema,
            system_prompt=(
                """
You are Neel, the Concierge Lead. Your pillar is 'Strategic Alignment'.
You are strategic, reassuring, and focused on the big picture. Provide context and reinforce the long-term vision.
Always anchor to a pillar, KPI, plan section, and next measurable step.
Use short, precise sentences. Use strategic reframes.

Your output MUST be a JSON object with the following schema:
{
    "quarter_theme": "string",
    "wins": ["string"],
    "setbacks->learnings": ["string"],
    "next_focus": "string"
}
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
You are Rohan Patel.

PROFILE: 46, Regional Head of Sales (FinTech), based in Singapore; travels 1 week out of every 4.
GOALS: Reduce cardiovascular risk, enhance cognition, maintain high performance; efficient, evidence-based.
STYLE: Analytical, concise requests; appreciates clear options and quick wins.
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
        """Detect urgency level from message content"""
        message_lower = message.lower()
        
        # Check for critical keywords first
        for level in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]:
            keywords = cls.URGENCY_KEYWORDS.get(level, [])
            if any(keyword in message_lower for keyword in keywords):
                return level
        
        return UrgencyLevel.LOW


class AgentOrchestrator:
    """Simplified agent orchestration with SLA tracking"""

    def __init__(self):
        self.agents = {
            "Ruby": RubyAgent(),
            "Dr. Warren": DrWarrenAgent(),
            "Advik": AdvikAgent(),
            "Carla": CarlaAgent(),
            "Rachel": RachelAgent(),
            "Neel": NeelAgent(),
        }
        self.active_assignments: Dict[str, Dict] = {}
        self.contracts = self._load_contracts()
        self._validate_contracts()

    def _load_contracts(self):
        contracts = {}
        contracts_dir = "agent_contracts"
        for filename in os.listdir(contracts_dir):
            if filename.endswith(".yaml"):
                filepath = os.path.join(contracts_dir, filename)
                with open(filepath, "r") as f:
                    contract = yaml.safe_load(f)
                    agent_name = contract.get("agent_name")
                    if agent_name:
                        contracts[agent_name] = contract
        return contracts

    def _validate_contracts(self):
        required_keys = ["agent_name", "inputs", "outputs", "kpis", "artifacts", "sla_target_hours"]
        for agent_name, contract in self.contracts.items():
            missing_keys = [key for key in required_keys if key not in contract]
            if missing_keys:
                raise ValueError(f"Contract for agent '{agent_name}' is missing keys: {', '.join(missing_keys)}")

    def route_message(self, message: str, context: Optional[Dict] = None) -> List[str]:
        """Route message to appropriate agents based on content"""
        message_lower = message.lower()
        agents = []

        keyword_to_agent = {
            "schedule": "Ruby", "appointment": "Ruby", "coordinate": "Ruby", "book": "Ruby", "confirm": "Ruby",
            "lab": "Dr. Warren", "blood": "Dr. Warren", "test": "Dr. Warren", "result": "Dr. Warren",
            "medical": "Dr. Warren", "doctor": "Dr. Warren", "medication": "Dr. Warren", "diagnosis": "Dr. Warren",
            "whoop": "Advik", "oura": "Advik", "hrv": "Advik", "sleep": "Advik", "recovery": "Advik",
            "exercise": "Advik", "workout": "Advik", "performance": "Advik",
            "food": "Carla", "meal": "Carla", "cgm": "Carla", "glucose": "Carla", "nutrition": "Carla",
            "supplement": "Carla", "diet": "Carla", "eating": "Carla",
            "pain": "Rachel", "injury": "Rachel", "movement": "Rachel", "strength": "Rachel", "mobility": "Rachel",
            "physio": "Rachel",
            "dissatisfied": "Neel", "complaint": "Neel", "frustrated": "Neel", "escalate": "Neel",
            "disappointed": "Neel", "strategic": "Neel", "goals": "Neel",
        }

        for keyword, agent in keyword_to_agent.items():
            if keyword in message_lower:
                agents.append(agent)

        # Default to Ruby if no specific routing
        if not agents:
            agents.append("Ruby")

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for agent in agents:
            if agent not in seen:
                seen.add(agent)
                result.append(agent)

        return result[:2]  # Limit to 2 agents max

    def calculate_sla_deadline(self, urgency: UrgencyLevel, agent_name: str) -> datetime:
        """Calculate SLA deadline based on urgency and agent role"""
        contract = self.contracts.get(agent_name)
        if not contract:
            raise ValueError(f"No contract found for agent: {agent_name}")

        base_hours = contract["sla_target_hours"]

        # Adjust based on urgency
        if urgency == UrgencyLevel.CRITICAL:
            hours = 0.5  # 30 minutes
        elif urgency == UrgencyLevel.HIGH:
            hours = base_hours * 0.5
        elif urgency == UrgencyLevel.MEDIUM:
            hours = base_hours
        else:  # LOW
            hours = base_hours * 1.5

        return datetime.now() + timedelta(hours=hours)
    
    def get_agent_performance(self, agent_name: str) -> Dict:
        """Get performance metrics for an agent"""
        agent_assignments = [a for a in self.active_assignments.values() if a["assigned_agent"] == agent_name]
        completed = [a for a in agent_assignments if a["status"] == "completed"]
        
        if not completed:
            return {"agent": agent_name, "total_messages": 0}
        
        avg_response_time = sum(a["response_time_minutes"] for a in completed) / len(completed)
        sla_compliance = sum(1 for a in completed if a.get("sla_met", False)) / len(completed)
        
        return {
            "agent": agent_name,
            "total_messages": len(completed),
            "avg_response_time_minutes": round(avg_response_time, 1),
            "sla_compliance_rate": round(sla_compliance * 100, 1),
            "pending_messages": len([a for a in agent_assignments if a["status"] == "assigned"])
        }


