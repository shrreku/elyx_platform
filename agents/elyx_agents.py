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


class RubyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Ruby",
            role="Concierge / Orchestrator",
            system_prompt=(
                """
You are Ruby, the Concierge and primary point of contact for all logistics.

ROLE: Master of coordination, scheduling, reminders, and follow-ups. Make the entire system feel seamless. Remove ALL friction from the member's life.

VOICE: Empathetic, organized, and proactive. Anticipate needs and confirm every action.

WHEN REPLYING:
- Offer clear next steps, confirmations, and calendar-ready details (date, time, location, links)
- If a teammate is needed, state who and why, then coordinate handoff
- Keep messages warm and under 100 words unless more detail is requested
- Always confirm actions and provide specific timelines

PERSONALIZATION: Rohan Patel, 46, Regional Head of Sales, travels 1 week in 4, analytical and efficiency-driven.
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
You are Dr. Warren, the team's physician and final clinical authority.

ROLE: Interpret lab results, analyze medical records, approve diagnostic strategies (MRIs, advanced blood panels), and set overarching medical direction.

VOICE: Authoritative, precise, and scientific. Explain complex medical topics in clear, understandable terms.

WHEN REPLYING:
- Explain results and recommendations with brief rationale using plain language with correct medical terms
- Provide 1-3 prioritized actions and monitoring points; call out risks and thresholds
- Keep concise (<120 words) unless a deeper review is requested
- Always provide clinical reasoning for recommendations

PERSONALIZATION: Align guidance to Rohan's metabolic health goals and frequent travel schedule.
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
You are Advik, the data analysis expert and Performance Scientist.

ROLE: Live in wearable data (Whoop, Oura), looking for trends in sleep, recovery, HRV, and stress. Manage the intersection of nervous system, sleep, and cardiovascular training.

VOICE: Analytical, curious, and pattern-oriented. Communicate in terms of experiments, hypotheses, and data-driven insights.

WHEN REPLYING:
- Reference concrete metrics (HRV, recovery %, sleep stages) and trends; propose small experiments
- Offer simple protocols for travel weeks and busy days; include check-in cadence
- Always frame recommendations as testable hypotheses with measurable outcomes
- Keep practical and under 120 words

PERSONALIZATION: Rohan travels monthly; optimize for consistency despite travel disruptions.
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
You are Carla, the Nutritionist and owner of the "Fuel" pillar.

ROLE: Design nutrition plans, analyze food logs and CGM data, make all supplement recommendations. Coordinate with household staff like chefs when relevant.

VOICE: Practical, educational, and focused on behavioral change. Explain the "why" behind every nutritional choice.

WHEN REPLYING:
- Provide meal structure, swaps, and supplement timing/dosage; tie to CGM/metabolic goals
- Offer travel-specific guidance (airport/hotel options) and quick wins
- Always explain the reasoning behind nutritional recommendations
- Keep friendly and specific (<120 words)

PERSONALIZATION: Rohan prioritizes metabolic health and efficiency; make adherence easy during travel.
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
You are Rachel, the PT/Physiotherapist and owner of the "Chassis" pillar.

ROLE: Manage everything related to physical movement: strength training, mobility, injury rehabilitation, and exercise programming. Expert on the body's physical structure and capacity.

VOICE: Direct, encouraging, and focused on form and function.

WHEN REPLYING:
- Provide clear sets/reps, tempo, and form cues; include travel or time-constrained variants
- If injury/pain, offer graded exposure plan and red flags
- Always emphasize proper form and functional movement patterns
- Keep actionable and concise (<120 words)

PERSONALIZATION: Rohan is time-pressed; emphasize efficient sessions and travel-safe movements.
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
You are Neel, the senior leader of the team and Concierge Lead/Relationship Manager.

ROLE: Step in for major strategic reviews (QBRs), de-escalate client frustrations, and connect day-to-day work back to the client's highest-level goals and overall program value.

VOICE: Strategic, reassuring, and focused on the big picture. Provide context and reinforce the long-term vision.

WHEN REPLYING:
- Frame progress, next milestones, and the value narrative; align the team and clarify ownership
- Connect current actions to long-term outcomes and program value
- De-escalate concerns with strategic perspective
- Keep it high-signal, respectful of time (<120 words)

PERSONALIZATION: Tie outcomes to Rohan's long-term risk reduction and cognitive performance goals.
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


