import json
from typing import List, Dict
from .elyx_agents import *
from .llm_router import LLMRouter
from .travel_protocol import generate_travel_protocol, extract_travel_details_with_llm
from monitoring.observability import track_routing_decision

class GroupChatSystem:
    def __init__(self):
        self.agents = {
            "Ruby": RubyAgent(),
            "Dr. Warren": DrWarrenAgent(),
            "Advik": AdvikAgent(),
            "Carla": CarlaAgent(),
            "Rachel": RachelAgent(),
            "Neel": NeelAgent(),
            "Rohan": RohanAgent()
        }
        self.conversation_history = []
        self.router = LLMRouter()

    def send_message(self, sender: str, message: str, context: Dict = None):
        # Handle event injection
        if self.handle_event_injection(message):
            return

        # Log the message
        self.conversation_history.append({
            "sender": sender,
            "message": message,
            "timestamp": self.get_timestamp(),
            "context": context
        })

        # If message from Rohan, get agent responses
        if sender == "Rohan":
            responses = self.get_agent_responses(message, context)
            for agent_name, response in responses.items():
                self.conversation_history.append({
                    "sender": agent_name,
                    "message": response,
                    "timestamp": self.get_timestamp(),
                    "context": context
                })

        return self.conversation_history[-5:]  # Return recent messages

    def get_agent_responses(self, message: str, context: Dict = None) -> Dict[str, str]:
        responses = {}

        # Determine which agents should respond based on keywords
        responding_agents = self.router.route(message, context)

        # Log the routing decision
        track_routing_decision(
            message=message,
            chosen_intent="inferred_intent",  # Placeholder
            selected_playbook="inferred_playbook",  # Placeholder
            involved_agents=responding_agents,
            prompts_used=self.router._build_route_prompt(message, context),
            artifacts_written={},  # Placeholder
            calendar_updates=[],  # Placeholder
        )

        # Generate travel protocol if needed
        if any(word in message.lower() for word in ["travel", "trip", "fly"]):
            travel_context = extract_travel_details_with_llm(message)
            travel_protocol = generate_travel_protocol(**travel_context)
            if context is None:
                context = {}
            context["travel_protocol"] = json.loads(travel_protocol)

        for agent_name in responding_agents:
            try:
                response = self.agents[agent_name].respond(message, context)
                responses[agent_name] = response
            except Exception as e:
                responses[agent_name] = f"Error: {str(e)}"

        return responses

    def handle_event_injection(self, message: str) -> bool:
        if "@inject" in message.lower():
            # Handle different event types
            if "injury" in message.lower():
                self.inject_injury_event()
            elif "travel" in message.lower():
                self.inject_travel_event()
            return True
        return False

    def inject_injury_event(self):
        injury_context = {
            "event_type": "leg_injury",
            "severity": "moderate",
            "location": "right calf",
            "impact": "affects mobility and exercise routine"
        }

        self.conversation_history.append({
            "sender": "System",
            "message": "Event Injected: Leg injury reported",
            "timestamp": self.get_timestamp(),
            "context": injury_context
        })

    def inject_travel_event(self):
        # This will be implemented later
        pass

    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
