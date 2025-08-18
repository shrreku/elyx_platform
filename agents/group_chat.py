import os
from typing import Dict, List, Optional, Union

try:
    from .crewai_orchestrator import CrewOrchestrator
    from .elyx_agents import AgentOrchestrator
    from .llm_router import LLMRouter
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    CrewOrchestrator = None
    AgentOrchestrator = None
    LLMRouter = None


class GroupChatSystem:
    def __init__(self, use_crewai: bool = True):
        self.conversation_history: List[Dict] = []
        self.use_crewai = use_crewai and AI_AVAILABLE
        
        if AI_AVAILABLE:
            if use_crewai:
                try:
                    self.crew_orchestrator = CrewOrchestrator()
                    self.agent_router = AgentOrchestrator()
                except Exception as e:
                    print(f"Warning: AI system initialization failed: {e}")
                    self.use_crewai = False
                    self.crew_orchestrator = None
                    self.agent_router = None
            else:
                self.router = LLMRouter()
        else:
            print("Warning: AI dependencies not available, using mock responses")
            self.use_crewai = False
            self.crew_orchestrator = None
            self.agent_router = None

    def send_message(
        self,
        sender: str,
        message: str,
        context: Optional[Dict] = None,
    ) -> Union[List[Dict], None]:
        print(f"\n💬 {sender}: {message}")

        if sender == "Rohan":
            self.conversation_history.append({"sender": sender, "message": message})

            if self.use_crewai and self.crew_orchestrator and self.agent_router:
                try:
                    # 1. Route message to get the right agent
                    agent_name = self.agent_router.route_message(message, context)[0]

                    # 2. Ask the selected agent
                    response_text = self.crew_orchestrator.ask(agent_name, message, context)
                    response = {"agent": agent_name, "message": response_text}
                except Exception as e:
                    print(f"Warning: AI response failed: {e}")
                    response = self._get_mock_response(message, context)
            else:
                response = self._get_mock_response(message, context)

            if response:
                self.conversation_history.append(
                    {"sender": response["agent"], "message": response["message"]}
                )
                print(f"🤖 {response['agent']}: {response['message']}")
                return [
                    {"sender": sender, "message": message},
                    {"sender": response["agent"], "message": response["message"]},
                ]
            return [{"sender": sender, "message": message}]
        return None

    def _get_mock_response(self, message: str, context: Optional[Dict] = None) -> Dict:
        """Generate mock responses when AI is not available"""
        import random
        
        mock_agents = ["Ruby", "Dr. Warren", "Advik", "Carla", "Rachel", "Neel"]
        agent_name = random.choice(mock_agents)
        
        # Generate context-aware mock responses
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["sleep", "hrv", "recovery", "whoop"]):
            agent_name = "Advik"
            response = "Based on your data, I recommend focusing on sleep hygiene and reducing evening screen time. Your HRV patterns suggest stress accumulation."
        elif any(word in message_lower for word in ["glucose", "sugar", "cgm", "diet"]):
            agent_name = "Carla"
            response = "I see your glucose patterns. Let's adjust your meal timing and consider protein-first approach for better metabolic control."
        elif any(word in message_lower for word in ["pain", "injury", "back", "mattress"]):
            agent_name = "Rachel"
            response = "Your symptoms suggest we need to address your sleep environment. Let's schedule a consultation to assess your setup."
        elif any(word in message_lower for word in ["schedule", "appointment", "call"]):
            agent_name = "Ruby"
            response = "I'll coordinate with the team to schedule your appointment. What time works best for you?"
        elif any(word in message_lower for word in ["test", "results", "medical"]):
            agent_name = "Dr. Warren"
            response = "Let me review your latest results. I'll schedule a follow-up to discuss any concerning trends."
        else:
            response = "Thank you for the update. I'm monitoring your progress and will reach out if any action is needed."
        
        return {"agent": agent_name, "message": response}

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history
