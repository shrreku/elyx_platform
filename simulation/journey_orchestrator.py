import random
from typing import Dict, List, Optional

from data.persistence import PersistenceManager
from agents.group_chat import GroupChatSystem

class JourneyOrchestrator:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.chat_system = GroupChatSystem()
        self.current_state = self.persistence.load_journey_state()

    def simulate_week(self, week: int, user_message: Optional[str] = None) -> Dict:
        print(f"=== Simulating Week {week} ===")

        events = self.generate_weekly_events(week)

        if user_message:
            user_messages = [user_message]
        else:
            user_messages = self.generate_user_messages(week, events)

        weekly_conversations: List[Dict] = []
        for message in user_messages:
            context = {"week": week, "events": events}
            conversation = self.chat_system.send_message("Rohan", message, context)
            if conversation:
                weekly_conversations.extend(conversation)

        report = self.generate_weekly_report(week, events, weekly_conversations)

        self.persistence.save_weekly_report(week, report)
        self.persistence.save_conversation_history(self.chat_system.conversation_history)

        return report

    def generate_weekly_events(self, week: int) -> List[str]:
        events: List[str] = []
        if week == 1:
            events.append("onboarding_complete")
            events.append("initial_blood_test_high_sugar")
        elif week in [12, 24, 34]:
            events.append("quarterly_diagnostic_test")
        elif week == 10:
            events.append("leg_injury_reported")
        elif week % 4 == 0:
            events.append("business_travel")
        elif week % 2 == 0:
            events.append("exercise_plan_update")
        return events

    def generate_user_messages(self, week: int, events: List[str]) -> List[str]:
        messages: List[str] = []
        num_messages = random.randint(3, 5)
        for _ in range(num_messages):
            if "leg_injury_reported" in events:
                messages.append("I twisted my leg at the hotel gym - it's painful and swollen. What should I do?")
            elif "business_travel" in events:
                messages.append("I'm traveling to Singapore next week. How should I adjust my plan?")
            elif "quarterly_diagnostic_test" in events:
                messages.append("Just got my test results back. Can we review them together?")
            else:
                curiosity_messages = [
                    "I read about CGM devices. Should I get one for better blood sugar monitoring?",
                    "What's the latest research on intermittent fasting for diabetes?",
                    "How does sleep quality affect blood sugar levels?",
                    "Can stress really impact my A1C levels?",
                    "What supplements should I consider for better metabolic health?",
                ]
                messages.append(random.choice(curiosity_messages))
        return messages

    def generate_weekly_report(self, week: int, events: List[str], conversations: List[Dict]) -> Dict:
        adherence = random.choice([0.3, 0.5, 0.7, 0.8])
        blood_sugar_improvement = max(0, (week - 1) * 2)
        
        # Extract agent responses from conversations
        agent_responses = []
        for conv in conversations:
            if conv.get("sender") != "Rohan":
                agent_responses.append(f"{conv['sender']}: {conv['message']}")
        
        return {
            "week": week,
            "events": events,
            "conversations_count": len(conversations),
            "adherence_rate": adherence,
            "agent_response": "; ".join(agent_responses) if agent_responses else "No agent response",
            "health_metrics": {
                "blood_sugar_avg": max(120, 180 - blood_sugar_improvement),
                "a1c": max(5.5, 6.2 - (week * 0.02)),
                "weight": 75 - (week * 0.1) if week > 4 else 75,
            },
            "agent_actions": {
                "doctor_hours": random.randint(8, 15),
                "coach_hours": random.randint(10, 20),
            },
            "recommendations": self.extract_recommendations(conversations),
        }

    def extract_recommendations(self, conversations: List[Dict]) -> List[str]:
        recommendations = []
        for conv in conversations:
            if conv.get("sender") != "Rohan":
                message = conv.get("message", "").lower()
                sender = conv.get("sender", "")
                
                # Look for actionable content patterns
                action_patterns = [
                    "schedule", "book", "order", "call", "email", "confirm", "arrange",
                    "test", "measure", "track", "monitor", "adjust", "change", "try",
                    "implement", "start", "begin", "continue", "follow up", "review",
                    "set up", "bring", "pencil in", "send", "add", "create", "plan",
                    "consider", "aim for", "target", "goal", "focus on", "practice",
                    "exercise", "workout", "session", "routine", "protocol", "diet",
                    "meal", "nutrition", "supplement", "medication", "appointment",
                    "consultation", "assessment", "evaluation", "check-in", "reminder"
                ]
                
                # Check if message contains actionable content
                has_actionable_content = any(pattern in message for pattern in action_patterns)
                
                # Also check for specific time references and concrete next steps
                time_patterns = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                               "am", "pm", "morning", "afternoon", "evening", "tonight", "tomorrow",
                               "next week", "this week", "daily", "weekly", "monthly"]
                
                has_time_reference = any(pattern in message for pattern in time_patterns)
                
                # Check for specific quantities and measurements
                quantity_patterns = ["2 large eggs", "1 cup", "30-45 min", "7-8h", "100-150 mg/dl",
                                   "60-70%", "15-30 min", "3-4 days", "1-2 hrs"]
                
                has_quantities = any(pattern in message for pattern in quantity_patterns)
                
                # If message has actionable content, time references, or specific quantities, it's a recommendation
                if has_actionable_content or has_time_reference or has_quantities:
                    # Truncate long messages but keep the actionable part
                    truncated_msg = conv['message'][:200] + "..." if len(conv['message']) > 200 else conv['message']
                    recommendations.append(f"{sender}: {truncated_msg}")
        
        return recommendations

    def run_full_journey(self):
        for week in range(1, 35):
            report = self.simulate_week(week)
            print(f"Week {week} completed - Adherence: {report['adherence_rate']:.1%}")
        print("8-month journey simulation completed!")


