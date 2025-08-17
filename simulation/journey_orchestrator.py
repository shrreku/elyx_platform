import random
from typing import Dict, List

from data.models import DiagnosticPanel, MicroReplan
from data.persistence import PersistenceManager
from agents.group_chat import GroupChatSystem


class JourneyOrchestrator:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.chat_system = GroupChatSystem()
        self.current_state = self.persistence.load_journey_state()

        # Handle migration for old state files
        if "diagnostic_panels" not in self.current_state:
            self.current_state["diagnostic_panels"] = []
        if "micro_replans" not in self.current_state:
            self.current_state["micro_replans"] = []

    def simulate_week(self, week: int) -> Dict:
        print(f"=== Simulating Week {week} ===")

        events = self.generate_weekly_events(week)
        user_messages = self.generate_user_messages(week, events)

        weekly_conversations: List[Dict] = []

        if "quarterly_diagnostic_test" in events:
            self._handle_diagnostic_panel(week, weekly_conversations)

        for message in user_messages:
            context = {"week": week, "events": events}
            conversation = self.chat_system.send_message("Rohan", message, context)
            if conversation:
                weekly_conversations.extend(conversation)

        report = self.generate_weekly_report(week, events, weekly_conversations)

        self.persistence.save_weekly_report(week, report)
        self.persistence.save_conversation_history(self.chat_system.conversation_history)
        self.persistence.save_journey_state(self.current_state)

        return report

    def _handle_diagnostic_panel(self, week: int, weekly_conversations: List[Dict]):
        # 1. Pre-check window
        weekly_conversations.append(
            {
                "sender": "Ruby",
                "message": f"Hi Rohan, it's time for your quarterly check-in. I'll be scheduling your blood panel for next week. I'll send over the details shortly.",
            }
        )

        # 2. Schedule phlebotomy with fasting guidance
        weekly_conversations.append(
            {
                "sender": "Ruby",
                "message": "Your phlebotomy appointment is scheduled for next Tuesday at 8 AM. Please remember to fast for 12 hours prior. No food or drink other than water.",
            }
        )

        # 3. On result arrival, auto-trigger Dr. Warren synthesis playbook
        results = self._simulate_lab_results(week)
        previous_panels = self.current_state["diagnostic_panels"]
        delta = {}
        if previous_panels:
            delta = self._calculate_delta(results, previous_panels[-1]["results"])

        panel = DiagnosticPanel(week=week, results=results, delta=delta)
        self.current_state["diagnostic_panels"].append(panel.__dict__)

        dr_warren_summary = f"Hi Rohan, your lab results are in. Overall, we're seeing some great progress. Your A1C is down to {results['A1C']}, and your triglycerides are {results['Triglycerides']}. The main area to focus on is {results['Focus Area']}. I'll loop in the team with some recommendations."
        weekly_conversations.append({"sender": "Dr. Warren", "message": dr_warren_summary})

        # 4. Create “delta” view against last panel; assign action items to Carla/Rachel/Advik; plan retest window.
        weekly_conversations.append(
            {
                "sender": "Dr. Warren",
                "message": "Team, here are the latest results. Please provide your recommendations. Carla, let's focus on nutrition to address the triglycerides. Rachel, please update Rohan's strength training plan. Advik, please check his latest wearable data for any correlations.",
            }
        )
        weekly_conversations.append({"sender": "Carla", "message": "Got it. I'll send over some meal plan adjustments."})
        weekly_conversations.append(
            {"sender": "Rachel", "message": "Understood. I'll add a new workout to his plan for next week."}
        )
        weekly_conversations.append(
            {"sender": "Advik", "message": "On it. I'll analyze his latest sleep and HRV data."}
        )

    def _simulate_lab_results(self, week: int) -> Dict[str, str]:
        # Simulate improvement over time
        a1c = 6.2 - (week * 0.03)
        triglycerides = 150 - (week * 1.5)
        return {
            "A1C": f"{a1c:.2f}",
            "Triglycerides": f"{triglycerides:.0f}",
            "Vitamin D": f"{random.randint(25, 45)}",
            "Focus Area": random.choice(["LDL Cholesterol", "Inflammation (hs-CRP)"]),
        }

    def _calculate_delta(self, current_results: Dict, previous_results: Dict) -> Dict:
        delta = {}
        for key, value in current_results.items():
            if key in previous_results:
                try:
                    current_val = float(value)
                    previous_val = float(previous_results[key])
                    change = current_val - previous_val
                    delta[key] = f"{change:+.2f}"
                except ValueError:
                    delta[key] = "N/A"
        return delta

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
        adherence = self._calculate_adherence(events)

        if adherence < 0.6:
            self._generate_micro_replan(week, adherence, conversations)

        blood_sugar_improvement = max(0, (week - 1) * 2)
        return {
            "week": week,
            "events": events,
            "conversations_count": len(conversations),
            "adherence_rate": adherence,
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

    def _calculate_adherence(self, events: List[str]) -> float:
        base_adherence = 0.85
        if "business_travel" in events:
            base_adherence -= 0.2
        if "leg_injury_reported" in events:
            base_adherence -= 0.3
        # Add some randomness to simulate real life
        base_adherence -= random.uniform(0, 0.15)
        return max(0, base_adherence)

    def _generate_micro_replan(self, week: int, adherence: float, weekly_conversations: List[Dict]):
        plan_version = f"v{week}.{len(self.current_state['micro_replans']) + 1}"
        reason = f"Detected low adherence ({adherence:.0%}) this week."
        changes = [
            "Shorter duration workouts (20-30 mins).",
            "Swap evening workout for a morning walk.",
            "Focus on recovery: 10 mins of stretching before bed.",
        ]

        replan = MicroReplan(week=week, version=plan_version, reason=reason, changes=changes)
        self.current_state["micro_replans"].append(replan.__dict__)

        message = (
            f"Hi Rohan, I noticed things were a bit hectic this week. No problem at all, that's completely normal. "
            f"Let's adjust the plan to make it more manageable. Here's a micro-replan for the coming week ({plan_version}):\n\n"
            f"- {changes[0]}\n- {changes[1]}\n- {changes[2]}\n\n"
            f"Let's focus on consistency, not intensity. How does this sound?"
        )
        weekly_conversations.append({"sender": "Neel", "message": message})

    def extract_recommendations(self, conversations: List[Dict]) -> List[str]:
        recommendations: List[str] = []
        for conv in conversations:
            if conv.get("sender") != "Rohan" and "recommend" in conv.get("message", "").lower():
                recommendations.append(f"{conv['sender']}: {conv['message'][:100]}...")
        return recommendations

    def run_full_journey(self):
        for week in range(1, 35):
            report = self.simulate_week(week)
            print(f"Week {week} completed - Adherence: {report['adherence_rate']:.1%}")
        print("8-month journey simulation completed!")


