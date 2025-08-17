from typing import Dict, List
from simulation.journey_orchestrator import JourneyOrchestrator

class CompleteJourney:
    def __init__(self, messages: List[str], num_months: int = 8):
        self.messages = messages
        self.num_weeks = num_months * 4
        self.orchestrator = JourneyOrchestrator()

    def run(self) -> Dict:
        """
        Runs the complete journey simulation using the provided messages.
        """
        print(f"🚀 Starting Complete Journey Simulation for {self.num_weeks} weeks...")

        journey_data = []

        for week in range(1, self.num_weeks + 1):
            # Use a message for the week if available
            if self.messages:
                user_message = self.messages.pop(0)
            else:
                user_message = "Just checking in for the week."

            print(f"--- Week {week}: Rohan says: '{user_message}' ---")

            weekly_report = self.orchestrator.simulate_week(week, user_message)
            journey_data.append(weekly_report)

            print(f"✅ Week {week} completed.")

        print("🎉 Complete Journey Simulation finished!")

        return {
            "conversation_history": self.orchestrator.chat_system.get_conversation_history(),
            "journey_data": journey_data,
        }
