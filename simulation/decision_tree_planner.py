from typing import Dict, Optional

class DecisionTreePlanner:
    def get_next_action(self, health_metrics: Dict) -> Optional[str]:
        """
        A simple rule-based planner to suggest the next action based on health metrics.
        """
        if health_metrics.get("blood_sugar_avg", 150) > 160:
            return "High blood sugar detected. Suggest a consultation with Dr. Warren."

        if health_metrics.get("adherence_rate", 0.8) < 0.5:
            return "Low adherence detected. Suggest a check-in with Ruby to discuss challenges."

        if health_metrics.get("weight", 75) > 76:
            return "Weight gain detected. Suggest a review of the nutrition plan with Carla."

        return None
