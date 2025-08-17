import json
from typing import Dict, List
from .base_agent import BaseAgent

def extract_travel_details_with_llm(message: str) -> Dict:
    """
    Uses an LLM to extract travel details from a user message.
    """
    prompt = f"""
You are an expert travel detail extractor. Your task is to extract travel information from a user's message and return it as a JSON object.
The JSON object should have the following keys: "origin", "destination", "dates", "time_zones", "training_split", "hotel_hub", "gym_requirements".
If a value is not present in the message, you should use a default value or "N/A".

User message: {message}

JSON output:
"""

    # Use a temporary lightweight agent to call the LLM
    extractor_agent = BaseAgent(name="Extractor", role="Util", system_prompt="")
    response = extractor_agent.call_openrouter([{"role": "user", "content": prompt}])

    try:
        # Extract the JSON part of the response
        json_part = response[response.find('{'):response.rfind('}')+1]
        return json.loads(json_part)
    except json.JSONDecodeError:
        # Return a default structure on failure
        return {
            "origin": "N/A",
            "destination": "N/A",
            "dates": [],
            "time_zones": {},
            "training_split": "N/A",
            "hotel_hub": "N/A",
            "gym_requirements": "N/A",
        }

class TravelProtocolGenerator:
    def generate(
        self,
        origin: str,
        destination: str,
        dates: List[str],
        time_zones: Dict[str, str],
        training_split: str,
        hotel_hub: str,
        gym_requirements: str,
    ) -> Dict:
        """
        Generates a travel protocol based on the provided details.
        """
        protocol = {
            "light_exposure_schedule": self._generate_light_exposure(time_zones),
            "in_flight_meals_hydration": self._generate_in_flight_nutrition(),
            "mobility_routines": self._generate_mobility_routines(),
            "local_gym_options": self._generate_local_gym_options(gym_requirements),
            "contingency_clinician_contact": self._generate_contingency_contact(),
            "arrival_day_plan": self._generate_arrival_day_plan(),
            "calendar_blocks": self._generate_calendar_blocks(dates),
        }
        return protocol

    def _generate_light_exposure(self, time_zones: Dict[str, str]) -> List[str]:
        # Placeholder logic
        return [
            "Morning: Seek bright light upon waking.",
            "Afternoon: Avoid bright light 2 hours before bedtime.",
        ]

    def _generate_in_flight_nutrition(self) -> List[str]:
        # Placeholder logic
        return [
            "Drink 250ml of water every hour.",
            "Avoid alcohol and caffeine.",
            "Eat a light, protein-rich meal.",
        ]

    def _generate_mobility_routines(self) -> List[str]:
        # Placeholder logic
        return [
            "Perform ankle circles and leg raises every hour.",
            "Stretch your hamstrings and hip flexors upon arrival.",
        ]

    def _generate_local_gym_options(self, gym_requirements: str) -> List[str]:
        # Placeholder logic
        return [
            "Pure Fitness, Singapore",
            "Fitness First, Singapore",
        ]

    def _generate_contingency_contact(self) -> str:
        # Placeholder logic
        return "Dr. Warren (+65 1234 5678)"

    def _generate_arrival_day_plan(self) -> List[str]:
        # Placeholder logic
        return [
            "Upon arrival, have a light meal.",
            "Engage in light physical activity, like a walk.",
            "Go to bed at your regular local time.",
        ]

    def _generate_calendar_blocks(self, dates: List[str]) -> List[str]:
        # Placeholder logic
        if not dates:
            return ["Travel Day: N/A", "Arrival Day: N/A"]
        return [
            f"Travel Day: {dates[0]}",
            f"Arrival Day: {dates[1] if len(dates) > 1 else 'N/A'}",
        ]

def generate_travel_protocol(
    origin: str,
    destination: str,
    dates: List[str],
    time_zones: Dict[str, str],
    training_split: str,
    hotel_hub: str,
    gym_requirements: str,
) -> str:
    """
    A wrapper function to generate the travel protocol and return it as a JSON string.
    """
    generator = TravelProtocolGenerator()
    protocol = generator.generate(
        origin,
        destination,
        dates,
        time_zones,
        training_split,
        hotel_hub,
        gym_requirements,
    )
    return json.dumps(protocol, indent=2)
