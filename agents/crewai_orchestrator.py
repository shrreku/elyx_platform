import os
from typing import Dict, Optional

try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    Agent = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]
    Crew = None  # type: ignore[assignment]
    ChatOpenAI = None  # type: ignore[assignment]


ROLE_TO_PROMPT: Dict[str, str] = {
    "Ruby": (
        "You are Ruby, the Elyx Concierge Orchestrator. You are empathetic, organized, "
        "and proactive. You coordinate all logistics, scheduling, and follow-ups. Your job is to remove friction from the member's life."
    ),
    "Dr. Warren": (
        "You are Dr. Warren, the Medical Strategist. You interpret lab results, analyze medical records, "
        "and set medical direction. You explain complex topics clearly and are authoritative and precise."
    ),
    "Advik": (
        "You are Advik, the Performance Scientist. You analyze wearable data, manage sleep/recovery/HRV, "
        "and cardiovascular training. You communicate in terms of experiments, hypotheses, and data-driven insights."
    ),
    "Carla": (
        "You are Carla, the Nutritionist. You design nutrition plans, analyze food logs and CGM data, "
        "and make supplement recommendations. Focus on behavioral change and explain the why behind nutritional choices."
    ),
    "Rachel": (
        "You are Rachel, the Physiotherapist. You manage physical movement, strength training, mobility, and injury rehabilitation. "
        "You are direct, encouraging, and focused on form and function."
    ),
    "Neel": (
        "You are Neel, the Concierge Lead. You handle strategic reviews, client frustrations, and connect day-to-day work to long-term goals. "
        "You provide context and reinforce the long-term vision."
    ),
}


class CrewOrchestrator:
    """Lightweight wrapper around CrewAI to get single-agent responses.

    This assumes OpenRouter via litellm-compatible env:
    - OPENAI_API_KEY = your OpenRouter key
    - OPENAI_API_BASE = https://openrouter.ai/api/v1
    - model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    """

    def __init__(self):
        if ChatOpenAI is None:
            raise RuntimeError("langchain_openai not installed or failed to import")
        # CrewAI/LiteLLM use OpenAI-compatible config; we set base to OpenRouter via env
        # Model stays as configured (e.g., "openai/gpt-oss-20b:free")
        self.llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        self._agents: Dict[str, object] = {}

    def _get_or_create_agent(self, name: str):
        if Agent is None:
            raise RuntimeError("crewai not installed or failed to import")
        if name in self._agents:
            return self._agents[name]
        system_prompt = ROLE_TO_PROMPT.get(name, f"You are {name}, an Elyx agent.")
        agent = Agent(
            role=name,
            goal=f"Provide {name} expertise to the member in a concise, empathetic way.",
            backstory=system_prompt,
            allow_delegation=False,
            verbose=False,
            llm=self.llm,
        )
        self._agents[name] = agent
        return agent

    def ask(self, agent_name: str, message: str, context: Optional[Dict] = None) -> str:
        """
        Ask the named agent for a response via CrewAI.

        Improvements:
        - Retries with exponential backoff on transient failures (e.g., provider 503).
        - Logs warnings/errors and returns a graceful fallback string if the provider is unavailable.
        """
        if Task is None or Crew is None:
            raise RuntimeError("crewai not installed or failed to import")
        agent = self._get_or_create_agent(agent_name)
        desc = message if not context else f"Context: {context}\n\nMessage: {message}"
        task = Task(
            description=desc,
            agent=agent,
            expected_output="A precise, helpful, role-aligned response to the member. Keep it under 150 words.",
        )

        crew = Crew(agents=[agent], tasks=[task])

        import time
        import logging

        max_attempts = 3
        delay = 1.0

        for attempt in range(1, max_attempts + 1):
            try:
                result = crew.kickoff()
                return str(result)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                logging.warning("Crew kickoff attempt %d/%d failed: %s", attempt, max_attempts, msg)
                # Detect common transient/provider HTML responses or 5xx
                transient = False
                if "503" in msg or "Service Temporarily Unavailable" in msg or "<html" in msg:
                    transient = True

                if attempt < max_attempts and transient:
                    time.sleep(delay)
                    delay *= 2
                    continue

                logging.error("Crew kickoff failed permanently: %s", msg)
                # Graceful fallback: return a role-aligned short fallback message
                fallback = ROLE_TO_PROMPT.get(agent_name, f"You are {agent_name}, an Elyx agent. We're experiencing temporary service issues; please try again later.")
                # Keep fallback short
                return f"{agent_name}: {fallback}"
