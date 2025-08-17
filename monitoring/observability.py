import os

try:
    from langfuse.decorators import observe
    from langfuse import Langfuse
except Exception:  # noqa: BLE001
    # Fallback no-op for environments without langfuse configured
    def observe():  # type: ignore[override]
        def _decorator(func):
            return func

        return _decorator

    class Langfuse:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass


langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
)


@observe()
def track_agent_interaction(agent_name: str, message: str, response: str, context: dict | None = None):  # type: ignore[valid-type]
    return {"agent": agent_name, "input": message, "output": response, "context": context}


import json
from typing import List, Dict, Any

@observe()
def track_journey_milestone(week: int, milestone: str, metrics: dict | None = None):  # type: ignore[valid-type]
    return {"week": week, "milestone": milestone, "metrics": metrics}


def track_routing_decision(message: str, chosen_intent: str, selected_playbook: str, involved_agents: List[str], prompts_used: List[Dict[str, str]], artifacts_written: Dict[str, Any], calendar_updates: List[str]):
    trace = {
        "message": message,
        "chosen_intent": chosen_intent,
        "selected_playbook": selected_playbook,
        "involved_agents": involved_agents,
        "prompts_used": prompts_used,
        "artifacts_written": artifacts_written,
        "calendar_updates": calendar_updates,
    }
    with open("monitoring/traces.json", "r+") as f:
        traces = json.load(f)
        traces.append(trace)
        f.seek(0)
        json.dump(traces, f, indent=4)
