import os
import warnings

# Suppress TracerProvider warnings
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

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

# Initialize Langfuse with proper error handling
try:
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    )
except Exception as e:
    # Fallback to no-op if initialization fails
    warnings.warn(f"Failed to initialize Langfuse: {e}")
    langfuse = Langfuse()


@observe()
def track_agent_interaction(agent_name: str, message: str, response: str, context: dict | None = None):  # type: ignore[valid-type]
    return {"agent": agent_name, "input": message, "output": response, "context": context}


@observe()
def track_journey_milestone(week: int, milestone: str, metrics: dict | None = None):  # type: ignore[valid-type]
    return {"week": week, "milestone": milestone, "metrics": metrics}


