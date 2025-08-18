import json
import os
from typing import List, Dict, Any


class PersistenceManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_conversation_history(self, history: List[Dict[str, Any]]):
        """
        Save conversation history in two forms:
         - conversation_history_raw.json: the raw list (used by backend APIs)
         - conversation_history.json: analysis-friendly object expected by the UI
        """
        # Normalize
        if history is None:
            history = []

        # 1) Save raw list (backwards-compatible for internal usage)
        raw_filename = os.path.join(self.data_dir, "conversation_history_raw.json")
        try:
            with open(raw_filename, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            # Best-effort: ignore write errors here but continue to try saving wrapper
            pass

        # 2) Save analysis-friendly wrapper for the visualizer UI
        try:
            max_day = max((m.get("Day", 0) for m in history if isinstance(m, dict)), default=0)
            simulation_period = f"{max_day} days" if max_day else "N/A"
        except Exception:
            simulation_period = "N/A"

        wrapper = {
            "total_messages": len(history),
            "simulation_period": simulation_period,
            "conversation_history": history
        }

        filename = os.path.join(self.data_dir, "conversation_history.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, indent=2, ensure_ascii=False)

    def load_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Load conversation history. This function supports multiple on-disk formats for
        compatibility:
         - conversation_history.json (wrapper with conversation_history key)
         - conversation_history_raw.json (raw list)
         - conversation_history.json containing raw list (legacy)
        Returns a list of message dicts.
        """
        wrapper_path = os.path.join(self.data_dir, "conversation_history.json")
        raw_path = os.path.join(self.data_dir, "conversation_history_raw.json")

        # Try wrapper first
        try:
            if os.path.exists(wrapper_path):
                with open(wrapper_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If it's the wrapper object
                    if isinstance(data, dict) and "conversation_history" in data:
                        ch = data.get("conversation_history", [])
                        if isinstance(ch, list):
                            return ch
                        # fallthrough to attempt other formats
                    # If it's already a list (legacy), return it
                    if isinstance(data, list):
                        return data
                    # If dict with 'messages' key (some endpoints expect that)
                    if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
                        return data["messages"]
                    # Unexpected structure: continue to try raw file
        except json.JSONDecodeError as e:
            print(f"Error parsing {wrapper_path}: {e}")
        except Exception as e:
            print(f"Unexpected error loading {wrapper_path}: {e}")

        # Fallback: try raw file
        try:
            if os.path.exists(raw_path):
                with open(raw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
                        return data["messages"]
        except json.JSONDecodeError as e:
            print(f"Error parsing {raw_path}: {e}")
        except Exception as e:
            print(f"Unexpected error loading {raw_path}: {e}")

        # If nothing found or parse failed, return empty list
        return []

    def save_journey_state(self, state: Dict[str, Any]):
        filename = os.path.join(self.data_dir, "journey_state.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_journey_state(self) -> Dict[str, Any]:
        filename = os.path.join(self.data_dir, "journey_state.json")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"current_week": 1, "total_weeks": 34}
        except json.JSONDecodeError:
            return {"current_week": 1, "total_weeks": 34}
        except Exception:
            return {"current_week": 1, "total_weeks": 34}

    def save_weekly_report(self, week: int, report: Dict[str, Any]):
        filename = os.path.join(self.data_dir, f"week_{week:02d}_report.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
