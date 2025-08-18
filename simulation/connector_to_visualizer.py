#!/usr/bin/env python3
"""
Connector: conversation_history.json -> conversation_history_for_visualizer.json

Creates an enriched JSON file that `simulation/decision_tree_visualizer.py`
expects. Fields added per-message:

- S.No.
- Sender
- Message
- Day        (days since start, integer)
- Date       (YYYY-MM-DD)
- Time       (HH:MM:SS)
- Reply_Needed (bool)
- Recommendations (list)

Usage:
    python3 simulation/connector_to_visualizer.py

Output:
    data/conversation_history_for_visualizer.json
"""

import json
from datetime import datetime

INPUT_PATH = "data/conversation_history.json"
OUTPUT_PATH = "data/conversation_history_for_visualizer.json"

# Explicit dates/times used in docs/conversation_journey.md
# If you want to change the timeline, edit these lists (must match number of messages).
DATES = [
    "2024-12-18",
    "2025-01-18",
    "2025-02-18",
    "2025-03-18",
    "2025-04-18",
    "2025-05-18",
    "2025-06-18",
    "2025-07-18",
]

TIMES = [
    "09:15:00",
    "10:05:00",
    "11:20:00",
    "09:45:00",
    "08:55:00",
    "14:00:00",
    "07:50:00",
    "20:10:00",
]

def load_input(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_question(text: str) -> bool:
    return "?" in text

def build_recommendations(sender: str, message: str):
    # For this dataset we treat agent recommendation messages as recommendations.
    # If an agent message contains actionable text, include it as a single recommendation.
    agents_that_recommend = {"Advik", "Dr. Warren", "Ruby", "Neel"}
    if sender in agents_that_recommend:
        # Heuristic: if message contains advice-like phrasing, expose it as a recommendation.
        if any(keyword in message.lower() for keyword in ["recommend", "recommendation", "focus", "suggest", "should"]):
            return [message]
    return []

def main():
    data = load_input(INPUT_PATH)
    conversation = data.get("conversation_history", [])
    total = len(conversation)

    if total != len(DATES) or total != len(TIMES):
        print("⚠️  Number of dates/times doesn't match number of messages.")
        print(f"Messages: {total}, Dates: {len(DATES)}, Times: {len(TIMES)}")
        print("If this is expected, edit DATES/TIMES in this script to match.")
        # continue anyway, we'll clamp to the shorter length

    start_date = datetime.strptime(DATES[0], "%Y-%m-%d").date()

    enriched = []
    for i, msg in enumerate(conversation):
        idx = i
        date = DATES[idx] if idx < len(DATES) else DATES[-1]
        time = TIMES[idx] if idx < len(TIMES) else TIMES[-1]

        # compute Day as days since start_date (1-based)
        try:
            cur_date = datetime.strptime(date, "%Y-%m-%d").date()
            day = (cur_date - start_date).days + 1
        except Exception:
            day = i + 1

        sender = msg.get("sender") or msg.get("Sender") or "Unknown"
        message_text = msg.get("message") or msg.get("Message") or ""

        recommendations = build_recommendations(sender, message_text)
        reply_needed = is_question(message_text) or (sender == "User") or (len(recommendations) > 0 and sender != "User")

        enriched.append({
            "S.No.": i + 1,
            "Sender": sender,
            "Message": message_text,
            "Day": day,
            "Date": date,
            "Time": time,
            "Reply_Needed": reply_needed,
            "Recommendations": recommendations
        })

    out = {
        "total_messages": len(enriched),
        "simulation_period": "8 months",
        "conversation_history": enriched
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote visualizer input: {OUTPUT_PATH} ({len(enriched)} messages)")

if __name__ == "__main__":
    main()
