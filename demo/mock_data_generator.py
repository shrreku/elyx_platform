#!/usr/bin/env python3
"""
Mock Data Generator for Elyx Platform Demo
Generate realistic episodes, decisions, and experiments for testing
"""

import json
import requests
from datetime import datetime, timedelta
import uuid

BASE_URL = "http://127.0.0.1:8787"

def generate_mock_episodes():
    """Generate mock episodes following the Implementation Guide scenarios"""
    episodes = [
        {
            "user_id": "rohan",
            "title": "Member Dissatisfaction - Service Feedback",
            "trigger_type": "member_initiated",
            "trigger_description": "Frustration about scheduling delays and lack of communication",
            "trigger_timestamp": "2025-01-15T08:30:00Z",
            "status": "resolved",
            "priority": 3,
            "member_state_before": "frustrated",
            "member_state_after": "satisfied"
        },
        {
            "user_id": "rohan",
            "title": "Sleep Quality Concerns",
            "trigger_type": "member_initiated", 
            "trigger_description": "Poor sleep latency and frequent disturbances reported",
            "trigger_timestamp": "2025-01-10T22:15:00Z",
            "status": "in_progress",
            "priority": 2,
            "member_state_before": "tired",
            "member_state_after": "optimizing"
        },
        {
            "user_id": "rohan",
            "title": "Travel Week Preparation",
            "trigger_type": "scheduled_checkup",
            "trigger_description": "Upcoming business trip to Tokyo - need protocol adjustments",
            "trigger_timestamp": "2025-01-20T10:00:00Z",
            "status": "open",
            "priority": 1,
            "member_state_before": "planning"
        }
    ]
    
    for episode in episodes:
        try:
            response = requests.post(f"{BASE_URL}/episodes", json=episode)
            if response.status_code == 200:
                print(f"✅ Created episode: {episode['title']}")
            else:
                print(f"❌ Failed to create episode: {response.text}")
        except Exception as e:
            print(f"❌ Error creating episode: {e}")

def generate_mock_decisions():
    """Generate mock decisions with why-traceability"""
    decisions = [
        {
            "type": "medication_recommendation",
            "content": "Start magnesium threonate 400mg before bed",
            "timestamp": "2025-01-12T22:30:00Z",
            "responsible_agent": "Dr. Warren",
            "rationale": "Poor sleep latency (25min avg) and member preference for natural interventions",
            "evidence": [
                {
                    "evidence_type": "biomarker_data",
                    "source": "whoop_sleep_analysis",
                    "data_json": {
                        "sleepLatency": 25,
                        "deepSleep": 40,
                        "disturbances": 12,
                        "recoveryScore": 65
                    },
                    "timestamp": "2025-01-11T07:00:00Z"
                }
            ],
            "messages": [
                {
                    "message_index": 145,
                    "message_timestamp": "2025-01-12T08:30:00Z"
                }
            ]
        },
        {
            "type": "nutrition_protocol",
            "content": "Implement protein-first meal sequencing",
            "timestamp": "2025-01-18T14:15:00Z",
            "responsible_agent": "Carla",
            "rationale": "CGM data shows 40mg/dL glucose spikes with current meal structure",
            "evidence": [
                {
                    "evidence_type": "cgm_data",
                    "source": "freestyle_libre",
                    "data_json": {
                        "peakGlucose": 165,
                        "timeToBaseline": 180,
                        "auc": 2400
                    },
                    "timestamp": "2025-01-17T12:30:00Z"
                }
            ]
        }
    ]
    
    for decision in decisions:
        try:
            response = requests.post(f"{BASE_URL}/decisions", json=decision)
            if response.status_code == 200:
                print(f"✅ Created decision: {decision['content']}")
            else:
                print(f"❌ Failed to create decision: {response.text}")
        except Exception as e:
            print(f"❌ Error creating decision: {e}")

def generate_mock_experiments():
    """Generate mock experiments"""
    experiments = [
        {
            "template": "SUPPLEMENT_TRIAL",
            "hypothesis": "Magnesium threonate 400mg before bed will improve sleep quality",
            "protocol_json": {
                "intervention": "Take magnesium threonate 400mg 1 hour before bedtime",
                "duration": "3 weeks",
                "measurements": ["sleep_latency", "sleep_quality_score", "morning_alertness"],
                "adherence_tracking": "daily_log"
            },
            "member_id": "rohan",
            "status": "running",
            "intervention": "Take magnesium threonate 400mg 1 hour before bedtime",
            "decision": "progress",
            "plan_section_updated": "Supplement Protocol"
        },
        {
            "template": "CGM_MEAL_TEST",
            "hypothesis": "Eating protein before carbs will reduce glucose spike by 30%",
            "protocol_json": {
                "intervention": "Consume protein portion 10 minutes before carbohydrates",
                "duration": "10 days",
                "measurements": ["glucose_peak", "glucose_auc", "time_to_baseline"],
                "adherence_tracking": "meal_photo_timestamps"
            },
            "member_id": "rohan",
            "status": "completed",
            "outcome": "Average glucose peak reduced by 32%, hypothesis confirmed",
            "success": True,
            "intervention": "Consume protein portion 10 minutes before carbohydrates",
            "decision": "progress",
            "plan_section_updated": "Nutrition Plan"
        },
        {
            "template": "EXERCISE_PROTOCOL",
            "hypothesis": "Morning HIIT sessions will improve HRV without affecting recovery",
            "protocol_json": {
                "intervention": "15-minute HIIT session 3x/week at 7am",
                "duration": "4 weeks",
                "measurements": ["hrv", "recovery_score", "workout_performance"],
                "adherence_tracking": "whoop_data"
            },
            "member_id": "rohan",
            "status": "planned",
            "intervention": "15-minute HIIT session 3x/week at 7am",
            "decision": "progress",
            "plan_section_updated": "Exercise Protocol"
        }
    ]

    for experiment in experiments:
        try:
            response = requests.post(f"{BASE_URL}/experiment-ledger", json=experiment)
            if response.status_code == 200:
                print(f"✅ Created experiment: {experiment['hypothesis'][:50]}...")
            else:
                print(f"❌ Failed to create experiment: {response.text}")
        except Exception as e:
            print(f"❌ Error creating experiment: {e}")

def add_experiment_measurements():
    """Add some mock measurements to running experiments"""
    # This would need the actual experiment IDs from the database
    # For demo purposes, you can run this after creating experiments
    pass

def main():
    print("🚀 Generating mock data for Elyx platform...")
    print(f"Target backend: {BASE_URL}")
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Backend not responding. Please start the backend first.")
            return
    except:
        print("❌ Cannot connect to backend. Please start the backend first.")
        return
    
    print("\n📊 Creating Episodes...")
    generate_mock_episodes()
    
    print("\n🧠 Creating Decisions...")
    generate_mock_decisions()
    
    print("\n🧪 Creating Experiments...")
    generate_mock_experiments()
    
    print("\n✅ Mock data generation complete!")
    print("\nYou can now:")
    print("1. View episodes, decisions, and experiments in the UI")
    print("2. Click 'Why?' buttons on decisions to see traceability")
    print("3. Add more data using the API endpoints")

if __name__ == "__main__":
    main()
