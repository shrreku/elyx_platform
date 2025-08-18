from typing import Dict, List
from simulation.journey_orchestrator import JourneyOrchestrator
from simulation.decision_tree_planner import DecisionTreePlanner
from simulation.xml_parser import XMLParser
from simulation.journey_analyzer import JourneyAnalyzer
import csv
import os
import json
from datetime import datetime, timedelta

class CompleteJourney:
    def __init__(self, xml_content: str, num_months: int = 8):
        self.num_weeks = num_months * 4
        self.orchestrator = JourneyOrchestrator()
        self.planner = DecisionTreePlanner()
        self.parser = XMLParser(xml_content)
        self.episodes = self.parser.parse_episodes()
        self.analyzer = JourneyAnalyzer()

    def run(self) -> Dict:
        """
        Runs the complete journey simulation for 242 days.
        """
        print(f"🚀 Starting Complete Journey Simulation for 242 days...")

        start_date = datetime.now()

        journey_data = []
        conversation_data = []  # For CSV export

        # Create a flat mapping of day -> message with metadata
        scheduled_messages = []
        for episode in self.episodes:
            for message in episode['messages']:
                # defensive parsing: skip if no day
                if not message.get("day"):
                    continue
                day = int(message["day"])
                
                message_data = {
                    "day": day,
                    "text": message.get("text", ""),
                    "date": message.get("date", ""),
                    "time": message.get("time", ""),
                    "sender": message.get("sender", "Rohan")
                }
                scheduled_messages.append(message_data)

        # Build a day->message map; later messages overwrite earlier ones
        day_map = {}
        for m in scheduled_messages:
            day_map[int(m["day"])] = m

        # Keep the last user message and reuse it for days without new user input
        last_user_message = "Just checking in."
        last_message_data = None

        for day in range(1, 243):
            # default to the last known user message
            user_message = last_user_message
            message_date = start_date + timedelta(days=day-1)
            message_time = message_date.strftime("%H:%M:%S")
            formatted_date = message_date.strftime("%Y-%m-%d")
            
            if day in day_map:
                message_data = day_map[day]
                user_message = message_data["text"]
                last_user_message = user_message
                last_message_data = message_data
                # Use the actual date/time from the message if available
                if message_data.get("date"):
                    formatted_date = message_data["date"]
                if message_data.get("time"):
                    try:
                        # Parse time like "06:19 AM" and convert to 24-hour format
                        parsed_time = datetime.strptime(message_data["time"], "%I:%M %p")
                        message_time = parsed_time.strftime("%H:%M:%S")
                    except ValueError:
                        # If parsing fails, keep the default time
                        pass
            elif last_message_data:
                # Reuse the last message for continuity
                user_message = last_message_data["text"]

            print(f"--- Day {day}: Rohan says: '{user_message}' ---")

            current_date = start_date + timedelta(days=day-1)
            message_time = current_date.strftime("%H:%M:%S")
            message_date = current_date.strftime("%Y-%m-%d")

            # Simulate the agent response to Rohan's message
            weekly_report = self.orchestrator.simulate_week(day // 7 + 1, user_message)

            # Add timestamp to conversation history
            for conv in self.orchestrator.chat_system.conversation_history:
                if 'date' not in conv:
                    conv['date'] = message_date
                    conv['time'] = message_time

            # Determine if Rohan should respond based on agent recommendations
            reply_needed = self._should_rohan_reply(weekly_report)
            
            if reply_needed:
                print(f"--- Rohan needs to respond to agent recommendations ---")
                # Generate a natural follow-up response from Rohan
                rohan_follow_up = self._generate_rohan_follow_up(weekly_report, user_message)
                print(f"--- Rohan responds: '{rohan_follow_up}' ---")
                
                # Simulate the agent's response to Rohan's follow-up
                follow_up_report = self.orchestrator.simulate_week(day // 7 + 1, rohan_follow_up)
                
                # Add the follow-up conversation to the data
                conversation_data.append({
                    "day": day,
                    "date": formatted_date,
                    "time": message_time,
                    "rohan_message": rohan_follow_up,
                    "agent_response": follow_up_report.get("agent_response", ""),
                    "reply_needed": False,  # Follow-up responses don't need further replies
                    "recommendations": follow_up_report.get("recommendations", []),
                    "suggested_action": follow_up_report.get("suggested_action", ""),
                    "rohan_follow_up": "",
                    "agent_follow_up_response": ""
                })
                
                # Update the main conversation data with follow-up info
                if len(conversation_data) >= 2:
                    conversation_data[-2]["rohan_follow_up"] = rohan_follow_up
                    conversation_data[-2]["agent_follow_up_response"] = follow_up_report.get("agent_response", "")

            next_action = self.planner.get_next_action(weekly_report)
            if next_action:
                weekly_report["suggested_action"] = next_action
                print(f"🧠 Planner suggestion: {next_action}")

            # Store conversation data for CSV export
            conversation_data.append({
                "day": day,
                "date": formatted_date,
                "time": message_time,
                "rohan_message": user_message,
                "agent_response": weekly_report.get("agent_response", ""),
                "reply_needed": reply_needed,
                "recommendations": weekly_report.get("recommendations", []),
                "suggested_action": weekly_report.get("suggested_action", ""),
                "rohan_follow_up": "",
                "agent_follow_up_response": ""
            })

            journey_data.append(weekly_report)

        print("🎉 Complete Journey Simulation finished!")

        # Analyze the journey
        journey_summary = self.analyzer.analyze(self.orchestrator.chat_system.get_conversation_history())

        # Export conversation data to CSV
        self._export_to_csv(conversation_data)
        
        # Export conversation history to JSON for decision tree analysis
        self._export_to_json(conversation_data)
        
        # Perform decision tree analysis
        decision_tree = self._analyze_decision_tree(conversation_data)
        
        return {
            "conversation_history": self.orchestrator.chat_system.get_conversation_history(),
            "journey_data": journey_data,
            "journey_summary": journey_summary,
            "conversation_data": conversation_data,
            "decision_tree": decision_tree,
        }

    def _should_rohan_reply(self, weekly_report: Dict) -> bool:
        """
        Determine if Rohan should respond based on agent recommendations.
        Returns True if the agent provided actionable recommendations that require follow-up.
        """
        recommendations = weekly_report.get("recommendations", [])
        
        # Check if there are actionable recommendations
        if not recommendations:
            return False
            
        # Look for keywords that suggest action is needed
        action_keywords = [
            "schedule", "book", "order", "call", "email", "confirm", "arrange",
            "test", "measure", "track", "monitor", "adjust", "change", "try",
            "implement", "start", "begin", "continue", "follow up", "review",
            "set up", "bring", "pencil in", "send", "add", "create", "plan",
            "consider", "aim for", "target", "goal", "focus on", "practice"
        ]
        
        # Time-related keywords that suggest scheduling or timing
        time_keywords = [
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "am", "pm", "morning", "afternoon", "evening", "tonight", "tomorrow",
            "next week", "this week", "daily", "weekly", "monthly", "recurring"
        ]
        
        # Specific action patterns
        action_patterns = [
            "let me know", "confirm", "reach out", "contact", "get back to",
            "follow up", "check in", "update", "report back", "keep track",
            "log", "record", "monitor", "observe", "watch for", "pay attention"
        ]
        
        for rec in recommendations:
            rec_text = rec.lower()
            
            # Check for actionable keywords
            if any(keyword in rec_text for keyword in action_keywords):
                return True
                
            # Check for time references (suggesting scheduling)
            if any(keyword in rec_text for keyword in time_keywords):
                return True
                
            # Check for action patterns
            if any(pattern in rec_text for pattern in action_patterns):
                return True
                
            # Check for specific quantities and measurements (suggesting actionable plans)
            if any(char.isdigit() for char in rec_text) and any(word in rec_text for word in ["mg", "min", "h", "hr", "cup", "g", "kcal", "am", "pm"]):
                return True
                
        return False

    def _generate_rohan_follow_up(self, weekly_report: Dict, user_message: str) -> str:
        """
        Generates a natural follow-up response from Rohan based on the agent's recommendations.
        """
        recommendations = weekly_report.get("recommendations", [])
        
        if not recommendations:
            return "Okay, I'll keep you updated."

        # Analyze the recommendations to generate contextually appropriate responses
        for rec in recommendations:
            rec_text = rec.lower()
            
            # Check for scheduling/appointment related content
            if any(word in rec_text for word in ["schedule", "appointment", "monday", "tuesday", "am", "pm"]):
                return "Perfect, that time works for me. I'll confirm and add it to my calendar."
            
            # Check for meal/nutrition related content
            if any(word in rec_text for word in ["meal", "diet", "nutrition", "breakfast", "eggs", "protein", "carbs"]):
                return "Got it! I'll share these details with my cook and let you know how it goes."
            
            # Check for exercise/workout related content
            if any(word in rec_text for word in ["exercise", "workout", "training", "recovery", "hrv", "sleep"]):
                return "Understood. I'll implement these changes and track my progress."
            
            # Check for travel/jetlag related content
            if any(word in rec_text for word in ["travel", "jetlag", "seoul", "flight", "timezone"]):
                return "Thanks for the travel tips! I'll start adjusting my schedule a few days before departure."
            
            # Check for monitoring/tracking related content
            if any(word in rec_text for word in ["monitor", "track", "measure", "log", "record"]):
                return "I'll make sure to track these metrics and report back to you."
            
            # Check for specific time-based recommendations
            if any(word in rec_text for word in ["daily", "weekly", "recurring", "routine"]):
                return "I'll set up this routine and stick to it consistently."
        
        # Default response for general recommendations
        return "Thanks for the guidance. I'll implement these suggestions and keep you posted on my progress."

    def _export_to_csv(self, conversation_data: List[Dict]):
        """
        Export conversation data to CSV format with chronological message sequence.
        Format: S.No., Message, Sender, Date, Time
        """
        csv_filename = "conversation_simulation.csv"
        
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['S.No.', 'Message', 'Sender', 'Date', 'Time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                message_counter = 1
                base_time = datetime.strptime("09:00:00", "%H:%M:%S")
                
                # Process each conversation entry chronologically
                for entry in conversation_data:
                    # Calculate realistic time gaps
                    current_time = base_time + timedelta(minutes=message_counter * 2)  # 2 min gap between messages
                    
                    # Add Rohan's initial message
                    clean_message = self._clean_message_for_csv(entry['rohan_message'])
                    writer.writerow({
                        'S.No.': message_counter,
                        'Message': clean_message,
                        'Sender': 'User',
                        'Date': entry['date'],
                        'Time': current_time.strftime("%H:%M:%S")
                    })
                    message_counter += 1
                    current_time += timedelta(minutes=1)  # 1 min gap for agent response
                    
                    # Add agent's response
                    if entry.get('agent_response'):
                        clean_agent_message = self._clean_message_for_csv(entry['agent_response'])
                        writer.writerow({
                            'S.No.': message_counter,
                            'Message': clean_agent_message,
                            'Sender': self._extract_agent_name(entry['agent_response']),
                            'Date': entry['date'],
                            'Time': current_time.strftime("%H:%M:%S")
                        })
                        message_counter += 1
                        current_time += timedelta(minutes=2)  # 2 min gap for follow-up
                    
                    # Add Rohan's follow-up if exists
                    if entry.get('rohan_follow_up'):
                        clean_follow_up = self._clean_message_for_csv(entry['rohan_follow_up'])
                        writer.writerow({
                            'S.No.': message_counter,
                            'Message': clean_follow_up,
                            'Sender': 'User',
                            'Date': entry['date'],
                            'Time': current_time.strftime("%H:%M:%S")
                        })
                        message_counter += 1
                        current_time += timedelta(minutes=1)  # 1 min gap for agent follow-up
                        
                        # Add agent's follow-up response
                        if entry.get('agent_follow_up_response'):
                            clean_follow_up_response = self._clean_message_for_csv(entry['agent_follow_up_response'])
                            writer.writerow({
                                'S.No.': message_counter,
                                'Message': clean_follow_up_response,
                                'Sender': self._extract_agent_name(entry['agent_follow_up_response']),
                                'Date': entry['date'],
                                'Time': current_time.strftime("%H:%M:%S")
                            })
                            message_counter += 1
                            current_time += timedelta(minutes=2)  # 2 min gap for next conversation
                    
            print(f"📊 Conversation data exported to {csv_filename} with {message_counter-1} messages")
            
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")
    
    def _clean_message_for_csv(self, message: str) -> str:
        """Clean message for CSV export by removing line breaks and extra spaces"""
        if not message:
            return ""
        
        # Remove line breaks and normalize spaces
        cleaned = message.replace('\n', ' ').replace('\r', ' ')
        cleaned = ' '.join(cleaned.split())  # Normalize multiple spaces to single space
        
        # Truncate if too long (CSV best practice)
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
        
        return cleaned
    
    def _extract_agent_name(self, response: str) -> str:
        """Extract agent name from response string"""
        if not response:
            return "Unknown"
        
        # Look for agent names in the response
        agent_names = ["Ruby", "Dr. Warren", "Advik", "Carla", "Rachel", "Neel"]
        
        for agent in agent_names:
            if agent in response:
                return agent
        
        return "Unknown"
    
    def _export_to_json(self, conversation_data: List[Dict]):
        """
        Export conversation history to JSON format for decision tree analysis.
        """
        json_filename = "data/conversation_history.json"
        
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Prepare conversation history for JSON export
            conversation_history = []
            message_counter = 1
            base_time = datetime.strptime("09:00:00", "%H:%M:%S")
            
            for entry in conversation_data:
                # Calculate realistic time gaps
                current_time = base_time + timedelta(minutes=message_counter * 2)
                
                # Add Rohan's initial message
                conversation_history.append({
                    'S.No.': message_counter,
                    'Message': entry['rohan_message'],
                    'Sender': 'User',
                    'Date': entry['date'],
                    'Time': current_time.strftime("%H:%M:%S"),
                    'Day': entry['day'],
                    'Reply_Needed': entry['reply_needed'],
                    'Recommendations': entry.get('recommendations', []),
                    'Message_Type': 'User_Initial'
                })
                message_counter += 1
                current_time += timedelta(minutes=1)
                
                # Add agent's response
                if entry.get('agent_response'):
                    conversation_history.append({
                        'S.No.': message_counter,
                        'Message': entry['agent_response'],
                        'Sender': self._extract_agent_name(entry['agent_response']),
                        'Date': entry['date'],
                        'Time': current_time.strftime("%H:%M:%S"),
                        'Day': entry['day'],
                        'Reply_Needed': False,
                        'Recommendations': entry.get('recommendations', []),
                        'Message_Type': 'Agent_Response'
                    })
                    message_counter += 1
                    current_time += timedelta(minutes=2)
                
                # Add Rohan's follow-up if exists
                if entry.get('rohan_follow_up'):
                    conversation_history.append({
                        'S.No.': message_counter,
                        'Message': entry['rohan_follow_up'],
                        'Sender': 'User',
                        'Date': entry['date'],
                        'Time': current_time.strftime("%H:%M:%S"),
                        'Day': entry['day'],
                        'Reply_Needed': False,
                        'Recommendations': [],
                        'Message_Type': 'User_Follow_Up'
                    })
                    message_counter += 1
                    current_time += timedelta(minutes=1)
                    
                    # Add agent's follow-up response
                    if entry.get('agent_follow_up_response'):
                        conversation_history.append({
                            'S.No.': message_counter,
                            'Message': entry['agent_follow_up_response'],
                            'Sender': self._extract_agent_name(entry['agent_follow_up_response']),
                            'Date': entry['date'],
                            'Time': current_time.strftime("%H:%M:%S"),
                            'Day': entry['day'],
                            'Reply_Needed': False,
                            'Recommendations': [],
                            'Message_Type': 'Agent_Follow_Up'
                        })
                        message_counter += 1
                        current_time += timedelta(minutes=2)
            
            # Write to JSON file
            with open(json_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump({
                    'total_messages': len(conversation_history),
                    'simulation_period': f"{len(conversation_data)} days",
                    'conversation_history': conversation_history,
                    'export_timestamp': datetime.now().isoformat()
                }, jsonfile, indent=2, ensure_ascii=False)
                
            print(f"📄 Conversation history exported to {json_filename}")
            
        except Exception as e:
            print(f"❌ Failed to export JSON: {e}")
    
    def _analyze_decision_tree(self, conversation_data: List[Dict]) -> Dict:
        """
        Analyze conversation data to identify decision tree structure.
        Identifies branching paths and decision points in the conversation.
        """
        decision_tree = {
            'total_decision_points': 0,
            'decision_points': [],
            'branching_paths': [],
            'conversation_flow': []
        }
        
        decision_counter = 1
        
        for entry in conversation_data:
            if entry.get('reply_needed') and entry.get('recommendations'):
                # This is a decision point
                decision_point = {
                    'id': decision_counter,
                    'day': entry['day'],
                    'date': entry['date'],
                    'time': entry['time'],
                    'user_message': entry['rohan_message'],
                    'agent_response': entry['agent_response'],
                    'agent_name': self._extract_agent_name(entry['agent_response']),
                    'recommendations': entry['recommendations'],
                    'possible_paths': self._identify_possible_paths(entry['recommendations']),
                    'chosen_path': entry.get('rohan_follow_up', 'No follow-up'),
                    'outcome': entry.get('agent_follow_up_response', 'No outcome')
                }
                
                decision_tree['decision_points'].append(decision_point)
                decision_tree['total_decision_points'] += 1
                decision_counter += 1
                
                # Add to conversation flow
                decision_tree['conversation_flow'].append({
                    'type': 'decision_point',
                    'data': decision_point
                })
            else:
                # Regular conversation flow
                decision_tree['conversation_flow'].append({
                    'type': 'conversation',
                    'data': {
                        'day': entry['day'],
                        'date': entry['date'],
                        'time': entry['time'],
                        'user_message': entry['rohan_message'],
                        'agent_response': entry.get('agent_response', ''),
                        'agent_name': self._extract_agent_name(entry.get('agent_response', ''))
                    }
                })
        
        # Identify branching patterns
        decision_tree['branching_paths'] = self._identify_branching_patterns(decision_tree['decision_points'])
        
        return decision_tree
    
    def _identify_possible_paths(self, recommendations: List[str]) -> List[str]:
        """
        Identify possible paths from agent recommendations.
        """
        paths = []
        
        for rec in recommendations:
            rec_text = rec.lower()
            
            # Identify different types of recommendations
            if any(word in rec_text for word in ["sleep", "rest", "bed"]):
                paths.append("Sleep/Rest")
            elif any(word in rec_text for word in ["meditate", "mindfulness", "breathing"]):
                paths.append("Meditation/Mindfulness")
            elif any(word in rec_text for word in ["read", "book", "study"]):
                paths.append("Reading/Study")
            elif any(word in rec_text for word in ["exercise", "workout", "training"]):
                paths.append("Exercise/Workout")
            elif any(word in rec_text for word in ["diet", "nutrition", "meal"]):
                paths.append("Diet/Nutrition")
            elif any(word in rec_text for word in ["schedule", "appointment", "consultation"]):
                paths.append("Medical Consultation")
            elif any(word in rec_text for word in ["monitor", "track", "measure"]):
                paths.append("Monitoring/Tracking")
            elif any(word in rec_text for word in ["travel", "jetlag", "adjust"]):
                paths.append("Travel Adjustment")
            else:
                paths.append("General Health Management")
        
        return list(set(paths))  # Remove duplicates
    
    def _identify_branching_patterns(self, decision_points: List[Dict]) -> List[Dict]:
        """
        Identify patterns in how conversations branch at decision points.
        """
        branching_patterns = []
        
        for i, dp in enumerate(decision_points):
            if i > 0:
                # Compare with previous decision point
                prev_dp = decision_points[i-1]
                
                pattern = {
                    'from_decision': prev_dp['id'],
                    'to_decision': dp['id'],
                    'days_between': dp['day'] - prev_dp['day'],
                    'path_continuity': self._analyze_path_continuity(prev_dp, dp),
                    'health_trend': self._analyze_health_trend(prev_dp, dp)
                }
                
                branching_patterns.append(pattern)
        
        return branching_patterns
    
    def _analyze_path_continuity(self, prev_decision: Dict, curr_decision: Dict) -> str:
        """
        Analyze if the chosen path continues or changes direction.
        """
        prev_paths = set(prev_decision['possible_paths'])
        curr_paths = set(curr_decision['possible_paths'])
        
        if prev_paths.intersection(curr_paths):
            return "Path Continuation"
        else:
            return "Path Change"
    
    def _analyze_health_trend(self, prev_decision: Dict, curr_decision: Dict) -> str:
        """
        Analyze health trend between decision points.
        """
        days_between = curr_decision['day'] - prev_decision['day']
        
        if days_between <= 3:
            return "Immediate Follow-up"
        elif days_between <= 7:
            return "Weekly Review"
        else:
            return "Long-term Planning"
