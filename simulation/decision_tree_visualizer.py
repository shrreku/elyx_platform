#!/usr/bin/env python3
"""
Decision Tree Visualizer for Elyx Platform Conversation Analysis
Analyzes conversation data to identify decision points and branching paths.
Enhanced to include decision reasons and configurable journey duration (months).
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import re
from datetime import datetime, timedelta

class DecisionTreeVisualizer:
    """
    Visualizes decision tree structure from conversation analysis.
    Identifies branching paths, decision points, and conversation flow.
    """

    def __init__(self, conversation_data_path: str = "data/conversation_history.json", journey_months: int = 8):
        """
        conversation_data_path: path to the conversation JSON
        journey_months: duration of the journey in months (default 8). Used for summary reporting / scaling.
        """
        self.conversation_data_path = conversation_data_path
        self.conversation_data = None
        self.decision_tree = None
        self.journey_months = journey_months

    def load_conversation_data(self) -> bool:
        """Load conversation data from JSON file."""
        try:
            if os.path.exists(self.conversation_data_path):
                with open(self.conversation_data_path, 'r', encoding='utf-8') as f:
                    self.conversation_data = json.load(f)
                total = self.conversation_data.get('total_messages', 'Unknown')
                print(f"✅ Loaded conversation data: {total} messages")
                return True
            else:
                print(f"❌ Conversation data file not found: {self.conversation_data_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load conversation data: {e}")
            return False

    def analyze_decision_tree(self) -> Dict:
        """Analyze conversation data to build decision tree structure."""
        if not self.conversation_data:
            print("❌ No conversation data loaded")
            return {}

        conversation_history = self.conversation_data.get('conversation_history', [])

        decision_tree = {
            'total_decision_points': 0,
            'decision_points': [],
            'branching_paths': [],
            'conversation_flow': [],
            'health_domains': {},
            'agent_interactions': {},
            'temporal_patterns': {},
            'journey_months': self.journey_months
        }

        decision_counter = 1
        current_conversation_flow = []

        # Normalize keys for safety (allow both PascalCase and snake_case)
        normalized_history = []
        for idx, message in enumerate(conversation_history):
            Sender = message.get('Sender') or message.get('sender') or 'Unknown'
            Message = message.get('Message') or message.get('message') or ''
            Date = message.get('Date') or message.get('date') or ''
            Time = message.get('Time') or message.get('time') or ''
            Day = message.get('Day') or message.get('day') or (idx // 10) + 1
            Reply_Needed = message.get('Reply_Needed') if 'Reply_Needed' in message else message.get('reply_needed', None)
            Recommendations = message.get('Recommendations') if 'Recommendations' in message else message.get('recommendations', [])

            normalized_history.append({
                'S.No.': message.get('S.No.') or message.get('s_no') or idx + 1,
                'Sender': Sender,
                'Message': Message,
                'Day': Day,
                'Date': Date,
                'Time': Time,
                'Reply_Needed': Reply_Needed,
                'Recommendations': Recommendations
            })

        # Iterate normalized history to detect decision points and capture context
        for i, message in enumerate(normalized_history):
            # Track conversation flow (short preview)
            current_conversation_flow.append({
                'S.No.': message['S.No.'],
                'Sender': message['Sender'],
                'Message': (message['Message'][:100] + "...") if len(message['Message']) > 100 else message['Message'],
                'Day': message['Day'],
                'Date': message['Date'],
                'Time': message['Time']
            })

            # Decision flagging: existing flags or presence of recommendations / heuristics
            # Prefer explicit Recommendations field, otherwise extract heuristics from message text
            recs_field = message.get('Recommendations') or message.get('recommendations') or []
            recs_from_text = []

            if not recs_field and isinstance(message.get('Message'), str) and message.get('Message').strip():
                # Extract numbered or bullet lines and lines containing key action phrases
                lines = re.split(r'\\r?\\n', message.get('Message'))
                for line in lines:
                    trimmed = line.strip()
                    if re.search(r'^[0-9]+\\.|^\\u2022|^\\-|next steps|recommend|action plan|what to do', trimmed, re.IGNORECASE):
                        cleaned = re.sub(r'^[0-9]+\\.|^\\u2022|^\\-\\s*', '', trimmed).strip()
                        if len(cleaned) > 6:
                            recs_from_text.append(cleaned)

                # Also pull sentences that contain common action verbs as possible recommendations
                sentences = re.split(r'[\\.\\!\\?]\\s+', message.get('Message'))
                action_verbs = ['schedule','check','measure','hydrate','log','track','monitor','add','remove','snack','eat','sleep','rest','walk','exercise','consult','appointment','follow up','follow-up']
                for s in sentences:
                    s_clean = s.strip()
                    for v in action_verbs:
                        # use re.escape for the verb to be safe
                        if re.search(r'\\b' + re.escape(v) + r'\\b', s_clean, re.IGNORECASE):
                            if len(s_clean) > 6 and s_clean not in recs_from_text:
                                recs_from_text.append(s_clean)
                            # once we matched this sentence to one verb, stop checking other verbs
                            break

            # Final recommendations list
            recommendations = recs_field if recs_field else recs_from_text
            has_recs = len(recommendations) > 0

            # Prefer explicit Reply_Needed; if absent, infer from question/ask-words or if recommendations exist
            reply_needed = message.get('Reply_Needed') is True
            if message.get('Reply_Needed') is None:
                if re.search(r'\\?|\\bhelp\\b|\\bshould\\b|\\bwhen\\b|\\bhow\\b|\\badvise\\b|\\bneed\\b', message.get('Message', ''), re.IGNORECASE) or has_recs:
                    reply_needed = True
                else:
                    reply_needed = False

            # If recommendations are provided directly, mark as decision point candidate
            if reply_needed and has_recs:
                context = current_conversation_flow[-6:]  # last few messages as context
                decision_point = self._create_decision_point(message, decision_counter, context)
                decision_tree['decision_points'].append(decision_point)
                decision_tree['total_decision_points'] += 1
                decision_counter += 1

                # Add to conversation flow as decision point
                decision_tree['conversation_flow'].append({
                    'type': 'decision_point',
                    'data': decision_point,
                    'conversation_context': context
                })

                # Reset conversation flow for next decision point
                current_conversation_flow = []
            else:
                # Regular conversation flow
                decision_tree['conversation_flow'].append({
                    'type': 'conversation',
                    'data': message
                })

        # Analyze patterns
        decision_tree['branching_paths'] = self._analyze_branching_patterns(decision_tree['decision_points'])
        decision_tree['health_domains'] = self._analyze_health_domains(decision_tree['decision_points'])
        decision_tree['agent_interactions'] = self._analyze_agent_interactions(decision_tree['decision_points'])
        decision_tree['temporal_patterns'] = self._analyze_temporal_patterns(decision_tree['decision_points'])

        self.decision_tree = decision_tree
        return decision_tree

    def _create_decision_point(self, message: Dict, decision_id: int, context: Optional[List[Dict]] = None) -> Dict:
        """Create a decision point from message data, including reasons derived from conversation analyzer."""
        recommendations = message.get('Recommendations') or message.get('recommendations') or []
        context = context or []

        dp = {
            'id': decision_id,
            'day': message.get('Day'),
            'date': message.get('Date'),
            'time': message.get('Time'),
            'user_message': message.get('Message'),
            'agent_name': message.get('Sender') if message.get('Sender') != 'User' else 'Unknown',
            'recommendations': recommendations,
            'possible_paths': self._identify_possible_paths(recommendations),
            'health_domain': self._categorize_health_domain(recommendations),
            'urgency_level': self._assess_urgency_level(recommendations),
            'complexity': self._assess_complexity(recommendations)
        }

        # Determine reasons using analyzer heuristics
        reasons = self._determine_decision_reasons(message, context, decision_id)
        dp['reasons'] = reasons

        return dp

    def _determine_decision_reasons(self, message: Dict, context: List[Dict], decision_id: int) -> List[str]:
        """
        Analyze the message + nearby context and produce a list of reasoning statements
        explaining why this message was flagged as a decision point.
        """
        reasons = []
        msg_text = message.get('Message', '') or ''
        recs = message.get('Recommendations') or message.get('recommendations') or []

        # 1) Explicit flags
        if message.get('Reply_Needed') is True:
            reasons.append("Flagged as requiring a reply (Reply_Needed).")

        # 2) Recommendations present
        if recs:
            reasons.append(f"Agent provided {len(recs)} recommendation(s): " + "; ".join(
                (r if isinstance(r, str) else str(r)) for r in recs[:3]
            ))

        # 3) Question or ask-words
        if re.search(r'\?|\bhelp\b|\bshould\b|\bwhen\b|\bhow\b|\bcan\b|\bcould\b', msg_text, re.IGNORECASE):
            reasons.append("User asked a question or requested help.")

        # 4) Urgency keywords
        if re.search(r'\b(immediate|urgent|emergency|asap|now|critical)\b', msg_text, re.IGNORECASE) or \
           any(re.search(r'\b' + k + r'\b', " ".join(recs), re.IGNORECASE) for k in ["immediate", "urgent", "emergency", "now", "asap"]):
            reasons.append("Contains urgency indicators triggering a higher-priority decision.")

        # 5) Contextual continuation (previous agent recommendations)
        last_agent_recs = []
        for c in reversed(context):
            if c.get('Sender') and c.get('Sender') != 'User' and c.get('Message'):
                # look for recommendation-like lines in agent messages
                if re.search(r'\b(recommend|next steps|plan|consider|should|aim|schedule|appointment)\b', c.get('Message'), re.IGNORECASE):
                    last_agent_recs.append(c.get('Message'))
            if len(last_agent_recs) >= 2:
                break
        if last_agent_recs:
            reasons.append("Follow-up to prior agent guidance: " + (last_agent_recs[0][:140] + ("..." if len(last_agent_recs[0]) > 140 else "")))

        # 6) Data-driven triggers (if message mentions metrics)
        if re.search(r'\b(hrv|glucose|cgm|sleep|heart rate|blood sugar|hr|glucose)\b', msg_text, re.IGNORECASE):
            reasons.append("Decision driven by monitored metrics mentioned in the message (e.g., HRV, glucose).")

        # 7) Fallback reason if none found
        if not reasons:
            reasons.append("Heuristic detection: message matched decision-point patterns (reply needed / recommendations / question).")

        # Deduplicate and keep the most relevant (shorten)
        seen = set()
        cleaned = []
        for r in reasons:
            if r not in seen:
                cleaned.append(r)
                seen.add(r)
        return cleaned

    def _identify_possible_paths(self, recommendations: List[str]) -> List[str]:
        """Identify possible paths from agent recommendations."""
        paths = []

        for rec in recommendations:
            rec_text = rec.lower()

            # Health domain categorization
            if any(word in rec_text for word in ["sleep", "rest", "bed", "hrv", "recovery"]):
                paths.append("Sleep & Recovery")
            elif any(word in rec_text for word in ["meditate", "mindfulness", "breathing", "stress"]):
                paths.append("Mental Health & Stress Management")
            elif any(word in rec_text for word in ["read", "book", "study", "cognitive"]):
                paths.append("Cognitive Health & Learning")
            elif any(word in rec_text for word in ["exercise", "workout", "training", "cardio"]):
                paths.append("Physical Activity & Exercise")
            elif any(word in rec_text for word in ["diet", "nutrition", "meal", "glucose", "protein"]):
                paths.append("Nutrition & Diet Management")
            elif any(word in rec_text for word in ["schedule", "appointment", "consultation", "doctor"]):
                paths.append("Medical Consultation & Care")
            elif any(word in rec_text for word in ["monitor", "track", "measure", "data"]):
                paths.append("Health Monitoring & Tracking")
            elif any(word in rec_text for word in ["travel", "jetlag", "adjust", "routine"]):
                paths.append("Lifestyle & Routine Adjustment")
            else:
                paths.append("General Health Management")

        return list(dict.fromkeys(paths))  # preserve order, remove duplicates

    def _categorize_health_domain(self, recommendations: List[str]) -> str:
        """Categorize the health domain of recommendations."""
        paths = self._identify_possible_paths(recommendations)

        if len(paths) == 1:
            return paths[0]
        elif len(paths) <= 3:
            return "Multi-Domain"
        else:
            return "Complex Multi-Domain"

    def _assess_urgency_level(self, recommendations: List[str]) -> str:
        """Assess the urgency level of recommendations."""
        rec_text = " ".join(recommendations).lower()

        urgent_keywords = ["immediate", "urgent", "emergency", "critical", "now", "asap"]
        high_keywords = ["schedule", "appointment", "consultation", "test", "measure"]
        moderate_keywords = ["consider", "plan", "prepare", "adjust", "modify"]

        if any(keyword in rec_text for keyword in urgent_keywords):
            return "High Urgency"
        elif any(keyword in rec_text for keyword in high_keywords):
            return "Medium Urgency"
        elif any(keyword in rec_text for keyword in moderate_keywords):
            return "Low Urgency"
        else:
            return "Information Only"

    def _assess_complexity(self, recommendations: List[str]) -> str:
        """Assess the complexity of recommendations."""
        if len(recommendations) <= 2:
            return "Simple"
        elif len(recommendations) <= 4:
            return "Moderate"
        else:
            return "Complex"

    def _analyze_branching_patterns(self, decision_points: List[Dict]) -> List[Dict]:
        """Analyze patterns in how conversations branch at decision points."""
        branching_patterns = []

        for i, dp in enumerate(decision_points):
            if i > 0:
                prev_dp = decision_points[i-1]

                pattern = {
                    'from_decision': prev_dp['id'],
                    'to_decision': dp['id'],
                    'days_between': (dp.get('day') or 0) - (prev_dp.get('day') or 0),
                    'domain_transition': f"{prev_dp.get('health_domain')} → {dp.get('health_domain')}",
                    'urgency_change': f"{prev_dp.get('urgency_level')} → {dp.get('urgency_level')}",
                    'complexity_change': f"{prev_dp.get('complexity')} → {dp.get('complexity')}"
                }

                branching_patterns.append(pattern)

        return branching_patterns

    def _analyze_health_domains(self, decision_points: List[Dict]) -> Dict:
        """Analyze distribution of health domains across decision points."""
        domain_counts = {}
        domain_urgency = {}

        for dp in decision_points:
            domain = dp.get('health_domain', 'Unknown')
            urgency = dp.get('urgency_level', 'Unknown')

            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            if domain not in domain_urgency:
                domain_urgency[domain] = []
            domain_urgency[domain].append(urgency)

        return {
            'domain_distribution': domain_counts,
            'domain_urgency_patterns': domain_urgency
        }

    def _analyze_agent_interactions(self, decision_points: List[Dict]) -> Dict:
        """Analyze agent interaction patterns."""
        agent_counts = {}
        agent_domains = {}

        for dp in decision_points:
            agent = dp.get('agent_name', 'Unknown')

            agent_counts[agent] = agent_counts.get(agent, 0) + 1

            if agent not in agent_domains:
                agent_domains[agent] = []
            agent_domains[agent].append(dp.get('health_domain', 'Unknown'))

        return {
            'agent_frequency': agent_counts,
            'agent_specializations': agent_domains
        }

    def _analyze_temporal_patterns(self, decision_points: List[Dict]) -> Dict:
        """Analyze temporal patterns in decision making."""
        if len(decision_points) < 2:
            return {}

        intervals = []
        weekly_patterns = {}

        for i in range(1, len(decision_points)):
            interval = (decision_points[i].get('day') or 0) - (decision_points[i-1].get('day') or 0)
            intervals.append(interval)

            week = (decision_points[i].get('day') or 0) // 7
            if week not in weekly_patterns:
                weekly_patterns[week] = 0
            weekly_patterns[week] += 1

        return {
            'average_interval_days': sum(intervals) / len(intervals) if intervals else 0,
            'interval_distribution': {
                '1-3 days': sum(1 for x in intervals if x <= 3),
                '4-7 days': sum(1 for x in intervals if 4 <= x <= 7),
                '8+ days': sum(1 for x in intervals if x > 7)
            },
            'weekly_decision_frequency': weekly_patterns
        }

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of the decision tree analysis."""
        if not self.decision_tree:
            return "No decision tree data available"

        report = []
        report.append("=" * 80)
        report.append("🌳 ELYX PLATFORM DECISION TREE ANALYSIS REPORT")
        report.append("=" * 80)

        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS:")
        report.append(f"   • Total Decision Points: {self.decision_tree['total_decision_points']}")
        report.append(f"   • Total Conversation Flow: {len(self.decision_tree['conversation_flow'])} interactions")

        # Use conversation_data.simulation_period if present, otherwise use journey_months
        sim_period = self.conversation_data.get('simulation_period') or f"{self.journey_months} months"
        report.append(f"   • Simulation Period: {sim_period}")

        # Decision point analysis
        if self.decision_tree['decision_points']:
            report.append(f"\n🎯 DECISION POINT ANALYSIS:")
            report.append(f"   • Decision Points Found: {len(self.decision_tree['decision_points'])}")

            # Show top decision points
            for i, dp in enumerate(self.decision_tree['decision_points'][:8], 1):
                report.append(f"     {i}. Day {dp.get('day')} - {dp.get('agent_name')}")
                report.append(f"        Domain: {dp.get('health_domain')}")
                report.append(f"        Urgency: {dp.get('urgency_level')}")
                report.append(f"        Complexity: {dp.get('complexity')}")
                report.append(f"        Possible Paths: {', '.join(dp.get('possible_paths') or [])}")
                # Reasons (short)
                if dp.get('reasons'):
                    report.append(f"        Reasons:")
                    for r in dp.get('reasons'):
                        report.append(f"           - {r}")
                report.append("")

            if len(self.decision_tree['decision_points']) > 8:
                report.append(f"     ... and {len(self.decision_tree['decision_points']) - 8} more decision points")

        # Health domain analysis
        if self.decision_tree['health_domains']:
            report.append(f"\n🏥 HEALTH DOMAIN ANALYSIS:")
            domains = self.decision_tree['health_domains']['domain_distribution']
            for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   • {domain}: {count} decision points")

        # Agent interaction analysis
        if self.decision_tree['agent_interactions']:
            report.append(f"\n🤖 AGENT INTERACTION ANALYSIS:")
            agents = self.decision_tree['agent_interactions']['agent_frequency']
            for agent, count in sorted(agents.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   • {agent}: {count} interactions")

        # Temporal patterns
        if self.decision_tree['temporal_patterns']:
            report.append(f"\n⏰ TEMPORAL PATTERN ANALYSIS:")
            temp = self.decision_tree['temporal_patterns']
            report.append(f"   • Average Interval: {temp['average_interval_days']:.1f} days")
            report.append(f"   • Interval Distribution:")
            for interval, count in temp['interval_distribution'].items():
                report.append(f"     - {interval}: {count} occurrences")

        # Branching patterns
        if self.decision_tree['branching_paths']:
            report.append(f"\n🔄 BRANCHING PATTERN ANALYSIS:")
            report.append(f"   • Total Branching Patterns: {len(self.decision_tree['branching_paths'])}")

            # Analyze domain transitions
            transitions = {}
            for pattern in self.decision_tree['branching_paths']:
                transition = pattern['domain_transition']
                transitions[transition] = transitions.get(transition, 0) + 1

            report.append(f"   • Top Domain Transitions:")
            for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"     - {transition}: {count} times")

        report.append(f"\n📁 DATA FILES:")
        report.append(f"   • conversation_simulation.csv - Chronological message sequence")
        report.append(f"   • {self.conversation_data_path} - Decision tree analysis data")

        report.append(f"\n🎉 Ready for decision tree visualization and workflow analysis!")
        report.append("=" * 80)

        return "\n".join(report)

    def save_analysis_report(self, output_path: str = "decision_tree_analysis_report.txt"):
        """Save the analysis report to a text file."""
        try:
            report = self.generate_summary_report()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Analysis report saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save analysis report: {e}")

def main():
    """Main function to run decision tree analysis."""
    print("🌳 Starting Decision Tree Analysis...")

    # Default journey_months can be overridden by environment or by passing an argument manually
    visualizer = DecisionTreeVisualizer()

    if visualizer.load_conversation_data():
        decision_tree = visualizer.analyze_decision_tree()

        if decision_tree:
            print("\n" + visualizer.generate_summary_report())
            visualizer.save_analysis_report()
        else:
            print("❌ Failed to analyze decision tree")
    else:
        print("❌ Failed to load conversation data")

if __name__ == "__main__":
    main()
