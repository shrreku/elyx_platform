import json
import os
import sys
from simulation.complete_journey import CompleteJourney

def main():
    # Set a dummy API key for simulation purposes
    os.environ["OPENAI_API_KEY"] = "dummy_key"
    
    # Check if XML file is provided as command line argument
    xml_file = None
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    
    # If no XML file provided, use the default test_messages.xml
    if not xml_file:
        xml_file = "test_messages.xml"
    
    # Check if the XML file exists
    if not os.path.exists(xml_file):
        print(f"❌ XML file '{xml_file}' not found!")
        print("Usage: python run_simulation.py [xml_file]")
        print("If no file is specified, 'test_messages.xml' will be used.")
        return
    
    print(f"🎬 Starting Complete Journey Simulation with XML file: {xml_file}")
    
    try:
        # Read the XML file
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Initialize the CompleteJourney with the XML content
        complete_journey = CompleteJourney(xml_content=xml_content, num_months=8)

        # Run the simulation
        results = complete_journey.run()
        
        print("\n" + "="*60)
        print("🎯 SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display conversation summary
        conversation_data = results.get("conversation_data", [])
        decision_tree = results.get("decision_tree", {})
        
        print(f"\n📊 CONVERSATION SUMMARY:")
        print(f"   • Total Days Simulated: {len(conversation_data)}")
        print(f"   • Total Decision Points: {decision_tree.get('total_decision_points', 0)}")
        print(f"   • Conversation Flow: {len(decision_tree.get('conversation_flow', []))} interactions")
        
        # Display decision tree insights
        if decision_tree.get('decision_points'):
            print(f"\n🌳 DECISION TREE ANALYSIS:")
            print(f"   • Decision Points Found: {len(decision_tree['decision_points'])}")
            
            for i, dp in enumerate(decision_tree['decision_points'][:5], 1):  # Show first 5
                print(f"     {i}. Day {dp['day']} - {dp['agent_name']}: {len(dp['possible_paths'])} possible paths")
                print(f"        Chosen: {dp['chosen_path'][:50]}...")
            
            if len(decision_tree['decision_points']) > 5:
                print(f"     ... and {len(decision_tree['decision_points']) - 5} more decision points")
        
        # Display branching patterns
        if decision_tree.get('branching_paths'):
            print(f"\n🔄 BRANCHING PATTERNS:")
            print(f"   • Path Continuations: {sum(1 for p in decision_tree['branching_paths'] if p['path_continuity'] == 'Path Continuation')}")
            print(f"   • Path Changes: {sum(1 for p in decision_tree['branching_paths'] if p['path_continuity'] == 'Path Change')}")
        
        print(f"\n📁 OUTPUT FILES GENERATED:")
        print(f"   • conversation_simulation.csv - Chronological message sequence")
        print(f"   • data/conversation_history.json - Decision tree analysis data")
        
        print(f"\n🎉 Ready for decision tree visualization and workflow analysis!")
        print("="*60)

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
