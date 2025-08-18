#!/usr/bin/env python3
"""
Run DecisionTreeVisualizer with the enriched conversation file generated
by simulation/connector_to_visualizer.py.

This loader uses importlib to load the visualizer module by path to avoid
package import issues when running as a script.
"""

import importlib.util
from pathlib import Path

ENRICHED_PATH = "data/conversation_history_for_visualizer.json"
REPORT_PATH = "decision_tree_analysis_report_from_enriched.txt"
VISUALIZER_PATH = Path(__file__).resolve().parent / "decision_tree_visualizer.py"

def load_visualizer_class(path):
    spec = importlib.util.spec_from_file_location("decision_tree_visualizer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "DecisionTreeVisualizer")

def main():
    DecisionTreeVisualizer = load_visualizer_class(VISUALIZER_PATH)
    visualizer = DecisionTreeVisualizer(conversation_data_path=ENRICHED_PATH)
    if not visualizer.load_conversation_data():
        print("Failed to load enriched conversation data.")
        return

    decision_tree = visualizer.analyze_decision_tree()
    if not decision_tree:
        print("No decision tree produced.")
        return

    report = visualizer.generate_summary_report()
    print(report)
    visualizer.save_analysis_report(output_path=REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
