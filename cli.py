# cli.py
from inference_dag import create_dag, log_event
from config import THRESHOLD_CONFIDENCE
import os

os.makedirs("logs", exist_ok=True)

dag_app = create_dag()

def run_cli():
    print("🤖 Self-Healing Classifier Ready!")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Review: ").strip()
        if user_input.lower() == "quit":
            print("👋 Goodbye!")
            break
        if not user_input:
            continue

        # Start the state as a dictionary
        initial_state = {
            "text": user_input,
            "raw_output": None,
            "predicted_label": None,
            "confidence": 0.0,
            "final_label": None,
            "needs_fallback": False,
            "user_feedback": None
        }

        # Run DAG
        try:
            result = dag_app.invoke(initial_state)
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            continue

        # Output final answer
        print(f"✅ Final Label: {result.get('final_label', 'Unknown')}")
        if result.get("confidence", 1.0) < THRESHOLD_CONFIDENCE:
            print("(Corrected via user clarification)")
        print("-" * 50)

        # Log final decision
        log_event("final_decision", {
            "input": user_input,
            "final_label": result.get("final_label"),
            "confidence": result.get("confidence")
        })

if __name__ == "__main__":
    run_cli()