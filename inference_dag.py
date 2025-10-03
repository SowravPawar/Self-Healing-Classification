# inference_dag.py
from langgraph.graph import StateGraph, END
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import *
from logging_setup import log_event
import torch

# Load model once
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    top_k=None,
    function_to_apply="softmax"
)


# Define the structure of our state as a TypedDict-like schema
class State(dict):
    text: str
    raw_output: list
    predicted_label: str
    confidence: float
    final_label: str
    needs_fallback: bool
    user_feedback: str


# Node 1: Inference
def inference_node(state: State):
    print("[InferenceNode] Running classification...")
    result = classifier(state["text"])

    # Extract scores using config labels
    pos_score = None
    neg_score = None

    for item in result[0]:
        label = item['label']
        score = item['score']

        if label == "Positive" or (label == "LABEL_1" and hasattr(model.config, 'id2label')):
            pos_score = score
        elif label == "Negative" or (label == "LABEL_0" and hasattr(model.config, 'id2label')):
            neg_score = score

    # Fallback: use index if names missing
    if pos_score is None and neg_score is None:
        if len(result[0]) >= 2:
            # Assume order: [negative, positive] based on training
            neg_score = result[0][0]['score']
            pos_score = result[0][1]['score']
        else:
            raise ValueError("Could not extract two class scores from model output")

    # Decide prediction
    if pos_score > neg_score:
        predicted_label = "Positive"
        confidence = pos_score
    else:
        predicted_label = "Negative"
        confidence = neg_score

    log_event("prediction", f"Predicted: {predicted_label}, Confidence: {confidence:.2f}")
    return {
        "raw_output": result,
        "predicted_label": predicted_label,
        "confidence": confidence
    }


# Node 2: Confidence Check
def confidence_check_node(state: State):
    conf = state["confidence"]
    print(f"[ConfidenceCheckNode] Confidence: {conf:.2f}")
    if conf < THRESHOLD_CONFIDENCE:
        log_event("fallback_triggered", f"Low confidence ({conf:.2f} < {THRESHOLD_CONFIDENCE})")
        return {"needs_fallback": True}
    else:
        return {
            "needs_fallback": False,
            "final_label": state["predicted_label"]
        }


# Node 3: Fallback – Ask User!
def fallback_node(state: State):
    print("[FallbackNode] Could you clarify your intent? Was this a negative review?")
    feedback = input("You: ").strip().lower()

    if any(word in feedback for word in ["yes", "negative", "bad", "hate"]):
        final_label = "Negative"
    elif any(word in feedback for word in ["no", "positive", "good", "love"]):
        final_label = "Positive"
    else:
        final_label = state["predicted_label"]  # fallback to original guess

    log_event("user_interaction", f"User said: {feedback}, Final decision: {final_label}")
    return {"final_label": final_label, "user_feedback": feedback}


# Build DAG
def create_dag():
    workflow = StateGraph(State)

    workflow.add_node("inference", inference_node)
    workflow.add_node("confidence_check", confidence_check_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("inference")
    workflow.add_edge("inference", "confidence_check")

    def should_fallback(state: State):
        return "fallback" if state["needs_fallback"] else END

    workflow.add_conditional_edges(
        "confidence_check",
        should_fallback,
        {
            "fallback": "fallback",
            END: END
        }
    )
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    return app