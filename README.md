# ATG Technical Assignment: Self-Healing Classification DAG  
**Machine Learning Intern Deliverable**

A LangGraph-based classification pipeline that performs sentiment analysis with a **self-healing fallback mechanism**. When prediction confidence is low, the system asks for human clarification — ensuring reliability over blind automation.

Built using:
- Fine-tuned `distilbert-base-uncased` on IMDB dataset
- LoRA (PEFT) for efficient training
- LangGraph for DAG workflow
- CLI interface with structured logging

---

## 📋 Requirements

Ensure you have Python 3.8+ and pip installed.

Install dependencies:

```bash
pip install -r requirements.txt