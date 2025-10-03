# logging_setup.py
import json
import os
from datetime import datetime

log_file = "./logs/dag_log.json"

os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_event(event_type, message):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "message": message
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")