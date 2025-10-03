# config.py
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
THRESHOLD_CONFIDENCE = 0.7  # Must be >70% confident
CHECKPOINT_DIR = "./model"
LOG_FILE = "./logs/dag_log.json"
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"