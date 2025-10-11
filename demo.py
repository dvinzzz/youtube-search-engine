# demo.py
from src.pipeline.training_pipeline import TrainingPipeline
from src.constants import SLANG_FILE, EMOTICONS_FILE

if __name__ == "__main__":
    pipeline = TrainingPipeline(slang_path=SLANG_FILE, emoticon_path=EMOTICONS_FILE)
    pipeline.run()