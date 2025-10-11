# src/pipeline/training_pipeline.py
import pandas as pd
from src.components.data_preprocessing import TextPreprocessor
from src.constants import DATA_PATH, OUTPUT_PATH, MODEL_SAVE_PATH, PROCESSED_COMBINED_CSV
from src.components.model_training import ModelTrainer
import os 

class TrainingPipeline:
    def __init__(self, slang_path: str, emoticon_path: str):
        # Preprocessing component
        self.text_preprocessor = TextPreprocessor(
            slang_file_path=slang_path,
            emoticons_json_path=emoticon_path
        )
        # Model training component
        self.trainer = ModelTrainer()

    # -------------------- Preprocessing --------------------
    def data_preprocessing(self, DATA_PATH: str, OUTPUT_PATH: str = None) -> pd.DataFrame:
        # Step 1: Read raw data
        df = pd.read_csv(DATA_PATH)

        # Step 2: Run preprocessing
        df_processed = self.text_preprocessor.initiate_preprocessing(df, text_column='text')

        # Step 3: Save preprocessed data (optional)
        if OUTPUT_PATH:
            df_processed.to_csv(OUTPUT_PATH, index=False)

        return df_processed

    # -------------------- Model Training --------------------
    def train_model(self, processed_data_path: str, save_path: str = MODEL_SAVE_PATH):
        # Prepare datasets
        train_dataset, val_dataset = self.trainer.data_preprocessing(processed_data_path)

        # Train
        self.trainer.train(train_dataset, val_dataset)

        # Save
        self.trainer.save_model(save_path)

    # -------------------- Master pipeline --------------------
    def run(self):
        """
        Master pipeline runner: calls preprocessing and model training automatically.
        Skips preprocessing if a processed CSV already exists.
        
        Args:
            processed_csv_path: Optional path to an existing processed CSV.
                                If provided and exists, preprocessing is skipped.
        """
        # Step 1: Load or preprocess data
        if PROCESSED_COMBINED_CSV and os.path.exists(PROCESSED_COMBINED_CSV):
            print(f"✅ Found existing processed CSV at {PROCESSED_COMBINED_CSV}. Skipping preprocessing.")
            processed_df=PROCESSED_COMBINED_CSV
        else:
            print("Starting data preprocessing...")
            processed_df = self.data_preprocessing(DATA_PATH, OUTPUT_PATH)
            print("✅ Data preprocessing complete.")
        if not os.path.exists(PROCESSED_COMBINED_CSV):
            processed_df.to_csv(PROCESSED_COMBINED_CSV, index=False)
        # Step 2: Train the model using the processed DataFrame
        processed_df= PROCESSED_COMBINED_CSV
        print("Starting model training...")
        self.train_model(processed_df)
        print("✅ Model training complete.")