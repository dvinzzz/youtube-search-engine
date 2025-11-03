# YouTube Video Sentiment Ranking Engine 🎯

A sophisticated machine learning-powered application that analyzes YouTube video comments to rank videos based on community sentiment. This project demonstrates end-to-end ML engineering, from data ingestion and preprocessing to model inference and web deployment.

## 🚀 Project Overview

This application fetches comments from multiple YouTube videos, applies advanced text preprocessing, and uses a hybrid sentiment analysis model to rank videos. It's designed to handle real-world challenges like slang, multilingual content, and API limitations, making it a robust tool for content analysis.

### Key Highlights
- **Hybrid ML Approach**: Combines fine-tuned DistilBERT (70% weight) with VADER sentiment analysis (30% weight) for accurate sentiment detection
- **Production-Ready Pipeline**: Modular architecture with preprocessing, model training, and inference components
- **Real-Time Processing**: Fetches up to 3,000 comments per video via YouTube Data API v3
- **Web Interface**: Flask-based UI for easy video analysis and ranking visualization
- **GPU Acceleration**: CUDA support for faster inference on compatible hardware. 

## 🏗️ Architecture Overview

The project follows a clean, modular architecture separating concerns into distinct components:

```
SVRE_P1/
├── app.py                          # Main Flask application with routing and inference
├── demo.py                         # Model training entry point
├── test.py                         # Batch processing script for multiple videos
├── src/
│   ├── components/
│   │   ├── data_preprocessing.py   # TextPreprocessor class for text cleaning
│   │   └── model_training.py       # ModelTrainer class for fine-tuning
│   ├── pipeline/
│   │   └── training_pipeline.py    # End-to-end training pipeline
│   ├── constants.py                # Configuration constants and paths
│   └── utils/                      # Helper utilities
├── models/                         # Pre-trained and fine-tuned model weights
├── data/                           # Raw and processed datasets
├── templates/                      # HTML templates for web interface
├── static/                         # CSS and static assets
└── requirements.txt                # Python dependencies
```

### Core Components

#### 1. TextPreprocessor (`src/components/data_preprocessing.py`)
A comprehensive text cleaning pipeline that handles:
- **Slang Expansion**: Converts abbreviations (e.g., "lol" → "laughing out loud") using a JSON dictionary
- **Emoticon & Emoji Processing**: Expands emoticons (:) → positive signal) and emojis (😍 → positive signal)
- **Contraction Expansion**: "don't" → "do not" using the `contractions` library
- **Multilingual Support**: Detects non-English text and translates to English using Google Translate API
- **Normalization**: HTML tag removal, lowercasing, punctuation reduction, URL/mention/email removal, whitespace cleanup, and repeated character normalization

#### 2. Sentiment Prediction Pipeline
- **Model**: Fine-tuned DistilBERT for sequence classification (3 classes: negative, neutral, positive)
- **Inference**: Batch processing with GPU acceleration, mapping predictions to scores (-1, 0, 1)
- **Hybrid Scoring**: Combines DistilBERT predictions with VADER compound scores for robustness
- **Confidence Calculation**: Percentage of comments with strong sentiment signals (|score| > 0.1)

#### 3. Training Pipeline (`src/pipeline/training_pipeline.py`)
- **Data Preparation**: Loads raw CSV, applies preprocessing, and splits into train/validation sets
- **Model Fine-Tuning**: Uses Hugging Face Transformers for DistilBERT fine-tuning on custom datasets
- **Evaluation**: Tracks validation accuracy during training

## 🔧 Technical Implementation

### Sentiment Scoring Algorithm
```python
# Hybrid scoring formula
combined_score = (distilbert_score * 0.7) + (vader_score * 0.3)
# Score mapping: {0: -1, 1: 0, 2: 1} for DistilBERT predictions
```

### Key Technologies
- **Backend**: Flask for web framework
- **ML Framework**: PyTorch with Hugging Face Transformers
- **Models**: DistilBERT (lightweight BERT variant), VADER (lexicon-based sentiment)
- **APIs**: Google YouTube Data API v3 for comment fetching
- **Data Processing**: Pandas, NumPy for data manipulation
- **Text Processing**: `contractions`, `emoji`, `deep-translator`, `langdetect`
- **Frontend**: HTML/CSS with Jinja2 templating

### Model Details
- **Base Model**: `distilbert-base-uncased` fine-tuned on educational video comments
- **Training Data**: Custom dataset with labeled comments (positive/neutral/negative)
- **Hyperparameters**: Batch size 16, learning rate 5e-5, 3 epochs
- **Performance**: Domain-specific accuracy on educational content

## 📊 How It Works

1. **Input Processing**: User provides YouTube URLs via web form or batch script
2. **Comment Fetching**: YouTube API retrieves up to 3,000 comments per video
3. **Text Preprocessing**: Applies comprehensive cleaning pipeline to normalize text
4. **Sentiment Analysis**: Runs hybrid model inference on preprocessed comments
5. **Aggregation**: Computes average sentiment score and confidence per video
6. **Ranking**: Sorts videos by sentiment score (highest to lowest)
7. **Visualization**: Displays ranked results with scores and confidence metrics

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 12.1+ (optional, for GPU acceleration)
- YouTube Data API v3 key (free tier available)

### Setup Steps
1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd SVRE_P1
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv svre-env
   source svre-env/bin/activate  # Windows: svre-env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   - Obtain YouTube API key from Google Cloud Console
   - Update `YOUTUBE_API_KEY` in `src/constants/__init__.py`

5. **Download/Train Model**
   - Pre-trained model available in `models/distillbert_finetuned_v2/`
   - To retrain: `python demo.py` (requires training data in `data/`)

## 🎬 Usage

### Web Application
```bash
python app.py
```
- Access at `http://localhost:6969`
- Paste YouTube URLs (one per line)
- Click "Analyze & Rank" for real-time results

### Batch Processing
```bash
python test.py
```
- Processes predefined video URLs in `test.py`
- Outputs ranked results to console

### Model Training
```bash
python demo.py
```
- Fine-tunes DistilBERT on your dataset
- Saves model to `models/distillbert_finetuned_v2/`

## ⚠️ Known Limitations & Considerations

- **API Quota**: Limited to 10,000 queries/day (YouTube API free tier)
- **Domain Specificity**: Model trained on educational video comments; may not generalize perfectly to other domains
- **GPU Memory**: Batch size adjustable to prevent OOM errors
- **Multilingual Handling**: Translation API calls may incur latency for non-English content


## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

This project was developed by : **Vineeth Dhagey** and **Hemanth Kumar Anne**

**Email**: dhageyvineeth@gmail.com,  hemanth.anne101@gmail.com
          
          

---



