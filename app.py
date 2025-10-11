from flask import Flask, render_template, request
import pandas as pd
from googleapiclient.discovery import build
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from src.components.data_preprocessing import TextPreprocessor
from src.constants import YOUTUBE_API_KEY, SLANG_FILE, EMOTICONS_FILE, MODEL_SAVE_PATH

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- CONFIG ----------------
app = Flask(__name__)
DISTILBERT_WEIGHT = 0.7
VADER_WEIGHT = 0.3

# Load model and tokenizer once
model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

preprocessor = TextPreprocessor(slang_file_path=SLANG_FILE, emoticons_json_path=EMOTICONS_FILE)
vader_analyzer = SentimentIntensityAnalyzer()

# ---------------- HELPERS ----------------
def get_youtube_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")

def fetch_comments(video_url: str, max_comments: int = 1000):
    video_id = get_youtube_video_id(video_url)
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments, next_page_token = [], None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments, columns=["text"])

def preprocess_comments(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["text"])
    return preprocessor.initiate_preprocessing(df, text_column="text")

# ---------------- PREDICTION ----------------
def predict_score(df: pd.DataFrame):
    if df.empty or len(df) == 0:
        return 0.0, 0.0  # safe fallback

    texts = df["text"].tolist()
    batch_size = 32
    all_preds = []

    # DistilBERT predictions
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    id2score = {0: -1, 1: 0, 2: 1}
    df["distilbert_score"] = [id2score[p] for p in all_preds]

    # VADER predictions
    vader_scores = [vader_analyzer.polarity_scores(text)["compound"] for text in texts]
    df["vader_score"] = vader_scores

    # Combined score
    df["combined_score"] = df["distilbert_score"] * DISTILBERT_WEIGHT + df["vader_score"] * VADER_WEIGHT

    avg_score = df["combined_score"].mean()
    if pd.isna(avg_score):
        avg_score = 0.0

    confidence = (df["combined_score"].abs() > 0.1).sum() / len(df)
    return avg_score, confidence

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        urls = request.form.get("video_urls").splitlines()
        results = []

        for url in urls:
            url = url.strip()
            if not url:
                continue
            try:
                df_comments = fetch_comments(url)
                df_clean = preprocess_comments(df_comments)
                score, confidence = predict_score(df_clean)
                results.append({
                    "url": url,
                    "title": f"Video {get_youtube_video_id(url)}",
                    "score": round(score, 3),
                    "confidence": round(confidence * 100, 1)
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "title": "Error fetching video",
                    "score": 0.0,
                    "confidence": 0.0
                })

        # Rank videos safely
        ranked = sorted(results, key=lambda x: x["score"] if x["score"] is not None else 0.0, reverse=True)
        return render_template("results.html", results=ranked)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=6969, debug=True)
