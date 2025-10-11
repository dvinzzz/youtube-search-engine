# video_sentiment_ranking.py
import os
import pandas as pd
from googleapiclient.discovery import build
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from concurrent.futures import ThreadPoolExecutor
from src.components.data_preprocessing import TextPreprocessor  # Adjust path if needed

# ----------------------- CONFIG -----------------------
YOUTUBE_API_KEY = "AIzaSyCoz9NrmBu5mFRm_-qD4XoTFaqu7AGvGeU"  # Replace with your key


MODEL_PATH = r"C:\Users\Vineeth\Desktop\SVRE_P1\models\distillbert_finetuned_v2"
SLANG_FILE = r"C:\Users\Vineeth\Desktop\SVRE_P1\data\slang_words.json"
EMOTICON_FILE = r"C:\Users\Vineeth\Desktop\SVRE_P1\data\emoticons.json"

MAX_COMMENTS = 3000

# ----------------------- FUNCTIONS -----------------------
def get_youtube_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")

def fetch_comments(video_url: str, api_key: str, max_comments: int = MAX_COMMENTS):
    video_id = get_youtube_video_id(video_url)
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(comments, columns=["text"])
    print(f"✅ Loaded {len(df)} comments from {video_url}")
    return df

def preprocess_comments(df: pd.DataFrame):
    preprocessor = TextPreprocessor(slang_file_path=SLANG_FILE, emoticons_json_path=EMOTICON_FILE)
    clean_df = preprocessor.initiate_preprocessing(df, text_column="text")
    return clean_df

def predict_video_score(df: pd.DataFrame, model, tokenizer, device):
    texts = df["text"].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    id2score = {0: -1, 1: 0, 2: 1}
    df["prediction"] = [id2score[p] for p in preds]

    # Normalized score for ranking: (positive - negative) / total comments
    pos = (df["prediction"] == 1).sum()
    neg = (df["prediction"] == -1).sum()
    total = len(df)
    normalized_score = (pos - neg) / total if total > 0 else 0
    return normalized_score

def process_video(video_url, model, tokenizer, device, api_key):
    df_comments = fetch_comments(video_url, api_key)
    df_clean = preprocess_comments(df_comments)
    score = predict_video_score(df_clean, model, tokenizer, device)
    return {"video_url": video_url, "score": score}

# ----------------------- MAIN -----------------------
# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    # List of video URLs
    video_urls = [
    "https://www.youtube.com/watch?v=liBWJp2OfUU&list=PLXj4XH7LcRfDlQklXu3Hrtru-bm2dJ9Df&index=3",
    "https://www.youtube.com/watch?v=-0_5ZkERRMg",
    "https://www.youtube.com/watch?v=wXsgJPnr1nQ&list=PLLOxZwkBK52BCOXC7wpI_U81W_eklMFE3",
    "https://www.youtube.com/watch?v=gLptmcuCx6Q",
    "https://www.youtube.com/watch?v=FYvTNBB5bJA"
    ]

    # Load model and tokenizer once
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    video_scores = []

    for video_url in video_urls:
        print(f"🚀 Processing video: {video_url}")

        # 1. Fetch comments
        df_comments = fetch_comments(video_url, YOUTUBE_API_KEY)

        # 2. Preprocess
        df_clean = preprocess_comments(df_comments)

        # 3. Predict and compute score
        texts = df_clean["text"].tolist()
        batch_size = 32  # Prevent GPU OOM
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # Map predictions to scores
        id2score = {0: -1, 1: 0, 2: 1}
        df_clean["prediction"] = [id2score[p] for p in all_preds]

        # Compute normalized score
        norm_score = df_clean["prediction"].mean()
        video_scores.append((video_url, norm_score))
        print(f"📊 Video Score: {norm_score:.3f}\n")

    # Rank videos by score
    ranked_videos = sorted(video_scores, key=lambda x: x[1], reverse=True)
    print("🏆 Video Ranking (best to worst):")
    for rank, (url, score) in enumerate(ranked_videos, start=1):
        print(f"{rank}. {url} → Score: {score:.3f}")

    print("\n✅ Done! All processing in real-time sequentially.")

