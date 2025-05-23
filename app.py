from flask import Flask, request, jsonify, render_template
import googleapiclient.discovery
import re
import pandas as pd
import json
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import contractions
from googletrans import Translator
from indic_transliteration import sanscriptpip install beautifulsoup4
from indic_transliteration.sanscript import transliterate
import pycld3 as cld3

app = Flask(__name__)

# Initialize YouTube API
api_key = 'AIzaSyC1XMRCkSRnNf-XnWvYgJZUTLaARwFeO68'  # Replace with your YouTube API key
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

# Load necessary data files
with open('C:/Users/Vineeth/Desktop/ytnlp-20241114T185833Z-001/ytnlp/static/slang_words.json', 'r') as file:
    slang_words = json.load(file)

with open('C:/Users/Vineeth/Desktop/ytnlp-20241114T185833Z-001/ytnlp/static/lang_code_fullname.json', 'r') as file:
    lang_code_fullname = json.load(file)

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Initialize NLP models and tools
bert_sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
vader_analyzer = SentimentIntensityAnalyzer()
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
english_word_set = set(nltk.corpus.words.words())

# Load LSTM sentiment analysis model and tokenizer
loaded_model = load_model("/Users/Vineeth/Desktop/ytnlp-20241114T185833Z-001/ytnlp/static/sentiment_model.h5")
with open("/Users/Vineeth/Desktop/ytnlp-20241114T185833Z-001/ytnlp/static/tokenizer.pkl", "rb") as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# Helper functions for processing
def preprocess_text(review):
    words = review.split()
    for word in words:
        if word in slang_words:
            review = review.replace(word, slang_words[word])

    review = re.sub(r"http\S+", "", review)
    review = re.sub(r'@[\w]+', ' ', review)
    review = BeautifulSoup(review, 'lxml').get_text()
    review = contractions.fix(review)
    review = re.sub(r'\d+', "", review).strip()
    review = re.sub('[^A-Za-z]+', ' ', review)
    review = review.lower()
    review = [lemmatizer.lemmatize(token, "v") for token in review]
    return "".join(review)

def detect_language(text):
    _, _, _, detected_languages = cld3.detect(text, returnVectors=True)
    return detected_languages

def detect_language_googletrans(text):
    translator = Translator()
    detected_lang = translator.detect(text)
    return lang_code_fullname.get(detected_lang.lang)

def transliterate_words(words, detected_lang):
    script_constant = sanscript.DEVANAGARI if detected_lang.upper() == "HINDI" else getattr(sanscript, detected_lang.upper(), sanscript.ITRANS)
    return [transliterate(word, sanscript.ITRANS, script_constant) for word in words]

def translate_to_english1(sentence, source_language='auto', target_language='en'):
    translator = Translator()
    try:
        translation = translator.translate(sentence, src=source_language, dest=target_language)
        return translation.text if translation and translation.text else sentence
    except Exception as e:
        print(f"Translation error: {e}")
        return sentence

# Sentiment analysis functions
def bert_sentiment_analysis(text):
    result = bert_sentiment_pipeline(text)[0]
    return convert_to_label(result)

def vader_sentiment_analysis(text):
    sentiment_score = vader_analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def lstm_sentiment_analysis(text):
    sequences = loaded_tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=loaded_model.input_shape[1])
    probability = loaded_model.predict(padded_sequences)[0]
    return "Positive" if probability > 0.5 else "Negative"

def convert_to_label(sentiment_score):
    if sentiment_score['label'] == '1 star' or sentiment_score['label'] == '2 stars':
        return 'Negative'
    elif sentiment_score['label'] == '4 stars' or sentiment_score['label'] == '5 stars':
        return 'Positive'
    else:
        return 'Neutral'

def get_ensemble_sentiment(sentiment_scores):
    votes = [sentiment_scores['bert_sentiment'], sentiment_scores['vader_sentiment'], sentiment_scores['lstm_sentiment']]
    counts = {vote: votes.count(vote) for vote in set(votes)}
    max_votes = max(counts.values())
    winners = [sentiment for sentiment, count in counts.items() if count == max_votes]
    return winners[0] if len(winners) == 1 else 'Neutral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_videos', methods=['POST'])
def submit_videos():
    data = request.get_json()
    video_links = data.get('video_links', [])
    
    if not video_links:
        return jsonify({"success": False, "error": "No video links received"}), 400

    video_ids = [link.split('=')[-1] for link in video_links]
    
    video_details = []
    for video_id in video_ids:
        try:
            video_response = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()

            if 'items' in video_response and len(video_response['items']) > 0:
                video = video_response['items'][0]
                title = video['snippet']['title']
                tags = video['snippet'].get('tags', [])
                comments = fetch_comments(video_id)
                video_details.append({
                    'Title': title,
                    'Tags': tags,
                    'Comments': comments
                })
        except Exception as e:
            print(f'Error processing video ID {video_id}: {str(e)}')

    rankings = process_and_rank_videos(video_details)

    # Create a formatted response with priority rankings
    priority_ranking = [{"rank": idx + 1, "title": title} for idx, title in enumerate(rankings)]
    
    return jsonify({"success": True, "message": "Video details processed and ranked successfully", "ranking": priority_ranking})

def fetch_comments(video_id):
    comments = []
    nextPageToken = None
    while True:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,
            pageToken=nextPageToken
        ).execute()
        for comment in comments_response['items']:
            text_display = comment['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(text_display)
        nextPageToken = comments_response.get('nextPageToken')
        if not nextPageToken:
            break
    return comments

def process_and_rank_videos(video_details):
    sortt = {}
    for video in video_details:
        video_name = video['Title']
        comments = video['Comments']

        processed_comments = []
        for comment in comments:
            preprocessed_text = preprocess_text(comment)
            translated_text = translate_to_english1(preprocessed_text)
            sentiment_scores = {
                'bert_sentiment': bert_sentiment_analysis(translated_text),
                'vader_sentiment': vader_sentiment_analysis(translated_text),
                'lstm_sentiment': lstm_sentiment_analysis(translated_text)
            }
            ensemble_sentiment = get_ensemble_sentiment(sentiment_scores)
            processed_comments.append(ensemble_sentiment)

        positive_count = processed_comments.count("Positive")
        negative_count = processed_comments.count("Negative")
        total_comments = len(processed_comments)

        score = (positive_count - negative_count) / total_comments if negative_count else 1.0

        # Use list to handle multiple videos with the same score
        if score in sortt:
            sortt[score].append(video_name)
        else:
            sortt[score] = [video_name]

    sorted_scores = sorted(sortt.keys(), reverse=True)
    sorted_videos = [video for score in sorted_scores for video in sortt[score]]
    return sorted_videos

if __name__ == '__main__':
    app.run()
