import pandas as pd
import emoji
import json
import re
import os

# Path to your saved emoticons.json
EMOTICONS_JSON_PATH = r"C:\Users\Vineeth\Desktop\SVRE\SVRE\data\emoticons.json"

def load_emoticons_map(json_path: str) -> dict:
    """
    Loads emoticon map from a JSON file.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            emot_map = json.load(f)
        print(f"INFO: Successfully loaded emoticons map from {json_path}")
        return emot_map
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at {json_path}. No emoticon expansion will occur.")
        return {}
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON from {json_path}. No emoticon expansion will occur.")
        return {}

# Load emoticons map once
EMOTICON_MAP = load_emoticons_map(EMOTICONS_JSON_PATH)

def expand_emoticons(text: str) -> str:
    """
    Expands text-based emoticons using the loaded EMOTICON_MAP.
    """
    if pd.isna(text) or not text:
        return text
    
    # Sort by length descending to prevent partial replacements (e.g., :) inside :-))
    for emoticon in sorted(EMOTICON_MAP.keys(), key=len, reverse=True):
        text = re.sub(re.escape(emoticon), EMOTICON_MAP[emoticon], text)
    return text

def expand_emojis(text: str) -> str:
    """
    Expands Unicode emojis into text.
    """
    if pd.isna(text) or not text:
        return text
    return emoji.demojize(text, delimiters=(" ", " "))

def expand_emojis_and_emoticons(text: str) -> str:
    """
    Expands both emojis and emoticons in the text.
    """
    text = expand_emoticons(text)
    text = expand_emojis(text)
    return text

def expand_in_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applies emoji and emoticon expansion on a DataFrame column.
    """
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].astype(str).apply(expand_emojis_and_emoticons)
    return df_copy

# ---------------- Example Testing ----------------
if __name__ == "_main_":
    test_comments = [
        "I am so happy today :) 😂",
        "This is sad :(",
        "LOL XD that was funny",
        "I love you <3",
        "Winking ;) and playful :P",
        "Surprised :-O and crying T_T",
        "Angel O:) and devil 3:)"
    ]

    print("\n--- Testing emoji and emoticon expansion on examples ---")
    for i, comment in enumerate(test_comments, 1):
        expanded = expand_emojis_and_emoticons(comment)
        print(f"Original {i}: {comment}")
        print(f"Expanded {i}: {expanded}\n")

    # Example: applying to comments2.csv
    csv_file = r"C:\Users\Vineeth\Desktop\SVRE\SVRE\data\raw\comments2.csv"
    if os.path.exists(csv_file):
        df_raw = pd.read_csv(csv_file)
        df_expanded = expand_in_dataframe(df_raw, text_column='text')  # replace 'text' with your actual column name
        print("\n--- First 5 rows after expansion ---")
        print(df_expanded.head())