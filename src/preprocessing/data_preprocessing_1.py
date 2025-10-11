import re
import json
import os
from typing import Dict, List

# Adjust the path relative to your notebook folder
SLANG_FILE_PATH = os.path.join("..", "..", "data", "slang_words.json")

def _load_slang_map() -> Dict[str, str]:
    """
    Load the slang map from JSON file, fallback to hardcoded map if it fails.
    All keys are converted to lowercase for case-insensitive matching.
    """
    FALLBACK_SLANG_MAP: Dict[str, str] = {
        #"lol": "laughing out loud",
        "brb": "be right back",
        #"btw": "by the way",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        #"thx": "thanks",
        "ty": "thank you",
        "tldr": "too long didn't read",
        "ikr": "i know, right",
        "fomo": "fear of missing out",
        "smh": "shaking my head",
        #"omg": "oh my god",
        "wtf": "what the hell",
        "lmao": "laughing my a** off",
        "np": "no problem",
        "afk": "away from keyboard",
        "gg": "good game",
        "gtg": "got to go",
        "rofl": "rolling on the floor laughing",
    }
    
    try:
        with open(SLANG_FILE_PATH, 'r', encoding='utf-8') as f:
            slang_data = json.load(f)
            print(f"INFO: Successfully loaded slang map from {SLANG_FILE_PATH}")
            # Ensure all keys are lowercase
            return {k.lower(): v for k, v in slang_data.items()}
    except Exception as e:
        print(f"WARNING: Could not load slang file ({e}). Using fallback map.")
        return FALLBACK_SLANG_MAP

# Load the slang map once
SLANG_MAP = _load_slang_map()

def expand_slang_and_abbreviations(text: str) -> str:
    """
    Expand slang/abbreviations in text based on SLANG_MAP.
    Preserves punctuation and handles case-insensitive matching.
    """
    if not text:
        return text

    words = text.split()
    expanded_words: List[str] = []

    for word in words:
        # Separate trailing punctuation
        match = re.match(r"(\w+)(\W*)", word)
        if match:
            core_word, trailing_punc = match.groups()
        else:
            core_word, trailing_punc = word, ''

        # Check slang map (case-insensitive)
        expansion = SLANG_MAP.get(core_word.lower())
        if expansion:
            expansion_words = expansion.split()
            # Reattach punctuation to last word of expansion
            if trailing_punc:
                expansion_words[-1] += trailing_punc
            expanded_words.extend(expansion_words)
        else:
            expanded_words.append(word)

    return ' '.join(expanded_words)

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    test_comment_1 = "OMG, lol, brb, fr this is the best tutorial ever lol! IKR?"
    test_comment_2 = "BTW, I think this is too long. TLDR. SMH."
    test_comment_3 = "thx for the help. np."

    print(f"Original 1: {test_comment_1}")
    print(f"Expanded 1: {expand_slang_and_abbreviations(test_comment_1)}\n")
    
    print(f"Original 2: {test_comment_2}")
    print(f"Expanded 2: {expand_slang_and_abbreviations(test_comment_2)}\n")

    print(f"Original 3: {test_comment_3}")
    print(f"Expanded 3: {expand_slang_and_abbreviations(test_comment_3)}\n")
    