import re
import json
import os
import pandas as pd
import contractions
import emoji
from deep_translator import GoogleTranslator
from langdetect import detect
from typing import Dict, List

class TextPreprocessor:
    """
    Complete text preprocessing pipeline for comments.
    Handles:
    1. Slang & abbreviation expansion
    2. Contraction expansion
    3. Emoji & emoticon expansion
    4. Translation to English (langdetect + DeepTranslator)
    5. Normalization techniques (HTML, lowercase, punctuation, URLs, mentions, emails,
       whitespace, repeated characters)
    """

    def __init__(self, slang_file_path: str, emoticons_json_path: str):
        # Load slang map
        self.SLANG_MAP = self._load_slang_map(slang_file_path)

        # Load emoticon map
        self.EMOTICON_MAP = self._load_emoticons_map(emoticons_json_path)

        # Translator (DeepTranslator)
        self.translator = GoogleTranslator(source="auto", target="en")

    # -------------------- SLANG --------------------
    def _load_slang_map(self, file_path: str) -> Dict[str, str]:
        FALLBACK_SLANG_MAP: Dict[str, str] = {
            "brb": "be right back",
            "imo": "in my opinion",
            "imho": "in my humble opinion",
            "ty": "thank you",
            "tldr": "too long didn't read",
            "ikr": "i know, right",
            "fomo": "fear of missing out",
            "smh": "shaking my head",
            "wtf": "what the hell",
            "lmao": "laughing my a** off",
            "np": "no problem",
            "afk": "away from keyboard",
            "gg": "good game",
            "gtg": "got to go",
            "rofl": "rolling on the floor laughing",
        }
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k.lower(): v for k, v in data.items()}
        except Exception as e:
            print(f"WARNING: Could not load slang file ({e}). Using fallback map.")
            return FALLBACK_SLANG_MAP

    def expand_slang_and_abbreviations(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        expanded_words: List[str] = []

        for word in words:
            match = re.match(r"(\w+)(\W*)", word)
            if match:
                core_word, trailing_punc = match.groups()
            else:
                core_word, trailing_punc = word, ''

            expansion = self.SLANG_MAP.get(core_word.lower())
            if expansion:
                expansion_words = expansion.split()
                if trailing_punc:
                    expansion_words[-1] += trailing_punc
                expanded_words.extend(expansion_words)
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    # -------------------- CONTRACTIONS --------------------
    def expand_contractions(self, text: str) -> str:
        if pd.isna(text) or not text:
            return text
        return contractions.fix(text)

    # -------------------- EMOJIS & EMOTICONS --------------------
    def _load_emoticons_map(self, file_path: str) -> Dict[str, str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                emot_map = json.load(f)
            return emot_map
        except Exception as e:
            print(f"WARNING: Could not load emoticons map ({e}). No expansion will occur.")
            return {}

    def expand_emoticons(self, text: str) -> str:
        if pd.isna(text) or not text:
            return text
        for emoticon in sorted(self.EMOTICON_MAP.keys(), key=len, reverse=True):
            text = re.sub(re.escape(emoticon), self.EMOTICON_MAP[emoticon], text)
        return text

    def expand_emojis(self, text: str) -> str:
        if pd.isna(text) or not text:
            return text
        return emoji.demojize(text, delimiters=(" ", " "))

    def expand_emojis_and_emoticons(self, text: str) -> str:
        text = self.expand_emoticons(text)
        text = self.expand_emojis(text)
        return text

    # -------------------- TRANSLATION --------------------
    def translate_text(self, text: str) -> str:
        """
        Translate text to English only if it is non-English.
        """
        if pd.isna(text) or not text.strip():
            return text

        try:
            # Detect language
            lang = detect(text)
            if lang == 'en':
                return text  # Already English
            # Translate non-English text
            translated = self.translator.translate(text)
            return translated
        except Exception:
            return text  # Fail-safe: return original text

    # -------------------- NORMALIZATION TECHNIQUES --------------------
    def normalization_techniques(self, text: str) -> str:
        if pd.isna(text) or not text:
            return text

        # 1. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # 2. Lowercasing
        text = text.lower()

        # 3. Punctuation normalization (reduce repeated punctuation 3+ → 1)
        text = re.sub(r'([!?.]){2,}', r'\1', text)

        # 4. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # 5. Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)

        # 6. Remove emails
        text = re.sub(r'\S+@\S+', '', text)

        # 7. Whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()

        # 8. Repeated-character normalization (more than 2 repeats → 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        return text

    # -------------------- INITIATE PREPROCESSING --------------------
    def initiate_preprocessing(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Applies all preprocessing steps in order on the specified text column.
        Returns a DataFrame with the same columns intact.
        """
        df_copy = df.copy()

        # Apply each step sequentially
        df_copy[text_column] = df_copy[text_column].astype(str)\
            .apply(self.expand_slang_and_abbreviations)\
            .apply(self.expand_contractions)\
            .apply(self.expand_emojis_and_emoticons)\
            .apply(self.translate_text)\
            .apply(self.normalization_techniques)

        return df_copy