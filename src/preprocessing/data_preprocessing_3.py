"""
data_translation.py

This module translates only the 'text' column of a DataFrame into English
using the deep_translator library. Other columns remain unchanged.
"""
import pandas as pd
from deep_translator import GoogleTranslator

def translate_text_series(series: pd.Series, target_lang: str = "en", source_lang: str = "auto") -> pd.Series:
    """
    Translate text in a Pandas Series into the target language, leaving text
    as-is if it's already in the target language or translation fails.
    
    Args:
        series (pd.Series): Input Series of strings (e.g., comments).
        target_lang (str): Target language code (default: 'en').
        source_lang (str): Source language code (default: 'auto').

    Returns:
        pd.Series: Series with translated text.
    """
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    def safe_translate(text):
        if not text or pd.isna(text):
            return text
        try:
            return translator.translate(text)
        except Exception:
            return text  # fallback: keep original

    return series.apply(safe_translate)

def translate_dataframe(df: pd.DataFrame, target_lang: str = "en") -> pd.DataFrame:
    """
    Translate only the 'text' column in the given DataFrame into English.
    
    Args:
        df (pd.DataFrame): Input DataFrame with at least a 'text' column.
        target_lang (str): Target language (default: 'en').
    
    Returns:
        pd.DataFrame: DataFrame with 'text' column translated.
    """
    df_copy = df.copy()
    df_copy['text'] = translate_text_series(df_copy['text'], target_lang=target_lang)
    return df_copy

# ---------------- Example ----------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "text": [
            "This is already English.",
            "बहुत अच्छा",        # Hindi
            "C'est la vie",       # French
            "Hola amigo"          # Spanish
        ],
        "label": ["neutral", "positive", "neutral", "neutral"]  # stays untouched
    })

    print("\n--- Original DataFrame ---")
    print(df)

    df_translated = translate_dataframe(df)

    print("\n--- After Translation ---")
    print(df_translated)