import pandas as pd
import contractions
from pathlib import Path

def expand_contractions_text(text: str) -> str:
    """
    Expands all contractions in a given text string using the ⁠ contractions ⁠ library.
    
    Args:
        text: Input string (e.g., "I can't do this").
    
    Returns:
        The string with contractions expanded (e.g., "I cannot do this").
    """
    if pd.isna(text) or not text:
        return text
    return contractions.fix(text)

def preprocess_comments_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads a CSV file containing a 'text' column, expands contractions for each comment,
    and returns a DataFrame with only the processed text.

    Args:
        csv_path: Path to the 'comments2.csv' file.

    Returns:
        DataFrame with a single column 'text' containing expanded comments.
    """
    df_raw = pd.read_csv(csv_path)
    
    # Apply contraction expansion to all rows
    df_processed = pd.DataFrame()
    df_processed['text'] = df_raw['text'].astype(str).apply(expand_contractions_text)
    
    return df_processed

# ---------------- Testing / Example ----------------
if __name__ == "_main_":
    # Example individual sentences
    test_comments = [
        "I can't believe it's already done!",
        "You're going to love this, don't worry.",
        "He'd've finished if he had more time.",
        "We're here, but it's too late.",
        "This won't work, I think.",
        "I'll be there ASAP, but I might be late."
    ]
    
    print("\n--- Testing contraction expansion on individual examples ---")
    for i, comment in enumerate(test_comments, 1):
        expanded = expand_contractions_text(comment)
        print(f"Original {i}: {comment}")
        print(f"Expanded {i}: {expanded}\n")

    # Process comments2.csv
    csv_file = Path(r"C:/Users/Vineeth/Desktop/SVRE/SVRE/data/raw/comments2.csv")
    print(f"Loading raw data from CSV: {csv_file}")
    df_expanded = preprocess_comments_from_csv(csv_file)
    
    print("\n--- First 5 rows after expanding contractions from CSV ---")
    print(df_expanded.head())
    
    print(f"\nTotal records processed from CSV: {len(df_expanded)}")