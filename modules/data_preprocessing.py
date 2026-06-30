"""
Data Preprocessing Module - Cleans and prepares raw social media data for analysis.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.helpers import log_message

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


class DataPreprocessing:
    """
    Handles data cleaning and preprocessing for social media text.
    """

    def __init__(self):
        """Initialize the preprocessing module."""
        self.stop_words = set(stopwords.words("english"))
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"  # alchemical symbols
            "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
            "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
            "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
            "\U0001fa00-\U0001fa6f"  # Chess Symbols
            "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
        self.mention_pattern = re.compile(r"@\w+", re.IGNORECASE)
        self.hashtag_pattern = re.compile(r"#\w+", re.IGNORECASE)
        log_message("Data Preprocessing module initialized")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = self.url_pattern.sub("", text)

        # Remove mentions
        text = self.mention_pattern.sub("", text)

        # Remove emojis
        text = self.emoji_pattern.sub("", text)

        # Remove special characters and digits (keeping letters and spaces)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.

        Args:
            text: Text string

        Returns:
            Text without stopwords
        """
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [
                word for word in tokens if word.lower() not in self.stop_words
            ]
            return " ".join(filtered_tokens)
        except:
            return text

    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.

        Args:
            text: Text string

        Returns:
            List of hashtags
        """
        return self.hashtag_pattern.findall(text)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to entire DataFrame.

        Args:
            df: DataFrame with raw social media data

        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            log_message("Empty DataFrame provided for preprocessing", level="WARNING")
            return df

        # Create a copy to avoid modifying the original
        df_processed = df.copy()

        # Handle missing values
        df_processed["text"] = df_processed["text"].fillna("")
        df_processed["title"] = df_processed["title"].fillna("")

        # Combine title and text for analysis
        df_processed["full_text"] = df_processed["title"] + " " + df_processed["text"]

        # Clean text
        df_processed["cleaned_text"] = df_processed["full_text"].apply(self.clean_text)
        df_processed["cleaned_text"] = df_processed["cleaned_text"].apply(
            self.remove_stopwords
        )

        # Extract hashtags
        df_processed["hashtags"] = df_processed["full_text"].apply(
            self.extract_hashtags
        )

        # Calculate text length features
        df_processed["text_length"] = df_processed["cleaned_text"].str.len()
        df_processed["word_count"] = df_processed["cleaned_text"].str.split().str.len()

        # Convert timestamps
        if "created_utc" in df_processed.columns:
            df_processed["created_utc"] = pd.to_datetime(df_processed["created_utc"])
            df_processed["date"] = df_processed["created_utc"].dt.date
            df_processed["hour"] = df_processed["created_utc"].dt.hour
            df_processed["day_of_week"] = df_processed["created_utc"].dt.dayofweek

        # Remove rows with empty text after cleaning
        df_processed = df_processed[df_processed["cleaned_text"].str.len() > 0]

        log_message(f"Preprocessed {len(df_processed)} rows")
        return df_processed

    def prepare_for_llm(self, text: str, max_length: int = 500) -> str:
        """
        Prepare text for LLM processing by truncating and cleaning.

        Args:
            text: Input text
            max_length: Maximum length of text for LLM

        Returns:
            Prepared text for LLM
        """
        cleaned = self.clean_text(text)
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        return cleaned

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from processed data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with additional features
        """
        if df.empty:
            return df

        df_features = df.copy()

        # Engagement metrics
        if "score" in df_features.columns and "num_comments" in df_features.columns:
            df_features["total_engagement"] = (
                df_features["score"] + df_features["num_comments"]
            )
            df_features["engagement_per_word"] = df_features[
                "total_engagement"
            ] / df_features["word_count"].replace(0, 1)

        # Time-based features
        if "hour" in df_features.columns:
            df_features["is_night"] = (
                (df_features["hour"] >= 22) | (df_features["hour"] <= 5)
            ).astype(int)
            df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)

        # Hashtag features
        if "hashtags" in df_features.columns:
            df_features["hashtag_count"] = df_features["hashtags"].apply(len)

        log_message(f"Extracted features from {len(df_features)} rows")
        return df_features
