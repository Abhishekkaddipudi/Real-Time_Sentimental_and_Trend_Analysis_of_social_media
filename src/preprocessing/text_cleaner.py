"""
Text cleaning and preprocessing module for social media data.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and preprocesses social media text for analysis.
    """

    def __init__(self):
        """Initialize the text cleaner with stopwords."""
        self.stopwords = set(stopwords.words("english"))
        # Add social media specific stopwords
        self.social_stopwords = {
            "http",
            "https",
            "www",
            "com",
            "org",
            "net",
            "rt",
            "re",
            "amp",
            "via",
            "u",
            "ur",
            "lol",
            "omg",
            "tbh",
            "imo",
            "imho",
        }
        self.stopwords.update(self.social_stopwords)

    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = True,
        remove_emojis: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        expand_contractions: bool = True,
    ) -> str:
        """
        Clean and preprocess text.

        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_emojis: Remove emojis
            remove_numbers: Remove numbers
            remove_stopwords: Remove stopwords
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            expand_contractions: Expand contractions

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Initial cleaning
        cleaned = text.strip()

        # Remove URLs
        if remove_urls:
            cleaned = re.sub(r"http\S+|www\S+|\S+\.com\S+", "", cleaned)

        # Remove mentions
        if remove_mentions:
            cleaned = re.sub(r"@\w+", "", cleaned)

        # Remove hashtags (but keep the text)
        if remove_hashtags:
            cleaned = re.sub(r"#", "", cleaned)

        # Remove emojis
        if remove_emojis:
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "\U0001f680-\U0001f6ff"  # transport & map symbols
                "\U0001f1e0-\U0001f1ff"  # flags (iOS)
                "\U00002702-\U000027b0"
                "\U000024c2-\U0001f251"
                "]+",
                flags=re.UNICODE,
            )
            cleaned = emoji_pattern.sub("", cleaned)

        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove numbers
        if remove_numbers:
            cleaned = re.sub(r"\d+", "", cleaned)

        # Expand contractions
        if expand_contractions:
            cleaned = self._expand_contractions(cleaned)

        # Convert to lowercase
        if lowercase:
            cleaned = cleaned.lower()

        # Remove punctuation
        if remove_punctuation:
            cleaned = re.sub(r"[^\w\s]", "", cleaned)

        # Remove stopwords
        if remove_stopwords:
            words = cleaned.split()
            cleaned = " ".join([word for word in words if word not in self.stopwords])

        # Final cleaning
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _expand_contractions(self, text: str) -> str:
        """
        Expand common contractions.

        Args:
            text: Input text with contractions

        Returns:
            Text with expanded contractions
        """
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "mightn't": "might not",
            "mustn't": "must not",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
        }

        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)

        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        try:
            return word_tokenize(text)
        except:
            return text.split()

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "text", **kwargs
    ) -> pd.DataFrame:
        """
        Clean all text in a DataFrame column.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            **kwargs: Arguments for clean_text

        Returns:
            DataFrame with cleaned text
        """
        df = df.copy()

        if text_column in df.columns:
            df[f"{text_column}_cleaned"] = df[text_column].apply(
                lambda x: self.clean_text(str(x), **kwargs)
            )

        return df

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract text features for analysis.

        Args:
            text: Input text

        Returns:
            Dictionary of text features
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)

        return {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "word_count": len(tokens),
            "unique_words": len(set(tokens)),
            "avg_word_length": (
                sum(len(t) for t in tokens) / len(tokens) if tokens else 0
            ),
            "has_question": "?" in text,
            "has_exclamation": "!" in text,
            "has_hashtag": "#" in text,
            "has_mention": "@" in text,
            "has_url": bool(re.search(r"http\S+|www\S+", text)),
            "has_emoji": bool(
                re.search(
                    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]",
                    text,
                )
            ),
        }

    def batch_extract_features(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Extract features for all text in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with extracted features
        """
        df = df.copy()

        if text_column in df.columns:
            features_df = df[text_column].apply(
                lambda x: pd.Series(self.extract_features(str(x)))
            )
            df = pd.concat([df, features_df], axis=1)

        return df
