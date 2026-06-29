"""
Sentiment analysis module using Google Gemini API for zero-shot classification.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import google.generativeai as genai

from config.settings import settings
from src.preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Performs zero-shot sentiment analysis using Google Gemini LLM.
    """

    def __init__(self):
        """Initialize the sentiment analyzer with Gemini API."""
        self.cleaner = TextCleaner()
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize the Google Gemini API client."""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    def analyze_sentiment(self, text: str, return_json: bool = True) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using Gemini API.

        Args:
            text: Input text to analyze
            return_json: Return structured JSON response

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "emotion": "unknown",
                "confidence": 0.0,
                "key_feature": "none",
                "explanation": "Empty or invalid text",
            }

        # Clean the text
        cleaned_text = self.cleaner.clean_text(text)

        if len(cleaned_text.split()) < 2:
            return {
                "sentiment": "neutral",
                "emotion": "unknown",
                "confidence": 0.0,
                "key_feature": "none",
                "explanation": "Text too short for analysis",
            }

        try:
            # Build the prompt
            prompt = self._build_sentiment_prompt(cleaned_text)

            # Get response from Gemini
            response = self.model.generate_content(prompt)

            # Parse the response
            result = self._parse_response(response.text)

            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "neutral",
                "emotion": "unknown",
                "confidence": 0.0,
                "key_feature": "none",
                "explanation": f"Error: {str(e)}",
            }

    def _build_sentiment_prompt(self, text: str) -> str:
        """
        Build the prompt for Gemini API.

        Args:
            text: Cleaned input text

        Returns:
            Prompt string
        """
        return f"""
        Analyze the following social media post and provide a structured JSON response.
        
        Post: "{text}"
        
        Your response MUST be a valid JSON object with the following structure:
        {{
            "sentiment": "positive OR negative OR neutral",
            "emotion": "one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, frustration, excitement, satisfaction, disappointment, concern, calm, curiosity",
            "confidence": "a float between 0.0 and 1.0 indicating your confidence in the analysis",
            "key_feature": "the main subject, product, topic, or entity being discussed",
            "explanation": "a brief explanation of your reasoning"
        }}
        
        Important:
        1. Consider the context and nuance of the language.
        2. Detect sarcasm, irony, and implied meanings.
        3. If the post is neutral or factual, classify as neutral with appropriate emotion.
        4. Extract the key feature or topic being discussed.
        5. Return ONLY the JSON object, no additional text.
        """

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the Gemini API response.

        Args:
            response_text: Raw response from Gemini

        Returns:
            Parsed sentiment analysis results
        """
        try:
            # Find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Ensure all fields are present
                required_fields = [
                    "sentiment",
                    "emotion",
                    "confidence",
                    "key_feature",
                    "explanation",
                ]
                for field in required_fields:
                    if field not in result:
                        result[field] = "unknown" if field != "confidence" else 0.0

                # Validate sentiment
                valid_sentiments = ["positive", "negative", "neutral"]
                if result["sentiment"].lower() not in valid_sentiments:
                    result["sentiment"] = "neutral"

                return result
            else:
                logger.warning("No JSON found in response")
                return self._fallback_analysis(response_text)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._fallback_analysis(response_text)

    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback analysis when JSON parsing fails.

        Args:
            text: Input text or response

        Returns:
            Basic sentiment analysis
        """
        # Simple keyword-based fallback
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "awesome",
            "love",
            "best",
            "perfect",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "worst",
            "poor",
            "disappointing",
        }

        text_lower = text.lower()
        words = set(text_lower.split())

        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "emotion": "unknown",
            "confidence": 0.5,
            "key_feature": "unknown",
            "explanation": "Fallback analysis due to API response parsing error",
        }

    def analyze_batch(
        self, texts: List[str], batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process per batch

        Returns:
            List of sentiment analysis results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)

        return results

    def analyze_dataframe(
        self, df: pd.DataFrame, text_column: str = "text", batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            batch_size: Number of texts to process per batch

        Returns:
            DataFrame with sentiment analysis results
        """
        df = df.copy()

        if text_column in df.columns:
            # Extract texts
            texts = df[text_column].fillna("").astype(str).tolist()

            # Analyze sentiments
            results = self.analyze_batch(texts, batch_size)

            # Add results to DataFrame
            result_df = pd.DataFrame(results)

            # Rename columns to avoid conflicts
            for col in result_df.columns:
                df[f"sentiment_{col}"] = result_df[col]

        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate sentiment distribution from a DataFrame.

        Args:
            df: DataFrame with sentiment analysis results

        Returns:
            Dictionary with sentiment distribution statistics
        """
        sentiment_col = None
        for col in df.columns:
            if "sentiment" in col.lower() and "sentiment" in col:
                sentiment_col = col
                break

        if not sentiment_col:
            return {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "total": 0,
                "distribution": {},
            }

        distribution = df[sentiment_col].value_counts().to_dict()
        total = len(df)

        return {
            "positive": distribution.get("positive", 0),
            "negative": distribution.get("negative", 0),
            "neutral": distribution.get("neutral", 0),
            "total": total,
            "distribution": (
                {k: (v / total * 100) for k, v in distribution.items()}
                if total > 0
                else {}
            ),
        }
