"""
Sentiment Analysis Module - Uses LLM for zero-shot sentiment classification.
Includes fallback options: Transformers (BERT-based) and Lexicon-based (VADER).
"""

import json
import time
import re
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from utils.helpers import log_message, safe_json_parse
import nltk

# Try importing Gemini
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log_message("Google Generative AI not installed", level="WARNING")

# Try importing transformers for deep learning fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

    TRANSFORMERS_AVAILABLE = True
    log_message("Transformers library available for deep learning fallback")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    log_message("Transformers not installed, using simpler fallbacks", level="WARNING")

# Try importing VADER for lexicon-based fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
    log_message("VADER available for lexicon-based fallback")
except ImportError:
    VADER_AVAILABLE = False
    log_message("VADER not installed", level="WARNING")

# Try importing NLTK for basic fallback
try:
    from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA

    nltk.download("vader_lexicon", quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False


class SentimentAnalysis:
    """
    Handles sentiment analysis with multiple fallback options:
    1. Gemini API (primary)
    2. Transformers/BERT (fallback 1)
    3. VADER lexicon-based (fallback 2)
    4. NLTK (fallback 3)
    5. Rule-based (final fallback)
    """

    def __init__(
        self,
        model_name: str = "google/gemini-1.5-flash",
        use_fallback: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the Sentiment Analysis module.

        Args:
            model_name: Model name for Gemini or transformers
            use_fallback: Whether to use fallback methods
            device: Device for transformers ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.device = device

        # Initialize primary model (Gemini)
        self.gemini_available = False
        self.gemini_model = None
        self._init_gemini()

        # Initialize transformers fallback
        self.transformers_available = False
        self.transformers_pipeline = None
        self._init_transformers()

        # Initialize VADER fallback
        self.vader_available = False
        self.vader_analyzer = None
        self._init_vader()

        # Initialize NLTK fallback
        self.nltk_available = False
        self._init_nltk()

        # Track which method is being used
        self.active_method = "none"

        log_message(
            f"Sentiment Analysis module initialized. "
            f"Gemini: {self.gemini_available}, "
            f"Transformers: {self.transformers_available}, "
            f"VADER: {self.vader_available}, "
            f"NLTK: {self.nltk_available}"
        )

    def _init_gemini(self):
        """Initialize the Gemini model."""
        try:
            from config.config import Config

            if not GEMINI_AVAILABLE:
                log_message("Gemini library not available", level="WARNING")
                return

            genai.configure(api_key=Config.GEMINI_API_KEY)

            # Try different model names that might work
            model_names = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "models/gemini-1.0-pro",
            ]

            for name in model_names:
                try:
                    self.gemini_model = genai.GenerativeModel(name)
                    # Test the model with a simple request
                    test_response = self.gemini_model.generate_content("Test")
                    if test_response and hasattr(test_response, "text"):
                        self.gemini_available = True
                        self.model_name = name
                        log_message(f"Gemini model initialized successfully: {name}")
                        break
                except Exception as e:
                    log_message(
                        f"Failed to initialize Gemini with {name}: {e}", level="DEBUG"
                    )
                    continue

            if not self.gemini_available:
                log_message("Could not initialize any Gemini model", level="WARNING")

        except Exception as e:
            log_message(f"Error initializing Gemini: {e}", level="WARNING")
            self.gemini_available = False

    def _init_transformers(self):
        """Initialize transformers-based sentiment classifier."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                return

            # Use a smaller, faster model for sentiment analysis
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self.transformers_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            )
            self.transformers_available = True
            log_message("Transformers sentiment classifier initialized successfully")

        except Exception as e:
            log_message(f"Error initializing transformers: {e}", level="WARNING")
            self.transformers_available = False

    def _init_vader(self):
        """Initialize VADER sentiment analyzer."""
        try:
            if VADER_AVAILABLE:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.vader_available = True
                log_message("VADER initialized successfully")
            else:
                self.vader_available = False
        except Exception as e:
            log_message(f"Error initializing VADER: {e}", level="WARNING")
            self.vader_available = False

    def _init_nltk(self):
        """Initialize NLTK sentiment analyzer."""
        try:
            if NLTK_AVAILABLE:
                self.nltk_analyzer = NLTK_SIA()
                self.nltk_available = True
                log_message("NLTK initialized successfully")
            else:
                self.nltk_available = False
        except Exception as e:
            log_message(f"Error initializing NLTK: {e}", level="WARNING")
            self.nltk_available = False

    def analyze_post(self, text: str, max_retries: int = 3) -> Dict:
        """
        Analyze a single post using the best available method.

        Args:
            text: Post text to analyze
            max_retries: Maximum retry attempts for API calls

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or len(text.strip()) < 3:
            return self._get_empty_result("Text too short or empty")

        # Try Gemini first
        if self.gemini_available:
            result = self._analyze_with_gemini(text, max_retries)
            if result and result.get("sentiment") in [
                "positive",
                "negative",
                "neutral",
            ]:
                self.active_method = "gemini"
                return result

        # Try Transformers fallback
        if self.transformers_available:
            result = self._analyze_with_transformers(text)
            if result:
                self.active_method = "transformers"
                return result

        # Try VADER fallback
        if self.vader_available:
            result = self._analyze_with_vader(text)
            if result:
                self.active_method = "vader"
                return result

        # Try NLTK fallback
        if self.nltk_available:
            result = self._analyze_with_nltk(text)
            if result:
                self.active_method = "nltk"
                return result

        # Final rule-based fallback
        self.active_method = "rule_based"
        return self._analyze_rule_based(text)

    def _analyze_with_gemini(self, text: str, max_retries: int = 3) -> Dict:
        """Analyze using Gemini API."""
        if not self.gemini_model:
            return None

        prompt = self._construct_prompt(text)

        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                if response and hasattr(response, "text"):
                    result = self._parse_response(response.text)
                    if result.get("sentiment") in ["positive", "negative", "neutral"]:
                        return result
                time.sleep(1)
            except Exception as e:
                log_message(
                    f"Gemini error (attempt {attempt + 1}): {e}", level="WARNING"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return None

        return None

    def _analyze_with_transformers(self, text: str) -> Dict:
        """Analyze using transformers-based classifier."""
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]

            result = self.transformers_pipeline(text)[0]
            sentiment = result["label"].lower()
            confidence = result["score"]

            # Map to our sentiment categories
            if "positive" in sentiment:
                sentiment = "positive"
            elif "negative" in sentiment:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "emotion": self._map_sentiment_to_emotion(sentiment),
                "confidence": float(confidence),
                "key_feature": self._extract_key_feature(text),
                "emotional_driver": self._extract_emotional_driver(text, sentiment),
                "summary": f"Analyzed using transformers with {confidence:.2f} confidence",
            }
        except Exception as e:
            log_message(f"Transformers error: {e}", level="WARNING")
            return None

    def _analyze_with_vader(self, text: str) -> Dict:
        """Analyze using VADER sentiment analyzer."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores["compound"]

            # Determine sentiment
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            confidence = abs(compound)

            return {
                "sentiment": sentiment,
                "emotion": self._map_sentiment_to_emotion(sentiment),
                "confidence": float(min(confidence, 1.0)),
                "key_feature": self._extract_key_feature(text),
                "emotional_driver": self._extract_emotional_driver(text, sentiment),
                "summary": f"VADER analysis: compound score {compound:.3f}",
            }
        except Exception as e:
            log_message(f"VADER error: {e}", level="WARNING")
            return None

    def _analyze_with_nltk(self, text: str) -> Dict:
        """Analyze using NLTK sentiment analyzer."""
        try:
            scores = self.nltk_analyzer.polarity_scores(text)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            confidence = abs(compound)

            return {
                "sentiment": sentiment,
                "emotion": self._map_sentiment_to_emotion(sentiment),
                "confidence": float(min(confidence, 1.0)),
                "key_feature": self._extract_key_feature(text),
                "emotional_driver": self._extract_emotional_driver(text, sentiment),
                "summary": f"NLTK analysis: compound score {compound:.3f}",
            }
        except Exception as e:
            log_message(f"NLTK error: {e}", level="WARNING")
            return None

    def _analyze_rule_based(self, text: str) -> Dict:
        """Analyze using simple rule-based approach."""
        # Define sentiment word lists
        positive_words = set(
            [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "beautiful",
                "love",
                "happy",
                "best",
                "awesome",
                "fantastic",
                "brilliant",
                "perfect",
                "nice",
                "superb",
                "outstanding",
            ]
        )
        negative_words = set(
            [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "worst",
                "hate",
                "sad",
                "angry",
                "disappointed",
                "poor",
                "rubbish",
                "useless",
                "pathetic",
                "disgusting",
                "unacceptable",
                "terrible",
            ]
        )

        text_lower = text.lower()
        words = re.findall(r"\w+", text_lower)

        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.7, 0.3 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.7, 0.3 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.3

        return {
            "sentiment": sentiment,
            "emotion": self._map_sentiment_to_emotion(sentiment),
            "confidence": float(confidence),
            "key_feature": self._extract_key_feature(text),
            "emotional_driver": self._extract_emotional_driver(text, sentiment),
            "summary": f"Rule-based analysis: {positive_count} positive, {negative_count} negative words",
        }

    def _construct_prompt(self, text: str) -> str:
        """Construct the prompt for the LLM."""
        return f"""
        Analyze the sentiment and emotions expressed in the following social media post.
        Return your response as a valid JSON object with these exact keys:
        - "sentiment": either "positive", "negative", or "neutral"
        - "emotion": one of "joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", or "neutral"
        - "confidence": a number between 0 and 1 indicating confidence level
        - "key_feature": the main topic or subject being discussed
        - "emotional_driver": what is driving the sentiment (e.g., frustration, excitement, satisfaction, concern, disappointment)
        - "summary": a brief 1-2 sentence summary of the post's sentiment in context
        
        Post: {text[:500]}
        
        JSON Response:
        """

    def _parse_response(self, response_text: str) -> Dict:
        """Parse the LLM response."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return self._validate_result(result)
            else:
                return None
        except Exception as e:
            log_message(f"Error parsing response: {e}", level="WARNING")
            return None

    def _validate_result(self, result: Dict) -> Dict:
        """Validate and normalize the analysis result."""
        return {
            "sentiment": self._validate_sentiment(result.get("sentiment", "neutral")),
            "emotion": self._validate_emotion(result.get("emotion", "neutral")),
            "confidence": float(result.get("confidence", 0.5)),
            "key_feature": str(result.get("key_feature", "not specified"))[:100],
            "emotional_driver": str(result.get("emotional_driver", "not specified"))[
                :100
            ],
            "summary": str(result.get("summary", "No summary provided"))[:200],
        }

    def _validate_sentiment(self, sentiment: str) -> str:
        """Validate sentiment category."""
        sentiment = sentiment.lower()
        if sentiment in ["positive", "negativo"]:
            return "positive"
        elif sentiment in ["negative", "negativo"]:
            return "negative"
        else:
            return "neutral"

    def _validate_emotion(self, emotion: str) -> str:
        """Validate emotion category."""
        valid_emotions = [
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "trust",
            "anticipation",
            "neutral",
        ]
        emotion = emotion.lower()
        return emotion if emotion in valid_emotions else "neutral"

    def _map_sentiment_to_emotion(self, sentiment: str) -> str:
        """Map sentiment to a basic emotion."""
        mapping = {"positive": "joy", "negative": "sadness", "neutral": "neutral"}
        return mapping.get(sentiment, "neutral")

    def _extract_key_feature(self, text: str) -> str:
        """Extract key feature from text (simplified)."""
        # Look for common topics
        topics = [
            "product",
            "service",
            "company",
            "brand",
            "tech",
            "phone",
            "car",
            "movie",
            "game",
            "book",
            "music",
            "food",
            "travel",
            "politics",
            "climate",
            "economy",
            "health",
            "education",
        ]

        text_lower = text.lower()
        for topic in topics:
            if topic in text_lower:
                return topic
        return "general"

    def _extract_emotional_driver(self, text: str, sentiment: str) -> str:
        """Extract emotional driver (simplified)."""
        text_lower = text.lower()

        if sentiment == "positive":
            drivers = ["excitement", "satisfaction", "happiness", "gratitude", "hope"]
        elif sentiment == "negative":
            drivers = ["frustration", "disappointment", "concern", "anger", "sadness"]
        else:
            drivers = ["neutral", "indifferent", "uncertain"]

        for driver in drivers:
            if driver in text_lower:
                return driver

        return drivers[0]

    def _get_empty_result(self, error_message: str = "") -> Dict:
        """Return an empty result dictionary."""
        return {
            "sentiment": "neutral",
            "emotion": "neutral",
            "confidence": 0.0,
            "key_feature": "error",
            "emotional_driver": "error",
            "summary": f"Analysis failed: {error_message}",
        }

    def analyze_posts_batch(
        self, df: pd.DataFrame, text_column: str = "cleaned_text", batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Analyze multiple posts in batch.

        Args:
            df: DataFrame containing posts
            text_column: Column name containing text to analyze
            batch_size: Number of posts to process in one batch

        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty:
            return df

        df_results = df.copy()

        # Initialize sentiment columns
        df_results["sentiment"] = ""
        df_results["emotion"] = ""
        df_results["confidence"] = 0.0
        df_results["key_feature"] = ""
        df_results["emotional_driver"] = ""
        df_results["sentiment_summary"] = ""
        df_results["analysis_method"] = ""

        total_posts = len(df_results)
        log_message(
            f"Analyzing {total_posts} posts with method: {self.active_method or 'auto'}"
        )

        for i in range(0, total_posts, batch_size):
            batch = df_results.iloc[i : i + batch_size]

            for idx, row in batch.iterrows():
                text = (
                    row[text_column] if text_column in row else row.get("full_text", "")
                )

                if text and len(text.strip()) >= 3:
                    result = self.analyze_post(text)
                    df_results.loc[idx, "sentiment"] = result["sentiment"]
                    df_results.loc[idx, "emotion"] = result["emotion"]
                    df_results.loc[idx, "confidence"] = result["confidence"]
                    df_results.loc[idx, "key_feature"] = result["key_feature"]
                    df_results.loc[idx, "emotional_driver"] = result["emotional_driver"]
                    df_results.loc[idx, "sentiment_summary"] = result["summary"]
                    df_results.loc[idx, "analysis_method"] = self.active_method
                else:
                    df_results.loc[idx, "sentiment"] = "neutral"
                    df_results.loc[idx, "emotion"] = "neutral"
                    df_results.loc[idx, "confidence"] = 0.0
                    df_results.loc[idx, "key_feature"] = "empty_text"
                    df_results.loc[idx, "emotional_driver"] = "empty_text"
                    df_results.loc[idx, "sentiment_summary"] = (
                        "Empty or very short text"
                    )
                    df_results.loc[idx, "analysis_method"] = "none"

            processed = min(i + batch_size, total_posts)
            log_message(f"Processed {processed}/{total_posts} posts")

            # Rate limiting for Gemini
            if self.gemini_available and self.active_method == "gemini":
                time.sleep(0.5)

        log_message(f"Completed sentiment analysis for {total_posts} posts")
        return df_results

    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
        """Get sentiment distribution from analyzed DataFrame."""
        if "sentiment" not in df.columns:
            return {}

        distribution = df["sentiment"].value_counts().to_dict()
        total = sum(distribution.values())

        percentages = {k: (v / total) * 100 for k, v in distribution.items()}

        return {"counts": distribution, "percentages": percentages, "total": total}

    def get_emotion_distribution(self, df: pd.DataFrame) -> Dict:
        """Get emotion distribution from analyzed DataFrame."""
        if "emotion" not in df.columns:
            return {}

        distribution = df["emotion"].value_counts().to_dict()
        total = sum(distribution.values())

        percentages = {k: (v / total) * 100 for k, v in distribution.items()}

        return {"counts": distribution, "percentages": percentages, "total": total}

    def get_active_method(self) -> str:
        """Get the currently active analysis method."""
        return self.active_method
