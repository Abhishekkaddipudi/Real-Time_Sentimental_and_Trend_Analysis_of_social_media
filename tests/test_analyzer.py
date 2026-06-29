"""
Unit tests for the sentiment and trend analyzers.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.trend_analyzer import TrendAnalyzer
from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    """Test cases for TextCleaner."""

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello, World! This is a test."
        cleaned = self.cleaner.clean_text(text)
        self.assertEqual(cleaned, "hello world test")

    def test_clean_text_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("http", cleaned)

    def test_clean_text_mentions(self):
        """Test mention removal."""
        text = "Hello @username how are you?"
        cleaned = self.cleaner.clean_text(text, remove_mentions=True)
        self.assertNotIn("@username", cleaned)

    def test_clean_text_hashtags(self):
        """Test hashtag handling."""
        text = "I love #Python programming!"
        cleaned = self.cleaner.clean_text(text, remove_hashtags=True)
        self.assertNotIn("#", cleaned)
        self.assertIn("python", cleaned)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer."""

    @patch("src.analysis.sentiment_analyzer.genai")
    def test_analyze_sentiment_positive(self, mock_genai):
        """Test positive sentiment analysis."""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = '{"sentiment": "positive", "emotion": "joy", "confidence": 0.95, "key_feature": "product", "explanation": "User expresses happiness"}'

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("I absolutely love this product!")

        self.assertEqual(result["sentiment"], "positive")
        self.assertIn("emotion", result)
        self.assertGreater(result["confidence"], 0.5)

    @patch("src.analysis.sentiment_analyzer.genai")
    def test_analyze_sentiment_negative(self, mock_genai):
        """Test negative sentiment analysis."""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = '{"sentiment": "negative", "emotion": "frustration", "confidence": 0.90, "key_feature": "service", "explanation": "User complains about service"}'

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("Terrible customer service!")

        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["emotion"], "frustration")

    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("")

        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["emotion"], "unknown")


class TestTrendAnalyzer(unittest.TestCase):
    """Test cases for TrendAnalyzer."""

    def setUp(self):
        self.analyzer = TrendAnalyzer()

        # Create sample data
        base_date = datetime.now()
        self.sample_data = []
        for i in range(30):
            self.sample_data.append(
                {
                    "created_utc": base_date + timedelta(hours=i),
                    "engagement": 10 + i * 2 + (i % 5) * 3,
                    "id": f"post_{i}",
                }
            )

        self.df = pd.DataFrame(self.sample_data)

    def test_calculate_trend_velocity(self):
        """Test trend velocity calculation."""
        result = self.analyzer.calculate_trend_velocity(self.df)

        self.assertIsNotNone(result)
        self.assertIn("velocity", result.columns)
        self.assertIn("phase", result.columns)

    def test_detect_trend_lifecycle(self):
        """Test trend lifecycle detection."""
        result = self.analyzer.detect_trend_lifecycle(self.df, "test")

        self.assertEqual(result["keyword"], "test")
        self.assertIn("total_posts", result)
        self.assertIn("growth_rate", result)

    def test_calculate_trend_score(self):
        """Test trend score calculation."""
        score = self.analyzer.calculate_trend_score(self.df)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 10)

    def test_find_emerging_trends(self):
        """Test emerging trends detection."""
        # Add keyword column
        self.df["keyword"] = "test"

        trends = self.analyzer.find_emerging_trends(self.df, keyword_column="keyword")

        self.assertIsInstance(trends, list)


if __name__ == "__main__":
    unittest.main()
