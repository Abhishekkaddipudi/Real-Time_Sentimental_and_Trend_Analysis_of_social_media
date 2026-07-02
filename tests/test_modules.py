"""
Unit tests for the Social Media Trend and Sentiment Analysis System.
"""

import unittest
import pandas as pd
import json
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import (
    DataIngestion,
    DataPreprocessing,
    SentimentAnalysis,
    TrendAnalysis,
    DatabaseManager,
)
from config.config import Config


class TestDataIngestion(unittest.TestCase):
    """Test cases for DataIngestion module."""

    def setUp(self):
        self.ingestion = DataIngestion(source="synthetic")

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        posts = self.ingestion.fetch_from_api("synthetic", count=10)
        self.assertEqual(len(posts), 10)
        self.assertIn("id", posts[0])
        self.assertIn("title", posts[0])
        self.assertIn("text", posts[0])

    def test_keyword_filtering(self):
        """Test keyword filtering in synthetic data."""
        posts = self.ingestion.fetch_from_api("synthetic", count=20, keyword="AI")
        for post in posts:
            self.assertTrue("AI" in post["title"] or "AI" in post["text"])


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for DataPreprocessing module."""

    def setUp(self):
        self.preprocessor = DataPreprocessing()

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        text = "Check out this awesome post! https://example.com @user #awesome"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn("#awesome", cleaned)

    def test_stopword_removal(self):
        """Test stopword removal."""
        text = "This is a very interesting post about technology"
        cleaned = self.preprocessor.remove_stopwords(text)
        self.assertNotIn("is", cleaned)
        self.assertNotIn("a", cleaned)

    def test_hashtag_extraction(self):
        """Test hashtag extraction."""
        text = "This is #awesome and #great work"
        hashtags = self.preprocessor.extract_hashtags(text)
        self.assertEqual(len(hashtags), 2)
        self.assertIn("#awesome", hashtags)


class TestSentimentAnalysis(unittest.TestCase):
    """Test cases for SentimentAnalysis module."""

    def setUp(self):
        self.analyzer = SentimentAnalysis()

    def test_analyze_positive_text(self):
        """Test sentiment analysis on positive text."""
        result = self.analyzer.analyze_post("This is amazing! I love it!")
        self.assertIn(result["sentiment"], ["positive", "neutral"])

    def test_analyze_negative_text(self):
        """Test sentiment analysis on negative text."""
        result = self.analyzer.analyze_post("This is terrible! I hate it!")
        self.assertIn(result["sentiment"], ["negative", "neutral"])

    def test_empty_text(self):
        """Test sentiment analysis on empty text."""
        result = self.analyzer.analyze_post("")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["confidence"], 0.0)

    def test_validate_sentiment_aliases(self):
        """Test localized sentiment aliases normalize correctly."""
        self.assertEqual(self.analyzer._validate_sentiment("positivo"), "positive")
        self.assertEqual(self.analyzer._validate_sentiment("negativo"), "negative")


class TestTrendAnalysis(unittest.TestCase):
    """Test cases for TrendAnalysis module."""

    def setUp(self):
        self.trend_analyzer = TrendAnalysis()

    def test_velocity_calculation(self):
        """Test velocity calculation."""
        df = pd.DataFrame(
            {
                "created_utc": pd.date_range("2024-01-01", periods=10, freq="H"),
                "total_engagement": [1, 3, 6, 10, 15, 12, 8, 5, 3, 1],
                "score": [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                "num_comments": [0, 1, 3, 6, 10, 8, 5, 3, 2, 1],
            }
        )
        velocity_df = self.trend_analyzer.calculate_velocity(df)
        self.assertIn("velocity", velocity_df.columns)
        self.assertIn("trend_phase", velocity_df.columns)

    def test_lifecycle_identification(self):
        """Test trend lifecycle identification."""
        df = pd.DataFrame(
            {
                "created_utc": pd.date_range("2024-01-01", periods=10, freq="H"),
                "total_engagement": [1, 3, 6, 10, 15, 12, 8, 5, 3, 1],
            }
        )
        lifecycle = self.trend_analyzer.identify_trend_lifecycle(df)
        self.assertEqual(lifecycle["status"], "success")
        self.assertIn("duration_hours", lifecycle)
        self.assertIn("peak_engagement", lifecycle)


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager module."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")

    def test_insert_post(self):
        """Test inserting a single post."""
        post_data = {
            "id": "test_001",
            "title": "Test Post",
            "text": "This is a test post",
            "author": "test_user",
            "created_utc": datetime.now(),
            "score": 100,
            "num_comments": 10,
            "source": "test",
        }
        result = self.db.insert_post(post_data)
        self.assertTrue(result)

    def test_retrieve_posts(self):
        """Test retrieving posts."""
        # Insert test data
        for i in range(5):
            post_data = {
                "id": f"test_{i:03d}",
                "title": f"Test Post {i}",
                "text": f"This is test post {i}",
                "author": "test_user",
                "created_utc": datetime.now() - timedelta(hours=i),
                "score": 100 - i * 10,
                "num_comments": 10 - i,
                "source": "test",
            }
            self.db.insert_post(post_data)

        posts = self.db.get_posts(limit=3)
        self.assertEqual(len(posts), 3)

    def test_statistics(self):
        """Test statistics retrieval."""
        # Insert test data with different sentiments
        sentiments = ["positive", "negative", "neutral"]
        for i, sentiment in enumerate(sentiments):
            post_data = {
                "id": f"stat_{i:03d}",
                "title": f"Stat Test {i}",
                "text": f"This is a {sentiment} post",
                "author": "test_user",
                "created_utc": datetime.now(),
                "sentiment": sentiment,
                "score": 10,
                "num_comments": 5,
                "source": "test",
            }
            self.db.insert_post(post_data)

        stats = self.db.get_statistics()
        self.assertIn("total_posts", stats)
        self.assertEqual(stats["total_posts"], 3)


def run_tests():
    """Run all test cases."""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
