"""
Configuration module for the Real-Time Social Media Trend and Sentiment Analysis System.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Main configuration class for the application."""

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SocialMediaAnalyzer/1.0")

    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/social_media_data.db")

    # API Rate Limits
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 60))
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 1000))

    # Reddit Configuration
    REDDIT_LIMIT = 100  # Number of posts to fetch per request

    # Sentiment Categories
    SENTIMENT_CATEGORIES = ["positive", "negative", "neutral"]

    # Emotion Categories
    EMOTION_CATEGORIES = [
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

    # Trend Analysis
    TREND_INTERVALS = ["hourly", "daily", "weekly"]

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present."""
        required_vars = ["GEMINI_API_KEY"]
        missing_vars = []

        for var in required_vars:
            if not getattr(cls, var, None):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        return True

    @classmethod
    def get_database_uri(cls):
        """Get the database URI for SQLAlchemy."""
        return f"sqlite:///{cls.DATABASE_PATH}"
