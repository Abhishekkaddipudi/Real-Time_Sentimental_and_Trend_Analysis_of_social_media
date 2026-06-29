"""
Configuration settings for the Social Media Trend and Sentiment Analysis System.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini API
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # Reddit API
    reddit_client_id: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "SocialMediaAnalysis/1.0")
    
    # Twitter API
    twitter_bearer_token: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = os.getenv("TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = os.getenv("TWITTER_API_SECRET")
    
    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    mongodb_uri: Optional[str] = os.getenv("MONGODB_URI")
    
    # Application
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_posts_per_query: int = int(os.getenv("MAX_POSTS_PER_QUERY", "100"))
    
    # Analysis Settings
    sentiment_labels: List[str] = ["Positive", "Negative", "Neutral"]
    emotion_labels: List[str] = [
        "Joy", "Sadness", "Anger", "Fear", "Surprise", 
        "Disgust", "Trust", "Anticipation", "Frustration", 
        "Excitement", "Satisfaction", "Disappointment", "Concern"
    ]
    
    # Trend Analysis
    trend_time_window_hours: int = 24
    velocity_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()