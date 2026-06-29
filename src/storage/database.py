"""
Database management module for storing social media analysis data.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from config.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class Post(Base):
    """SQLAlchemy model for social media posts."""

    __tablename__ = "posts"

    id = Column(String(50), primary_key=True)
    platform = Column(String(20))
    text = Column(Text)
    cleaned_text = Column(Text)
    author = Column(String(100))
    created_at = Column(DateTime)
    engagement = Column(Integer, default=0)
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    url = Column(String(500))
    subreddit = Column(String(100))

    # Sentiment analysis results
    sentiment = Column(String(20))
    emotion = Column(String(30))
    sentiment_confidence = Column(Float)
    key_feature = Column(String(200))
    sentiment_explanation = Column(Text)

    # Additional metadata
    metadata = Column(JSONB)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "engagement": self.engagement,
            "score": self.score,
            "num_comments": self.num_comments,
            "url": self.url,
            "subreddit": self.subreddit,
            "sentiment": self.sentiment,
            "emotion": self.emotion,
            "sentiment_confidence": self.sentiment_confidence,
            "key_feature": self.key_feature,
            "sentiment_explanation": self.sentiment_explanation,
            "metadata": self.metadata,
        }


class DatabaseManager:
    """
    Manages database operations for storing and retrieving social media data.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            connection_string: Database connection string (optional)
        """
        self.connection_string = connection_string or settings.database_url
        self.engine = None
        self.Session = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        if not self.connection_string:
            logger.warning(
                "No database connection string provided. Using in-memory storage."
            )
            return

        try:
            self.engine = create_engine(self.connection_string)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.engine = None
            self.Session = None

    def _get_session(self):
        """Get a database session."""
        if not self.Session:
            raise Exception("Database not initialized")
        return self.Session()

    def insert_post(self, post_data: Dict[str, Any]) -> bool:
        """
        Insert a single post into the database.

        Args:
            post_data: Post data dictionary

        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.warning("Database not available. Data not persisted.")
            return False

        try:
            session = self._get_session()

            # Convert to model
            post = Post(
                id=post_data.get("id", f"post_{int(datetime.now().timestamp())}"),
                platform=post_data.get("platform", "unknown"),
                text=post_data.get("text", ""),
                cleaned_text=post_data.get("cleaned_text", ""),
                author=post_data.get("author", "unknown"),
                created_at=post_data.get("created_utc", datetime.now()),
                engagement=post_data.get("engagement", 0),
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                url=post_data.get("url", ""),
                subreddit=post_data.get("subreddit", ""),
                sentiment=post_data.get("sentiment", "neutral"),
                emotion=post_data.get("emotion", "unknown"),
                sentiment_confidence=post_data.get("sentiment_confidence", 0.0),
                key_feature=post_data.get("key_feature", ""),
                sentiment_explanation=post_data.get("sentiment_explanation", ""),
                metadata=post_data.get("metadata", {}),
            )

            # Merge or insert
            session.merge(post)
            session.commit()
            session.close()

            return True

        except Exception as e:
            logger.error(f"Error inserting post: {e}")
            return False

    def insert_batch(self, posts: List[Dict[str, Any]]) -> int:
        """
        Insert multiple posts into the database.

        Args:
            posts: List of post data dictionaries

        Returns:
            Number of successfully inserted posts
        """
        if not self.engine:
            logger.warning("Database not available. Data not persisted.")
            return 0

        success_count = 0
        for post in posts:
            if self.insert_post(post):
                success_count += 1

        return success_count

    def get_posts(
        self,
        platform: Optional[str] = None,
        keyword: Optional[str] = None,
        sentiment: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query posts from the database.

        Args:
            platform: Filter by platform
            keyword: Filter by keyword in text
            sentiment: Filter by sentiment
            limit: Maximum number of posts to return
            since: Filter by created_at >= since
            until: Filter by created_at <= until

        Returns:
            List of post dictionaries
        """
        if not self.engine:
            logger.warning("Database not available. Returning empty list.")
            return []

        try:
            session = self._get_session()
            query = session.query(Post)

            if platform:
                query = query.filter(Post.platform == platform)

            if sentiment:
                query = query.filter(Post.sentiment == sentiment)

            if since:
                query = query.filter(Post.created_at >= since)

            if until:
                query = query.filter(Post.created_at <= until)

            if keyword:
                query = query.filter(
                    (Post.text.ilike(f"%{keyword}%"))
                    | (Post.cleaned_text.ilike(f"%{keyword}%"))
                    | (Post.key_feature.ilike(f"%{keyword}%"))
                )

            results = query.order_by(Post.created_at.desc()).limit(limit).all()
            session.close()

            return [post.to_dict() for post in results]

        except Exception as e:
            logger.error(f"Error querying posts: {e}")
            return []

    def get_sentiment_stats(
        self,
        platform: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get sentiment statistics from the database.

        Args:
            platform: Filter by platform
            since: Filter by created_at >= since
            until: Filter by created_at <= until

        Returns:
            Dictionary with sentiment statistics
        """
        if not self.engine:
            return {}

        try:
            session = self._get_session()
            query = session.query(Post)

            if platform:
                query = query.filter(Post.platform == platform)

            if since:
                query = query.filter(Post.created_at >= since)

            if until:
                query = query.filter(Post.created_at <= until)

            # Get counts
            total = query.count()

            if total == 0:
                return {"total": 0}

            # Sentiment counts
            sentiment_counts = {}
            for sentiment in ["positive", "negative", "neutral"]:
                count = query.filter(Post.sentiment == sentiment).count()
                sentiment_counts[sentiment] = count

            # Emotion counts
            emotion_counts = {}
            emotions = query.with_entities(Post.emotion).distinct().all()
            for emotion in emotions:
                if emotion[0]:
                    count = query.filter(Post.emotion == emotion[0]).count()
                    emotion_counts[emotion[0]] = count

            session.close()

            return {
                "total": total,
                "sentiment_distribution": sentiment_counts,
                "sentiment_percentages": {
                    k: (v / total * 100) for k, v in sentiment_counts.items()
                },
                "emotion_distribution": emotion_counts,
            }

        except Exception as e:
            logger.error(f"Error getting sentiment stats: {e}")
            return {}

    def get_trend_data(
        self,
        keyword: str,
        time_column: str = "created_at",
        aggregation: str = "hour",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get time-series trend data for a keyword.

        Args:
            keyword: Keyword to analyze
            time_column: Column to use for time
            aggregation: Time aggregation (hour, day, week)
            since: Start date
            until: End date

        Returns:
            DataFrame with trend data
        """
        if not self.engine:
            return pd.DataFrame()

        try:
            # Query posts with keyword
            posts = self.get_posts(
                keyword=keyword, since=since, until=until, limit=1000
            )

            if not posts:
                return pd.DataFrame()

            df = pd.DataFrame(posts)

            if time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column])

                # Group by aggregation period
                if aggregation == "hour":
                    df["period"] = df[time_column].dt.floor("H")
                elif aggregation == "day":
                    df["period"] = df[time_column].dt.floor("D")
                elif aggregation == "week":
                    df["period"] = df[time_column].dt.to_period("W").dt.start_time
                else:
                    df["period"] = df[time_column]

                # Aggregate engagement
                grouped = (
                    df.groupby("period")
                    .agg(
                        {
                            "id": "count",
                            "engagement": (
                                "sum" if "engagement" in df.columns else "count"
                            ),
                            "sentiment": lambda x: (
                                x.mode()[0] if not x.empty else "neutral"
                            ),
                        }
                    )
                    .reset_index()
                )

                grouped.columns = [
                    "period",
                    "post_count",
                    "engagement",
                    "dominant_sentiment",
                ]

                return grouped

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting trend data: {e}")
            return pd.DataFrame()
