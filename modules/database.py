"""
Database Module - Manages database operations using SQLAlchemy and SQLite.
Includes automatic schema migration for backward compatibility.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    func,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from config.config import Config
from utils.helpers import log_message

Base = declarative_base()


class SocialMediaPost(Base):
    """SQLAlchemy model for social media posts."""

    __tablename__ = "social_media_posts"

    id = Column(String(50), primary_key=True)
    title = Column(Text)
    text = Column(Text)
    full_text = Column(Text)
    cleaned_text = Column(Text)
    author = Column(String(100))
    created_utc = Column(DateTime)
    date = Column(DateTime)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    score = Column(Integer)
    num_comments = Column(Integer)
    total_engagement = Column(Integer)
    url = Column(Text)
    source = Column(String(50))
    subreddit = Column(String(100))

    # Sentiment analysis results
    sentiment = Column(String(20))
    emotion = Column(String(30))
    confidence = Column(Float)
    key_feature = Column(Text)
    emotional_driver = Column(Text)
    sentiment_summary = Column(Text)
    analysis_method = Column(String(50), default="unknown")  # Added new column

    # Metadata
    hashtags = Column(Text)  # JSON string
    text_length = Column(Integer)
    word_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DatabaseManager:
    """Manages database operations for the sentiment analysis system."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or Config.DATABASE_PATH
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
        log_message(f"Database Manager initialized with path: {self.db_path}")

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Ensure data directory exists
            import os

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            self.engine = create_engine(
                f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False}
            )

            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)

            # Run schema migration
            self._migrate_schema()

            self.SessionLocal = sessionmaker(bind=self.engine)
            log_message("Database tables created/verified successfully")
        except Exception as e:
            log_message(f"Error initializing database: {e}", level="ERROR")
            raise

    def _migrate_schema(self):
        """
        Migrate database schema to add new columns if they don't exist.
        """
        try:
            inspector = inspect(self.engine)
            columns = [
                col["name"] for col in inspector.get_columns("social_media_posts")
            ]

            # Check if analysis_method column exists
            if "analysis_method" not in columns:
                log_message("Adding analysis_method column to database...")
                with self.engine.connect() as conn:
                    conn.execute(
                        "ALTER TABLE social_media_posts ADD COLUMN analysis_method VARCHAR(50) DEFAULT 'unknown'"
                    )
                    conn.commit()
                log_message("analysis_method column added successfully")

            # Check if other new columns exist and add them if needed
            new_columns = {
                "emotional_driver": "TEXT",
                "sentiment_summary": "TEXT",
                "analysis_method": "VARCHAR(50)",
            }

            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    log_message(f"Adding {col_name} column to database...")
                    with self.engine.connect() as conn:
                        conn.execute(
                            f"ALTER TABLE social_media_posts ADD COLUMN {col_name} {col_type}"
                        )
                        conn.commit()
                    log_message(f"{col_name} column added successfully")

        except Exception as e:
            log_message(f"Error during schema migration: {e}", level="WARNING")
            # Continue execution - schema migration is not critical for basic functionality

    def get_session(self) -> Session:
        """Get a new database session."""
        if not self.SessionLocal:
            self._initialize_database()
        return self.SessionLocal()

    def insert_post(self, post_data: Dict) -> bool:
        """
        Insert a single post into the database.

        Args:
            post_data: Dictionary containing post data

        Returns:
            Boolean indicating success
        """
        try:
            session = self.get_session()

            # Check if post already exists
            post_id = str(post_data.get("id", ""))
            existing = session.query(SocialMediaPost).filter_by(id=post_id).first()

            if existing:
                log_message(f"Post {post_id} already exists, updating...")
                return self.update_post(post_id, post_data)

            # Create new post
            post = self._dict_to_post(post_data)
            session.add(post)
            session.commit()
            log_message(f"Inserted post {post_id} successfully")
            return True

        except Exception as e:
            log_message(f"Error inserting post: {e}", level="ERROR")
            if "session" in locals():
                session.rollback()
            return False

    def insert_posts_batch(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Insert multiple posts in batch.

        Args:
            df: DataFrame containing post data

        Returns:
            Dictionary with success and failure counts
        """
        stats = {"success": 0, "failed": 0, "total": len(df)}

        if df.empty:
            log_message("Empty DataFrame provided for batch insert", level="WARNING")
            return stats

        # Process in batches for better performance
        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            for _, row in batch.iterrows():
                post_data = row.to_dict()
                if self.insert_post(post_data):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1

        log_message(f"Batch insert complete: {stats}")
        return stats

    def _dict_to_post(self, data: Dict) -> SocialMediaPost:
        """
        Convert dictionary to SocialMediaPost object.

        Args:
            data: Dictionary with post data

        Returns:
            SocialMediaPost object
        """
        # Handle hashtags as JSON string
        hashtags = data.get("hashtags", [])
        if isinstance(hashtags, list):
            import json

            hashtags = json.dumps(hashtags)
        elif not isinstance(hashtags, str):
            hashtags = "[]"

        # Get analysis method from data or default
        analysis_method = data.get("analysis_method", "unknown")
        if analysis_method == "none" or not analysis_method:
            analysis_method = "unknown"

        return SocialMediaPost(
            id=str(data.get("id", "")),
            title=str(data.get("title", ""))[:500],
            text=str(data.get("text", ""))[:1000],
            full_text=str(data.get("full_text", ""))[:2000],
            cleaned_text=str(data.get("cleaned_text", ""))[:2000],
            author=str(data.get("author", "unknown"))[:100],
            created_utc=self._parse_datetime(data.get("created_utc")),
            date=self._parse_datetime(data.get("date", data.get("created_utc"))),
            hour=int(data.get("hour", 0)),
            day_of_week=int(data.get("day_of_week", 0)),
            score=int(data.get("score", 0)),
            num_comments=int(data.get("num_comments", 0)),
            total_engagement=int(data.get("total_engagement", 0)),
            url=str(data.get("url", ""))[:500],
            source=str(data.get("source", "unknown"))[:50],
            subreddit=str(data.get("subreddit", "unknown"))[:100],
            sentiment=str(data.get("sentiment", "neutral"))[:20],
            emotion=str(data.get("emotion", "neutral"))[:30],
            confidence=float(data.get("confidence", 0.0)),
            key_feature=str(data.get("key_feature", ""))[:200],
            emotional_driver=str(data.get("emotional_driver", ""))[:200],
            sentiment_summary=str(data.get("sentiment_summary", ""))[:500],
            analysis_method=str(analysis_method)[:50],
            hashtags=hashtags,
            text_length=int(data.get("text_length", 0)),
            word_count=int(data.get("word_count", 0)),
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return pd.to_datetime(value).to_pydatetime()
            except:
                return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except:
                return None
        return None

    def update_post(self, post_id: str, data: Dict) -> bool:
        """
        Update an existing post.

        Args:
            post_id: ID of the post to update
            data: Updated data dictionary

        Returns:
            Boolean indicating success
        """
        try:
            session = self.get_session()
            post = session.query(SocialMediaPost).filter_by(id=post_id).first()

            if not post:
                log_message(f"Post {post_id} not found for update", level="WARNING")
                return False

            # Update fields
            for key, value in data.items():
                if hasattr(post, key):
                    if key == "created_utc":
                        value = self._parse_datetime(value)
                    setattr(post, key, value)

            post.updated_at = datetime.now()
            session.commit()
            log_message(f"Updated post {post_id} successfully")
            return True

        except Exception as e:
            log_message(f"Error updating post {post_id}: {e}", level="ERROR")
            if "session" in locals():
                session.rollback()
            return False

    def get_posts(
        self, filters: Optional[Dict] = None, limit: int = 100, offset: int = 0
    ) -> pd.DataFrame:
        """
        Retrieve posts from the database.

        Args:
            filters: Dictionary of filter conditions
            limit: Maximum number of posts to retrieve
            offset: Number of posts to skip

        Returns:
            DataFrame with retrieved posts
        """
        try:
            session = self.get_session()
            query = session.query(SocialMediaPost)

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(SocialMediaPost, key):
                        query = query.filter(getattr(SocialMediaPost, key) == value)

            # Apply ordering and limits
            query = query.order_by(SocialMediaPost.created_utc.desc())
            query = query.limit(limit).offset(offset)

            posts = query.all()
            result = self._posts_to_dataframe(posts)

            log_message(f"Retrieved {len(result)} posts")
            return result

        except Exception as e:
            log_message(f"Error retrieving posts: {e}", level="ERROR")
            return pd.DataFrame()

    def _posts_to_dataframe(self, posts: List[SocialMediaPost]) -> pd.DataFrame:
        """
        Convert list of posts to DataFrame.

        Args:
            posts: List of SocialMediaPost objects

        Returns:
            DataFrame with post data
        """
        if not posts:
            return pd.DataFrame()

        data = []
        for post in posts:
            post_dict = {
                "id": post.id,
                "title": post.title,
                "text": post.text,
                "full_text": post.full_text,
                "cleaned_text": post.cleaned_text,
                "author": post.author,
                "created_utc": post.created_utc,
                "date": post.date,
                "hour": post.hour,
                "day_of_week": post.day_of_week,
                "score": post.score,
                "num_comments": post.num_comments,
                "total_engagement": post.total_engagement,
                "url": post.url,
                "source": post.source,
                "subreddit": post.subreddit,
                "sentiment": post.sentiment,
                "emotion": post.emotion,
                "confidence": post.confidence,
                "key_feature": post.key_feature,
                "emotional_driver": post.emotional_driver,
                "sentiment_summary": post.sentiment_summary,
                "analysis_method": post.analysis_method,
                "text_length": post.text_length,
                "word_count": post.word_count,
            }

            # Parse hashtags
            try:
                import json

                post_dict["hashtags"] = (
                    json.loads(post.hashtags) if post.hashtags else []
                )
            except:
                post_dict["hashtags"] = []

            data.append(post_dict)

        return pd.DataFrame(data)

    def get_statistics(self) -> Dict:
        """
        Get database statistics with error handling for missing columns.

        Returns:
            Dictionary with statistics
        """
        try:
            session = self.get_session()

            # Total count
            total_posts = session.query(func.count(SocialMediaPost.id)).scalar() or 0

            stats = {
                "total_posts": total_posts,
                "sentiment_distribution": {},
                "emotion_distribution": {},
                "source_distribution": {},
                "method_distribution": {},
                "oldest_post": None,
                "newest_post": None,
            }

            # Try to get sentiment counts (may fail if column doesn't exist)
            try:
                sentiment_counts = (
                    session.query(
                        SocialMediaPost.sentiment,
                        func.count(SocialMediaPost.id).label("count"),
                    )
                    .group_by(SocialMediaPost.sentiment)
                    .all()
                )
                stats["sentiment_distribution"] = {
                    s or "unknown": c for s, c in sentiment_counts
                }
            except Exception as e:
                log_message(
                    f"Could not get sentiment distribution: {e}", level="WARNING"
                )

            # Try to get emotion counts
            try:
                emotion_counts = (
                    session.query(
                        SocialMediaPost.emotion,
                        func.count(SocialMediaPost.id).label("count"),
                    )
                    .group_by(SocialMediaPost.emotion)
                    .all()
                )
                stats["emotion_distribution"] = {
                    e or "unknown": c for e, c in emotion_counts
                }
            except Exception as e:
                log_message(f"Could not get emotion distribution: {e}", level="WARNING")

            # Try to get source counts
            try:
                source_counts = (
                    session.query(
                        SocialMediaPost.source,
                        func.count(SocialMediaPost.id).label("count"),
                    )
                    .group_by(SocialMediaPost.source)
                    .all()
                )
                stats["source_distribution"] = {
                    s or "unknown": c for s, c in source_counts
                }
            except Exception as e:
                log_message(f"Could not get source distribution: {e}", level="WARNING")

            # Try to get method counts
            try:
                method_counts = (
                    session.query(
                        SocialMediaPost.analysis_method,
                        func.count(SocialMediaPost.id).label("count"),
                    )
                    .group_by(SocialMediaPost.analysis_method)
                    .all()
                )
                stats["method_distribution"] = {
                    m or "unknown": c for m, c in method_counts
                }
            except Exception as e:
                log_message(f"Could not get method distribution: {e}", level="WARNING")

            # Date range
            try:
                stats["oldest_post"] = session.query(
                    func.min(SocialMediaPost.created_utc)
                ).scalar()
                stats["newest_post"] = session.query(
                    func.max(SocialMediaPost.created_utc)
                ).scalar()
            except Exception as e:
                log_message(f"Could not get date range: {e}", level="WARNING")

            log_message(
                f"Retrieved database statistics: {stats['total_posts']} total posts"
            )
            return stats

        except Exception as e:
            log_message(f"Error getting statistics: {e}", level="ERROR")
            return {"total_posts": 0, "error": str(e)}

    def get_sentiment_over_time(self, interval: str = "D") -> pd.DataFrame:
        """Get sentiment counts over time."""
        try:
            session = self.get_session()

            # Use SQLite date functions
            if interval == "H":
                date_format = "%Y-%m-%d %H:00:00"
            elif interval == "D":
                date_format = "%Y-%m-%d"
            elif interval == "W":
                date_format = "%Y-%W"
            elif interval == "M":
                date_format = "%Y-%m"
            else:
                date_format = "%Y-%m-%d"

            results = (
                session.query(
                    func.strftime(date_format, SocialMediaPost.created_utc).label(
                        "time_period"
                    ),
                    SocialMediaPost.sentiment,
                    func.count(SocialMediaPost.id).label("count"),
                )
                .group_by(
                    func.strftime(date_format, SocialMediaPost.created_utc),
                    SocialMediaPost.sentiment,
                )
                .order_by("time_period")
                .all()
            )

            data = []
            for r in results:
                data.append(
                    {"time_period": r[0], "sentiment": r[1] or "unknown", "count": r[2]}
                )

            return pd.DataFrame(data)

        except Exception as e:
            log_message(f"Error getting sentiment over time: {e}", level="ERROR")
            return pd.DataFrame()

    def get_top_features(self, limit: int = 10) -> List[Dict]:
        """Get top key features from posts."""
        try:
            session = self.get_session()

            results = (
                session.query(
                    SocialMediaPost.key_feature,
                    func.count(SocialMediaPost.id).label("count"),
                )
                .filter(
                    SocialMediaPost.key_feature != "",
                    SocialMediaPost.key_feature != "error",
                    SocialMediaPost.key_feature != "empty_text",
                    SocialMediaPost.key_feature.isnot(None),
                )
                .group_by(SocialMediaPost.key_feature)
                .order_by(func.count(SocialMediaPost.id).desc())
                .limit(limit)
                .all()
            )

            return [{"feature": f, "count": c} for f, c in results if f]

        except Exception as e:
            log_message(f"Error getting top features: {e}", level="ERROR")
            return []

    def get_engagement_stats(self) -> Dict:
        """Get engagement statistics."""
        try:
            session = self.get_session()

            stats = session.query(
                func.avg(SocialMediaPost.total_engagement).label("avg_engagement"),
                func.max(SocialMediaPost.total_engagement).label("max_engagement"),
                func.min(SocialMediaPost.total_engagement).label("min_engagement"),
                func.sum(SocialMediaPost.total_engagement).label("total_engagement"),
                func.avg(SocialMediaPost.score).label("avg_score"),
                func.avg(SocialMediaPost.num_comments).label("avg_comments"),
            ).first()

            if stats:
                return {
                    "avg_engagement": float(stats[0] or 0),
                    "max_engagement": int(stats[1] or 0),
                    "min_engagement": int(stats[2] or 0),
                    "total_engagement": int(stats[3] or 0),
                    "avg_score": float(stats[4] or 0),
                    "avg_comments": float(stats[5] or 0),
                }
            return {}

        except Exception as e:
            log_message(f"Error getting engagement stats: {e}", level="ERROR")
            return {}

    def cleanup_old_data(self, days: int = 30) -> int:
        """Delete posts older than specified days."""
        try:
            session = self.get_session()
            cutoff_date = datetime.now() - timedelta(days=days)

            deleted = (
                session.query(SocialMediaPost)
                .filter(SocialMediaPost.created_utc < cutoff_date)
                .delete()
            )

            session.commit()
            log_message(f"Deleted {deleted} old posts")
            return deleted

        except Exception as e:
            log_message(f"Error cleaning up old data: {e}", level="ERROR")
            if "session" in locals():
                session.rollback()
            return 0

    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            session = self.get_session()
            deleted = session.query(SocialMediaPost).delete()
            session.commit()
            log_message(f"Cleared {deleted} posts from database")
            return True
        except Exception as e:
            log_message(f"Error clearing database: {e}", level="ERROR")
            if "session" in locals():
                session.rollback()
            return False

    def get_table_info(self) -> Dict:
        """Get information about the database table schema."""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns("social_media_posts")
            return {
                "table_name": "social_media_posts",
                "columns": [
                    {"name": col["name"], "type": str(col["type"])} for col in columns
                ],
                "column_count": len(columns),
            }
        except Exception as e:
            log_message(f"Error getting table info: {e}", level="ERROR")
            return {"error": str(e)}


# Import timedelta for cleanup
from datetime import timedelta
