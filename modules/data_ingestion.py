"""
Data Ingestion Module - Handles fetching data from social media platforms.
Supports Reddit API integration and synthetic data generation for testing.
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import praw
import requests
from utils.helpers import log_message, validate_date


class DataIngestion:
    """
    Handles data ingestion from social media platforms.
    Supports Reddit API and synthetic data generation.
    """

    def __init__(self, source: str = "reddit", config: Optional[Dict] = None):
        """
        Initialize the Data Ingestion module.

        Args:
            source: Source platform ('reddit', 'synthetic', 'twitter')
            config: Configuration dictionary
        """
        self.source = source
        self.config = config or {}
        self.reddit_client = None

        if source == "reddit":
            self._initialize_reddit_client()
        elif source == "synthetic":
            log_message("Using synthetic data generator for testing")

    def _initialize_reddit_client(self):
        """Initialize the Reddit API client."""
        try:
            from config.config import Config

            self.reddit_client = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT,
            )
            log_message("Reddit client initialized successfully")
        except Exception as e:
            log_message(f"Error initializing Reddit client: {e}", level="ERROR")
            raise

    def fetch_reddit_posts(
        self,
        subreddit: str = "all",
        limit: int = 100,
        time_filter: str = "day",
        keyword: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch posts from Reddit.

        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year')
            keyword: Optional keyword to filter posts

        Returns:
            List of post dictionaries
        """
        if not self.reddit_client:
            log_message("Reddit client not initialized, falling back to synthetic data")
            return self._generate_synthetic_data(limit, keyword)

        try:
            posts = []
            subreddit_obj = self.reddit_client.subreddit(subreddit)

            # Fetch posts based on time filter
            if time_filter == "hour":
                fetched_posts = subreddit_obj.hot(limit=limit)
            elif time_filter == "day":
                fetched_posts = subreddit_obj.top(time_filter="day", limit=limit)
            else:
                fetched_posts = subreddit_obj.top(time_filter=time_filter, limit=limit)

            for post in fetched_posts:
                # Filter by keyword if provided
                if (
                    keyword
                    and keyword.lower() not in post.title.lower()
                    and keyword.lower() not in post.selftext.lower()
                ):
                    continue

                posts.append(
                    {
                        "id": post.id,
                        "title": post.title,
                        "text": post.selftext[:500] if post.selftext else "",
                        "author": str(post.author) if post.author else "deleted",
                        "created_utc": datetime.fromtimestamp(post.created_utc),
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "url": post.url,
                        "source": "reddit",
                        "subreddit": str(subreddit),
                    }
                )

                # Respect rate limits
                time.sleep(0.5)

            log_message(f"Fetched {len(posts)} posts from r/{subreddit}")
            return posts

        except Exception as e:
            log_message(f"Error fetching Reddit posts: {e}", level="ERROR")
            # Fallback to synthetic data
            return self._generate_synthetic_data(limit, keyword)

    def _generate_synthetic_data(
        self, count: int = 100, keyword: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate synthetic social media data for testing.

        Args:
            count: Number of posts to generate
            keyword: Optional keyword to filter posts

        Returns:
            List of synthetic post dictionaries
        """
        sample_posts = [
            {
                "id": f"post_{i}",
                "title": titles[i % len(titles)],
                "text": texts[i % len(texts)],
                "author": f"user_{random.randint(1000, 9999)}",
                "created_utc": datetime.now() - timedelta(hours=random.randint(0, 48)),
                "score": random.randint(0, 1000),
                "num_comments": random.randint(0, 200),
                "url": f"https://example.com/post_{i}",
                "source": "synthetic",
                "subreddit": "test_subreddit",
            }
            for i in range(count)
        ]

        # Filter by keyword if provided
        if keyword:
            sample_posts = [
                p
                for p in sample_posts
                if keyword.lower() in p["title"].lower()
                or keyword.lower() in p["text"].lower()
            ]

        log_message(f"Generated {len(sample_posts)} synthetic posts")

        return sample_posts

    def fetch_twitter_posts(self, query: str, count: int = 100) -> List[Dict]:
        """
        Placeholder for Twitter API integration.
        Currently returns synthetic data.
        """
        log_message(
            "Twitter API integration not fully implemented, using synthetic data"
        )
        return self._generate_synthetic_data(count, query)

    def fetch_from_api(self, platform: str, **kwargs) -> List[Dict]:
        """
        Generic method to fetch data from any supported platform.

        Args:
            platform: Platform name ('reddit', 'twitter', 'synthetic')
            **kwargs: Platform-specific parameters

        Returns:
            List of post dictionaries
        """
        if platform == "reddit":
            return self.fetch_reddit_posts(**kwargs)
        elif platform == "twitter":
            return self.fetch_twitter_posts(**kwargs)
        elif platform == "synthetic":
            return self._generate_synthetic_data(**kwargs)
        else:
            raise ValueError(f"Unsupported platform: {platform}")


# Sample data for synthetic generation
titles = [
    "AI Revolution: The Future of Technology",
    "Climate Change Impacts on Global Economy",
    "New Product Launch: Game Changer",
    "Political Debate Heats Up Online",
    "Tech Company Faces Backlash Over Privacy",
    "Exciting Breakthrough in Medical Research",
    "Sports Star Shines in Championship",
    "Movie Release Breaks Box Office Records",
    "Social Media Platform Introduces New Features",
    "Economic Growth Slows Down in Q3",
]

texts = [
    "I'm absolutely thrilled about this new development! This is going to change everything we know about technology. The potential for innovation is incredible and I can't wait to see what comes next. #excited #future",
    "This is terrible news for our community. The government's decision to cut funding for education is a huge mistake that will affect generations to come. Very disappointed in this outcome.",
    "I'm not sure what to think about this situation. It could be good or it could be bad, I need more information before making a decision. The data seems mixed so far.",
    "Finally! The company has released their long-awaited update and it's absolutely amazing. The performance improvements are significant and the new UI is beautiful.",
    "I can't believe they would release a product with so many defects. This is completely unacceptable for a major corporation. My trust in this brand has been shattered.",
    "The community is divided on this issue. Some people love it while others hate it. I think there's room for compromise but both sides need to be willing to listen.",
    "Just witnessed the most incredible performance of my life! The artists' dedication and skill are truly inspiring. This will be remembered as a historic moment in the industry.",
    "So disappointed with the service I received today. The customer support team was rude and unhelpful. I won't be returning to this business.",
    "The innovation in this space is mind-blowing! Every day brings new possibilities and I'm so excited to be part of this journey. The future is bright!",
    "This is concerning news for the industry. The new regulations could stifle innovation and hurt small businesses. We need to voice our concerns.",
]
