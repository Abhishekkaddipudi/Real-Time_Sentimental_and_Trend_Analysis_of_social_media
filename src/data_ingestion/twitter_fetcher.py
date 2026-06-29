"""
Twitter/X data fetcher module using Tweepy.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import tweepy
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

class TwitterFetcher:
    """
    Fetches data from Twitter/X API using Tweepy.
    """
    
    def __init__(self):
        """Initialize Twitter API client."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Tweepy Twitter client."""
        try:
            self.client = tweepy.Client(
                bearer_token=settings.twitter_bearer_token,
                consumer_key=settings.twitter_api_key,
                consumer_secret=settings.twitter_api_secret,
                wait_on_rate_limit=True
            )
            logger.info("Twitter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            raise
    
    def fetch_tweets_by_keyword(
        self,
        keyword: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch tweets based on a keyword search.
        
        Args:
            keyword: Search keyword or hashtag
            max_results: Maximum number of tweets to fetch (max 100)
            start_time: Start time for search
            end_time: End time for search
        
        Returns:
            List of tweet dictionaries with metadata
        """
        tweets = []
        
        try:
            # Build query
            query = f"{keyword} lang:en -is:retweet"
            
            # Search tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=["created_at", "public_metrics", "author_id", "lang", "context_annotations"],
                user_fields=["username", "name", "verified"],
                expansions=["author_id"]
            )
            
            if response.data:
                # Map users for author information
                users = {}
                if response.includes and "users" in response.includes:
                    for user in response.includes["users"]:
                        users[user.id] = user
                
                for tweet in response.data:
                    tweet_data = self._extract_tweet_data(tweet, users)
                    tweets.append(tweet_data)
                
            logger.info(f"Fetched {len(tweets)} tweets for keyword '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error fetching tweets for keyword '{keyword}': {e}")
        
        return tweets
    
    def fetch_tweets_by_user(
        self,
        username: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch tweets from a specific user.
        
        Args:
            username: Twitter username
            max_results: Maximum number of tweets to fetch
        
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        
        try:
            # Get user ID from username
            user_response = self.client.get_user(username=username)
            user_id = user_response.data.id
            
            # Get user's tweets
            response = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics", "lang", "context_annotations"]
            )
            
            if response.data:
                for tweet in response.data:
                    tweet_data = self._extract_tweet_data(tweet)
                    tweets.append(tweet_data)
                
            logger.info(f"Fetched {len(tweets)} tweets from user @{username}")
            
        except Exception as e:
            logger.error(f"Error fetching tweets from user @{username}: {e}")
        
        return tweets
    
    def _extract_tweet_data(self, tweet: tweepy.Tweet, users: Dict = None) -> Dict[str, Any]:
        """
        Extract relevant data from a tweet.
        
        Args:
            tweet: Tweepy Tweet object
            users: Optional user mapping
        
        Returns:
            Dictionary of extracted data
        """
        user_info = {}
        if users and tweet.author_id in users:
            user = users[tweet.author_id]
            user_info = {
                "username": user.username,
                "name": user.name,
                "verified": user.verified
            }
        
        return {
            "id": tweet.id,
            "text": tweet.text,
            "author_id": tweet.author_id,
            "created_at": tweet.created_at,
            "like_count": tweet.public_metrics.get("like_count", 0),
            "retweet_count": tweet.public_metrics.get("retweet_count", 0),
            "reply_count": tweet.public_metrics.get("reply_count", 0),
            "quote_count": tweet.public_metrics.get("quote_count", 0),
            "lang": tweet.lang,
            "platform": "twitter",
            **user_info
        }
    
    def to_dataframe(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert fetched tweets to a pandas DataFrame.
        
        Args:
            tweets: List of tweet dictionaries
        
        Returns:
            Pandas DataFrame with structured data
        """
        return pd.DataFrame(tweets)