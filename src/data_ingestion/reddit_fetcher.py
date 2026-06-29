"""
Reddit data fetcher module using PRAW (Python Reddit API Wrapper).
"""

import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import praw
import pandas as pd
from praw.models import Submission, Comment

from config.settings import settings

logger = logging.getLogger(__name__)

class RedditFetcher:
    """
    Fetches data from Reddit API using PRAW.
    """
    
    def __init__(self):
        """Initialize Reddit API client."""
        self.reddit = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the PRAW Reddit client."""
        try:
            self.reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent
            )
            logger.info("Reddit client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def fetch_posts_by_keyword(
        self,
        keyword: str,
        subreddit: str = "all",
        limit: int = 100,
        time_filter: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from Reddit based on a keyword search.
        
        Args:
            keyword: Search keyword or hashtag
            subreddit: Specific subreddit or 'all'
            limit: Maximum number of posts to fetch
            time_filter: Time filter (hour, day, week, month, year, all)
        
        Returns:
            List of post dictionaries with metadata
        """
        posts = []
        
        try:
            # Search for posts containing the keyword
            search_query = f"{keyword} site:reddit.com"
            submissions = self.reddit.subreddit(subreddit).search(
                query=keyword,
                sort="relevance",
                time_filter=time_filter,
                limit=limit
            )
            
            for submission in submissions:
                post_data = self._extract_submission_data(submission)
                posts.append(post_data)
                
            logger.info(f"Fetched {len(posts)} posts for keyword '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error fetching posts for keyword '{keyword}': {e}")
        
        return posts
    
    def fetch_hot_posts(
        self,
        subreddit: str = "all",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch hot posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            limit: Maximum number of posts to fetch
        
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            submissions = self.reddit.subreddit(subreddit).hot(limit=limit)
            
            for submission in submissions:
                post_data = self._extract_submission_data(submission)
                posts.append(post_data)
                
            logger.info(f"Fetched {len(posts)} hot posts from r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error fetching hot posts from r/{subreddit}: {e}")
        
        return posts
    
    def fetch_top_posts(
        self,
        subreddit: str = "all",
        limit: int = 50,
        time_filter: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Fetch top posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            limit: Maximum number of posts to fetch
            time_filter: Time filter (hour, day, week, month, year, all)
        
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            submissions = self.reddit.subreddit(subreddit).top(
                time_filter=time_filter,
                limit=limit
            )
            
            for submission in submissions:
                post_data = self._extract_submission_data(submission)
                posts.append(post_data)
                
            logger.info(f"Fetched {len(posts)} top posts from r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error fetching top posts from r/{subreddit}: {e}")
        
        return posts
    
    def fetch_comments_for_post(
        self,
        post_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a specific post.
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to fetch
        
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                comment_data = self._extract_comment_data(comment)
                comments.append(comment_data)
                
            logger.info(f"Fetched {len(comments)} comments for post {post_id}")
            
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
        
        return comments
    
    def _extract_submission_data(self, submission: Submission) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit submission.
        
        Args:
            submission: PRAW Submission object
        
        Returns:
            Dictionary of extracted data
        """
        return {
            "id": submission.id,
            "title": submission.title,
            "text": submission.selftext,
            "author": str(submission.author) if submission.author else "[deleted]",
            "created_utc": datetime.fromtimestamp(submission.created_utc),
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "url": submission.url,
            "permalink": submission.permalink,
            "subreddit": submission.subreddit.display_name,
            "flair": submission.link_flair_text,
            "is_self": submission.is_self,
            "award_count": submission.total_awards_received,
            "platform": "reddit"
        }
    
    def _extract_comment_data(self, comment: Comment) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit comment.
        
        Args:
            comment: PRAW Comment object
        
        Returns:
            Dictionary of extracted data
        """
        return {
            "id": comment.id,
            "text": comment.body,
            "author": str(comment.author) if comment.author else "[deleted]",
            "created_utc": datetime.fromtimestamp(comment.created_utc),
            "score": comment.score,
            "parent_id": comment.parent_id,
            "link_id": comment.link_id,
            "permalink": comment.permalink,
            "subreddit": comment.subreddit.display_name,
            "platform": "reddit"
        }
    
    def to_dataframe(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert fetched posts to a pandas DataFrame.
        
        Args:
            posts: List of post dictionaries
        
        Returns:
            Pandas DataFrame with structured data
        """
        return pd.DataFrame(posts)