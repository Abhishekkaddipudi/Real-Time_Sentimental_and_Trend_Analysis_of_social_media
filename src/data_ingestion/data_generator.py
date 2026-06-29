"""
Data generator for creating synthetic social media data for testing.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from config.settings import settings

class DataGenerator:
    """
    Generates synthetic social media data for testing and development.
    """
    
    def __init__(self):
        """Initialize the data generator with sample content."""
        self.sample_posts = self._initialize_sample_posts()
        self.sample_comments = self._initialize_sample_comments()
    
    def _initialize_sample_posts(self) -> List[Dict[str, Any]]:
        """Initialize sample post templates."""
        return [
            {
                "title": "Just tried the new {product} and it's amazing!",
                "text": "I've been waiting for this {product} for months. It completely exceeded my expectations. The {feature} is a game-changer. Highly recommend to everyone!",
                "sentiment": "positive",
                "emotion": "excitement"
            },
            {
                "title": "Very disappointed with {product}",
                "text": "I had high hopes for {product} but it's been a complete letdown. The {feature} doesn't work as advertised and customer support has been unhelpful. Save your money.",
                "sentiment": "negative",
                "emotion": "frustration"
            },
            {
                "title": "Interesting development in {topic}",
                "text": "Just read about the latest update in {topic}. It's hard to say if this is a good or bad thing. Time will tell how this impacts the community.",
                "sentiment": "neutral",
                "emotion": "curiosity"
            },
            {
                "title": "Absolutely love the new {product}!",
                "text": "I can't believe how good the new {product} is. The {feature} alone makes it worth every penny. I've already recommended it to all my friends!",
                "sentiment": "positive",
                "emotion": "joy"
            },
            {
                "title": "This {product} is overhyped",
                "text": "I don't understand what everyone is raving about. The {product} is average at best and the {feature} is actually quite annoying. Don't believe the hype.",
                "sentiment": "negative",
                "emotion": "disappointment"
            },
            {
                "title": "{topic} is going to change everything",
                "text": "I've been following the developments in {topic} and I'm genuinely impressed. The potential applications are endless. This could be bigger than anyone realizes.",
                "sentiment": "positive",
                "emotion": "anticipation"
            },
            {
                "title": "Worried about the direction of {topic}",
                "text": "I have serious concerns about where {topic} is heading. The recent changes feel like a step backward. I hope the community pushes back before it's too late.",
                "sentiment": "negative",
                "emotion": "concern"
            },
            {
                "title": "Just another day in {topic}",
                "text": "Nothing new to report in {topic} today. Just business as usual. It's nice to have a quiet moment before things inevitably get chaotic again.",
                "sentiment": "neutral",
                "emotion": "calm"
            }
        ]
    
    def _initialize_sample_comments(self) -> List[str]:
        """Initialize sample comment templates."""
        return [
            "I completely agree with this! {topic} is definitely the way to go.",
            "This is nonsense. You clearly don't understand {topic}.",
            "Interesting take. I hadn't thought about it that way before.",
            "Can you elaborate on this point? I'm curious about your perspective.",
            "Finally someone said it! I've been thinking the same thing about {topic}.",
            "Not sure I agree with this. There are other factors to consider.",
            "This is exactly what I needed to hear today. Thank you!",
            "Does anyone else feel like {topic} is being overblown?",
            "I've been following {topic} for years and this analysis is spot on.",
            "This is a game-changer for the {topic} community.",
            "I'm skeptical. There have been many false promises in {topic} before.",
            "The {topic} community has been waiting for this for so long!",
            "This is terrible news for {topic} enthusiasts.",
            "I think we need to wait and see how {topic} develops.",
            "Fantastic analysis! Keep up the great work on {topic}."
        ]
    
    def generate_posts(
        self,
        keyword: str,
        num_posts: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic posts related to a keyword.
        
        Args:
            keyword: Topic or keyword to generate posts about
            num_posts: Number of posts to generate
            start_date: Start date for posts
            end_date: End date for posts
        
        Returns:
            List of generated post dictionaries
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        posts = []
        date_range = (end_date - start_date).total_seconds()
        
        for i in range(num_posts):
            # Select a random template
            template = random.choice(self.sample_posts)
            
            # Fill in the keyword
            title = template["title"].format(product=keyword, topic=keyword)
            text = template["text"].format(
                product=keyword, 
                topic=keyword,
                feature=random.choice(["speed", "design", "battery", "camera", "interface", "performance"])
            )
            
            # Generate post
            post = {
                "id": f"gen_{i}_{int(datetime.now().timestamp())}",
                "title": title,
                "text": text,
                "author": f"user_{random.randint(1000, 9999)}",
                "created_utc": start_date + timedelta(seconds=random.random() * date_range),
                "score": random.randint(0, 1000),
                "upvote_ratio": random.uniform(0.3, 0.95),
                "num_comments": random.randint(0, 50),
                "url": f"https://reddit.com/r/{keyword}/comments/gen_{i}",
                "subreddit": f"r/{keyword.lower().replace(' ', '_')}",
                "is_self": True,
                "platform": random.choice(["reddit", "twitter"]),
                "sentiment": template["sentiment"],
                "emotion": template["emotion"],
                "engagement": random.randint(10, 5000)
            }
            posts.append(post)
        
        # Add some random comments to posts
        for post in posts:
            post["comments"] = self.generate_comments(
                post_id=post["id"],
                num_comments=min(random.randint(0, 20), post["num_comments"]),
                topic=keyword
            )
        
        return posts
    
    def generate_comments(
        self,
        post_id: str,
        num_comments: int = 10,
        topic: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic comments for a post.
        
        Args:
            post_id: ID of the parent post
            num_comments: Number of comments to generate
            topic: Topic for comment content
        
        Returns:
            List of comment dictionaries
        """
        comments = []
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        date_range = (end_date - start_date).total_seconds()
        
        for i in range(num_comments):
            comment_text = random.choice(self.sample_comments)
            comment_text = comment_text.format(topic=topic)
            
            comment = {
                "id": f"comment_{post_id}_{i}",
                "text": comment_text,
                "author": f"commenter_{random.randint(1000, 9999)}",
                "created_utc": start_date + timedelta(seconds=random.random() * date_range),
                "score": random.randint(-10, 50),
                "parent_id": post_id,
                "platform": random.choice(["reddit", "twitter"])
            }
            comments.append(comment)
        
        return comments
    
    def generate_trend_data(
        self,
        keyword: str,
        days: int = 7,
        posts_per_day: int = 20
    ) -> pd.DataFrame:
        """
        Generate time-series trend data for a keyword.
        
        Args:
            keyword: Topic or keyword
            days: Number of days to generate data for
            posts_per_day: Posts per day
        
        Returns:
            DataFrame with trend data
        """
        data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            day_date = base_date + timedelta(days=day)
            
            # Generate varying engagement levels for trend lifecycle
            # Simulate: rise -> peak -> decline
            progress = day / days
            if progress < 0.3:
                # Rising phase
                engagement_multiplier = progress / 0.3
            elif progress < 0.6:
                # Peak phase
                engagement_multiplier = 1.0
            else:
                # Declining phase
                engagement_multiplier = 1.0 - (progress - 0.6) / 0.4
            
            posts_for_day = int(posts_per_day * (0.5 + 0.5 * engagement_multiplier))
            
            for _ in range(posts_for_day):
                sentiment = random.choices(
                    ["positive", "negative", "neutral"],
                    weights=[0.4 + 0.3 * engagement_multiplier, 0.3, 0.3 - 0.3 * engagement_multiplier]
                )[0]
                
                emotion = random.choice(settings.emotion_labels)
                
                data.append({
                    "date": day_date + timedelta(hours=random.randint(0, 23)),
                    "keyword": keyword,
                    "engagement": int(50 * (0.5 + 0.5 * engagement_multiplier) * random.uniform(0.7, 1.3)),
                    "sentiment": sentiment,
                    "emotion": emotion,
                    "posts_count": posts_for_day
                })
        
        return pd.DataFrame(data)