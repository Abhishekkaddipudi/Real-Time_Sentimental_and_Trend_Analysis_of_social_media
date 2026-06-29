#!/usr/bin/env python
"""
Main pipeline script for running the social media trend and sentiment analysis.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.data_ingestion import RedditFetcher, TwitterFetcher, DataGenerator
from src.preprocessing import TextCleaner
from src.analysis import SentimentAnalyzer, TrendAnalyzer
from src.storage import DatabaseManager
from src.visualization import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SocialMediaAnalysisPipeline:
    """
    Main pipeline for social media trend and sentiment analysis.
    """

    def __init__(self):
        """Initialize the pipeline components."""
        self.reddit_fetcher = RedditFetcher()
        self.twitter_fetcher = TwitterFetcher()
        self.data_generator = DataGenerator()
        self.text_cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.database = DatabaseManager()
        self.dashboard = Dashboard()

        logger.info("Pipeline initialized successfully")

    def run_reddit_analysis(
        self,
        keyword: str,
        subreddit: str = "all",
        limit: int = 50,
        analyze_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Run analysis on Reddit data.

        Args:
            keyword: Keyword to search for
            subreddit: Subreddit to search in
            limit: Maximum number of posts
            analyze_sentiment: Whether to analyze sentiment

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Fetching Reddit data for keyword '{keyword}'...")

        # Fetch posts
        posts = self.reddit_fetcher.fetch_posts_by_keyword(
            keyword=keyword, subreddit=subreddit, limit=limit
        )

        if not posts:
            logger.warning(f"No posts found for keyword '{keyword}'")
            return {"error": "No posts found"}

        logger.info(f"Fetched {len(posts)} posts")

        # Convert to DataFrame
        df = self.reddit_fetcher.to_dataframe(posts)

        # Clean text
        df = self.text_cleaner.clean_dataframe(df, text_column="text")

        # Analyze sentiment if requested
        if analyze_sentiment:
            logger.info("Analyzing sentiment...")
            df = self.sentiment_analyzer.analyze_dataframe(
                df, text_column="text_cleaned"
            )

        # Analyze trends
        logger.info("Analyzing trends...")
        trend_summary = self.trend_analyzer.generate_trend_summary(df, keyword)

        # Store in database
        posts_with_sentiment = []
        for _, row in df.iterrows():
            post_data = row.to_dict()
            posts_with_sentiment.append(post_data)

        self.database.insert_batch(posts_with_sentiment)

        # Get sentiment distribution
        sentiment_dist = self.sentiment_analyzer.get_sentiment_distribution(df)

        return {
            "keyword": keyword,
            "platform": "reddit",
            "total_posts": len(posts),
            "trend_summary": trend_summary,
            "sentiment_distribution": sentiment_dist,
            "dataframe": df,
        }

    def run_generated_analysis(
        self,
        keyword: str,
        num_posts: int = 50,
        days: int = 7,
        analyze_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Run analysis on generated synthetic data.

        Args:
            keyword: Topic for generated data
            num_posts: Number of posts to generate
            days: Number of days of data
            analyze_sentiment: Whether to analyze sentiment

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Generating synthetic data for keyword '{keyword}'...")

        # Generate posts
        posts = self.data_generator.generate_posts(keyword=keyword, num_posts=num_posts)

        logger.info(f"Generated {len(posts)} posts")

        # Convert to DataFrame
        df = pd.DataFrame(posts)

        # Clean text
        df = self.text_cleaner.clean_dataframe(df, text_column="text")

        # Analyze sentiment if requested
        if analyze_sentiment:
            logger.info("Analyzing sentiment...")
            df = self.sentiment_analyzer.analyze_dataframe(
                df, text_column="text_cleaned"
            )

        # Analyze trends
        logger.info("Analyzing trends...")
        trend_summary = self.trend_analyzer.generate_trend_summary(df, keyword)

        # Get trend data
        trend_data = self.data_generator.generate_trend_data(keyword=keyword, days=days)

        # Store in database
        posts_with_sentiment = []
        for _, row in df.iterrows():
            post_data = row.to_dict()
            posts_with_sentiment.append(post_data)

        self.database.insert_batch(posts_with_sentiment)

        # Get sentiment distribution
        sentiment_dist = self.sentiment_analyzer.get_sentiment_distribution(df)

        return {
            "keyword": keyword,
            "platform": "generated",
            "total_posts": len(posts),
            "trend_summary": trend_summary,
            "trend_data": trend_data,
            "sentiment_distribution": sentiment_dist,
            "dataframe": df,
        }

    def generate_report(
        self, results: Dict[str, Any], output_file: str = "analysis_report.html"
    ) -> None:
        """
        Generate an HTML report from analysis results.

        Args:
            results: Analysis results dictionary
            output_file: Output file path
        """
        try:
            import plotly.io as pio

            # Extract data
            df = results.get("dataframe", pd.DataFrame())

            if df.empty:
                logger.warning("No data to generate report")
                return

            # Create dashboard
            fig = self.dashboard.create_combined_dashboard(
                sentiment_data=results.get("sentiment_distribution", {}).get(
                    "distribution", {}
                ),
                emotion_data={},  # No emotion data in this version
                trend_df=results.get("trend_data", pd.DataFrame()),
                posts_df=df,
            )

            # Save as HTML
            pio.write_html(fig, output_file)
            logger.info(f"Report saved to {output_file}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Real-Time Social Media Trend and Sentiment Analysis"
    )
    parser.add_argument(
        "--keyword", type=str, default="technology", help="Keyword or topic to analyze"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["reddit", "twitter", "generated"],
        default="generated",
        help="Data source",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of posts to fetch"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days for generated data"
    )
    parser.add_argument(
        "--no-sentiment", action="store_true", help="Skip sentiment analysis"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="analysis_report.html",
        help="Output report file path",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SocialMediaAnalysisPipeline()

    # Run analysis
    if args.source == "reddit":
        results = pipeline.run_reddit_analysis(
            keyword=args.keyword,
            limit=args.limit,
            analyze_sentiment=not args.no_sentiment,
        )
    elif args.source == "twitter":
        # Twitter implementation similar to Reddit
        logger.warning("Twitter integration requires additional setup")
        results = {"error": "Twitter integration not fully implemented"}
    else:
        results = pipeline.run_generated_analysis(
            keyword=args.keyword,
            num_posts=args.limit,
            days=args.days,
            analyze_sentiment=not args.no_sentiment,
        )

    # Generate report
    if results and "error" not in results:
        pipeline.generate_report(results, args.report)

        # Print summary
        print("\n" + "=" * 60)
        print(f"ANALYSIS SUMMARY: {args.keyword}")
        print("=" * 60)
        print(f"Total Posts: {results.get('total_posts', 0)}")

        sentiment_dist = results.get("sentiment_distribution", {})
        print(f"Positive: {sentiment_dist.get('positive', 0)}")
        print(f"Negative: {sentiment_dist.get('negative', 0)}")
        print(f"Neutral: {sentiment_dist.get('neutral', 0)}")

        trend_summary = results.get("trend_summary", {})
        print(f"Trend Score: {trend_summary.get('trend_score', 0):.2f}/10")
        print(f"Is Trending: {trend_summary.get('is_trending', False)}")
        print("=" * 60)
    else:
        logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
