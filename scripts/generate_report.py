#!/usr/bin/env python
"""
Script to generate analysis reports from database data.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import DatabaseManager
from src.analysis import TrendAnalyzer
from src.visualization import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_report(keyword: str, days: int = 7, output_file: str = "report.html"):
    """
    Generate a report from database data.

    Args:
        keyword: Keyword to analyze
        days: Number of days to look back
        output_file: Output file path
    """
    # Initialize components
    db = DatabaseManager()
    analyzer = TrendAnalyzer()
    dashboard = Dashboard()

    # Get data from database
    since = datetime.now() - timedelta(days=days)
    posts = db.get_posts(keyword=keyword, since=since, limit=500)

    if not posts:
        logger.warning(f"No posts found for keyword '{keyword}'")
        return

    logger.info(f"Found {len(posts)} posts for keyword '{keyword}'")

    # Convert to DataFrame
    import pandas as pd

    df = pd.DataFrame(posts)

    # Analyze sentiment distribution
    sentiment_counts = df["sentiment"].value_counts().to_dict()

    # Analyze trend
    trend_summary = analyzer.generate_trend_summary(df, keyword)

    # Get trend data
    trend_data = db.get_trend_data(keyword, since=since)

    # Create dashboard
    fig = dashboard.create_combined_dashboard(
        sentiment_data=sentiment_counts,
        emotion_data={},  # Would need to extract from data
        trend_df=trend_data,
        posts_df=df,
    )

    # Save as HTML
    import plotly.io as pio

    pio.write_html(fig, output_file)
    logger.info(f"Report saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"REPORT SUMMARY: {keyword}")
    print("=" * 60)
    print(f"Total Posts: {trend_summary.get('total_posts', 0)}")
    print(f"Sentiment: {sentiment_counts}")
    print(f"Trend Score: {trend_summary.get('trend_score', 0):.2f}/10")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate analysis report from database"
    )
    parser.add_argument("--keyword", type=str, required=True, help="Keyword to analyze")
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to look back"
    )
    parser.add_argument(
        "--output", type=str, default="report.html", help="Output file path"
    )

    args = parser.parse_args()
    generate_report(args.keyword, args.days, args.output)


if __name__ == "__main__":
    main()
