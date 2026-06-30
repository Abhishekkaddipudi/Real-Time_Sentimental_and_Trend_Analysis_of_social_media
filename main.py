#!/usr/bin/env python3
"""
Main entry point for the Real-Time Social Media Trend and Sentiment Analysis System.
"""

import warnings

warnings.filterwarnings("ignore")
import sys
import os
import json
import argparse
from datetime import datetime
import pandas as pd
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    DataIngestion,
    DataPreprocessing,
    SentimentAnalysis,
    TrendAnalysis,
    DatabaseManager,
)
from config.config import Config
from utils.helpers import log_message


def run_pipeline(
    keyword: str = "technology",
    platform: str = "reddit",
    limit: int = 50,
    save_to_db: bool = True,
):
    """
    Run the complete analysis pipeline.

    Args:
        keyword: Search keyword
        platform: Data source platform
        limit: Number of posts to fetch
        save_to_db: Whether to save results to database
    """
    log_message(f"Starting analysis pipeline for keyword: '{keyword}'")

    # Step 1: Data Ingestion
    log_message("Step 1: Fetching data...")
    ingestion = DataIngestion(source=platform)

    if platform == "reddit":
        posts = ingestion.fetch_reddit_posts(
            subreddit="all", limit=limit, keyword=keyword
        )
    else:
        posts = ingestion.fetch_from_api(platform, keyword=keyword, count=limit)

    if not posts:
        log_message("No posts fetched, exiting.", level="ERROR")
        return

    log_message(f"Fetched {len(posts)} posts")

    # Step 2: Data Preprocessing
    log_message("Step 2: Preprocessing data...")
    preprocessor = DataPreprocessing()
    df = pd.DataFrame(posts)
    df = preprocessor.preprocess_dataframe(df)
    df = preprocessor.extract_features(df)

    if df.empty:
        log_message("No valid posts after preprocessing, exiting.", level="ERROR")
        return

    log_message(f"Preprocessed {len(df)} posts")

    # Step 3: Sentiment Analysis
    log_message("Step 3: Performing sentiment analysis...")
    sentiment_analyzer = SentimentAnalysis()
    df = sentiment_analyzer.analyze_posts_batch(df)

    # Show sentiment distribution
    sentiment_dist = sentiment_analyzer.get_sentiment_distribution(df)
    log_message(f"Sentiment Distribution: {sentiment_dist}")

    emotion_dist = sentiment_analyzer.get_emotion_distribution(df)
    log_message(f"Emotion Distribution: {emotion_dist}")

    # Step 4: Trend Analysis
    log_message("Step 4: Analyzing trends...")
    trend_analyzer = TrendAnalysis()

    # Calculate engagement if not present
    if "total_engagement" not in df.columns:
        df["total_engagement"] = df["score"] + df["num_comments"]

    # Analyze lifecycle
    lifecycle = trend_analyzer.identify_trend_lifecycle(df)
    log_message(f"Trend Lifecycle: {lifecycle}")

    # Detect emerging trends
    emerging = trend_analyzer.detect_emerging_trends(df)
    log_message(f"Emerging Trends: {emerging}")

    # Step 5: Save to Database
    if save_to_db:
        log_message("Step 5: Saving results to database...")
        db = DatabaseManager()
        result = db.insert_posts_batch(df)
        log_message(f"Database save result: {result}")

        # Get statistics
        stats = db.get_statistics()
        log_message(f"Database Statistics: {stats}")

    # Step 6: Generate Report
    log_message("Generating analysis report...")
    report = generate_report(df, lifecycle, emerging, sentiment_dist, emotion_dist)

    # Save report
    report_filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log_message(f"Report saved to {report_filename}")

    # Show summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Keyword: {keyword}")
    print(f"Platform: {platform}")
    print(f"Total Posts Analyzed: {len(df)}")
    print(f"Sentiment: {sentiment_dist}")
    print(f"Top Emotions: {emotion_dist}")
    print(f"Trend Lifecycle: {lifecycle.get('duration_hours', 0):.1f} hours")
    print(f"Emerging Trends: {len(emerging)}")
    print(f"Report: {report_filename}")
    print("=" * 60)

    return df, report


def generate_report(
    df: pd.DataFrame,
    lifecycle: Dict,
    emerging: List[Dict],
    sentiment_dist: Dict,
    emotion_dist: Dict,
) -> Dict:
    """
    Generate comprehensive analysis report.

    Args:
        df: DataFrame with analysis results
        lifecycle: Trend lifecycle data
        emerging: Emerging trends
        sentiment_dist: Sentiment distribution
        emotion_dist: Emotion distribution

    Returns:
        Dictionary with report data
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_posts": len(df),
        "lifecycle": lifecycle,
        "emerging_trends": emerging,
        "sentiment_distribution": sentiment_dist,
        "emotion_distribution": emotion_dist,
        "top_mentions": {},
        "sample_posts": [],
    }

    # Top key features
    if "key_feature" in df.columns:
        top_features = (
            df[df["key_feature"] != "error"]["key_feature"].value_counts().head(10)
        )
        report["top_mentions"] = top_features.to_dict()

    # Sample posts (5 each sentiment)
    for sentiment in ["positive", "negative", "neutral"]:
        samples = df[df["sentiment"] == sentiment].head(3)
        if not samples.empty:
            report["sample_posts"].extend(
                [
                    {
                        "text": row.get("text", "")[:200],
                        "sentiment": row.get("sentiment", "neutral"),
                        "emotion": row.get("emotion", "neutral"),
                        "confidence": row.get("confidence", 0.0),
                    }
                    for _, row in samples.iterrows()
                ]
            )

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Social Media Trend and Sentiment Analysis"
    )
    parser.add_argument(
        "--keyword", "-k", type=str, default="technology", help="Keyword to search for"
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        choices=["reddit", "synthetic", "twitter"],
        default="synthetic",
        help="Data source platform",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=50, help="Number of posts to fetch"
    )
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")

    args = parser.parse_args()

    # Run the pipeline
    run_pipeline(
        keyword=args.keyword,
        platform=args.platform,
        limit=args.limit,
        save_to_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
