"""
Trend analysis module for detecting and tracking social media trends.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
from scipy import stats

from config.settings import settings

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyzes trends in social media data, including lifecycle and velocity.
    """

    def __init__(self):
        """Initialize the trend analyzer."""
        pass

    def calculate_trend_velocity(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        engagement_column: str = "engagement",
        window_hours: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate trend velocity (rate of change) over time.

        Args:
            df: Input DataFrame with time and engagement data
            time_column: Name of the time column
            engagement_column: Name of the engagement column
            window_hours: Time window for grouping

        Returns:
            DataFrame with velocity calculations
        """
        df = df.copy()

        # Ensure time column is datetime
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column])

            # Group by time window
            df["time_window"] = df[time_column].dt.floor(f"{window_hours}H")

            # Aggregate engagement per time window
            grouped = df.groupby("time_window")[engagement_column].sum().reset_index()
            grouped.columns = ["time_window", "engagement_total"]

            # Calculate velocity (derivative)
            grouped["velocity"] = grouped["engagement_total"].diff()
            grouped["velocity_normalized"] = (
                grouped["velocity"] / grouped["engagement_total"].shift(1)
            ).fillna(0)

            # Calculate trend phase
            grouped["phase"] = self._determine_trend_phase(grouped)

            return grouped

        return pd.DataFrame()

    def _determine_trend_phase(self, df: pd.DataFrame) -> List[str]:
        """
        Determine the phase of a trend (emerging, peak, declining, etc.).

        Args:
            df: DataFrame with velocity data

        Returns:
            List of phase labels
        """
        phases = []

        if len(df) < 3:
            return ["unknown"] * len(df)

        # Use a sliding window to detect phases
        window_size = min(3, len(df) // 2)

        for i in range(len(df)):
            if i < window_size:
                # Early phase
                velocities = df["velocity"].iloc[: i + 1].values
                if len(velocities) >= 2 and np.mean(velocities) > 0:
                    phases.append("emerging")
                else:
                    phases.append("emerging")
            elif i >= len(df) - window_size:
                # Late phase
                velocities = df["velocity"].iloc[i:].values
                if len(velocities) >= 2 and np.mean(velocities) < 0:
                    phases.append("declining")
                else:
                    phases.append("declining")
            else:
                # Middle phase
                window = (
                    df["velocity"].iloc[i - window_size : i + window_size + 1].values
                )
                mean_velocity = np.mean(window)
                if mean_velocity > 0.1:
                    phases.append("rising")
                elif mean_velocity < -0.1:
                    phases.append("declining")
                else:
                    phases.append("peak")

        return phases

    def detect_trend_lifecycle(
        self, df: pd.DataFrame, keyword: str, time_column: str = "created_utc"
    ) -> Dict[str, Any]:
        """
        Detect the lifecycle of a trend.

        Args:
            df: Input DataFrame with trend data
            keyword: Keyword or topic being analyzed
            time_column: Name of the time column

        Returns:
            Dictionary with lifecycle metrics
        """
        df = df.copy()

        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column])

            # Sort by time
            df = df.sort_values(time_column)

            # Calculate metrics
            total_posts = len(df)
            time_span = (df[time_column].max() - df[time_column].min()).total_seconds()

            # Find peak engagement time
            if "engagement" in df.columns:
                peak_idx = df["engagement"].idxmax()
                peak_time = df.loc[peak_idx, time_column]

                # Calculate growth rate (first half vs second half)
                mid_point = len(df) // 2
                early_engagement = df.iloc[:mid_point]["engagement"].sum()
                late_engagement = df.iloc[mid_point:]["engagement"].sum()
                growth_rate = (late_engagement - early_engagement) / (
                    early_engagement + 1
                )
            else:
                peak_time = df[time_column].mean()
                growth_rate = 0

            return {
                "keyword": keyword,
                "total_posts": total_posts,
                "start_time": df[time_column].min(),
                "end_time": df[time_column].max(),
                "time_span_hours": time_span / 3600,
                "peak_time": peak_time,
                "growth_rate": growth_rate,
                "is_trending": total_posts > 10 and growth_rate > 0.2,
            }

        return {"keyword": keyword, "error": "Time column not found in data"}

    def calculate_trend_score(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        engagement_column: str = "engagement",
    ) -> float:
        """
        Calculate a trend score based on engagement and velocity.

        Args:
            df: Input DataFrame
            time_column: Name of the time column
            engagement_column: Name of the engagement column

        Returns:
            Trend score (0-10)
        """
        if len(df) < 3:
            return 0.0

        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # Calculate various metrics
        total_engagement = (
            df[engagement_column].sum() if engagement_column in df.columns else len(df)
        )

        # Time span
        time_span = (df[time_column].max() - df[time_column].min()).total_seconds()
        time_span_hours = time_span / 3600

        # Recent engagement (last 25% of posts)
        recent_start = len(df) * 3 // 4
        recent_engagement = (
            df.iloc[recent_start:][engagement_column].sum()
            if engagement_column in df.columns
            else len(df) - recent_start
        )
        recent_ratio = recent_engagement / (total_engagement + 1)

        # Velocity
        velocity_df = self.calculate_trend_velocity(df, time_column, engagement_column)
        if not velocity_df.empty and len(velocity_df) > 1:
            avg_velocity = velocity_df["velocity"].mean()
            max_velocity = velocity_df["velocity"].max()
        else:
            avg_velocity = 0
            max_velocity = 0

        # Calculate score components (0-10 scale)
        engagement_score = min(10, (total_engagement / 100) * 5)
        velocity_score = min(10, (max_velocity / 50) * 5)
        recency_score = min(10, recent_ratio * 10)

        # Weighted average
        score = engagement_score * 0.4 + velocity_score * 0.3 + recency_score * 0.3

        return min(10, max(0, score))

    def find_emerging_trends(
        self,
        df: pd.DataFrame,
        keyword_column: str = "keyword",
        time_column: str = "created_utc",
        min_engagement: int = 10,
        lookback_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Find emerging trends in the data.

        Args:
            df: Input DataFrame
            keyword_column: Name of the keyword column
            time_column: Name of the time column
            min_engagement: Minimum engagement to consider
            lookback_hours: Hours to look back

        Returns:
            List of emerging trend dictionaries
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Filter by lookback period
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_df = df[df[time_column] >= cutoff_time]

        if recent_df.empty:
            return []

        trends = []
        for keyword, group in recent_df.groupby(keyword_column):
            if len(group) >= 3:  # Minimum posts to consider
                engagement = (
                    group["engagement"].sum()
                    if "engagement" in group.columns
                    else len(group)
                )
                if engagement >= min_engagement:
                    score = self.calculate_trend_score(group)
                    if score > 5:  # Threshold for emerging trend
                        trends.append(
                            {
                                "keyword": keyword,
                                "post_count": len(group),
                                "engagement": engagement,
                                "trend_score": score,
                                "start_time": group[time_column].min(),
                                "peak_time": (
                                    group[time_column].max() if len(group) > 0 else None
                                ),
                                "is_emerging": True,
                            }
                        )

        # Sort by trend score
        trends.sort(key=lambda x: x["trend_score"], reverse=True)

        return trends

    def generate_trend_summary(self, df: pd.DataFrame, keyword: str) -> Dict[str, Any]:
        """
        Generate a comprehensive trend summary.

        Args:
            df: Input DataFrame
            keyword: Keyword or topic being analyzed

        Returns:
            Dictionary with trend summary
        """
        # Basic statistics
        total_posts = len(df)

        if total_posts == 0:
            return {
                "keyword": keyword,
                "total_posts": 0,
                "trend_score": 0,
                "is_trending": False,
            }

        # Engagement statistics
        if "engagement" in df.columns:
            total_engagement = df["engagement"].sum()
            avg_engagement = df["engagement"].mean()
            max_engagement = df["engagement"].max()
        else:
            total_engagement = total_posts
            avg_engagement = 1
            max_engagement = 1

        # Time statistics
        if "created_utc" in df.columns:
            df["created_utc"] = pd.to_datetime(df["created_utc"])
            time_span = (
                df["created_utc"].max() - df["created_utc"].min()
            ).total_seconds() / 3600
            posts_per_hour = total_posts / max(1, time_span)
        else:
            time_span = 0
            posts_per_hour = total_posts

        # Sentiment distribution
        sentiment_dist = {}
        for col in df.columns:
            if "sentiment" in col.lower():
                sentiment_dist = df[col].value_counts().to_dict()
                break

        # Trend score
        trend_score = self.calculate_trend_score(df)

        return {
            "keyword": keyword,
            "total_posts": total_posts,
            "total_engagement": total_engagement,
            "avg_engagement": avg_engagement,
            "max_engagement": max_engagement,
            "time_span_hours": time_span,
            "posts_per_hour": posts_per_hour,
            "sentiment_distribution": sentiment_dist,
            "trend_score": trend_score,
            "is_trending": trend_score > 5,
            "trend_level": (
                "High" if trend_score > 7 else "Medium" if trend_score > 4 else "Low"
            ),
        }
