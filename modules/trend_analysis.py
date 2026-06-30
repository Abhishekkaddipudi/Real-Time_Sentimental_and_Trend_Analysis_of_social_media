"""
Trend Analysis Module - Analyzes trend lifecycle and velocity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils.helpers import log_message


class TrendAnalysis:
    """
    Analyzes trend lifecycle, velocity, and patterns in social media data.
    """

    def __init__(self):
        """Initialize the Trend Analysis module."""
        log_message("Trend Analysis module initialized")

    def calculate_velocity(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        engagement_column: str = "total_engagement",
        interval: str = "H",
    ) -> pd.DataFrame:
        """
        Calculate trend velocity over time.

        Args:
            df: DataFrame with timestamp and engagement data
            time_column: Column name for timestamps
            engagement_column: Column name for engagement metrics
            interval: Time interval for grouping ('H'=hourly, 'D'=daily, 'W'=weekly)

        Returns:
            DataFrame with trend velocity metrics
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure datetime format
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])

        # Group by time interval
        df_copy.set_index(time_column, inplace=True)
        grouped = (
            df_copy.resample(interval)
            .agg({engagement_column: "sum", "score": "sum", "num_comments": "sum"})
            .reset_index()
        )

        # Calculate engagement volume
        grouped["engagement_volume"] = grouped[engagement_column]

        # Calculate velocity (rate of change)
        grouped["velocity"] = grouped["engagement_volume"].diff()
        grouped["velocity_pct"] = grouped["engagement_volume"].pct_change() * 100

        # Identify trend phase
        grouped["trend_phase"] = self._identify_trend_phase(
            grouped["velocity"], grouped["engagement_volume"]
        )

        log_message(f"Calculated velocity for {len(grouped)} time intervals")
        return grouped

    def _identify_trend_phase(
        self, velocity: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Identify trend phases based on velocity and volume.

        Args:
            velocity: Velocity values
            volume: Volume values

        Returns:
            Series with phase labels
        """
        phases = []

        for i in range(len(velocity)):
            if i == 0:
                phases.append("initial")
                continue

            v = velocity.iloc[i]
            vol = volume.iloc[i]
            vol_prev = volume.iloc[i - 1]

            if vol == 0:
                phases.append("inactive")
            elif v > 0 and vol > vol_prev * 1.1:
                phases.append("climbing")
            elif v < 0 and vol < vol_prev * 0.9:
                phases.append("declining")
            elif abs(v) < volume.iloc[i] * 0.05:
                phases.append("peak_plateau")
            elif v < 0:
                phases.append("declining")
            else:
                phases.append("stabilizing")

        return pd.Series(phases, index=velocity.index)

    def identify_trend_lifecycle(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        engagement_column: str = "total_engagement",
    ) -> Dict:
        """
        Identify complete trend lifecycle.

        Args:
            df: DataFrame with timestamp and engagement data
            time_column: Column name for timestamps
            engagement_column: Column name for engagement metrics

        Returns:
            Dictionary with lifecycle metrics
        """
        if df.empty:
            return {"status": "no_data"}

        # Get velocity data
        velocity_df = self.calculate_velocity(df, time_column, engagement_column)

        if velocity_df.empty:
            return {"status": "insufficient_data"}

        # Find peak
        peak_idx = velocity_df["engagement_volume"].idxmax()
        peak_engagement = velocity_df.loc[peak_idx, "engagement_volume"]
        peak_time = velocity_df.loc[peak_idx, time_column]

        # Find start and end of trend
        active_data = velocity_df[velocity_df["engagement_volume"] > 0]
        if active_data.empty:
            return {"status": "no_active_trend"}

        start_time = active_data.iloc[0][time_column]
        end_time = active_data.iloc[-1][time_column]

        # Calculate trend duration
        duration = (end_time - start_time).total_seconds() / 3600  # in hours

        # Calculate growth rate
        first_volume = active_data.iloc[0]["engagement_volume"]
        last_volume = active_data.iloc[-1]["engagement_volume"]
        growth_rate = (
            ((last_volume - first_volume) / first_volume * 100)
            if first_volume > 0
            else 0
        )

        # Calculate peak velocity
        max_velocity = velocity_df["velocity"].max()
        avg_velocity = velocity_df["velocity"].mean()

        lifecycle = {
            "status": "success",
            "start_time": start_time,
            "peak_time": peak_time,
            "end_time": end_time,
            "duration_hours": duration,
            "peak_engagement": peak_engagement,
            "total_engagement": velocity_df["engagement_volume"].sum(),
            "growth_rate_percent": growth_rate,
            "max_velocity": max_velocity,
            "avg_velocity": avg_velocity,
            "total_data_points": len(velocity_df),
        }

        log_message(f"Identified trend lifecycle: {lifecycle}")
        return lifecycle

    def detect_emerging_trends(
        self, df: pd.DataFrame, threshold_percentile: int = 90
    ) -> List[Dict]:
        """
        Detect emerging trends based on engagement velocity.

        Args:
            df: DataFrame with analysis results
            threshold_percentile: Percentile threshold for considering a trend

        Returns:
            List of emerging trend candidates
        """
        if df.empty:
            return []

        trends = []

        # Calculate aggregated metrics
        if "hashtags" in df.columns:
            # Group by hashtags
            hashtag_groups = df.explode("hashtags")
            hashtag_groups = hashtag_groups[hashtag_groups["hashtags"].notna()]

            if not hashtag_groups.empty:
                grouped = (
                    hashtag_groups.groupby("hashtags")
                    .agg(
                        {
                            "total_engagement": "sum",
                            "sentiment": lambda x: (
                                x.value_counts().index[0] if not x.empty else "neutral"
                            ),
                        }
                    )
                    .reset_index()
                )

                # Calculate threshold
                threshold = grouped["total_engagement"].quantile(
                    threshold_percentile / 100
                )

                # Filter emerging trends
                emerging = grouped[grouped["total_engagement"] >= threshold]

                for _, row in emerging.iterrows():
                    trends.append(
                        {
                            "hashtag": row["hashtags"],
                            "engagement": row["total_engagement"],
                            "dominant_sentiment": row["sentiment"],
                        }
                    )

        # Also check by key_feature if available
        if "key_feature" in df.columns:
            feature_groups = (
                df[df["key_feature"].notna()]
                .groupby("key_feature")
                .agg({"total_engagement": "sum"})
                .reset_index()
            )

            if not feature_groups.empty:
                threshold = feature_groups["total_engagement"].quantile(
                    threshold_percentile / 100
                )
                emerging_features = feature_groups[
                    feature_groups["total_engagement"] >= threshold
                ]

                for _, row in emerging_features.iterrows():
                    # Check if feature already exists in trends
                    if not any(t["hashtag"] == row["key_feature"] for t in trends):
                        trends.append(
                            {
                                "hashtag": row["key_feature"],
                                "engagement": row["total_engagement"],
                                "dominant_sentiment": "unknown",
                            }
                        )

        log_message(f"Detected {len(trends)} emerging trends")
        return trends

    def analyze_trend_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trend patterns in the data.

        Args:
            df: DataFrame with analysis results

        Returns:
            Dictionary with trend pattern analysis
        """
        if df.empty:
            return {"status": "no_data"}

        patterns = {}

        # Time-based patterns
        if "hour" in df.columns and "sentiment" in df.columns:
            # Sentiment by hour
            sentiment_by_hour = (
                df.groupby("hour")["sentiment"].value_counts().unstack().fillna(0)
            )
            patterns["sentiment_by_hour"] = sentiment_by_hour.to_dict()

        if "day_of_week" in df.columns:
            sentiment_by_day = (
                df.groupby("day_of_week")["sentiment"]
                .value_counts()
                .unstack()
                .fillna(0)
            )
            patterns["sentiment_by_day"] = sentiment_by_day.to_dict()

        # Engagement patterns
        if "total_engagement" in df.columns:
            patterns["engagement_stats"] = {
                "mean": df["total_engagement"].mean(),
                "median": df["total_engagement"].median(),
                "max": df["total_engagement"].max(),
                "min": df["total_engagement"].min(),
                "std": df["total_engagement"].std(),
            }

        # Sentiment distribution over time
        if "created_utc" in df.columns and "sentiment" in df.columns:
            df_temp = df.copy()
            df_temp["date"] = df_temp["created_utc"].dt.date
            sentiment_over_time = (
                df_temp.groupby("date")["sentiment"].value_counts().unstack().fillna(0)
            )
            patterns["sentiment_over_time"] = sentiment_over_time.to_dict()

        log_message("Completed trend pattern analysis")
        return patterns
