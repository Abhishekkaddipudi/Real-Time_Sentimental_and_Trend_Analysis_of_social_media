"""
Dashboard module for creating interactive visualizations.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import settings

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Creates interactive visualizations for social media trend analysis.
    """

    def __init__(self):
        """Initialize the dashboard."""
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def create_sentiment_pie_chart(
        self, sentiment_data: Dict[str, int], title: str = "Sentiment Distribution"
    ) -> go.Figure:
        """
        Create a pie chart of sentiment distribution.

        Args:
            sentiment_data: Dictionary with sentiment counts
            title: Chart title

        Returns:
            Plotly Figure object
        """
        labels = list(sentiment_data.keys())
        values = list(sentiment_data.values())

        colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

        color_values = [colors.get(label, "#95a5a6") for label in labels]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker=dict(colors=color_values),
                    textinfo="label+percent",
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        )

        return fig

    def create_sentiment_bar_chart(
        self, sentiment_data: Dict[str, int], title: str = "Sentiment Counts"
    ) -> go.Figure:
        """
        Create a bar chart of sentiment distribution.

        Args:
            sentiment_data: Dictionary with sentiment counts
            title: Chart title

        Returns:
            Plotly Figure object
        """
        labels = list(sentiment_data.keys())
        values = list(sentiment_data.values())

        colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

        color_values = [colors.get(label, "#95a5a6") for label in labels]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=color_values,
                    text=values,
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=title, xaxis_title="Sentiment", yaxis_title="Count", showlegend=False
        )

        return fig

    def create_trend_line_chart(
        self,
        df: pd.DataFrame,
        x_column: str = "period",
        y_column: str = "engagement",
        title: str = "Trend Over Time",
    ) -> go.Figure:
        """
        Create a line chart of trend over time.

        Args:
            df: DataFrame with time-series data
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Convert to datetime if needed
        if x_column in df.columns:
            df[x_column] = pd.to_datetime(df[x_column])

        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=df[x_column],
                y=df[y_column],
                mode="lines+markers",
                name=y_column,
                line=dict(color="#3498db", width=2),
                marker=dict(size=8),
            )
        )

        # Add moving average if enough data
        if len(df) > 5:
            df["ma"] = (
                df[y_column].rolling(window=min(3, len(df) // 2), center=True).mean()
            )
            fig.add_trace(
                go.Scatter(
                    x=df[x_column],
                    y=df["ma"],
                    mode="lines",
                    name="Moving Average",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Engagement",
            hovermode="x unified",
        )

        return fig

    def create_emotion_heatmap(
        self, emotion_data: Dict[str, int], title: str = "Emotion Distribution"
    ) -> go.Figure:
        """
        Create a heatmap of emotion distribution.

        Args:
            emotion_data: Dictionary with emotion counts
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Sort emotions by count
        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)

        labels = [item[0] for item in sorted_emotions]
        values = [item[1] for item in sorted_emotions]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=values,
                    marker_colorscale="Blues",
                    text=values,
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=title, xaxis_title="Emotion", yaxis_title="Count", showlegend=False
        )

        return fig

    def create_daily_pattern_chart(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        title: str = "Activity Pattern by Hour",
    ) -> go.Figure:
        """
        Create a chart showing activity patterns by hour.

        Args:
            df: DataFrame with timestamp data
            time_column: Column with timestamps
            title: Chart title

        Returns:
            Plotly Figure object
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Extract hour
        df["hour"] = df[time_column].dt.hour

        # Count by hour
        hourly_counts = df["hour"].value_counts().sort_index()

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    mode="lines+markers",
                    fill="tozeroy",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=10, color="#2ecc71"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Number of Posts",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        )

        return fig

    def create_daily_pattern_heatmap(
        self,
        df: pd.DataFrame,
        time_column: str = "created_utc",
        title: str = "Activity Pattern by Day and Hour",
    ) -> go.Figure:
        """
        Create a heatmap showing activity patterns by day and hour.

        Args:
            df: DataFrame with timestamp data
            time_column: Column with timestamps
            title: Chart title

        Returns:
            Plotly Figure object
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Extract day of week and hour
        df["day_of_week"] = df[time_column].dt.day_name()
        df["hour"] = df[time_column].dt.hour

        # Create pivot table
        pivot = df.pivot_table(
            index="day_of_week",
            columns="hour",
            values="id",
            aggfunc="count",
            fill_value=0,
        )

        # Order days
        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        pivot = pivot.reindex(days_order)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="YlOrRd",
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=title, xaxis_title="Hour of Day", yaxis_title="Day of Week"
        )

        return fig

    def create_combined_dashboard(
        self,
        sentiment_data: Dict[str, int],
        emotion_data: Dict[str, int],
        trend_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        title: str = "Social Media Analysis Dashboard",
    ) -> go.Figure:
        """
        Create a combined dashboard with multiple charts.

        Args:
            sentiment_data: Sentiment distribution data
            emotion_data: Emotion distribution data
            trend_df: Trend time-series data
            posts_df: Posts data for activity patterns
            title: Dashboard title

        Returns:
            Plotly Figure object with subplots
        """
        # Create subplot grid
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Sentiment Distribution",
                "Emotion Distribution",
                "Trend Over Time",
                "Activity Pattern",
                "",
                "",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. Sentiment Pie Chart
        fig.add_trace(
            go.Pie(
                labels=list(sentiment_data.keys()),
                values=list(sentiment_data.values()),
                hole=0.3,
                textinfo="label+percent",
            ),
            row=1,
            col=1,
        )

        # 2. Emotion Bar Chart
        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
        fig.add_trace(
            go.Bar(
                x=[item[0] for item in sorted_emotions[:8]],
                y=[item[1] for item in sorted_emotions[:8]],
                marker_color="#3498db",
            ),
            row=1,
            col=2,
        )

        # 3. Trend Line Chart
        if not trend_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df.iloc[:, 0],
                    mode="lines+markers",
                    line=dict(color="#2ecc71"),
                ),
                row=1,
                col=3,
            )

        # 4. Activity Pattern
        if not posts_df.empty:
            posts_df[time_column] = pd.to_datetime(posts_df[time_column])
            posts_df["hour"] = posts_df[time_column].dt.hour
            hourly_counts = posts_df["hour"].value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    mode="lines+markers",
                    fill="tozeroy",
                    line=dict(color="#e74c3c"),
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=title, height=800, showlegend=False, template="plotly_white"
        )

        # Update axes
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Posts", row=2, col=1)

        return fig

    def create_streamlit_dashboard(
        self,
        sentiment_data: Dict[str, int],
        emotion_data: Dict[str, int],
        trend_data: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> None:
        """
        Create a Streamlit dashboard with all visualizations.

        Note: This requires streamlit to be installed.
        """
        try:
            import streamlit as st

            st.set_page_config(page_title="Social Media Trend Analysis", layout="wide")

            st.title("📊 Real-Time Social Media Trend & Sentiment Analysis")
            st.markdown("---")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Posts", sentiment_data.get("total", 0))

            with col2:
                positive = sentiment_data.get("positive", 0)
                total = sentiment_data.get("total", 1)
                st.metric(
                    "Positive Sentiment", f"{positive} ({positive/total*100:.1f}%)"
                )

            with col3:
                negative = sentiment_data.get("negative", 0)
                st.metric(
                    "Negative Sentiment", f"{negative} ({negative/total*100:.1f}%)"
                )

            with col4:
                trend_score = trend_data.get("trend_score", 0)
                st.metric("Trend Score", f"{trend_score:.1f}/10")

            st.markdown("---")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                if sentiment_data:
                    fig = self.create_sentiment_pie_chart(sentiment_data)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if emotion_data:
                    fig = self.create_emotion_heatmap(emotion_data)
                    st.plotly_chart(fig, use_container_width=True)

            # Trend chart
            if posts:
                df = pd.DataFrame(posts)
                fig = self.create_daily_pattern_chart(df)
                st.plotly_chart(fig, use_container_width=True)

            # Recent posts
            st.markdown("---")
            st.subheader("Recent Posts")
            if posts:
                recent_df = pd.DataFrame(posts[:20])

                # Select columns to display
                display_cols = ["text", "sentiment", "emotion", "created_at"]
                display_cols = [c for c in display_cols if c in recent_df.columns]

                if display_cols:
                    st.dataframe(recent_df[display_cols])

        except ImportError:
            logger.warning(
                "Streamlit not installed. Cannot create Streamlit dashboard."
            )
        except Exception as e:
            logger.error(f"Error creating Streamlit dashboard: {e}")
