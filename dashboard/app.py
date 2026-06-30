"""
Streamlit Dashboard for Real-Time Social Media Trend and Sentiment Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import Counter
import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import DatabaseManager, TrendAnalysis
from config.config import Config
from utils.helpers import log_message

# Optional libraries (degrade gracefully if not installed)
try:
    from wordcloud import WordCloud, STOPWORDS

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Social Media Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .section-divider {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

STOPWORDS_BASE = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "this",
        "that",
        "it",
        "its",
        "from",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "their",
        "your",
        "my",
        "our",
        "not",
        "no",
        "so",
        "if",
        "than",
        "then",
        "what",
        "which",
        "who",
        "about",
        "into",
        "out",
        "up",
        "down",
    ]
)


def load_data(limit: int = 1000, sentiment_filter: str = "all"):
    """
    Load data from database.

    Args:
        limit: Maximum number of records to load
        sentiment_filter: Filter by sentiment

    Returns:
        DataFrame with loaded data
    """
    try:
        db = DatabaseManager()
        df = db.get_posts(limit=limit)

        if df.empty:
            return df

        # Apply sentiment filter
        if sentiment_filter != "all" and "sentiment" in df.columns:
            df = df[df["sentiment"] == sentiment_filter]

        return df

    except Exception as e:
        log_message(f"Error loading data: {e}", level="ERROR")
        return pd.DataFrame()


def ensure_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure total_engagement column exists."""
    if "total_engagement" not in df.columns:
        df["total_engagement"] = df.get("score", 0) + df.get("num_comments", 0)
    return df


def get_text_column(df: pd.DataFrame) -> str:
    """Pick the best available text column."""
    for col in ["text", "title", "body"]:
        if col in df.columns:
            return col
    return ""


# ---------------------------------------------------------------------------
# Existing charts
# ---------------------------------------------------------------------------


def create_sentiment_chart(df: pd.DataFrame):
    """Create sentiment distribution chart."""
    if df.empty or "sentiment" not in df.columns:
        return go.Figure()

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

    fig = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=colors,
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400)
    return fig


def create_emotion_chart(df: pd.DataFrame):
    """Create emotion distribution chart."""
    if df.empty or "emotion" not in df.columns:
        return go.Figure()

    emotion_counts = df["emotion"].value_counts().head(10).reset_index()
    emotion_counts.columns = ["Emotion", "Count"]

    colors = px.colors.qualitative.Set3

    fig = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        title="Top Emotion Distribution",
        color="Emotion",
        color_discrete_sequence=colors,
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_trend_velocity_chart(df: pd.DataFrame):
    """Create trend velocity chart."""
    if df.empty or "created_utc" not in df.columns:
        return go.Figure()

    df_clean = df.dropna(subset=["created_utc"]).copy()
    if df_clean.empty:
        return go.Figure()

    df_clean["date"] = pd.to_datetime(df_clean["created_utc"]).dt.floor("H")
    df_clean = ensure_engagement(df_clean)

    grouped = (
        df_clean.groupby("date")
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

    grouped["velocity"] = grouped["total_engagement"].diff()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grouped["date"],
            y=grouped["total_engagement"],
            mode="lines+markers",
            name="Engagement Volume",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Bar(
            x=grouped["date"],
            y=grouped["velocity"],
            name="Velocity (Δ Engagement)",
            yaxis="y2",
            marker_color="rgba(231, 76, 60, 0.5)",
        )
    )

    fig.update_layout(
        title="Trend Lifecycle and Velocity",
        xaxis_title="Time",
        yaxis_title="Engagement Volume",
        yaxis2=dict(title="Velocity", overlaying="y", side="right"),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_sentiment_over_time(df: pd.DataFrame):
    """Create sentiment over time chart."""
    if df.empty or "created_utc" not in df.columns or "sentiment" not in df.columns:
        return go.Figure()

    df_clean = df.dropna(subset=["created_utc", "sentiment"]).copy()
    if df_clean.empty:
        return go.Figure()

    df_clean["date"] = pd.to_datetime(df_clean["created_utc"]).dt.floor("D")

    grouped = df_clean.groupby(["date", "sentiment"]).size().reset_index(name="count")

    fig = px.bar(
        grouped,
        x="date",
        y="count",
        color="sentiment",
        title="Sentiment Over Time",
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
    )
    fig.update_layout(
        height=400, barmode="stack", xaxis_title="Date", yaxis_title="Number of Posts"
    )
    return fig


def create_key_features_chart(df: pd.DataFrame):
    """Create key features chart."""
    if df.empty or "key_feature" not in df.columns:
        return go.Figure()

    features = df[df["key_feature"] != "error"]["key_feature"].value_counts().head(10)
    features = features.reset_index()
    features.columns = ["Feature", "Count"]

    fig = px.bar(
        features,
        x="Count",
        y="Feature",
        title="Top Key Features",
        orientation="h",
        color="Count",
        color_continuous_scale=px.colors.sequential.Blues_r,
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_engagement_scatter(df: pd.DataFrame):
    """Create engagement vs sentiment scatter plot."""
    if df.empty or "sentiment" not in df.columns:
        return go.Figure()

    df = ensure_engagement(df)

    hover_cols = [c for c in ["text", "title"] if c in df.columns]

    fig = px.scatter(
        df,
        x="total_engagement",
        y="confidence",
        color="sentiment",
        title="Engagement vs Confidence by Sentiment",
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
        hover_data=hover_cols,
    )
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# New charts: word cloud / keywords, heatmap, correlation
# ---------------------------------------------------------------------------


def extract_keywords(texts, top_n: int = 30):
    """Extract top keyword frequencies from a series of text."""
    words = []
    for t in texts.dropna().astype(str):
        tokens = re.findall(r"[a-zA-Z']+", t.lower())
        words.extend([w for w in tokens if len(w) > 2 and w not in STOPWORDS_BASE])
    return Counter(words).most_common(top_n)


def create_wordcloud_image(df: pd.DataFrame):
    """Create a word cloud image (PNG) from post text, if wordcloud lib available."""
    text_col = get_text_column(df)
    if df.empty or not text_col:
        return None

    corpus = " ".join(df[text_col].dropna().astype(str).tolist())
    if not corpus.strip():
        return None

    stopwords = (
        STOPWORDS.union(STOPWORDS_BASE) if WORDCLOUD_AVAILABLE else STOPWORDS_BASE
    )
    wc = WordCloud(
        width=900,
        height=400,
        background_color="white",
        stopwords=stopwords,
        colormap="viridis",
    ).generate(corpus)
    return wc.to_array()


def create_keywords_bar_chart(df: pd.DataFrame, top_n: int = 15):
    """Fallback / supplementary bar chart of top keywords."""
    text_col = get_text_column(df)
    if df.empty or not text_col:
        return go.Figure()

    counts = extract_keywords(df[text_col], top_n=top_n)
    if not counts:
        return go.Figure()

    kw_df = pd.DataFrame(counts, columns=["Keyword", "Count"])

    fig = px.bar(
        kw_df.sort_values("Count"),
        x="Count",
        y="Keyword",
        orientation="h",
        title="Top Keywords",
        color="Count",
        color_continuous_scale=px.colors.sequential.Tealgrn,
    )
    fig.update_layout(height=450, showlegend=False)
    return fig


def create_activity_heatmap(df: pd.DataFrame):
    """Create a day-of-week x hour-of-day activity heatmap."""
    if df.empty or "created_utc" not in df.columns:
        return go.Figure()

    df_clean = df.dropna(subset=["created_utc"]).copy()
    if df_clean.empty:
        return go.Figure()

    ts = pd.to_datetime(df_clean["created_utc"])
    df_clean["hour"] = ts.dt.hour
    df_clean["day_of_week"] = ts.dt.day_name()

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    pivot = (
        df_clean.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
        .pivot(index="day_of_week", columns="hour", values="count")
        .reindex(day_order)
        .fillna(0)
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Posts"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title="Posting Activity Heatmap (Day vs Hour)",
    )
    fig.update_layout(height=400)
    return fig


def create_correlation_heatmap(df: pd.DataFrame):
    """Create a correlation heatmap of numeric features."""
    if df.empty:
        return go.Figure()

    df = ensure_engagement(df)
    candidate_cols = [
        "score",
        "num_comments",
        "confidence",
        "total_engagement",
        "upvote_ratio",
    ]
    numeric_cols = [c for c in candidate_cols if c in df.columns]

    if len(numeric_cols) < 2:
        return go.Figure()

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Numeric Features",
    )
    fig.update_layout(height=400)
    return fig


def create_source_comparison_chart(df: pd.DataFrame):
    """Compare engagement and sentiment mix across sources/subreddits."""
    source_col = "subreddit" if "subreddit" in df.columns else None
    if df.empty or not source_col or "sentiment" not in df.columns:
        return go.Figure()

    df = ensure_engagement(df)
    grouped = (
        df.groupby([source_col, "sentiment"])
        .agg(count=("sentiment", "size"), engagement=("total_engagement", "sum"))
        .reset_index()
    )

    top_sources = (
        df.groupby(source_col)["total_engagement"].sum().nlargest(8).index.tolist()
    )
    grouped = grouped[grouped[source_col].isin(top_sources)]

    fig = px.bar(
        grouped,
        x=source_col,
        y="count",
        color="sentiment",
        title="Sentiment Mix by Source (Top 8 by Engagement)",
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        },
        barmode="stack",
    )
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def create_forecast_chart(df: pd.DataFrame, periods: int = 6, freq: str = "H"):
    """Simple linear-trend + moving-average forecast of engagement volume."""
    if df.empty or "created_utc" not in df.columns:
        return go.Figure(), None

    df_clean = df.dropna(subset=["created_utc"]).copy()
    if df_clean.empty:
        return go.Figure(), None

    df_clean = ensure_engagement(df_clean)
    df_clean["bucket"] = pd.to_datetime(df_clean["created_utc"]).dt.floor(freq)

    grouped = (
        df_clean.groupby("bucket")["total_engagement"]
        .sum()
        .reset_index()
        .sort_values("bucket")
    )

    if len(grouped) < 3:
        return go.Figure(), None

    grouped["t"] = np.arange(len(grouped))
    grouped["ma"] = (
        grouped["total_engagement"]
        .rolling(window=min(3, len(grouped)), min_periods=1)
        .mean()
    )

    # Linear regression fit (numpy, no extra dependency needed)
    coeffs = np.polyfit(grouped["t"], grouped["total_engagement"], deg=1)
    slope, intercept = coeffs[0], coeffs[1]

    last_t = grouped["t"].iloc[-1]
    last_time = grouped["bucket"].iloc[-1]
    future_t = np.arange(last_t + 1, last_t + 1 + periods)
    future_times = pd.date_range(start=last_time, periods=periods + 1, freq=freq)[1:]
    future_vals = slope * future_t + intercept
    future_vals = np.clip(future_vals, a_min=0, a_max=None)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["bucket"],
            y=grouped["total_engagement"],
            mode="lines+markers",
            name="Actual Engagement",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["bucket"],
            y=grouped["ma"],
            mode="lines",
            name="Moving Average",
            line=dict(color="#95a5a6", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_vals,
            mode="lines+markers",
            name="Forecast (Linear Trend)",
            line=dict(color="#e67e22", width=2, dash="dash"),
        )
    )

    direction = (
        "increasing 📈" if slope > 0 else ("decreasing 📉" if slope < 0 else "flat ➡️")
    )

    fig.update_layout(
        title="Engagement Forecast",
        xaxis_title="Time",
        yaxis_title="Total Engagement",
        height=420,
        hovermode="x unified",
    )

    summary = {
        "direction": direction,
        "slope": slope,
        "next_period_estimate": float(future_vals[0]) if len(future_vals) else None,
    }

    return fig, summary


# ---------------------------------------------------------------------------
# Topic clustering (optional, requires scikit-learn)
# ---------------------------------------------------------------------------


def create_topic_cluster_chart(df: pd.DataFrame, n_clusters: int = 5):
    """Cluster posts via TF-IDF + KMeans and visualize with PCA."""
    text_col = get_text_column(df)
    if df.empty or not text_col or not SKLEARN_AVAILABLE:
        return go.Figure()

    texts = df[text_col].dropna().astype(str)
    if len(texts) < n_clusters + 1:
        return go.Figure()

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(texts)

    k = min(n_clusters, len(texts) - 1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": [f"Cluster {c}" for c in labels],
            "text": texts.values,
        }
    )

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        title="Topic Clusters (TF-IDF + KMeans, PCA projection)",
        hover_data=["text"],
    )
    fig.update_layout(height=450)
    return fig


# ---------------------------------------------------------------------------
# Sidebar / data loading helpers
# ---------------------------------------------------------------------------


def load_dashboard_data(data_source: str, limit: int, sentiment_filter: str):
    """Load data based on selected source."""
    if data_source == "Synthetic":
        from modules.data_ingestion import DataIngestion

        ingestion = DataIngestion(source="synthetic")
        posts = ingestion.fetch_from_api("synthetic", count=limit)
        df = pd.DataFrame(posts)

        from modules.data_preprocessing import DataPreprocessing

        preprocessor = DataPreprocessing()
        df = preprocessor.preprocess_dataframe(df)
        df = preprocessor.extract_features(df)

        from modules.sentiment_analysis import SentimentAnalysis

        sentiment_analyzer = SentimentAnalysis()
        df = sentiment_analyzer.analyze_posts_batch(df)

        return df, True
    else:
        return load_data(limit, sentiment_filter), False


def apply_search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter dataframe rows by keyword search across text-like columns."""
    if not query:
        return df

    text_cols = [c for c in ["text", "title", "body", "key_feature"] if c in df.columns]
    if not text_cols:
        return df

    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask |= df[col].astype(str).str.contains(query, case=False, na=False)

    return df[mask]


def render_metrics(df: pd.DataFrame, prefix: str = ""):
    """Render the 4-metric row for a given dataframe."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">{prefix}Total Posts</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if "sentiment" in df.columns and len(df) > 0:
            positive_count = df[df["sentiment"] == "positive"].shape[0]
            positive_pct = positive_count / len(df) * 100
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{positive_pct:.1f}%</div>
                <div class="metric-label">{prefix}Positive Sentiment</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        df_eng = ensure_engagement(df) if not df.empty else df
        if "total_engagement" in df_eng.columns:
            total_engagement = int(df_eng["total_engagement"].sum())
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{total_engagement:,}</div>
                <div class="metric-label">{prefix}Total Engagement</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        if "subreddit" in df.columns and not df["subreddit"].empty:
            top_source = df["subreddit"].mode().iloc[0]
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{top_source}</div>
                <div class="metric-label">{prefix}Top Source</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    """Main dashboard function."""
    st.markdown(
        '<div class="main-header">📊 Social Media Trend & Sentiment Analysis</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Controls")

    data_source = st.sidebar.selectbox(
        "Data Source", ["Database", "Synthetic"], help="Select data source for analysis"
    )

    st.sidebar.subheader("Filters")

    limit = st.sidebar.slider(
        "Number of Posts", min_value=10, max_value=1000, value=100, step=10
    )

    sentiment_filter = st.sidebar.selectbox(
        "Sentiment Filter", ["all", "positive", "negative", "neutral"]
    )

    search_query = st.sidebar.text_input(
        "🔎 Keyword Search", value="", help="Filter posts containing this text"
    )

    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    st.sidebar.subheader("Comparison Mode")
    compare_enabled = st.sidebar.checkbox("Compare two date ranges", value=False)
    compare_start, compare_end = None, None
    if compare_enabled:
        c1, c2 = st.sidebar.columns(2)
        with c1:
            compare_start = st.date_input(
                "Compare Start", datetime.now() - timedelta(days=14), key="cmp_start"
            )
        with c2:
            compare_end = st.date_input(
                "Compare End", datetime.now() - timedelta(days=8), key="cmp_end"
            )

    st.sidebar.subheader("Forecasting")
    forecast_periods = st.sidebar.slider("Forecast Horizon (periods)", 1, 24, 6)

    refresh = st.sidebar.button("🔄 Refresh Data")

    # Load data
    with st.spinner("Loading data..."):
        df, is_synthetic = load_dashboard_data(data_source, limit, sentiment_filter)
        if is_synthetic:
            st.info("Using synthetic data for demonstration")

    if df.empty:
        st.warning(
            "No data available. Please check your data source or adjust filters."
        )
        return

    # Filter by date
    if "created_utc" in df.columns:
        df["date"] = pd.to_datetime(df["created_utc"]).dt.date
        df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    else:
        df_filtered = df

    # Apply keyword search
    df_filtered = apply_search_filter(df_filtered, search_query)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    df_filtered = ensure_engagement(df_filtered)

    # Main dashboard
    st.subheader("📈 Key Metrics")
    render_metrics(df_filtered)

    # Comparison mode panel
    if (
        compare_enabled
        and "created_utc" in df.columns
        and compare_start
        and compare_end
    ):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("⚖️ Period Comparison")

        df_compare = df[(df["date"] >= compare_start) & (df["date"] <= compare_end)]
        df_compare = (
            ensure_engagement(df_compare) if not df_compare.empty else df_compare
        )

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown(f"**Primary range:** {start_date} → {end_date}")
            render_metrics(df_filtered, prefix="")
        with cmp_col2:
            st.markdown(f"**Comparison range:** {compare_start} → {compare_end}")
            if df_compare.empty:
                st.info("No posts in comparison range.")
            else:
                render_metrics(df_compare, prefix="")

    # Drill-down by sentiment
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔬 Drill-down")
    drill_sentiment = st.selectbox(
        "Focus charts on a specific sentiment (drill-down)",
        ["all"]
        + (
            sorted(df_filtered["sentiment"].unique().tolist())
            if "sentiment" in df_filtered.columns
            else []
        ),
    )
    drill_df = (
        df_filtered
        if drill_sentiment == "all"
        else df_filtered[df_filtered["sentiment"] == drill_sentiment]
    )

    # Charts section
    st.subheader("📊 Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        if "sentiment" in drill_df.columns:
            st.plotly_chart(create_sentiment_chart(drill_df), use_container_width=True)
    with col2:
        if "emotion" in drill_df.columns:
            st.plotly_chart(create_emotion_chart(drill_df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "created_utc" in drill_df.columns and "total_engagement" in drill_df.columns:
            st.plotly_chart(
                create_trend_velocity_chart(drill_df), use_container_width=True
            )
    with col4:
        if "created_utc" in drill_df.columns and "sentiment" in drill_df.columns:
            st.plotly_chart(
                create_sentiment_over_time(drill_df), use_container_width=True
            )

    col5, col6 = st.columns(2)
    with col5:
        if "key_feature" in drill_df.columns:
            st.plotly_chart(
                create_key_features_chart(drill_df), use_container_width=True
            )
    with col6:
        if "sentiment" in drill_df.columns:
            st.plotly_chart(
                create_engagement_scatter(drill_df), use_container_width=True
            )

    # New: keyword / word cloud row
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔡 Text Insights")
    col7, col8 = st.columns(2)
    with col7:
        if WORDCLOUD_AVAILABLE:
            wc_img = create_wordcloud_image(drill_df)
            if wc_img is not None:
                st.image(wc_img, caption="Word Cloud")
            else:
                st.info("Not enough text data for a word cloud.")
        else:
            st.caption(
                "Install the `wordcloud` package to enable word cloud rendering."
            )
            st.plotly_chart(
                create_keywords_bar_chart(drill_df), use_container_width=True
            )
    with col8:
        st.plotly_chart(create_keywords_bar_chart(drill_df), use_container_width=True)

    # New: heatmap / correlation / source comparison
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔥 Activity & Correlation")
    col9, col10 = st.columns(2)
    with col9:
        if "created_utc" in drill_df.columns:
            st.plotly_chart(create_activity_heatmap(drill_df), use_container_width=True)
    with col10:
        st.plotly_chart(create_correlation_heatmap(drill_df), use_container_width=True)

    if "subreddit" in drill_df.columns:
        st.plotly_chart(
            create_source_comparison_chart(drill_df), use_container_width=True
        )

    # New: forecasting
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔮 Forecasting")
    if "created_utc" in drill_df.columns:
        forecast_fig, forecast_summary = create_forecast_chart(
            drill_df, periods=forecast_periods
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
        if forecast_summary:
            st.caption(
                f"Trend direction: {forecast_summary['direction']} • "
                f"Next-period estimate: {forecast_summary['next_period_estimate']:.0f}"
                if forecast_summary["next_period_estimate"] is not None
                else f"Trend direction: {forecast_summary['direction']}"
            )
        else:
            st.info("Not enough time-series data to generate a forecast.")
    else:
        st.info("No timestamp data available for forecasting.")

    # New: topic clustering (optional)
    if SKLEARN_AVAILABLE:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🧩 Topic Clusters")
        n_clusters = st.slider("Number of clusters", 2, 10, 5)
        st.plotly_chart(
            create_topic_cluster_chart(drill_df, n_clusters=n_clusters),
            use_container_width=True,
        )

    # Recent posts table
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📝 Recent Posts")

    display_cols = ["title", "sentiment", "emotion", "confidence", "created_utc"]
    available_cols = [col for col in display_cols if col in drill_df.columns]

    if available_cols:
        display_df = drill_df[available_cols].head(20).copy()
        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].round(2)
        if "created_utc" in display_df.columns:
            display_df["created_utc"] = pd.to_datetime(
                display_df["created_utc"]
            ).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, use_container_width=True)

    # Download data
    st.subheader("📥 Export Data")
    csv = drill_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"social_media_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
