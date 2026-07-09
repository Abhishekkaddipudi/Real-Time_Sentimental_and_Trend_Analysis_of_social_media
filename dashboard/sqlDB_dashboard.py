"""
SQLite Dashboard
Social Media Trend & Sentiment Analysis
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="Social Media Trend Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

.main-title{
    font-size:38px;
    font-weight:bold;
    text-align:center;
    color:#2196F3;
    margin-bottom:15px;
}

.metric-box{
    background:#F5F5F5;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 0px 8px rgba(0,0,0,0.15);
    text-align:center;
}

.metric-value{
    font-size:32px;
    font-weight:bold;
    color:#2196F3;
}

.metric-label{
    font-size:15px;
    color:#555;
}

.sidebar-title{
    font-size:20px;
    font-weight:bold;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Database
# -------------------------------------------------------

DATABASE = "social_media.db"


@st.cache_data
def load_data():

    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql(
        """
        SELECT *
        FROM social_media_posts
        """,
        conn,
    )

    conn.close()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["Date"] = df["Timestamp"].dt.date

    return df


df = load_data()

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.markdown(
    '<div class="main-title">📊 Social Media Trend & Sentiment Dashboard</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Filters")

platforms = sorted(df.Platform.unique())

keywords = sorted(df.Keyword.unique())

categories = sorted(df.Category.unique())

sentiments = sorted(df.Sentiment.unique())

selected_platform = st.sidebar.multiselect("Platform", platforms, default=platforms)

selected_keyword = st.sidebar.multiselect("Keyword", keywords, default=keywords)

selected_category = st.sidebar.multiselect("Category", categories, default=categories)

selected_sentiment = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

date_range = st.sidebar.date_input("Date Range", [df.Date.min(), df.Date.max()])

# -------------------------------------------------------
# Apply Filters
# -------------------------------------------------------

filtered = df.copy()

filtered = filtered[filtered.Platform.isin(selected_platform)]

filtered = filtered[filtered.Keyword.isin(selected_keyword)]

filtered = filtered[filtered.Category.isin(selected_category)]

filtered = filtered[filtered.Sentiment.isin(selected_sentiment)]

if len(date_range) == 2:

    start, end = date_range

    filtered = filtered[(filtered.Date >= start) & (filtered.Date <= end)]

# -------------------------------------------------------
# KPIs
# -------------------------------------------------------

total_posts = len(filtered)

total_likes = int(filtered["Likes"].sum())

total_comments = int(filtered["Comments"].sum())

avg_likes = round(filtered["Likes"].mean(), 2)

avg_comments = round(filtered["Comments"].mean(), 2)

avg_trend = round(filtered["Trend_score"].mean(), 2)

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

st.subheader("Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{total_posts}</div>
    <div class="metric-label">Posts</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{total_likes:,}</div>
    <div class="metric-label">Likes</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{total_comments:,}</div>
    <div class="metric-label">Comments</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{avg_likes}</div>
    <div class="metric-label">Avg Likes</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c5:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{avg_comments}</div>
    <div class="metric-label">Avg Comments</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c6:

    st.markdown(
        f"""
    <div class="metric-box">
    <div class="metric-value">{avg_trend}</div>
    <div class="metric-label">Avg Trend Score</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==========================================================
# PART 2 STARTS HERE
# ==========================================================

st.header("📈 Sentiment Analysis")

left, right = st.columns(2)

# ==========================================================
# SENTIMENT DISTRIBUTION (DONUT)
# ==========================================================

with left:

    sentiment_count = filtered["Sentiment"].value_counts().reset_index()

    sentiment_count.columns = ["Sentiment", "Posts"]

    fig = px.pie(
        sentiment_count,
        values="Posts",
        names="Sentiment",
        hole=0.60,
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"},
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# SENTIMENT OVER TIME
# ==========================================================

with right:

    sentiment_time = (
        filtered.groupby(["Date", "Sentiment"]).size().reset_index(name="Posts")
    )

    fig = px.line(
        sentiment_time,
        x="Date",
        y="Posts",
        color="Sentiment",
        markers=True,
        title="Sentiment Trend Over Time",
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

platform_col, keyword_col = st.columns(2)
# ==========================================================
# PLATFORM VS SENTIMENT
# ==========================================================

with platform_col:

    platform_sentiment = (
        filtered.groupby(["Platform", "Sentiment"]).size().reset_index(name="Posts")
    )

    fig = px.bar(
        platform_sentiment,
        x="Platform",
        y="Posts",
        color="Sentiment",
        barmode="group",
        title="Platform Wise Sentiment",
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# KEYWORD VS SENTIMENT
# ==========================================================

with keyword_col:

    keyword_sentiment = (
        filtered.groupby(["Keyword", "Sentiment"]).size().reset_index(name="Posts")
    )

    fig = px.bar(
        keyword_sentiment,
        x="Keyword",
        y="Posts",
        color="Sentiment",
        barmode="group",
        title="Keyword Wise Sentiment",
    )

    fig.update_layout(height=450, xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Sentiment Summary")
pivot = filtered.pivot_table(
    index="Keyword",
    columns="Sentiment",
    values="PostID",
    aggfunc="count",
    fill_value=0,
    margins=True,
)

st.dataframe(pivot, use_container_width=True)
st.markdown("---")

st.header("🔥 Trend Analysis")

left, right = st.columns(2)

pivot = filtered.pivot_table(
    index="Keyword",
    columns="Sentiment",
    values="PostID",
    aggfunc="count",
    fill_value=0,
    margins=True,
)

st.dataframe(pivot, use_container_width=True)
# ==========================================================
# TREND SCORE TIMELINE
# ==========================================================

with left:

    trend_time = filtered.groupby("Date", as_index=False)["Trend_score"].sum()

    fig = px.line(
        trend_time,
        x="Date",
        y="Trend_score",
        markers=True,
        title="Trend Score Over Time",
    )

    fig.update_traces(line=dict(width=3))

    fig.update_layout(height=430, xaxis_title="Date", yaxis_title="Trend Score")

    st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# POSTS PER DAY
# ==========================================================

with right:

    posts = filtered.groupby("Date").size().reset_index(name="Posts")

    fig = px.bar(posts, x="Date", y="Posts", title="Posts Per Day", text="Posts")

    fig.update_layout(height=430)

    st.plotly_chart(fig, use_container_width=True)
keyword_col, platform_col = st.columns(2)
# ==========================================================
# TREND SCORE BY KEYWORD
# ==========================================================

with keyword_col:

    keyword_trend = (
        filtered.groupby("Keyword", as_index=False)
        .agg({"Trend_score": "sum"})
        .sort_values("Trend_score", ascending=False)
    )

    fig = px.bar(
        keyword_trend,
        x="Keyword",
        y="Trend_score",
        color="Trend_score",
        title="Trend Score by Keyword",
        text="Trend_score",
    )

    fig.update_layout(height=450, xaxis_tickangle=-45, coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# TREND SCORE BY PLATFORM
# ==========================================================

with platform_col:

    platform_trend = filtered.groupby("Platform", as_index=False).agg(
        {"Trend_score": "sum"}
    )

    fig = px.bar(
        platform_trend,
        x="Platform",
        y="Trend_score",
        color="Platform",
        title="Trend Score by Platform",
        text="Trend_score",
    )

    fig.update_layout(height=450, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
top_col, summary_col = st.columns(2)
# ==========================================================
# TOP 10 TRENDING TOPICS
# ==========================================================

with top_col:

    st.subheader("Top Trending Topics")

    top_topics = (
        filtered.groupby("Keyword", as_index=False)
        .agg({"Trend_score": "sum", "Likes": "sum", "Comments": "sum"})
        .sort_values("Trend_score", ascending=False)
        .head(10)
    )

    st.dataframe(top_topics, use_container_width=True, hide_index=True)
# ==========================================================
# TREND SUMMARY
# ==========================================================

with summary_col:

    st.subheader("Trend Summary")

    summary = (
        filtered.groupby(["Platform", "Category"])
        .agg(
            Posts=("PostID", "count"),
            Likes=("Likes", "sum"),
            Comments=("Comments", "sum"),
            Trend=("Trend_score", "sum"),
        )
        .reset_index()
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)
st.markdown("---")

st.header("📋 Detailed Data")
st.markdown("---")
st.header("📂 Category Insights")

col1, col2 = st.columns(2)

with col1:

    category = filtered.groupby("Category").size().reset_index(name="Posts")

    fig = px.pie(
        category,
        names="Category",
        values="Posts",
        hole=0.55,
        title="Category Distribution",
    )

    fig.update_layout(height=430)

    st.plotly_chart(fig, use_container_width=True)
with col2:

    platform = filtered.groupby("Platform").size().reset_index(name="Posts")

    fig = px.bar(
        platform,
        x="Platform",
        y="Posts",
        color="Platform",
        text="Posts",
        title="Platform Distribution",
    )

    fig.update_layout(showlegend=False, height=430)

    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.header("🕒 Posting Activity")

filtered["Hour"] = filtered["Timestamp"].dt.hour
filtered["Day"] = filtered["Timestamp"].dt.day_name()

heat = filtered.pivot_table(
    index="Day", columns="Hour", values="PostID", aggfunc="count", fill_value=0
)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heat = heat.reindex(days)

fig = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="Blues",
    title="Posting Activity Heatmap",
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.header("🔍 Search Posts")

keyword = st.text_input("Search Post Text")

search_df = filtered.copy()

if keyword:

    search_df = search_df[
        search_df["Post_Text"].str.contains(keyword, case=False, na=False)
    ]
st.subheader("Recent Posts")

display = search_df.sort_values("Timestamp", ascending=False)

st.dataframe(
    display[
        [
            "Timestamp",
            "Platform",
            "Keyword",
            "Category",
            "Sentiment",
            "Likes",
            "Comments",
            "Trend_score",
            "Post_Text",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    height=450,
)
st.markdown("---")

csv = search_df.to_csv(index=False)

st.download_button(
    "📥 Download Filtered Data", csv, "social_media_analysis.csv", "text/csv"
)
st.sidebar.markdown("---")

refresh = st.sidebar.checkbox("Auto Refresh")

if refresh:

    st.cache_data.clear()

    st.rerun()
st.markdown("---")

st.markdown(
    """
<center>

### 📊 Real-Time Social Media Trend & Sentiment Analysis

Developed using

**Python • Streamlit • SQLite • Plotly**

MCA Final Project

</center>
""",
    unsafe_allow_html=True,
)
