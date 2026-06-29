# Real-Time Social Media Trend and Sentiment Analysis

A comprehensive system for analyzing social media trends and sentiment using Large Language Models (LLMs).

## Features

- **Data Ingestion**: Fetch data from Reddit, Twitter, or generate synthetic data
- **Text Preprocessing**: Clean and normalize social media text
- **Sentiment Analysis**: Zero-shot classification using Google Gemini API
- **Trend Analysis**: Detect emerging trends, calculate velocity, and track lifecycle
- **Visualization**: Interactive dashboards with Plotly and Streamlit
- **Database Storage**: PostgreSQL for structured storage of analysis results

## Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA INGESTION LAYER │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Reddit │ │ Twitter │ │ Data Generator │ │
│ │ Fetcher │ │ Fetcher │ │ (Synthetic Data) │ │
│ └──────────────┘ └──────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROCESSING & ANALYTICS LAYER │
│ ┌──────────────┐ ┌─────────────────┐ ┌──────────────────────────────┐ │
│ │ Text │ │ Sentiment │ │ Trend │ │
│ │ Cleaner │──▶ Analyzer │──▶ Analyzer │ │
│ │ │ │ (Gemini API) │ │ │ │
│ └──────────────┘ └─────────────────┘ └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STORAGE & PRESENTATION LAYER │
│ ┌──────────────────┐ ┌──────────────────────────────────────────────┐ │
│ │ Database │ │ Dashboard │ │
│ │ (PostgreSQL) │──▶ (Plotly / Streamlit / Power BI) │ │
│ └──────────────────┘ └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

text

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL (optional)
- Google Gemini API Key
- Reddit API Credentials (optional)
- Twitter API Credentials (optional)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/social-media-analysis.git
cd social-media-analysis
```
Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```
Run the pipeline:

```bash
python scripts/run_pipeline.py --keyword "artificial intelligence" --source generated
```
Usage
Command Line
bash
# Analyze generated data
python scripts/run_pipeline.py --keyword "technology" --source generated --limit 100

# Analyze Reddit data
python scripts/run_pipeline.py --keyword "python" --source reddit --limit 50

# Generate report
python scripts/run_pipeline.py --keyword "AI" --report report.html

# Skip sentiment analysis (faster)
python scripts/run_pipeline.py --keyword "tech" --no-sentiment
Jupyter Notebook
bash
jupyter notebook notebooks/analysis_demo.ipynb
Streamlit Dashboard
bash
# Create a Streamlit app (requires additional code)
streamlit run src/visualization/streamlit_app.py
Project Structure
text
real-time-social-media-analysis/
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── data_ingestion/      # Data fetching modules
│   │   ├── reddit_fetcher.py
│   │   ├── twitter_fetcher.py
│   │   └── data_generator.py
│   ├── preprocessing/       # Text cleaning
│   │   └── text_cleaner.py
│   ├── analysis/            # Sentiment and trend analysis
│   │   ├── sentiment_analyzer.py
│   │   └── trend_analyzer.py
│   ├── storage/             # Database operations
│   │   └── database.py
│   └── visualization/       # Dashboard creation
│       └── dashboard.py
├── scripts/
│   ├── run_pipeline.py      # Main pipeline script
│   └── generate_report.py   # Report generation
├── notebooks/
│   └── analysis_demo.ipynb  # Jupyter notebook demo
├── tests/
│   └── test_analyzer.py     # Unit tests
├── requirements.txt
├── .env
└── README.md
API Integration
Google Gemini API
The system uses Google's Gemini API for zero-shot sentiment classification:

python
from src.analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment("I love this product!")
# Returns: {'sentiment': 'positive', 'emotion': 'joy', ...}
Reddit API (PRAW)
python
from src.data_ingestion import RedditFetcher

fetcher = RedditFetcher()
posts = fetcher.fetch_posts_by_keyword("python", limit=50)
Testing
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Create a Pull Request