"""
Helper utility functions for the sentiment analysis system.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional, Dict, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SocialMediaAnalysis")


def log_message(message: str, level: str = "INFO"):
    """
    Log a message with the specified level.

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    level = level.upper()
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message)


def safe_json_parse(text: str, default: Optional[Dict] = None) -> Dict:
    """
    Safely parse JSON string.

    Args:
        text: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed dictionary or default
    """
    if default is None:
        default = {}

    try:
        # Clean up the text
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Try to parse as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON-like content
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass

        log_message(f"Failed to parse JSON: {text[:100]}...", level="WARNING")
        return default


def validate_date(date_str: str) -> Optional[datetime]:
    """
    Validate and parse date string.

    Args:
        date_str: Date string to validate

    Returns:
        Datetime object or None if invalid
    """
    if not date_str:
        return None

    date_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    try:
        # Try pandas parsing
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        log_message(f"Failed to parse date: {date_str}", level="WARNING")
        return None


def format_timestamp(timestamp: Union[datetime, int, float, str]) -> str:
    """
    Format timestamp for display.

    Args:
        timestamp: Timestamp value

    Returns:
        Formatted date string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        dt = validate_date(timestamp)
        if dt is None:
            return timestamp
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def calculate_engagement(row: pd.Series) -> int:
    """
    Calculate total engagement from score and comments.

    Args:
        row: DataFrame row with 'score' and 'num_comments'

    Returns:
        Total engagement score
    """
    score = row.get("score", 0)
    comments = row.get("num_comments", 0)
    return int(score) + int(comments)


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with default.

    Args:
        data: Dictionary to search
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value or default
    """
    keys = key.split(".")
    current = data

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
