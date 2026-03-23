"""
bot/history.py — Simple per-user conversation memory

Stores last N interactions per user (default = 3).
Used to provide context to LLM for better responses.
"""

from collections import defaultdict, deque
from typing import List, Dict

# Store history per user
# key = user_id, value = deque of messages
_user_history = defaultdict(lambda: deque(maxlen=6))


def add_user_message(user_id: int, role: str, content: str) -> None:
    """
    Add a message to user's history.

    Args:
        user_id: Telegram user ID
        role: "user" or "assistant"
        content: message text
    """
    _user_history[user_id].append({
        "role": role,
        "content": content
    })


def get_user_history(user_id: int) -> List[Dict]:
    """
    Get last N messages for user.

    Returns:
        List of messages (for LLM input)
    """
    return list(_user_history[user_id])


def clear_user_history(user_id: int) -> None:
    """
    Clear history for a user.
    """
    _user_history[user_id].clear()