
import logging
import tiktoken
import time
from dataclasses import dataclass
import os

# Logger configuration
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("cs_ner")

logger = setup_logging()

# Tokenizer setup
TOKEN_ENCODING_NAME = "cl100k_base"
try:
    encoding = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0

    def log_status(self):
        logger.info(
            f"Status: {self.num_tasks_succeeded} success, {self.num_tasks_failed} failed, "
            f"{self.num_tasks_in_progress} pending, {self.num_rate_limit_errors} rate limits"
        )

def count_tokens(text: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(text))
