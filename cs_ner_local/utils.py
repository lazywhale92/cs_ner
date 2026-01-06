
import logging
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

# Tokenizer setup (optional tiktoken)
# _tiktoken_encoding = None
# try:
#     import tiktoken
#     _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
#     logger.debug("tiktoken 사용: 정확한 토큰 카운팅")
# except ImportError:
#     logger.debug("tiktoken 미설치: 문자 수 기반 추정 사용")

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
    """
    Returns the number of tokens in a text string.
    tiktoken 설치 시 정확한 값, 미설치 시 문자 수 기반 추정.
    (한국어 기준 보수적으로 2자 = 1토큰으로 계산)
    """

    _tiktoken_encoding = None
    if _tiktoken_encoding is not None:
        return len(_tiktoken_encoding.encode(text))
    # Fallback: 문자 수 / 2 (한국어 보수적 추정)
    return max(1, len(text) // 2)
