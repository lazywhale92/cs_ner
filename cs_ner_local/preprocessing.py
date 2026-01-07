
import re
import pandas as pd
from typing import List, Optional

# Regex Patterns
PASSPORT_RE = re.compile(r'\bM[A-Za-z0-9]{8}\b')
PHONE_RE = re.compile(r'\b010-?\d{4}-?\d{4}\b')
PHONE_WIDE_RE = re.compile(r'(?:\+82[-\s\.]?)?0?1[0-9][-\s\.]?\d{3,4}[-\s\.]?\d{4}')

def mask_text_simple(text: str) -> str:
    """Masks phone numbers only (Air2/Package style)."""
    if not isinstance(text, str):
        return ""
    # Normalize newlines/tabs
    text = re.sub(r'[\r\n\t]+', ' ', text)
    return PHONE_WIDE_RE.sub("<MASKED_PHONE>", text)

def mask_text_advanced(text: str, mask_vals: List[str] = None) -> str:
    """Masks specific values + Passport + Phone (Air style)."""
    if not isinstance(text, str):
        return ""
    
    # 1. Mask specific column values if provided
    if mask_vals:
        for val in mask_vals:
            if val and str(val).strip():
                # Escape to treat as literal, replace with generic mask
                try:
                    text = re.sub(re.escape(str(val)), "<MASKED_VALUE>", text)
                except Exception:
                    pass

    # 2. Passport
    text = PASSPORT_RE.sub("<MASKED_PASSPORT>", text)
    
    # 3. Phone (Simple Air style)
    text = PHONE_RE.sub("<MASKED_PHONE>", text)
    
    return text

def mask_air(df: pd.DataFrame) -> pd.DataFrame:
    """
    Air 도메인 마스킹 (개별 ticket 레벨)
    - response_type 필터링
    - PII 마스킹 (여권번호, 전화번호, 개인정보)
    - 반환: 원본 행 수 유지, title_anon/content_anon 컬럼 추가
    """
    # Filter if column exists
    if "response_type" in df.columns:
        df = df[df["response_type"] == "RECEIVE"].copy()
    else:
        df = df.copy()

    # Columns to mask (if they exist)
    mask_cols = ["inquirer_id", "inquirer_name", "inquiry_status", "reservation_number", "destination"]

    def apply_mask(row):
        cols_vals = [row[c] for c in mask_cols if c in row]
        # Mask both title and content
        t = str(row.get("inquiry_title", ""))
        c = str(row.get("inquiry_content", ""))
        return mask_text_advanced(t, cols_vals), mask_text_advanced(c, cols_vals)

    # Apply masking
    mask_results = df.apply(apply_mask, axis=1)
    df["title_anon"] = [x[0] for x in mask_results]
    df["content_anon"] = [x[1] for x in mask_results]

    return df


def aggregate_by_thread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thread별 집계 (API 호출용)
    - groupby thread_id
    - content 시간순 연결
    - 반환: thread_id별 1행
    """
    if "thread_id" not in df.columns:
        raise ValueError("Air domain requires 'thread_id' column for aggregation.")

    # Sort by time first to ensure concat order
    if "inquiry_created_at" in df.columns:
        df = df.sort_values("inquiry_created_at")

    # helper for aggregation: join with space
    def join_text(x):
        return " ".join([str(s) for s in x if s])

    agg_rules = {
        "title_anon": join_text,
        "content_anon": join_text,
    }

    df_agg = df.groupby("thread_id", as_index=False).agg(agg_rules)

    # Rename to expected output: 'content' for the processor
    df_agg = df_agg.rename(columns={"content_anon": "content"})

    return df_agg


def preprocess_air(df: pd.DataFrame) -> pd.DataFrame:
    """
    Air Domain Preprocessing (하위 호환성 유지)
    1. 마스킹 적용
    2. Thread별 집계
    """
    masked_df = mask_air(df)
    return aggregate_by_thread(masked_df)

def preprocess_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Air2/Package Preprocessing:
    1. Anonymize inquiry_detail/content.
    """
    # Target column: user input should contain 'inquiry_detail' or mapped 'content'
    # In config, we map Excel 'inquiry_detail' -> 'content' at Load time.
    # So here we expect 'content'.
    
    if "content" not in df.columns:
        # Fallback if mapping failed? No, data_loader guarantees mapping based on config
        return df

    # Apply simple REGEX masking
    df["content"] = df["content"].apply(mask_text_simple)
    
    return df
