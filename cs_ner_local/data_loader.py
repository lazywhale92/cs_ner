
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from config import DomainConfig
from utils import logger

def load_data(file_path: str, config: DomainConfig, skip_preprocess: bool = False) -> pd.DataFrame:
    """
    Loads data from an Excel or CSV file and normalizes columns based on domain config.

    Args:
        file_path: 입력 파일 경로
        config: 도메인 설정
        skip_preprocess: True면 전처리 없이 원본 반환 (컬럼 정규화만 수행)

    Returns:
        DataFrame (skip_preprocess=True면 원본, False면 전처리 결과)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    logger.info(f"Loading data from {file_path}...")

    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")

    # Rename columns to standard internal names

    # 1. Positional Renaming (List)
    if isinstance(config.input_columns, list):
        if len(df.columns) == len(config.input_columns):
             df.columns = config.input_columns
             logger.info("Applied positional column renaming.")
        else:
             logger.warning(f"Column count mismatch. Expected {len(config.input_columns)}, got {len(df.columns)}. Skipping rename.")

    # 2. Dictionary Mapping (Dict)
    elif isinstance(config.input_columns, dict):
        df = df.rename(columns=config.input_columns)

    # skip_preprocess=True: 원본 데이터만 반환 (컬럼 정규화만 수행)
    if skip_preprocess:
        logger.info(f"Raw data loaded: {len(df)} rows, {len(df.columns)} columns (preprocessing skipped)")
        return df

    # 3. Run Domain Extraction / Preprocessing
    try:
        df_processed = config.preprocess_func(df)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.error(f"Available columns: {list(df.columns)}")
        raise e

    logger.info(f"Preprocessed data: {len(df_processed)} rows ready for API.")
    return df_processed

def load_categories(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads category definitions from an Excel file.
    Expected columns: level1, level2, level3, description, note
    (or similar, code will attempt to normalize)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Category file not found: {file_path}")
        
    logger.info(f"Loading categories from {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Normalize common column names
    normalization_map = {
        "유형_1": "level1", "유형_2": "level2", "유형_3": "level3",
        "설명": "description", "비고": "note",
        "Level1": "level1", "Level2": "level2", "Level3": "level3"
    }
    df = df.rename(columns=normalization_map)
    
    required = ["level1", "level2", "level3"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Category file missing required column: {col}")
    
    # Fill NaNs with empty strings for text fields to avoid JSON errors
    if "description" not in df.columns:
        df["description"] = ""
    if "note" not in df.columns:
        df["note"] = ""
        
    df = df.fillna("")
    
    records = df.to_dict(orient="records")
    logger.info(f"Loaded {len(records)} category rules.")
    return records

def save_results(df: pd.DataFrame, output_path: str):
    logger.info(f"Saving results to {output_path}...")
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        df.to_excel(output_path, index=False, engine='openpyxl')
    logger.info("Save complete.")
