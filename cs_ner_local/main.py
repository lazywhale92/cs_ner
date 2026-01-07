
import os
import argparse
import asyncio
import sys
import pandas as pd
import json
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from datetime import datetime
import httpx

from config import get_config
from data_loader import load_data, save_results, load_categories
from processor import process_api_requests
from utils import logger

# Load environment variables
load_dotenv()

# window 호환성
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio._WindowsSelectorEventLoopPolicy())


def create_openai_client(api_key: str, api_version: str, endpoint: str) -> AsyncAzureOpenAI:
    """
    OpenAI 클라이언트 생성 (SSL 인증서 지원)
    우선순위: SSL_CERT_PATH > SKIP_SSL_VERIFY > 시스템 기본값
    """
    ssl_cert_path = os.getenv("SSL_CERT_PATH")
    skip_ssl = os.getenv("SKIP_SSL_VERIFY", "false").lower() == "true"

    if ssl_cert_path:
        if os.path.exists(ssl_cert_path):
            logger.info(f"SSL 인증서 사용: {ssl_cert_path}")
            http_client = httpx.AsyncClient(verify=ssl_cert_path)
            return AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                http_client=http_client
            )
        else:
            logger.warning(f"SSL 인증서 파일을 찾을 수 없음: {ssl_cert_path}")

    if skip_ssl:
        logger.warning("SSL 검증 비활성화됨 (보안 위험)")
        http_client = httpx.AsyncClient(verify=False)
        return AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            http_client=http_client
        )

    # 기본값: 시스템 인증서
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )


def main():
    parser = argparse.ArgumentParser(description="Local CS NER Processor")
    parser.add_argument("--domain", required=True, choices=["air", "air2", "package"], help="Domain to process")
    parser.add_argument("--input", required=True, help="Input Excel/CSV file path")
    parser.add_argument("--categories", required=True, help="Category definition Excel file path")
    parser.add_argument("--output", help="Output file path (default: auto-generated name)")

    args = parser.parse_args()

    # 1. Load Configuration
    try:
        config = get_config(args.domain)
        logger.info(f"Starting processing for domain: {config.domain_name}")
    except ValueError as e:
        logger.error(str(e))
        return

    # 2. Check API Keys
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    if not api_key or not endpoint:
        logger.error("Missing AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT in .env file.")
        return

    # 3. Load Original Data (전처리 없이)
    try:
        original_df = load_data(args.input, config, skip_preprocess=True)
        logger.info(f"Loaded {len(original_df)} rows with {len(original_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    # 4. Prepare API Data (도메인별 분기)
    if config.domain_name == "air":
        from preprocessing import mask_air, aggregate_by_thread

        # 4-1. 마스킹 적용 (원본 행 수 유지)
        masked_df = mask_air(original_df)
        logger.info(f"After masking: {len(masked_df)} rows")

        # 4-2. Thread 집계 (API 호출용)
        api_df = aggregate_by_thread(masked_df)
        logger.info(f"After thread aggregation: {len(api_df)} unique threads")
    else:
        # air2, package: 기존 방식 유지
        api_df = config.preprocess_func(original_df)

    # 5. Load Categories
    try:
        categories = load_categories(args.categories)
        categories_json = json.dumps(categories, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return

    # 6. Prepare System Prompt
    system_msg = config.system_prompt_template.format(categories_json=categories_json)

    # 7. Initialize Client (SSL 인증서 지원)
    client = create_openai_client(api_key, api_version, endpoint)

    # 8. Run Processing
    logger.info("Initializing Async API Processing...")
    start_time = datetime.now()

    results = asyncio.run(process_api_requests(
        data_df=api_df,
        config=config,
        system_msg=system_msg,
        client=client,
        deployment_name=deployment
    ))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Processing finished in {duration:.1f} seconds.")

    # 9. Merge Results with ORIGINAL Data (핵심!)
    if not results:
        logger.warning("No results generated.")
        return

    results_df = pd.DataFrame(results)

    # 병합 키 결정
    id_col = "thread_id" if config.domain_name == "air" else "ticket_id"

    if id_col not in results_df.columns:
        logger.error(f"Results missing ID column {id_col}. Cannot merge.")
        save_results(results_df, "raw_results_error.csv")
        return

    # 타입 일치
    original_df[id_col] = original_df[id_col].astype(str)
    results_df[id_col] = results_df[id_col].astype(str)

    # LEFT JOIN: 원본 데이터 기준 병합 (핵심!)
    # Air: 99행 original + 64개 thread 결과 → 99행 출력
    final_df = pd.merge(original_df, results_df, on=id_col, how="left")

    logger.info(f"Final output: {len(final_df)} rows, {len(final_df.columns)} columns")

    # 10. Save Results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{input_name}_result_{timestamp}.csv"

    try:
        save_results(final_df, output_path)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
