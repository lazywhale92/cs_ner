
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
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini") # Default from original code
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    if not api_key or not endpoint:
        logger.error("Missing AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT in .env file.")
        return

    # 3. Load Data
    try:
        df = load_data(args.input, config)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    # 4. Load Categories
    try:
        categories = load_categories(args.categories)
        categories_json = json.dumps(categories, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return

    # 5. Prepare System Prompt
    system_msg = config.system_prompt_template.format(categories_json=categories_json)

    # 6. Initialize Client
    skip_ssl = os.getenv("SKIP_SSL_VERIFY", "false").lower() == "true"
    if skip_ssl:
        logger.warning("SSL 검증 비활성화됨")
        http_client = httpx.AsyncClient(verify=False)
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            http_client=http_client
        )
    else:
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

    # 7. Run Processing
    logger.info("Initializing Async API Processing...")
    start_time = datetime.now()
    
    results = asyncio.run(process_api_requests(
        data_df=df,
        config=config,
        system_msg=system_msg,
        client=client,
        deployment_name=deployment
    ))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Processing finished in {duration:.1f} seconds.")

    # 8. Merge and Save Results
    if not results:
        logger.warning("No results generated.")
        return

    results_df = pd.DataFrame(results)
    
    # Merge based on ID
    id_col = "thread_id" if config.domain_name == "air" else "ticket_id"
    
    if id_col not in results_df.columns:
        logger.error(f"Results missing ID column {id_col}. Cannot merge.")
        # Save raw results just in case
        save_results(results_df, "raw_results_error.csv")
        return

    # Ensure ID types match for merging
    df[id_col] = df[id_col].astype(str)
    results_df[id_col] = results_df[id_col].astype(str)

    # Left join original data with results
    final_df = pd.merge(df, results_df, on=id_col, how="left")
    
    # Define output path
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
