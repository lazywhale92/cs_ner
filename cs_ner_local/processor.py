
import asyncio
import json
import time
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import AsyncAzureOpenAI
import pandas as pd
# import nest_asyncio

from utils import logger, StatusTracker, count_tokens
from config import DomainConfig

# Apply nest_asyncio to allow nested event loops if necessary
# nest_asyncio.apply()

# Constants for Rate Limiting
BATCH_SIZE = 5
CHUNK_LOG = 10
SECONDS_TO_PAUSE_AFTER_RATE_LIMIT = 15
SECONDS_TO_SLEEP_EACH_LOOP = 0.001
CHECKPOINT_INTERVAL = 10  # 10 배치(50건)마다 체크포인트 저장


class CheckpointManager:
    """체크포인트 저장/복구 관리자"""

    def __init__(self, input_file: str, domain: str):
        # 입력 파일명 기반으로 체크포인트 파일명 생성
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        self.checkpoint_path = f"{base_name}_{domain}_checkpoint.json"
        self.domain = domain
        self.input_file = input_file

    def exists(self) -> bool:
        """체크포인트 파일 존재 여부"""
        return os.path.exists(self.checkpoint_path)

    def save(self, results: List[Dict], processed_batch_idx: int, total_batches: int):
        """현재 진행상황 저장"""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "input_file": self.input_file,
            "domain": self.domain,
            "processed_batch_idx": processed_batch_idx,
            "total_batches": total_batches,
            "results": results
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {processed_batch_idx}/{total_batches} batches ({len(results)} items)")

    def load(self) -> Optional[Dict]:
        """체크포인트 로드"""
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {data['processed_batch_idx']}/{data['total_batches']} batches")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def cleanup(self):
        """완료 후 체크포인트 삭제"""
        if self.exists():
            os.remove(self.checkpoint_path)
            logger.info(f"Checkpoint file removed: {self.checkpoint_path}")

@dataclass
class APIRequest:
    task_id: int
    batch_items: List[Dict[str, Any]]
    token_consumption: int
    attempts_left: int
    system_msg: str
    config: DomainConfig
    user_msg: str = field(init=False)
    result: List[Dict[str, Any]] = field(default_factory=list)
    error_msg: str = ""

    def __post_init__(self):
        # Use the domain-specific function to create user message
        self.user_msg = self.config.user_message_creator(self.batch_items)

    def get_batch_id(self) -> str:
        # Determine ID key based on domain
        id_key = "thread_id" if self.config.domain_name == "air" else "ticket_id"
        first_item = self.batch_items[0]
        return str(first_item.get(id_key, "unknown"))

    async def call_api(
        self,
        client: AsyncAzureOpenAI,
        deployment_name: str,
        retry_queue: asyncio.Queue,
        save_results: List[Dict[str, Any]],
        status_tracker: StatusTracker,
    ):
        batch_id = self.get_batch_id()
        logger.debug(f"[Request #{self.task_id}][Batch {batch_id}] Starting API call")
        
        error = None
        try:
            response = await client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": self.system_msg},
                    {"role": "user", "content": self.user_msg}
                ],
                temperature=0.0,
                max_tokens=1024
            )

            text = response.choices[0].message.content
            logger.debug(f"[Batch {batch_id}] Response received: {text[:80]}...")

            try:
                # Clean up JSON markdown code blocks if present
                clean_text = text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                
                parsed = json.loads(clean_text)
                
                if not (isinstance(parsed, list) and len(parsed) == len(self.batch_items)):
                    logger.warning(f"[Batch {batch_id}] Format Error: Item count mismatch")
                    raise ValueError("Response format error")
                self.result = parsed
                logger.debug(f"[Batch {batch_id}] Successfully processed")

            except json.JSONDecodeError as e:
                logger.error(f"[Batch {batch_id}] JSON Parse Error: {str(e)}")
                logger.debug(f"Raw response: {text}")
                error = e

        except Exception as e:
            import traceback
            logger.warning(f"[Request #{self.task_id}] Failed: {e}")
            logger.warning(f"[Request #{self.task_id}] traceback: \n{traceback.format_exc()}")
            status_tracker.num_api_errors += 1
            error = e
            if "rate limit" in str(e).lower():
                status_tracker.num_rate_limit_errors += 1
                # Decrement api_errors because we count rate limit separately
                status_tracker.num_api_errors -= 1 
                status_tracker.time_of_last_rate_limit_error = time.time()
                logger.warning(f"Rate limit error detected")

        if error:
            self.error_msg = str(error)
            if self.attempts_left > 0:
                self.attempts_left -= 1
                retry_queue.put_nowait(self)
            else:
                logger.error(f"[Request #{self.task_id}][Batch {batch_id}] Final Failure")
                # Create fallback empty results
                id_key = "thread_id" if self.config.domain_name == "air" else "ticket_id"
                fallback_data = [
                    {id_key: itm.get(id_key), "level1": None, "level2": None, "level3": None, "error": str(error)}
                    for itm in self.batch_items
                ]
                save_results.extend(fallback_data)
                status_tracker.num_tasks_failed += 1
                status_tracker.num_tasks_in_progress -= 1
        else:
            save_results.extend(self.result)
            status_tracker.num_tasks_succeeded += 1
            status_tracker.num_tasks_in_progress -= 1
            logger.debug(f"[Request #{self.task_id}][Batch {batch_id}] Result saved")


def task_id_generator_function():
    task_id = 0
    while True:
        yield task_id
        task_id += 1

async def process_api_requests(
    data_df: pd.DataFrame,
    config: DomainConfig,
    system_msg: str,
    client: AsyncAzureOpenAI,
    deployment_name: str,
    max_requests_per_minute: float = 100,
    max_tokens_per_minute: float = 200000,
    max_attempts: int = 5,
    input_file: str = "",
):
    """
    Main async processing loop with checkpoint support.
    """
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # Create batches
    batches = [
        data_df.iloc[i : i + BATCH_SIZE].to_dict(orient="records")
        for i in range(0, len(data_df), BATCH_SIZE)
    ]
    logger.info(f"Created {len(batches)} batches from {len(data_df)} items.")

    # 체크포인트 관리자 초기화
    checkpoint_mgr = CheckpointManager(input_file, config.domain_name) if input_file else None

    all_results = []
    batch_idx = 0
    batches_exhausted = False
    last_checkpoint_batch = 0

    # 체크포인트에서 복구
    if checkpoint_mgr and checkpoint_mgr.exists():
        checkpoint_data = checkpoint_mgr.load()
        if checkpoint_data:
            all_results = checkpoint_data.get("results", [])
            batch_idx = checkpoint_data.get("processed_batch_idx", 0)
            last_checkpoint_batch = batch_idx
            logger.info(f"Resuming from checkpoint: {batch_idx}/{len(batches)} batches ({len(all_results)} items already processed)")
            if batch_idx >= len(batches):
                batches_exhausted = True
    
    logger.debug("Entering main loop")
    
    while True:
        # Get next request
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logger.debug(f"Retrying request {next_request.task_id}")
            elif not batches_exhausted and batch_idx < len(batches):
                current_batch = batches[batch_idx]
                batch_idx += 1
                if batch_idx >= len(batches):
                    batches_exhausted = True
                
                # We need to approximate token count for rate limiting
                # We use the config's user msg creator to measure size
                user_msg_dummy = config.user_message_creator(current_batch)
                token_count = count_tokens(system_msg) + count_tokens(user_msg_dummy) + (100 * len(current_batch))
                
                next_request = APIRequest(
                    task_id=next(task_id_generator),
                    batch_items=current_batch,
                    token_consumption=token_count,
                    attempts_left=max_attempts,
                    system_msg=system_msg,
                    config=config
                )
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                
                if batch_idx % CHUNK_LOG == 0 or batch_idx == len(batches):
                    logger.info(f"Progress: {batch_idx}/{len(batches)} batches queued")
                    status_tracker.log_status()

                # 체크포인트 저장 (CHECKPOINT_INTERVAL 배치마다)
                if checkpoint_mgr and (batch_idx - last_checkpoint_batch) >= CHECKPOINT_INTERVAL:
                    # 비동기 작업 완료 대기
                    await asyncio.sleep(1.0)
                    # 실제 완료된 결과 수 기준으로 저장 (안전한 복구를 위해)
                    completed_batches = len(all_results) // BATCH_SIZE
                    checkpoint_mgr.save(all_results.copy(), completed_batches, len(batches))
                    last_checkpoint_batch = batch_idx
                    
        # Update Rate Limits
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time
        
        # Issue Request if valid
        if next_request:
            if (available_request_capacity >= 1 and 
                available_token_capacity >= next_request.token_consumption):
                
                available_request_capacity -= 1
                available_token_capacity -= next_request.token_consumption
                
                asyncio.create_task(
                    next_request.call_api(
                        client=client,
                        deployment_name=deployment_name,
                        retry_queue=queue_of_requests_to_retry,
                        save_results=all_results,
                        status_tracker=status_tracker
                    )
                )
                next_request = None
        
        if status_tracker.num_tasks_in_progress == 0 and batches_exhausted and queue_of_requests_to_retry.empty():
            break
            
        await asyncio.sleep(SECONDS_TO_SLEEP_EACH_LOOP)
        
        # Cool down if rate limited
        seconds_since_rate_limit = time.time() - status_tracker.time_of_last_rate_limit_error
        if (seconds_since_rate_limit < SECONDS_TO_PAUSE_AFTER_RATE_LIMIT 
            and status_tracker.time_of_last_rate_limit_error > 0):
            wait_time = SECONDS_TO_PAUSE_AFTER_RATE_LIMIT - seconds_since_rate_limit
            logger.warning(f"Rate limit cooldown: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            
    logger.info(f"Processing complete. Generated {len(all_results)} results.")

    # 완료 시 체크포인트 삭제
    if checkpoint_mgr:
        checkpoint_mgr.cleanup()

    return all_results
