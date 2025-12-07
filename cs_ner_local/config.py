
import os
import json
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Union, Optional

from .preprocessing import preprocess_air, preprocess_simple

@dataclass
class DomainConfig:
    domain_name: str
    # input_columns can be Dict (mapping) or List (positional rename)
    input_columns: Union[Dict[str, str], List[str]] 
    system_prompt_template: str
    user_message_creator: Callable[[List[Dict[str, Any]]], str]
    preprocess_func: Callable[[pd.DataFrame], pd.DataFrame]

# --- Air Domain Logic ---

def create_user_message_air(batch_items: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, itm in enumerate(batch_items, 1):
        content_esc = str(itm.get("content", "")).replace('"', '\\"')
        lines.append(f'{idx}. {{"thread_id":"{itm.get("thread_id")}","content":"{content_esc}"}}')

    return (
        "다음 문의들을 분류해주세요. 응답은 정확히 다음 항목들에 대해서만, 배열 형태 JSON으로 반환하세요.\n"
        "입력된 thread_id와 정확히 일치하는 thread_id만 결과에 포함해야 합니다.\n"
        "항목 개수가 반드시 입력과 동일해야 합니다.\n\n"
        + "\n".join(lines)
        + f"\n\n주의사항:\n"
        f"1. 입력된 문의만 분류하세요. 추가 문의를 만들지 마세요.\n"
        f"2. 응답은 반드시 아래 스키마에 맞춰주세요.\n"
        f"3. 응답은 정확히 {len(batch_items)}개 항목을 포함해야 합니다.\n\n"
        f"RESPONSE_SCHEMA:\n[{{\"thread_id\":\"...\",\"level1\":\"...\",\"level2\":\"...\",\"level3\":\"...\"}}, …]"
    )

AIR_SYSTEM_PROMPT = (
    "당신은 CS문의 후처리 유형을 예측하는 모델입니다.\n"
    "각 카테고리에는 level1(유형_1), level2(유형_2), level3(유형_3)와 함께 description(설명)과 note(비고) 필드가 포함되어 있습니다.\n\n"
    "분류 시 다음 단계를 따르세요:\n"
    "1. 문의 내용을 파악하여 가장 적합한 level1, level2, level3 조합을 찾으세요.\n"
    "2. 여러 유사한 카테고리가 있을 경우, description 필드를 참조하여 더 적합한 카테고리를 선택하세요.\n"
    "3. note 필드에 예외 상황이나 특별 지시사항이 있는지 확인하고 이를 우선적으로 적용하세요.\n"
    "4. 모호한 경우 가장 구체적인 description을 가진 카테고리를 선택하세요.\n\n"
    "5. level3는 상위 분류(level1, level2)와 의미적으로 정합되며, 동일한 표현이 중복되지 않도록 조합하세요.\n"
    "6. level3는 반드시 해당 level2의 하위 항목으로만 분류하세요. 동일한 level3가 여러 level2에 존재할 경우, 문의 내용과 문맥상 의미가 가장 정확히 일치하는 조합을 선택하세요.\n"
    "7. level1의 명칭이 level2 또는 level3로 사용되지 않도록 하며, 분류 체계의 상하 관계를 유지하세요.\n"
    "8. 분류는 반드시 AVAILABLE_CATEGORIES에 명시된 level1~3 조합 중에서만 선택해야 합니다.\n"
    "9. 적절한 조합이 없다고 판단될 경우, 가장 가까운 의미의 조합을 선택하고 새 항목은 절대 생성하지 마세요.\n"
    "AVAILABLE_CATEGORIES = {categories_json}"
)

# --- Package / Air 2 Domain Logic ---

def create_user_message_simple(batch_items: List[Dict[str, Any]]) -> str:
    def norm(x: Any) -> str:
        return "" if x is None else str(x)

    lines = []
    for idx, itm in enumerate(batch_items, 1):
        content_esc = norm(itm.get("content")).replace('"', '\\"')
        pre1 = norm(itm.get("pre_level1", "")).replace('"', '\\"')
        pre2 = norm(itm.get("pre_level2", "")).replace('"', '\\"')
        pre3 = norm(itm.get("pre_level3", "")).replace('"', '\\"')

        lines.append(
            f'{idx}. {{"ticket_id":"{itm.get("ticket_id")}",'
            f'"content":"{content_esc}",'
            f'"pre_level1":"{pre1}","pre_level2":"{pre2}","pre_level3":"{pre3}"}}'
        )

    return (
        "다음 문의들을 분류해주세요. 응답은 정확히 아래 항목들에 대해서만, "
        "배열 형태 JSON으로 반환하세요.\n"
        "입력된 ticket_id와 정확히 일치하는 ticket_id만 결과에 포함해야 합니다.\n"
        f"항목 개수가 반드시 입력과 동일해야 합니다.\n\n"
        + "\n".join(lines)
        + "\n\n주의사항:\n"
          "1) pre_level1~3은 상담사가 사전 부여한 힌트입니다. "
          "그러나 문의 텍스트(content)를 최우선으로 해석하여 실제 의미와 다르면 힌트를 무시하고 재판단하세요.\n"
          "2) level1~3은 반드시 AVAILABLE_CATEGORIES 중 하나의 조합이어야 합니다. 새 항목을 만들지 마세요.\n"
          "3) 반환 스키마는 아래와 같고, 추가 필드를 만들지 마세요.\n\n"
          "RESPONSE_SCHEMA:\n"
          '[{"ticket_id":"...","level1":"...","level2":"...","level3":"..."}]'
    )

SIMPLE_SYSTEM_PROMPT = (
    "당신은 CS문의 후처리 유형을 예측하는 모델입니다.\n"
    "각 카테고리에는 level1(유형_1), level2(유형_2), level3(유형_3)와 함께 description(설명)과 note(비고) 필드가 포함되어 있습니다.\n\n"
    "입력 항목에는 content(사용자 문의 텍스트)와 pre_level1~3(상담사 사전 분류 힌트)이 포함됩니다.\n"
    "pre_level1~3은 참고용 힌트일 뿐 정답이 아닙니다. "
    "content의 의미가 힌트와 충돌하면 content를 최우선으로 해석하여 재판단하세요.\n"
    "분류 시 다음 단계를 따르세요:\n"
    "1. 문의 내용을 파악하여 가장 적합한 level1, level2, level3 조합을 찾으세요.\n"
    "2. 여러 유사한 카테고리가 있을 경우, description 필드를 참조하여 더 적합한 카테고리를 선택하세요.\n"
    "3. note 필드에 예외 상황이나 특별 지시사항이 있는지 확인하고 이를 우선적으로 적용하세요.\n"
    "4. 모호한 경우 가장 구체적인 description을 가진 카테고리를 선택하세요.\n\n"
    "5. level3는 상위 분류(level1, level2)와 의미적으로 정합되며, 동일한 표현이 중복되지 않도록 조합하세요.\n"
    "6. level3는 반드시 해당 level2의 하위 항목으로만 분류하세요. 동일한 level3가 여러 level2에 존재할 경우, 문의 내용과 문맥상 의미가 가장 정확히 일치하는 조합을 선택하세요.\n"
    "7. level1의 명칭이 level2 또는 level3로 사용되지 않도록 하며, 분류 체계의 상하 관계를 유지하세요.\n"
    "8. 분류는 반드시 AVAILABLE_CATEGORIES에 명시된 level1~3 조합 중에서만 선택해야 합니다.\n"
    "9. 적절한 조합이 없다고 판단될 경우, 가장 가까운 의미의 조합을 선택하고 새 항목은 절대 생성하지 마세요.\n"
    "AVAILABLE_CATEGORIES = {categories_json}"
)

# --- Configuration Map ---

# Column Definitions from 1_renewal... scripts
AIR_COLS = [
    "ticket_id", "thread_id", "first_inquiry_date", "inquiry_created_at",
    "inquiry_title", "inquiry_content", "inquiry_type_code", "parent_type",
    "inquiry_type", "inquiry_type_name", "response_type", "inquirer_id",
    "inquirer_name", "inquiry_status", "reservation_number", "destination",
    "product_info"
]

AIR2_COLS = [
    "ticket_id", "channel", "call_type", "inquiry_created_at", 
    "reservation_number", "customer_type", "inquiry_type", 
    "main_category", "sub_category", "detail_category", 
    "content", # Originally inquiry_detail, mapped to content for processor
    "department", "agent_name", "manager_name"
] # Len 14

PACKAGE_COLS = [
    "ticket_id", "channel", "call_type", "inquiry_created_at",
    "reservation_number", "customer_type", "inquiry_type",
    "main_category", "sub_category", "detail_category",
    "content", # Originally inquiry_detail
    "department", "agent_name", "manager_name"
] # Assuming same as Air2 based on 14 column check usually

CONFIGS = {
    "air": DomainConfig(
        domain_name="air",
        input_columns=AIR_COLS,
        system_prompt_template=AIR_SYSTEM_PROMPT,
        user_message_creator=create_user_message_air,
        preprocess_func=preprocess_air
    ),
    "air2": DomainConfig(
        domain_name="air2",
        input_columns=AIR2_COLS,
        system_prompt_template=SIMPLE_SYSTEM_PROMPT,
        user_message_creator=create_user_message_simple,
        preprocess_func=preprocess_simple
    ),
    "package": DomainConfig(
        domain_name="package",
        input_columns=PACKAGE_COLS,
        system_prompt_template=SIMPLE_SYSTEM_PROMPT,
        user_message_creator=create_user_message_simple,
        preprocess_func=preprocess_simple
    )
}

def get_config(domain: str) -> DomainConfig:
    if domain not in CONFIGS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[domain]
