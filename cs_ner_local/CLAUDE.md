# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference
Responds in Korean when asked in Korean.

## Project Overview

CS NER (Customer Service Named Entity Recognition) is a Python-based tool that classifies customer service inquiries into predefined categories using Azure OpenAI API (GPT-4). It replaces Databricks notebook implementations with a standalone local application.

**Three supported domains:**
- **air**: Accommodation/Hotel inquiries (thread 집계는 API 호출용, 출력은 원본 ticket_id별 유지)
- **air2**: Transportation/Flight inquiries (individual ticket processing)
- **package**: Package/Delivery inquiries (individual ticket processing)

## Running the Application

```bash
# Install dependencies
pip install -r cs_ner_local/requirements.txt

# Run classification
python -m cs_ner_local.main --domain {air|air2|package} --input <file.xlsx> --categories <rules.xlsx> --output <output.csv>
```

**Required environment variables** (create `.env` from `.env.example`):
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

**Optional environment variables** (SSL 인증서 설정):
- `SSL_CERT_PATH`: 사내망 SSL 인증서 경로 (예: `C:\cert\cert_NOL_SSL_2025.crt`)
- `SKIP_SSL_VERIFY`: SSL 검증 비활성화 (비권장, `true`로 설정 시)

## Architecture

```
cs_ner_local/
├── main.py          # CLI entry point, orchestrates workflow
├── config.py        # Domain configurations (DomainConfig dataclass)
├── processor.py     # Async API processing with rate limiting
├── data_loader.py   # File I/O, column normalization
├── preprocessing.py # Text masking (PII) & thread aggregation
└── utils.py         # Logging, token counting, status tracking
```

**Processing Flow:**
1. Load & normalize input data (Excel/CSV) - 원본 데이터 보존
2. Preprocess: mask PII (air: 개별 ticket 레벨)
3. Air domain: thread별 집계 (API 호출용으로만 사용)
4. Batch records (5 items per batch)
5. Async API calls with rate limiting (100 req/min, 200k tokens/min)
6. Merge results with ORIGINAL data (LEFT JOIN) - 원본 행 수 유지
7. Save output (air: 99행 입력 → 99행 출력, 17컬럼 + 3컬럼 = 20컬럼)

## Key Implementation Details

**Rate Limiting:** `processor.py` implements token-aware rate limiting with retry logic (max 5 attempts, 15s cooldown on rate limit errors).

**Domain Configuration:** Each domain in `config.py` specifies:
- Column mapping rules
- System prompt template
- User message formatter
- Preprocessing function

**PII Masking:** `preprocessing.py` masks passport numbers, phone numbers, and domain-specific fields (reservation IDs, names).

**Air Domain Processing:**
- `mask_air()`: 개별 ticket 레벨에서 PII 마스킹 (원본 행 수 유지)
- `aggregate_by_thread()`: API 호출용 thread 집계 (같은 thread_id 내용 연결)
- 최종 병합: 원본 데이터에 분류 결과 LEFT JOIN (모든 ticket_id 유지)

## Adding a New Domain

1. Create `DomainConfig` in `config.py` with column mappings and prompts
2. Add preprocessing function in `preprocessing.py` if needed
3. Register domain in `DOMAIN_CONFIGS` dictionary

## Category File Format

Supports Korean or English column names:
- `유형_1`/`level1`, `유형_2`/`level2`, `유형_3`/`level3`
- `설명`/`description`, `비고`/`note`
