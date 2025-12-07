# CS NER 로컬 처리기 (Local Processor)

이 도구는 기존 데이터브릭스 노트북을 대체하여 CS 문의(**Air**, **Air2**, **Package**)를 분류하는 파이썬 프로그램입니다. 데이터브릭스나 스파크 없이 로컬 PC에서 바로 실행되며, 각 도메인에 맞는 전처리(병합, 마스킹)를 자동으로 수행한 뒤 Azure OpenAI API를 호출합니다.


## 주요 기능
- **로컬 실행**: 데이터브릭스/스파크 없이 실행 가능.
- **통합 로직**: `Air` (스레드 병합)와 `Air2`/`Package` (개별 티켓) 방식을 하나의 툴에서 지원.
- **고성능 비동기 처리**: API 속도 제한(Rate Limit)을 준수하며 빠르게 처리.
- **자동 전처리**: 개인정보(여권, 전화번호) 마스킹 및 대화 내용 병합 자동화.

## 설치 방법 (Setup)

1.  **Python 3.9 이상 설치**
2.  **패키지 설치**:
    터미널에서 다음 명령어를 실행하세요.
    ```bash
    pip install -r cs_ner_local/requirements.txt
    ```
3.  **환경 변수 설정**:
    - `cs_ner_local/.env.example` 파일을 복사해서 `cs_ner_local/.env` 파일을 만드세요.
    - `.env` 파일 안에 `AZURE_OPENAI_KEY`와 엔드포인트 주소를 입력하세요.

## 사용 방법 (Usage)

터미널(프로젝트 최상위 폴더)에서 아래 명령어로 실행합니다.

```bash
# Air 도메인 실행 예시 (같은 thread_id 끼리 내용을 합쳐서 처리)
python -m cs_ner_local.main \
  --domain air \
  --input "s3_data/air_cs_202503.xlsx" \
  --categories "s3_data/category_rule.xlsx" \
  --output "air_result.xlsx"

# Air2 도메인 실행 예시 (개별 티켓 처리)
python -m cs_ner_local.main \
  --domain air2 \
  --input "air2_data.xlsx" \
  --categories "category_rule.xlsx"

# Package 도메인 실행 예시
python -m cs_ner_local.main \
  --domain package \
  --input "package_data.xlsx" \
  --categories "package_categories.xlsx"
```

## 옵션 설명 (Arguments)

- `--domain`: 처리할 도메인 (`air`, `air2`, `package` 중 택 1) **(필수)**
- `--input`: 입력 엑셀(.xlsx) 또는 CSV 파일 경로 **(필수)**
- `--categories`: 카테고리 규칙이 정의된 엑셀 파일 경로 **(필수)**
- `--output`: 결과 파일 저장 경로 (생략 시 시간값 포함하여 자동 생성됨)

## 입력 파일 형식

### Air
- 필수 컬럼: `thread_id`, `inquiry_title`, `inquiry_content` (순서대로 있어도 인식함)
- **로직**: 동일한 `thread_id`를 가진 행들을 하나로 합쳐서 문맥을 파악합니다.

### Air2 / Package
- 필수 컬럼: `ticket_id`, `inquiry_detail` (순서대로 있어도 인식함)
- **로직**: 각 행(티켓)을 개별적으로 처리합니다.

## 카테고리 파일 형식
- 엑셀 파일에 다음 컬럼들이 포함되어 있어야 합니다: `level1`, `level2`, `level3`, `description`, `note`
- 한국어 컬럼명(`유형_1`, `설명`, `비고` 등)도 자동으로 인식해서 처리합니다.