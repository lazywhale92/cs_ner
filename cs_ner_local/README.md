# CS NER Local Processor

This tool replaces the Databricks notebooks for classifying CS inquiries (**Air**, **Air2**, **Package**). It runs locally using Python and automatically handles domain-specific preprocessing (aggregation, masking, etc.) before calling the Azure OpenAI API.

## Features
- **Local Execution**: No Databricks/Spark required.
- **Unified Logic**: Supports `air` (thread aggregation) and `air2`/`package` (ticket-based) in one tool.
- **Async Processing**: High-performance async API calls with rate limiting.
- **Preprocessing**: Automatic PII masking (Phone, Passport) and message aggregation.

## Setup

1.  **Install Python 3.9+**
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API**:
    - Copy `.env.example` to `.env`.
    - Fill in your `AZURE_OPENAI_KEY` and other settings.

## Usage

Run the tool from the command line:

```bash
# Example for Air domain (Aggregates messages by thread_id)
python -m cs_ner_local.main \
  --domain air \
  --input "s3_data/air_cs_202503.xlsx" \
  --categories "s3_data/category_rule.xlsx" \
  --output "air_result.xlsx"

# Example for Air2 domain (Individual tickets)
python -m cs_ner_local.main \
  --domain air2 \
  --input "air2_data.xlsx" \
  --categories "category_rule.xlsx"

# Example for Package domain
python -m cs_ner_local.main \
  --domain package \
  --input "package_data.xlsx" \
  --categories "package_categories.xlsx"
```

## Arguments

- `--domain`: `air`, `air2`, or `package` (Required)
- `--input`: Path to the input Excel or CSV file. (Required)
- `--categories`: Path to the Excel file defining categories (rules). (Required)
- `--output`: Path for the output file. If omitted, generates a name with timestamp.

## Input File Formats

### Air
- Required Columns: `thread_id`, `inquiry_title`, `inquiry_content`.
- **Logic**: Automatically aggregates all rows with the same `thread_id` into one context window.

### Air2 / Package
- Required Columns: `ticket_id`, `inquiry_detail`.
- **Logic**: Processes each row independently.

## Category File Format
- Excel file with columns: `level1`, `level2`, `level3`, `description`, `note`. 
- Korean column names like `유형_1`, `설명` are also supported.
