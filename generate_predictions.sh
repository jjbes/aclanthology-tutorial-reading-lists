# Current methods

## Semantic Scholar
python experiments/current_methods/search_engines/request_semantic_scholar.py \
    --annotations annotations \
    --output experiments/current_methods/search_engines/preds/semantic_scholar

## GPT-4o Base
python experiments/current_methods/instructs_models/request_openai_completion.py \
    --model gpt-4o-2024-08-06 \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06 \
    --output_type base \
    --rate_limit 5000
python experiments/current_methods/instructs_models/extract_json_from_markdown.py \
    --folder experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06 \
    --output experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06 \
    --parsing_type md_to_json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06

## GPT-4o JSON Mode
python experiments/current_methods/instructs_models/request_openai_completion.py \
    --model gpt-4o-2024-08-06 \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06_json \
    --output_type json_mode \
    --rate_limit 5000
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06_json \
    --output experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06_json \
    --parsing_type json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06_json

## GPT-4o Structured Output
python experiments/current_methods/instructs_models/request_openai_completion.py \
    --model gpt-4o-2024-08-06 \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06_structured_output \
    --output_type structured_output \
    --rate_limit 5000
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gpt-4o-2024-08-06_structured_output \
    --output experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06_structured_output \
    --parsing_type json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gpt-4o-2024-08-06_structured_output

## Gemini 1.5 Flash Base
python experiments/current_methods/instructs_models/request_google_genai.py \
    --model gemini-1.5-flash \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gemini-1.5-flash \
    --output_type base \
    --rate_limit 15
python experiments/current_methods/instructs_models/extract_json_from_markdown.py \
    --folder experiments/current_methods/instructs_models/results/gemini-1.5-flash
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gemini-1.5-flash \
    --output experiments/current_methods/instructs_models/preds/gemini-1.5-flash \
    --parsing_type md_to_json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gemini-1.5-flash

## Gemini 1.5 Flash JSON Mode
python experiments/current_methods/instructs_models/request_google_genai.py \
    --model gemini-1.5-flash \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gemini-1.5-flash_json \
    --output_type json_mode \
    --rate_limit 15
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gemini-1.5-flash_json \
    --output experiments/current_methods/instructs_models/preds/gemini-1.5-flash_json \
    --parsing_type json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gemini-1.5-flash_json

## Gemini 1.5 Pro JSON Mode
python experiments/current_methods/instructs_models/request_google_genai.py \
    --model gemini-1.5-pro \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/gemini-1.5-pro_json \
    --output_type json_mode \
    --rate_limit 2
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/gemini-1.5-pro_json \
    --output experiments/current_methods/instructs_models/preds/gemini-1.5-pro_json \
    --parsing_type json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/gemini-1.5-pro_json

## OLMoE 1B-7B-0924-Instruct
pyhton experiments/current_methods/instructs_models/request_transformer.py \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --annotations annotations \
    --output experiments/current_methods/instructs_models/results/OLMoE-1B-7B-0924-Instruct
python experiments/current_methods/instructs_models/extract_json_from_markdown.py \
    --folder experiments/current_methods/instructs_models/results/OLMoE-1B-7B-0924-Instruct
python experiments/current_methods/parse_results_json_preds.py \
    --input experiments/current_methods/instructs_models/results/OLMoE-1B-7B-0924-Instruct \
    --output experiments/current_methods/instructs_models/preds/OLMoE-1B-7B-0924-Instruct \
    --parsing_type md_to_json
python experiments/current_methods/match_title_to_semantic_ids.py \
    --folder experiments/current_methods/instructs_models/preds/OLMoE-1B-7B-0924-Instruct

# Classic methods

## Format ACL Dataset
python experiments/classic_methods/format_acl_anthology_collection.py \
    --dataset experiments/classic_methods/acl_anthology_dataset

## Semantic Scholar (For comparison)
python experiments/classic_methods/request_semantic_scholar.py \
    --annotations annotations \
    --output experiments/classic_methods/preds/semantic_scholar

## BM25
python experiments/classic_methods/predict_bm25.py \
    --annotations annotations \
    --output experiments/classic_methods/preds/bm25  \
    --dataset experiments/classic_methods/acl_anthology_dataset

## SPECTER2
python experiments/classic_methods/predict_specter2.py \
    --annotations annotations \
    --output experiments/classic_methods/preds/specter2  \
    --dataset experiments/classic_methods/acl_anthology_dataset
