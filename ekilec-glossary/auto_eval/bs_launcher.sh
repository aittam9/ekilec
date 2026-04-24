
# python bertscore_evaluation.py --model deepseek-chat-v3-0324 --output_dir embedder
# python bertscore_evaluation.py --model gemini-2.5-flash  --output_dir embedder
# python bertscore_evaluation.py --model gpt-4o  --output_dir embedder
# python bertscore_evaluation.py --model mistral-7b-instruct  --output_dir embedder
# python bertscore_evaluation.py --model minerva-7b-inst --output_dir embedder
# python bertscore_evaluation.py --model llama-3-8b-instruct --output_dir embedder
# python bertscore_evaluation.py --model llamantino-3-anita --output_dir embedder
# python bertscore_evaluation.py --model llamantino-3-anita --output_dir embedder

models=("mistral-7b-instruct"
        "minerva-7b-inst" 
        "llama-3-8b-instruct"
        "llamantino-3-anita" 
        "deepseek-chat-v3-0324"
        "gemini-2.5-flash"
        "gpt-4o"
        "gemini-3-flash-preview"
        "gpt-5.1-chat"
        "claude-sonnet-4.5"
        "llama-3.1-405b-instruct") # add here models to run iteratively on all datasets, e.g. "llama-3-8b-instruct" "mistral-7b-instruct" 


for m in "${models[@]}"; do
    python bertscore_evaluation.py --model $m --bert_type "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" --output_dir embedder
done

