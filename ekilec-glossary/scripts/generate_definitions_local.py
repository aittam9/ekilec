
import os
import sys
import json
import argparse
import time 

import torch
import pandas as pd
from tqdm import tqdm

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

import sys 
sys.path.append("../")
sys.path.append("../ekilec-bench")

from models import OPEN_ITA_MODELS, GEN_PROMPTS

BASE_INDIR = "./data"
BASE_OUTDIR = "./results/generation" # "../autoeval/results/generation"
CACHE_DIR = "/extra/mattia.proietti/hf_models"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


BATCHES = {"1": "0-250", "2" : "250-500", "3" : "500-750","4" : "750-1000", "5" : "1000-1250"}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", choices = list(OPEN_ITA_MODELS.keys()), default = "gpt-4o-mini",  help = "The openrouter model to test")
    parser.add_argument("--prompt", "-p", default = "0_SHOTS_v2")
    parser.add_argument("--is-pilot", "-ip", action= "store_true", default= False, help = "Run experiments on pilot data")
    parser.add_argument("--all-batches", "-ab", action= "store_true", default = False, help = "process all batches")
    parser.add_argument("--batch", "-b", help = "The batch to process")

    args = parser.parse_args()
    
    model_id = OPEN_ITA_MODELS[args.model]
    model_name = model_id.split("/")[-1].strip()

    file_name = "batch_"+ BATCHES[args.batch] +".json"
    data_path = os.path.join(BASE_INDIR, f"./batches4search/{file_name}")
    outfile_name = "batch_"+ BATCHES[args.batch] +".tsv"

    OUTPUT_DIR_PATH = os.path.join(BASE_OUTDIR, model_name)

     # buiild output directory
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)

    #load data
    with open(data_path, "r", encoding="utf-8") as f:
         data = json.load(f)

    max_len = 1000 
     # run the queries and save results
    base_prompt = GEN_PROMPTS[args.prompt]
    print(f"Processing batch: {BATCHES[args.batch]}")
    print(f"Using prompt: {args.prompt}")
    print(f"Querying {args.model} | HF model id = {model_id}")

    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR).to(DEVICE)
    outline_model = outlines.from_transformers(model, tokenizer)

    # run the queries and save results
    model_answers = []
    start = time.time()
    for item in tqdm(data):
        concept = item["term"]
        prompt = base_prompt.replace("{term}", concept)
        outputs = outline_model.generate(prompt, max_new_tokens=max_len, temperature=0., do_sample=False)
        model_answers.append({
            "term": item['term'],
            "definition": item['term_definition'],
            "model_response": outputs.replace("\n", " ")
        })
        
    #build dataframe to save results and check how many responses end with complete sentences
    response_df = pd.DataFrame(model_answers)
   
    # check roughly how many generations end with complete sentences (.) and print the number 
    complete_sentences = response_df['model_response'].apply(lambda x: x.strip().endswith(".") or x.strip().endswith('."')).sum()
    total_responses = response_df.shape[0] 
    print(f"\nOut of {total_responses} responses, {complete_sentences} ({(complete_sentences/total_responses)*100:.2f}%) end with a proper end-of-sentence char.\n")

    #save responses to tsv
    response_df.to_csv(os.path.join(OUTPUT_DIR_PATH, outfile_name), index=False, sep = "\t")

    end = time.time()
    elapsed_time = divmod(end - start, 60)
    print(f"Finished in: {elapsed_time[0]} minutes and {elapsed_time[1]:.2f} seconds")
