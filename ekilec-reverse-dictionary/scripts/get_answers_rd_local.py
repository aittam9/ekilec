
from collections import Counter
import os
import sys
import json
import argparse
import re
from typing_extensions import Literal
import time 
import sys 
sys.path.append("../../common_utils")
from utils import  prepare_data,  get_correct_labels
from models import PROMPT, OPEN_ITA_MODELS

import pandas as pd
from tqdm import tqdm

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()



CACHE_DIR = "/extra/mattia.proietti/hf_models"
DEVICE = "cuda:0"
OUTPUT_TYPE = Literal["A", "B", "C", "D"]

BASE_DIR = "../data"



# #helper format a single prompt
# def format_prompt(prompt_template, text, choices):
#     return prompt_template.replace("{quesito_}", text ).replace("{risposte_possibili}", "\n".join(choices))


# #format the whole dataset
# def prepare_data(prompt, shuffled_dataset):
#     all_prompts = []
#     for k in shuffled_dataset:
#         text = shuffled_dataset[k]["question_text"]
#         choices = shuffled_dataset[k]["choices"]
#         all_prompts.append(format_prompt(prompt, text, choices))
#     return all_prompts


# #store all correct answers and labels after the shuffling
# def get_correct_labels(shuffled_dataset):
#     correct_labels = []
#     full_answers = []
#     for k in shuffled_dataset:
#         full_answers.append(shuffled_dataset[k]["correct_answer"])
#         correct_labels.append(shuffled_dataset[k]["correct_answer"][0])
#     return correct_labels, full_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type = str, default = "rd_dataset_formatted.json", help = "The dataset to shuffle. Refer to the last folder it is in.")
    parser.add_argument("--model", "-m", choices=list(OPEN_ITA_MODELS.keys()), help = "The model to test")
    args = parser.parse_args()

    start = time.monotonic()
    
    model_id = OPEN_ITA_MODELS.get(args.model, "The model is not present or the key is wrong. Check models.py")
    
    
    inpath = os.path.join(BASE_DIR, args.data)
    outpath = f"../results/open_models/{args.model}_rd.tsv"
    

    #load data and store correct answers and labels
    data = json.load(open(inpath, "r"))
    print(f"Length of eval data: {len(data)}")
    correct_labels, full_answers = get_correct_labels(data)
    all_prompts = prepare_data(PROMPT, data)
    
    
    # #load hf model and tokenizer
    # hf_model, hf_tokenizer = load_hf_model(model_id)
    print(f"Loading {model_id}...")
    model = outlines.from_transformers(
                AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE, cache_dir=CACHE_DIR),
                AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
                )
    
    # get model responses
    all_model_answers = []
    for p in tqdm(all_prompts):        
        response = model(p, OUTPUT_TYPE, max_new_tokens=10, temperature=0., do_sample=False)
        all_model_answers.append(response)

    
    #print accuracy for a fast check
    accuracy = len([(a,l) for a,l in zip(all_model_answers, correct_labels) if a == l]) / len(all_model_answers) * 100
    print(f"Accuracy of {args.model} on {args.data}: {accuracy}")
    print(f"Answer distribution: {Counter(all_model_answers)}")

    #write results in a log
    with open("log.txt", "a") as ofile:
        ofile.write(f"Accuracy of {args.model} on {args.data}: {accuracy}\n" +
        f"Answer distribution: {Counter(all_model_answers)}\n\n")
    
    #store results
    results = {}
    results["question_id"] = list(data.keys())
    results["model_answers"] = all_model_answers
    results["correct_answers"] = correct_labels

    df = pd.DataFrame( results)
    #save results file
    df.to_csv(os.path.join(outpath), sep = "\t")
    print(f"Saved at {os.path.join(outpath)}")


    end = time.monotonic() 
    elapsed_time = divmod(end - start, 60)
    print(f"Total time: {elapsed_time[0]}m {elapsed_time[1]:.2f}s")