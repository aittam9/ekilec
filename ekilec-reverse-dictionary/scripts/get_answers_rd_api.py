import requests
import json
import argparse
import os
from collections import Counter


from tqdm import tqdm 
import pandas as pd

# handle paths of env and common utils
from dotenv import load_dotenv
import sys 
sys.path.append("../../common_utils")
from utils import format_prompt, prepare_data, chat, get_correct_labels
from models import PROMPT, API_MODELS
#load the openrouter key 
load_dotenv("../common_utils/.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")

DATA_DIR = f"../data"
RESULTS_DIR = f"../results/api_models"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type = str, default = "rd_dataset_formatted.json", help = "The dataset to shuffle. Refer to the last folder it is in.")
    parser.add_argument("--model", "-m", choices=list(API_MODELS.keys()), help = "The model to test")
    args = parser.parse_args() 

    model_id = API_MODELS[args.model]
    print(f"Evaluating {args.model} on {args.data}")

    
    inpath = f"{DATA_DIR}/{args.data}"
    outpath = RESULTS_DIR
   
    
    #load data and store correct answers and labels
    data = json.load(open(inpath, "r"))
    print(f"Length of eval data: {len(data)}")
    correct_labels, full_answers = get_correct_labels(data)
    
    if "gpt-5" or "gemini" in model_id:
        max_tokens = 1000 #necessary bc gpt-5 consumes a lot of tokens on reasoning
    else:
        max_tokens = 1
    # format prompts and get model responses
    all_prompts = prepare_data(PROMPT, data)
    all_model_answers = []
    for p in tqdm(all_prompts):
        response = chat(p, model_id, max_tokens=max_tokens)
        max_retry = 3
        while max_retry > 0 and not response:
            try:
                print("Something went wrong, retrying once...")
                response = chat(p, model_id=model_id, max_tokens=max_tokens)
            except:
                print("Retry failed, moving to next term.")
                print(response)
                max_retry -= 1
                response = None
   
        all_model_answers.append(response)
        
       
    #print accuracy for a fast check
    accuracy = len([(a,l) for a,l in zip(all_model_answers, correct_labels) if a == l]) / len(all_model_answers) * 100
    print(f"Accuracy of {args.model} on {args.data}: {accuracy}")
    print(f"Answer distribution: {Counter(all_model_answers)}")
    
    #store results
    results = {}
    results["question_id"] = list(data.keys())
    results["model_answers"] = all_model_answers
    results["correct_answers"] = correct_labels

    df = pd.DataFrame(results)
    #save results file
    outfile_name = f"{args.model}_rd.tsv"
    df.to_csv(os.path.join(outpath, outfile_name), sep = "\t")
    print(f"Saved at {os.path.join(outpath, outfile_name)}")







    