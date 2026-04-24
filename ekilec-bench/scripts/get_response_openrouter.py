import requests
import json
import argparse
import os
from dotenv import load_dotenv
from src.utils import format_prompt, prepare_data, get_correct_labels
from src.labels_format import PROMPT, MODEL_NAME_2_MODEL_ID
from tqdm import tqdm 
from collections import Counter
import pandas as pd

#load the openrouter key 
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
BASE_DIR = f"./converted_data/giustizia_amministrativa"

def chat(user_prompt:str, model_id,max_tokens: int):
    response = requests.post(
                            url="https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            },
                            data=json.dumps({
                                "model": model_id, 
                                "messages": [
                                {
                                    "role": "user",
                                    "content": f"{user_prompt}"
                                }
                                ],
                                "max_tokens" : max_tokens,
                                "temperature": 0.1,
                                "seed": 42

                            })
                            
                            )
    try:
        answer = response.json()["choices"][0]["message"]["content"]
        return answer
    except:
        print(f"\nOops something went wrong due to the following API error:\n {response.json()}")
        return response.json()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type = str, choices = os.listdir(BASE_DIR), help = "The dataset to shuffle. Refer to the last folder it is in.")
    parser.add_argument("--model", "-m", choices=list(MODEL_NAME_2_MODEL_ID.keys()), help = "The model to test")
    parser.add_argument("--shuffle", "-s", type = str,  choices = ["1","2","3","4"], help = "The shuffle dataset to load")
    args = parser.parse_args() 

    model_id = MODEL_NAME_2_MODEL_ID[args.model]
    print(f"Evaluating {args.model} on {args.data} with shuffle {args.shuffle}")

    file_name = f"{args.data}_shuffle{args.shuffle}"
    indir = f"evaluation/shuffled_mqa/{args.data}"
    outdir = f"evaluation/results/{args.model}/{args.data}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #load data and store correct answers and labels
    data = json.load(open(os.path.join(indir, file_name+".json" ), "r"))
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
            # one shot-retry
            try:
                print("Something went wrong, retrying once...")
                response = chat(p, model_id=model_id, max_tokens=max_tokens)
            except:
                print("Retry failed, moving to next term.")
                print(response)
                max_retry -= 1
                response = None
        #print(response)
   
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

    df = pd.DataFrame( results)
    #save results file
    df.to_csv(os.path.join(outdir, file_name+".tsv"), sep = "\t")
    print(f"Saved at {os.path.join(outdir, file_name+'.tsv')}")







    