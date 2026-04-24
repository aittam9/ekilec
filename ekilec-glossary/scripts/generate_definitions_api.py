import requests
import json
import argparse
import os
import time

from dotenv import load_dotenv
from tqdm import tqdm 

import pandas as pd
from prompts import PROMPTS, MODELS

#load the openrouter key 
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")


def chat(user_prompt:str, model_id, max_tokens: int):
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
        return None
      


      
# helper to run queries 
def run_queries(data,model_id, base_prompt, max_len):
    #base_prompt = PROMPTS["0_SHOTS_v2"]
    model_answers = []
    for item in tqdm(data):
        current_prompt = base_prompt.format(term=item['term']) #.format(term=item['term']) #.replace("\n", " ")
        response = chat(current_prompt, model_id= model_id, max_tokens= max_len)
        
       
        
        if not response:
            # one shot-retry
            try:
                print("Something went wrong, retrying once...")
                response = chat(current_prompt, model_id= model_id, max_tokens= max_len)
            except:
                print("Retry failed, moving to next term.")
                response = "Failed"
        
        # # greedy retrying until we get a response
        #  trial_count = 0
        # while not response:
        #     print("Something went wrong, retrying...")
        #     response = chat(current_prompt, model_id= model_id, max_tokens= max_len)
                # trial_count += 1
                # if trial_count >=5:
                #     print("Max retries reached, moving to next term.")
                #     response = "Failed"
                #     break
            
        model_answers.append({
            "term": item['term'],
            "definition": item['term_definition'],
            "model_response": response.replace("\n", " ")
        })
    return pd.DataFrame(model_answers)

# def set_working_directories():
#     if os.getcwd().split("/")[-1].strip() == "scripts":

#         BASE_IN_DIR = "../data/"
#         BASE_OUT_DIR = "../results/generation/"
#     elif os.getcwd().split("/")[-1].strip() == "ekilec-glossary":
#         BASE_IN_DIR = "./data/"
#         BASE_OUT_DIR = "./results/generation/"
#     else:
#         print("Launch script from an allowed directory: scripts or parent")
#         print(os.getcwd())
#         exit(1)
#     return BASE_IN_DIR, BASE_OUT_DIR

def set_working_directories():
    if os.getcwd().split("\\")[-1].strip() == "scripts":

        BASE_IN_DIR = "..\\data\\"
        BASE_OUT_DIR = "..\\results\\generation\\"
    elif os.getcwd().split("\\")[-1].strip() == "ekilec-glossary":
        BASE_IN_DIR = ".\\data/"
        BASE_OUT_DIR = ".\\results\\generation\\s"
    else:
        print("Launch script from an allowed directory: scripts or parent")
        print(os.getcwd())
        exit(1)
    return BASE_IN_DIR, BASE_OUT_DIR


BATCHES = {"1": "0-250", "2" : "250-500", "3" : "500-750","4" : "750-1000", "5" : "1000-1250"}
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", choices = MODELS.keys(), default = "gpt-4o-mini",  help = "The openrouter model to test")
    parser.add_argument("--prompt", "-p", default = "0_SHOTS_v2")
    parser.add_argument("--is-pilot", "-ip", action= "store_true", default= False, help = "Run experiments on pilot data")
    parser.add_argument("--all-batches", "-ab", action= "store_true", default = False, help = "process all batches")
    parser.add_argument("--batch", "-b", help = "The batch to process")
    
    args = parser.parse_args() 
    print(os.getcwd())
    BASE_IN_DIR, BASE_OUT_DIR = set_working_directories()

    model_id = MODELS[args.model]
    model_name = model_id.split("/")[-1].strip()

    # dinamically handle input data  accounting for pilot and batches and output path
    if args.is_pilot:
        # TODO: fix and improve
        # data_path = "pilot_sample.json"
        data_path = os.path.join(BASE_IN_DIR, "pilot/pilot.json")
        # with open(data_path, "r", encoding="utf-8") as f:
        #     data = json.load(f)
        OUTPUT_DIR_PATH = os.path.join(BASE_OUT_DIR,"pilot", model_name)
        outfile_name = "pilot_responses.tsv"
    
    elif args.all_batches:
        data_path = os.path.join(BASE_IN_DIR, "processed/filtered_definitions.json")
        outfile_name = "full_data_responses" +".tsv"

    else:
        file_name = "batch_"+ BATCHES[args.batch] +".json"
        data_path = os.path.join(BASE_IN_DIR, f"processed/batches4search/{file_name}")
        outfile_name = "batch_"+ BATCHES[args.batch] +".tsv"
    
    OUTPUT_DIR_PATH = os.path.join(BASE_OUT_DIR, model_name)
        
    # buiild output directory
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)

    #load data
    with open(data_path, "r", encoding="utf-8") as f:
         data = json.load(f)

    # get max length of definitions to set max tokens
    
    # TODO: this is is in character, it should be in tokens!! maybe we can approximate it by computing word length and adding 
    # a factor (like 30%, considering that on average the word/token ratio may be around 0.75, stated here https://platform.openai.com/tokenizer)
    # max_len = max([len(item['term_definition'].split()) for item in data])
    # max_len = int(round(max_len + (max_len * 0.30))) ### This is not good as it abruptly cuts the definitions
    
    max_len = 1000
    # pd.DataFrame(model_answers).to_csv(f"../results/generations/model_responses_{args.model.replace('/', '_')}.tsv", index=False, sep = "\t")

    # run the queries and save results
    base_prompt = PROMPTS[args.prompt]
    print(f"Processing batch: {BATCHES[args.batch] if not args.all_batches and not args.is_pilot else 'pilot' if args.is_pilot else 'full data'}")
    print(f"Using prompt: {args.prompt}")
    print(f"Querying {args.model} | OpenRouter id = {model_id}")
    
    # get responses for batch
    #TODO add elapsed time computation 
    start = time.time()
    response_df = run_queries(data, model_id, base_prompt=base_prompt, max_len = max_len)
    end = time.time()
    elapsed_time = end - start
    print(f"Finished in: {elapsed_time/60:.2f} minutes")
    # check roughly how many generations end with complete sentences (.) and print the number 
    complete_sentences = response_df['model_response'].apply(lambda x: x.strip().endswith(".") or x.strip().endswith('."')).sum()
    total_responses = response_df.shape[0] 
    print(f"\nOut of {total_responses} responses, {complete_sentences} ({(complete_sentences/total_responses)*100:.2f}%) end with a proper end-of-sentence char.\n")

    #save responses to tsv
    response_df.to_csv(os.path.join(OUTPUT_DIR_PATH, outfile_name), index=False, sep = "\t")