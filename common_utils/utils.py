import re
import os 
import json
import random
import pandas as pd
from typing import Dict, List, Tuple, Literal
import requests
from dotenv import load_dotenv


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")

random_states = [1,2,3,4]
#shuffling function
def shuffle_data(data:Dict, labels:List[str] = ["A:", "B:", "C:", "D:"]) -> Dict:
    """
    Shuffle the dataset avoiding that the correct answer is always in the first place. Labels can be customized, but must be strings of length 2.
    Args:
    data:dict = the data to shuffle
    labels:List[str] = the labels to attach to answers
    """
    shuffled_dataset = {}
    for question in data:
        shuffled_dataset[question] = {}
        risp = data[question]["choices"]
        correct = data[question]["correct_answer"]
        shuffled_dataset[question]["question_text"] = data[question]["question_text"]
        random.shuffle(risp)
        shuffled_dataset[question]["choices"] = [f"{lab} {risp}" for lab,risp in zip(labels, risp)]
        shuffled_dataset[question]["correct_answer"] = [i for i in shuffled_dataset[question]["choices"] if i[3:] == correct][0]
    return shuffled_dataset 

#helper format a single prompt
def format_prompt(prompt_template, text, choices):
    return prompt_template.replace("{quesito_}", text ).replace("{risposte_possibili}", "\n".join(choices))


#format the whole dataset
def prepare_data(prompt, shuffled_dataset):
    all_prompts = []
    for k in shuffled_dataset:
        text = shuffled_dataset[k]["question_text"]
        choices = shuffled_dataset[k]["choices"]
        all_prompts.append(format_prompt(prompt, text, choices))
    return all_prompts


#store all correct answers and labels after the shuffling
def get_correct_labels(shuffled_dataset):
    correct_labels = []
    full_answers = []
    for k in shuffled_dataset:
        full_answers.append(shuffled_dataset[k]["correct_answer"])
        correct_labels.append(shuffled_dataset[k]["correct_answer"][0])
    return correct_labels, full_answers


#interact with the openrouter API
def chat(user_prompt:str, model_id:str, max_tokens: int) -> str | None:
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


def compute_accuracy(df:pd.DataFrame) -> Tuple[float, float]:
    correct = df["correct_answers"].tolist()
    predictions = df["model_answers"].tolist()
    accuracy = (len([i for i,e in zip(correct, predictions) if i == e]) / df.shape[0]) *100
    oos_answers = len([i for i in predictions if not i in ["A", "B", "C", "D"]]) / df.shape[0] *100
    return round(accuracy,2), round(oos_answers,5)