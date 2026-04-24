import re 
import  os 
import json 
import argparse 
import itertools

import pandas as pd 


# read the data and unpack nested list
def load_data(data_path):
    data = json.load(open(data_path, "r"))
    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]
    return data


# helper to shorten the definition and extract only the first part
def shorten_definitions(data):
    rem = re.compile(r"(\w+[aeiouàò]\.)")
    new_dict_list = []
    for el in data:
        new_dict = {}
        new_dict["term"] = el["term"]
        new_dict["term_definition"] = "".join(re.split(rem, el["term_definition"])[:2]).replace("\n", "").strip()
        # new_dict["term_url"] = el["term_url"]
        # new_dict["related_articles"] = el["related_articles"]
        new_dict_list.append(new_dict)
    return new_dict_list

# helper to compute definitions length stats
def compute_len(data):
    definitions_len = [len(item['term_definition']) for item in data]
    print(f"Max: {max(definitions_len)} -- Min: {min(definitions_len)}")
    avg_def_len = round((sum(definitions_len) / len(definitions_len)), 3)
    print(f"Avg len: {avg_def_len}")
    return avg_def_len

# helper to filter in only average length definitions
def get_avg_len_def(data, window_size = 51):
    avg_def_len = compute_len(data)
    avg_definitions = [{"term" : item["term"], 
                        "term_definition": item['term_definition']}
                        for item in data if len(item['term_definition']) > avg_def_len -window_size and len(item['term_definition']) < avg_def_len +window_size ]
    print(f"Found {len(avg_definitions)} definitions in the range {avg_def_len} +-{window_size} ")
    return avg_definitions


DATA_DIR = "../data"
OUT_DIR = "../data/processed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help = "the dataset to process")
    parser.add_argument("--window_size", "-ws", default = 51)
    parser.add_argument("--output_dir", "-od", default = OUT_DIR, help = "The output data dir")
    parser.add_argument("--make_batches", "-mb", default=False, action= "store_true", help = "Wether output batches to perform google search later")
    args = parser.parse_args()
    
    if not args.data:
        file_name = "vocabulary_terms_wo_articles.json"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # load data and shorten the definitions
    
    data = load_data(file_path)
    shorten_def_dict = shorten_definitions(data)
    avg_len_def = get_avg_len_def(shorten_def_dict, window_size= args.window_size)
    print(f"Reduced defintion numbers from {len(data)} to {len(avg_len_def)}")
    # save the file
    outfile_name = "filtered_definitions.json"
    if args.output_dir != DATA_DIR:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    # saving processed file
    with open(os.path.join(args.output_dir, outfile_name), "w") as outfile:
        json.dump(avg_len_def, outfile, indent = 1)

    if args.make_batches:
        BATCHES_DIR = "../data/processed/batches4search"
        if not os.path.exists(BATCHES_DIR):
            os.makedirs(BATCHES_DIR)
        
        # produce batches
        print(f"Making batches...")
        batch_size = 250
        batches = []
        for i in range(0,len(avg_len_def), batch_size):
            current_batch = avg_len_def[i:i+batch_size]
            batches.append(current_batch)
            with open(os.path.join(BATCHES_DIR, f"batch_{i}-{i+batch_size}.json"), "w") as outfile:
                json.dump(current_batch, outfile, indent = 1)
                outfile.close()
        print(f"{len(batches)} batches saved in {BATCHES_DIR}")
    
    
    