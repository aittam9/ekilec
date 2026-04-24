import http.client
import json
import os
from tqdm import tqdm
import dotenv
import argparse

#//.*\.\w+/ = regex to shorten urls   --> //www.brocardi.it/
#\.\w+\.\w+ = regex to get domain --> .brocardi.it

# initiate input and output paths and API key
DATA_PATH = "../data/processed/batches4search"
OUTPUT_PATH = "../data/processed/batches_searched"
dotenv.load_dotenv()
API_KEY = os.getenv("SERPER_KEY")

#search one query
def serper_search(query:str, num_results:int=10, api_key:str = API_KEY)->dict:
    """
    Perform a search using the Serper.dev API.
    
    :param query: The search query string
    :param num_results: The number of search results to return
    :param api_key: The API key for authentication
    :return: The search results as a dictionary
    """
    assert api_key is not None, "API key must be provided"
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
      "q": query,
      "gl": "it",
      "hl": "it",
      "num": num_results
    })
    headers = {
      'X-API-KEY': api_key,
      'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

#extract from one batch file
def extract_from_batch(entry:str)->None:
    """
    Loads a batch file, performs searches for each term, and saves the results.
    :param entry: the file name of the batch to process
    """
    print(f"Processing file: {entry}")
    with open(os.path.join(DATA_PATH, entry), 'r', encoding='utf-8') as f:
            file_data = json.load(f)
    all_sources = []
    for item in tqdm(file_data):
        term = item['term']
        term_source = {}
        term_source["term"] = term
        term_source["sources"] = []

        # formating query
        query = f"Cosa significa {term} in diritto italiano?"
        
        # getting response from serper.dev
        response = serper_search(query)
        # looping over organic results
        print(f"Number of organic results for term '{term}': {len(response['organic'])}")
        for result in response["organic"]:
            term_source["sources"].append({i:v for i,v in result.items() if i in ["link", "position"]})
            all_sources.append(term_source)
    print(f"Finished processing file: {entry}")
    # saved results for batch file
    with open(os.path.join(OUTPUT_PATH, entry), 'w', encoding='utf-8') as f:
        json.dump(all_sources, f, ensure_ascii=False, indent=1)
  
def search_wiki_pool(list_of_terms:list)->dict:
    """
    Search for a list of terms in the wiktionary pool and return the results.
    :param list_of_terms: List of terms to search for
    :return: A dictionary with terms as keys and search results as values
    """
    results = []
    for term in tqdm(list_of_terms):
        term_source = {}
        term_source["term"] = term
        term_source["sources"] = []
        query = f"Cosa significa {term} in diritto italiano?"
        response = serper_search(query)
        for result in response["organic"]:
            term_source["sources"].append({i:v for i,v in result.items() if i in ["link", "position"]})
            results.append(term_source)
    
    filtered_results = []
    for item in results:
            if not item in filtered_results:
                filtered_results.append(item)
    
    return filtered_results

BATCH_MAP = {i:file_name for i,file_name in enumerate(os.listdir(DATA_PATH),1)}
CHOICES = list(BATCH_MAP.keys()) + ["all"]
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Search terms using Serper.dev API")
  parser.add_argument("--batch", type=int, choices=CHOICES, help="Path to input data files")
  parser.add_argument("--wiki_pool",  help="search only terms in the wiktionary pool", action="store_true")
  args = parser.parse_args()
  
  
  if args.batch == "all":
     
    for entry in tqdm(os.listdir(DATA_PATH)):
        extract_from_batch(entry)
  # search for the wikitionary pool of terms
  elif args.wiki_pool:
    with open("../notebooks/wikitionary_pool.txt", "r") as f:
        wiktionary_pool = [line.strip() for line in f.readlines()]
    results = search_wiki_pool(wiktionary_pool)

    with open(os.path.join(OUTPUT_PATH, "wiki_pool_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
  
  else:
    extract_from_batch(BATCH_MAP[args.batch])
      