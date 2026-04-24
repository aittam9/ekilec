from typing import List, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor


from tqdm import tqdm
import pandas as pd
import json 

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.brocardi.it/dizionario/?page="

# desired entry schema
# {                     
# "term" :"term_text",
# "term_definition" : "definition_text",
# "term_url" :"term_url",
# "related_articles": ["related_article_1", "related_article_2" ]
# }

# function to extract definitions of terms present in a page
def get_terms_from_page(page_number:int) -> List[Dict]:
    
    """ Extract terms and their definitions from a given page number.
    Args:
        page_number (int): The page number to scrape.
        Returns:
        list: A list of dictionaries containing term details."""
    
    terms = []
    page_url = BASE_URL + str(page_number)
    
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    terms = []
    entries_list = soup.find("ul", class_="terms-list")
    
    try: 
        for n, el in enumerate(entries_list.find_all("li")): # soup.find("ul", class_="terms-list").find_all("li")):
            entry = {}
            term_link = "https://www.brocardi.it/"+ el.find("a")["href"]
            term_name = el.find("a").text.strip()
            
            # Fetch and parse the term's page
            term_response = requests.get(term_link)
            term_soup = BeautifulSoup(term_response.content, "html.parser")

            # Extract the definition
            definition_div = term_soup.find("div", class_="corpoDelTesto")
            if definition_div:
                definition_text = definition_div.get_text(separator="\n").strip()

            related_articles = []
            try:
                arts_section = term_soup.find("div", class_="text articolo-dictionary")
                for li in arts_section.find("ul").find_all("li"):
                    related_articles.append(li.text.strip())
            except:
                pass
            
            # populate dictionary
            #entry["term_id"] = n
            entry["term"] = term_name
            entry["term_definition"] = definition_text
            entry["term_url"] = term_link 
            entry["related_articles"] = related_articles  
        
            terms.append(entry)
    except Exception as e:
        pass
    
    return terms

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scrape vocabulary terms from a website.")
    parser.add_argument("--output", type=str, default="vocabulary_terms.json", help="Output JSON file name")
    args = parser.parse_args()

    # set pages number (brocardi dictionary has 73 pages as of 10/9/2925)
    pages = list(range(1, 74))

    # launch requests with multi threading
    terms = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map ISBNs to fetch_description function concurrently
        descriptions = list(tqdm(executor.map(get_terms_from_page, pages), total=len(pages)))
        terms.extend(descriptions)
    
    
    # flatten list of lists and sort by term name
    flatten_terms = [i for page in terms for i in page]
    flatten_terms_sorted = sorted(flatten_terms, key = lambda x: x['term'])
    
    print(flatten_terms_sorted[0])
    print(f"Total terms extracted: {len(flatten_terms_sorted)}")
    
    # save results to json file
    with open(f"data/{args.output}", "w") as f:
        json.dump(flatten_terms_sorted, f, indent=4, ensure_ascii=True)
    print(f"results saved as {args.output}")




    