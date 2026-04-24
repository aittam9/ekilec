import pymupdf 
import pandas as pd 
import json 
import re 
import os 
import numpy as np
import argparse
from tqdm import tqdm


class TabledDocumentProcessor:
    """
    Class to process a document containing question-answering tests downloaed from the website Giustizia Amministrativa.
    Take a path pointinig to a pdf file formatted as a table with id,question,ans1,answ2,answ3,answ4
    Build two other files:
    a json and a csv containing the same information corresponding to the questions and their answers.
    """
    def __init__(self, path):
        self.doc = pymupdf.open(path)
        
        self.file_name = path.split("\\")[-1].split(".")[0]

    def extract_tables(self):
        print("Extracting tables from pages...")
        self.page_tables = []
        for page in tqdm(self.doc):
            tabs = page.find_tables()
            if tabs.tables:
                self.page_tables.append(tabs[0].extract())
        return self.page_tables
     
    def get_questions_from_tables(self):
        self.questions_from_tables = []
        print("Handling table rows...")
        for page in tqdm(self.page_tables):
            for row in page:
                if  row[0].isdigit():
                    self.questions_from_tables.append(row)
        
        return self.questions_from_tables 
    
    def make_structured_dataset(self, df):
        print("Making a structured dataset to be saved in json")
        structured = {}
        for id, row in df.iterrows():
            try:
                id = f"Question {id}"
                structured[id] = {}
                structured[id]["question_text"] = row.question_text.replace("\n", " ")
                structured[id]["choices"] = [i.replace("\n", " ") for i in row[2:]]
                structured[id]["correct_answer"] = row.correct_answer.replace("\n", " ")   
            except:
                pass
        return structured

    def save_csv(self,df, outpath):
        file_name = f"{self.file_name}_fromga.tsv"
        df.to_csv(os.path.join(outpath, file_name), sep = "\t")
 

    def save_json(self,structured,  outpath):
        file_name = f"{self.file_name}_fromga.json"
        with open(os.path.join(outpath, file_name), "w") as outfile:
            json.dump(structured, outfile, indent = 2)
         

    def process_document(self, outpath):
        self.extract_tables()
        questions = self.get_questions_from_tables()
        rows = np.array(questions)
        df = pd.DataFrame(rows, columns = ["ID", "question_text", "correct_answer", "wrong_answ1", "wrong_answ2", "wrong_answ3"]) #.apply(lambda x: x.replace("\n", " "))
        structured_df = self.make_structured_dataset(df)                                                                               
        
        self.save_csv(df, outpath)
        self.save_json(structured_df, outpath)
        print(f"Reusults saved at {outpath}")
        return structured_df


def make_out_dir():
    pass
    
INDIR = "../input_dir/tabled_docs"
OUTDIR = "../converted_data/giustizia_amministrativa"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help = "The path to document to process")
    parser.add_argument("--input_names", "-n", action = "store_true", help = "Print the list of possible inputs")
    #parser.add_argument("--output", "-o", help = "The output to save the processed document")
    args = parser.parse_args()
    
    #list the
    if args.input_names:
        for i in os.listdir(INDIR):
            if i.endswith(".pdf"):
                print(i)
        
    else:
        #handle input/output paths
        final_out_dir = os.path.join(OUTDIR, args.input.split(".")[0])
        inpath = os.path.join(INDIR, args.input)
        if not os.path.exists(final_out_dir):
            os.makedirs(final_out_dir)
        
        #process the document
        processor = TabledDocumentProcessor(inpath)
        processor.process_document(final_out_dir)
    




        
        
