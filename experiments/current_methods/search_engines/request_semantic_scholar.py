import os
import json
import dotenv
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Optional

dotenv.load_dotenv()

""" Retrieve articles from S2 API"""
def get_papers(query:str, max_year:Optional[int]=None) -> dict:   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']
    payload = {
        "query":query,
        "offset":0,
        "limit":20,
        "fields":"title,year",
        "year": f"-{max_year}" if max_year else "",
    }
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=payload, headers={"x-api-key":apiKey})
    return r.json()

""" Fetch S2 API using query and a maximum year to request """
def fetch_s2(query:str, max_year:Optional[int]=None) -> list:
    results = []
    response = get_papers(query, max_year=max_year)
    if "data" in response :
        for paper in response["data"]:
            results.append({
                "id": paper["paperId"],
                "title": paper["title"],
                "year": paper["year"]
            })
    return results     

""" Request S2 for each annotations """
def process_s2_request(annotations_folder:str, output_folder:str) -> None :
    for annotator_num in [1, 2, 3]:
        annotator_queries = pd.read_csv(f"{annotations_folder}/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
        preds_annot = {query["id"]: fetch_s2(query["query_keywords"], max_year=query["year"]) for query in tqdm(annotator_queries, desc=f"A{annotator_num}")}
        with open(f"{output_folder}/preds_annot{annotator_num}.json", "w") as fp:
            json.dump(preds_annot , fp) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request Semantic Scholar API for the generation of a reading list')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')

    args = parser.parse_args()
    
    print(f"Requesting Semantic Scholar.")
    process_s2_request(args.annotations, args.output)
    print(f"Requested.")