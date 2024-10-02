import os
import json
import dotenv
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Optional
dotenv.load_dotenv()

# S2 API rate limit is 1RPM, async is not needed
""" Retrieve articles from S2 API"""
def get_papers(query:str, max_year:Optional[int]=None, offset:int=0, limit:int=20) -> dict:   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']
    payload = {
        "query":query,
        "offset":offset,
        "limit":limit,
        "fields":"title,year,externalIds",
        "year": f"-{max_year}" if max_year else "",
    }
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=payload, headers={"x-api-key":apiKey})
    return r.json()

""" Fetch S2 API using query and a maximum year to request """
def fetch_s2(query:str, max_year:Optional[int]=None) -> list:
    results_acl = []
    results_any = []
    for offset in range(0, 1100, 100):
        if(results_acl == 20):
            break
        response = get_papers(query, max_year=max_year, offset=offset, limit=100)
        if "data" in response :
            for paper in response["data"]:
                # Try to get 20 ACL results out of 1000 first results
                if "ACL" in paper["externalIds"].keys() and len(results_acl) < 20: 
                    results_acl.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
                # first 20 results for comparison
                if len(results_any) < 20: 
                    results_any.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
    return {"semantic_scholar_any":results_any, "semantic_scholar_acl":results_acl}    

""" Request S2 for each annotations, both all results and ACL only results  """
def process_s2_request(annotations_folder:str, output_folder:str) -> None :
    for annotator_num in [1, 2, 3]:
        annotator_queries = pd.read_csv(f"{annotations_folder}/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
        semantic_scholar_any = {}
        semantic_scholar_acl = {}

        for query in tqdm(annotator_queries, desc=f"Requesting S2 (A{annotator_num})"):
            generation = fetch_s2(query["query_keywords"], max_year=query["year"])
            semantic_scholar_any[query["id"]] = generation["semantic_scholar_any"]
            semantic_scholar_acl[query["id"]] = generation["semantic_scholar_acl"]
        
        os.makedirs(f"{output_folder}/any/", exist_ok=True)
        with open(f"{output_folder}/any/preds_annot{annotator_num}.json", "w") as file:
            json.dump(semantic_scholar_any , file)
        
        os.makedirs(f"{output_folder}/acl/", exist_ok=True)
        with open(f"{output_folder}/acl/preds_annot{annotator_num}.json", "w") as file:
            json.dump(semantic_scholar_acl , file)  

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