import os
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

def query_scholar(query, max_year=None):   
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

def generate_list(query, max_year=None):
    results = []
    response = query_scholar(query, max_year=max_year)
    if "data" in response :
        for paper in response["data"]:
            results.append({
                "id": paper["paperId"],
                "title": paper["title"],
                "year": paper["year"]
            })
    return results     

for annotator_i in [1,2,3]:
    print(f"Requesting annotator {annotator_i}")
    annotator_queries = pd.read_csv(f"../../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
    preds_annot = {query["id"]: generate_list(query["query_keywords"], max_year=query["year"]) for query in tqdm(annotator_queries)}
    with open(f"preds/semantic_scholar/preds_annot{annotator_i}.json", "w") as fp:
        json.dump(preds_annot , fp) 