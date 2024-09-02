import os
import json
import requests
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def query_scholar(query, max_year=None, offset=0):   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']

    payload = {
        "query":query,
        "offset":offset,
        "limit":100,
        "fields":"title,year,externalIds",
        "year": f"-{max_year}" if max_year else "",
    }
    
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=payload, headers={"x-api-key":apiKey})
    return r.json()

def generate_list(query, max_year=None): # Ensure 20 results
    results = []
    offset = 0
    
    while len(results) < 20 and offset < 1000:
        response = query_scholar(query, max_year=max_year, offset=offset)
        if "data" in response :
            for paper in response["data"]:
                 if "ACL" in paper["externalIds"].keys() and len(results) < 20:
                    results.append({
                        "title": paper["title"],
                        "year": paper["year"]
                    })
        offset += 100
    return results     

for annotator_i in [1,2,3]:
    annotator_queries = pd.read_csv(f"../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
    preds_annot = {query["id"]: generate_list(query["query_keywords"], max_year=query["year"]) for query in tqdm(annotator_queries) }
    with open(f"preds/semantic_scholar/preds_annot{annotator_i}.json", "w") as fp:
        json.dump(preds_annot , fp) 
