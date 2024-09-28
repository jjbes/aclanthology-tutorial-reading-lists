import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def query_scholar(query, max_year=None, offset=0, limit=20):   
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

def generate(query, max_year=None): # Ensure 20 results
    results_acl = []
    results_any = []
    offset = 0
    while len(results_any) < 20 and offset < 1000:
        response = query_scholar(query, max_year=max_year, offset=offset, limit=100)
        if "data" in response :
            for paper in response["data"]:
                if "ACL" in paper["externalIds"].keys() and len(results_any) < 20:
                    results_acl.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
                if len(results_acl) < 20:
                    results_any.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
        offset += 100
    return {"semantic_scholar_any":results_any, "semantic_scholar_acl":results_acl} 

for annotator_i in [1,2,3]:
    print(f"Requesting annotator {annotator_i}")
    annotator_queries = pd.read_csv(f"../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')

    preds = defaultdict(lambda : defaultdict(list))
    for query in tqdm(annotator_queries):
        generation = generate(query["query_keywords"], max_year=query["year"])
        preds["semantic_scholar_any"][query["id"]] = generation["semantic_scholar_any"]
        preds["semantic_scholar_acl"][query["id"]] = generation["semantic_scholar_acl"]
    
    for model in ["semantic_scholar_any", "semantic_scholar_acl"]:
        FOLDER_PATH = f"preds/{model}/"
        if not os.path.exists(FOLDER_PATH):
            os.makedirs(FOLDER_PATH)
        FILE_PATH = f"{FOLDER_PATH}/preds_annot{annotator_i}.json"

        if not Path(FILE_PATH).is_file():
            with open(FILE_PATH, "w") as fp:
                json.dump(preds[model] , fp) 