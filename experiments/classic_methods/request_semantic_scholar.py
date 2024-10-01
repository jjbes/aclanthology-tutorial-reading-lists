import os
import json
import dotenv
import requests
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import typing_extensions as typing 

dotenv.load_dotenv()

# S2 API rate limit is 1RPM, async is not needed

def get_papers(query:str, max_year:typing.Optional[int]=None, offset:int=0, limit:int=20) -> typing.Dict:   
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

def fetch_s2(query:str, max_year:typing.Optional[int]=None) -> list:
    results_acl = []
    results_any = []
    offset = 0
    while len(results_any) < 20 and offset < 1000:
        response = get_papers(query, max_year=max_year, offset=offset, limit=100)
        if "data" in response :
            for paper in response["data"]:
                # Try to get 20 ACL results out of 1000 first results
                if "ACL" in paper["externalIds"].keys() and len(results_any) < 20: 
                    results_acl.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
                # first 20 results for comparison
                if len(results_acl) < 20: 
                    results_any.append({
                        "id": paper["paperId"],
                        "title": paper["title"],
                        "year": paper["year"]
                    })
        offset += 100
    return {"semantic_scholar_any":results_any, "semantic_scholar_acl":results_acl}    

def process_s2_request() -> None :
    for annotator_i in [1, 2, 3]:
        print(f"Requesting S2's top 20 results")
        annotator_queries = pd.read_csv(f"../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
        preds = defaultdict(lambda : defaultdict(list))
        for query in tqdm(annotator_queries):
            generation = fetch_s2(query["query_keywords"], max_year=query["year"])
            preds["semantic_scholar_any"][query["id"]] = generation["semantic_scholar_any"]
            preds["semantic_scholar_acl"][query["id"]] = generation["semantic_scholar_acl"]
        with open(f"preds/semantic_scholar_any/preds_annot{annotator_i}.json", "w") as fp:
            json.dump(preds["semantic_scholar_any"] , fp)
        with open(f"preds/semantic_scholar_acl/preds_annot{annotator_i}.json", "w") as fp:
            json.dump(preds["semantic_scholar_acl"] , fp)  

if __name__ == "__main__":
    process_s2_request()