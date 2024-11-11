import os
import json
import dotenv
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Optional
from collections import defaultdict
dotenv.load_dotenv()

# S2 API rate limit is 1RPM, async is not needed
""" Retrieve articles from S2 API"""
def search_papers(query:str, max_year:Optional[int]=None, offset:int=0, limit:int=20) -> dict:   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']
    payload = {
        "query":query,
        "offset":offset,
        "limit":limit,
        "fields":"title,year,externalIds,citationCount,authors"
    }
    if max_year:
        payload["year"] = f"-{max_year}"

    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=payload, headers={"x-api-key":apiKey})
    return r.json()

""" Fetch S2 API top k results using query and a maximum year to request """
def fetch_s2(query:str, k:int=1000, max_year:Optional[int]=None) -> list:
    if k > 1000:
        k = 1000
    results_topk = []
    for offset in range(0, k, 100):
        response = search_papers(query, max_year=max_year, offset=offset, limit=100)
        if "data" in response :
            for paper in response["data"]:
                results_topk.append({
                    "id": paper["externalIds"]["CorpusId"],
                    "externalIds": paper["externalIds"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "citationCount": paper["citationCount"],
                    "authors": [author["authorId"] for author in paper["authors"]]
                })
        if "next" not in response:
            break
    return results_topk

def get_top_papers(papers:list, k:int=20, acl_only:bool=False) -> list:
    if acl_only:
        return [paper for paper in papers if "ACL" in paper["externalIds"].keys()][0:k]
    else:
        return [paper for paper in papers][0:k]
    
def filter_cols(papers:list, cols:list=["id", "title", "year", "citationCount", "authors"]) -> list:
    return [{key: paper[key] for key in cols} for paper in papers]

""" Request S2 for each annotations """
def process_s2_request(annotations_folder:str, output_folder:str) -> None :
    for annotator_num in [1, 2, 3]:
        annotator_queries = pd.read_csv(f"{annotations_folder}/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
        result = defaultdict(lambda: defaultdict(dict))

        for query in tqdm(annotator_queries, desc=f"Requesting S2 (A{annotator_num})"):
            top1000 = fetch_s2(query["query_keywords"], k=1000, max_year=query["year"])
            result["any"][query["id"]] = filter_cols(get_top_papers(top1000, k=20, acl_only=False))
            result["acl"][query["id"]] = filter_cols(get_top_papers(top1000, k=20, acl_only=True))
            result["any_most_cited"][query["id"]] = filter_cols(
                sorted(
                    get_top_papers(top1000, k=100, acl_only=False), 
                    key=lambda d: d['citationCount'], reverse=True)[0:20])
            result["acl_most_cited"][query["id"]] = filter_cols(
                sorted(
                    get_top_papers(top1000, k=100, acl_only=True), 
                    key=lambda d: d['citationCount'], reverse=True)[0:20])

        for k, v in result.items(): 
            os.makedirs(f"{output_folder}/{k}/", exist_ok=True)
            with open(f"{output_folder}/{k}/preds_annot{annotator_num}.json", "w") as file:
                json.dump(v, file)

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