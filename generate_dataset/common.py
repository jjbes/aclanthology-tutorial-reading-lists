

import os
import re
import json
import dotenv
import requests
from pathlib import Path

dotenv.load_dotenv()

""" Clean the string by removing non-alphanumeric characters and converting to lowercase """
def clean_string(string:str) -> str:
    return re.sub(r'\W+', '', string).lower()

""" Request S2 API for metadata of a list of IDs """
def request_metadata(ids:list, fields:str="corpusId") -> list:
    response = requests.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params={'fields': fields},
        json={"ids": ids},
        headers={"x-api-key":os.environ['SEMANTIC_SCHOLAR_API']}
    )
    if response.status_code != 200:
        print(response.text)

    return response.json()

""" Load a JSON file """
def load_json(path:str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

""" Load metadata files as JSON or create file if it doesn't exists """
def load_metadata(file_path:str) -> dict:
    Path(file_path).touch(exist_ok=True) # Ensure the file exists
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}
    
""" Gather sorted list of paths for JSON files of selected years """
def gather_paths_years(path:str, years:list[str]) -> list[Path]:
    paths = []
    for year in years:
        paths.extend(sorted(Path(f"{path}/{year}").glob('**/*.json')))
    return paths