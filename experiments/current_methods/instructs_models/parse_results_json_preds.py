import os
import json
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def parse(path):
    with open(str(path), "r") as file:      
        try:
            data = json.load(file)
            return [{"title": item.get("title"), "year": item.get("year")} for item in data]
        except ValueError:
            return []

for model in ["gpt-4o_json", "gpt-4o-2024-08-06", "gemini-1.5-flash"]:
    for annotator_i in [1,2,3]:
            files_json = sorted(Path(f"results/{model}/annotator{annotator_i}/").glob(f'*.json'))
            preds = {path.parts[-1].replace(f".json", ""): parse(path) for path in tqdm(files_json)}
            FOLDER_PATH = f"preds/{model}/"
            if not os.path.exists(FOLDER_PATH):
                os.makedirs(FOLDER_PATH)
            with open(f"{FOLDER_PATH}/preds_annot{annotator_i}.json", "w") as fp:
                json.dump(preds , fp) 
