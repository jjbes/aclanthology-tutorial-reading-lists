import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
import typing_extensions as typing
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

genai.configure()
model = genai.GenerativeModel('gemini-1.5-flash')

class Article(typing.TypedDict):
    title: str
    authors: list[str]
    year: int

def request(query, max_year):
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json", 
        response_schema=list[Article], 
        temperature=0
    )
    response = model.generate_content(query, generation_config=generation_config)
    return response.text

MODEL_NAME = "gemini-1.5-flash"
for annotator_i in [1,2,3]:
    print(f"Requesting annotator {annotator_i}")
    annotator_queries = pd.read_csv(f"../../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
    for query in tqdm(annotator_queries):
        FOLDER_PATH = f"results/{MODEL_NAME}/annotator{annotator_i}/"
        if not os.path.exists(FOLDER_PATH):
            os.makedirs(FOLDER_PATH)
        if not Path(f"{FOLDER_PATH}/{query['id']}.json").is_file():
            preds = request(query["query_sentence"], query["year"])
            with open(f"{FOLDER_PATH}/{query['id']}.json", "w") as f:
                f.write(preds)
