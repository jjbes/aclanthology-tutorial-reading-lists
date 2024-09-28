import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def request(query, max_year):
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": query
            }
            ]
        }
        ],
        functions= [
            {
            "name": "generate_reading_list",
            "description": "Generate a reading list following the Article and ReadingList schema.",
            "parameters": {
                "type": "object",
                "properties": {
                "reading_list": {
                    "type": "array",
                    "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                        "type": "string",
                        "description": "The title of the article."
                        },
                        "authors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of authors for the article."
                        },
                        "year": {
                        "type": "integer",
                        "description": "Year the article was published."
                        }
                    },
                    "required": ["title", "authors", "year"]
                    }
                }
                },
                "required": ["reading_list"]
            }
            }
        ],
        function_call= { "name": "generate_reading_list" },
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = completion.choices[0].message
    return message.function_call.arguments


MODEL_NAME = "gpt-4o_json"
for annotator_i in [1,2,3]:
    print(f"Requesting annotator {annotator_i}")
    annotator_queries = pd.read_csv(f"../../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
    for query in tqdm(annotator_queries):
        FOLDER_PATH = f"results/{MODEL_NAME}/annotator{annotator_i}/"
        if not os.path.exists(FOLDER_PATH):
            os.makedirs(FOLDER_PATH)
        if not Path(f"{FOLDER_PATH}/{query['id']}.json").is_file():
            preds = request(query["query_sentence"], query["year"])
            preds = json.loads(preds)["reading_list"]
            with open(f"{FOLDER_PATH}/{query['id']}.json", "w") as f:
                json.dump(preds, f)