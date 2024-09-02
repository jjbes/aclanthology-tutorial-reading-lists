import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def request_gpt(query, max_year):
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    response = client.chat.completions.create(
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
        temperature=0.3,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


for annotator_i in [1,2,3]:
    print(f"Requesting annotator {annotator_i}")
    annotator_queries = pd.read_csv(f"../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
    for run_i in [1,2,3]:
        print(f"Run {run_i}")
        for query in tqdm(annotator_queries):

            file = Path(f"search_engine_results/gpt-4o/annotator{annotator_i}/run{run_i}/{query['id']}.md")
            if not file.is_file():
                preds_annot = request_gpt(query["query_sentence"], query["year"])
                with open(f"search_engine_results/gpt-4o/annotator{annotator_i}/run{run_i}/{query['id']}.md", "w") as f:
                    f.write(preds_annot)