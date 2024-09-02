import re
import json
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def generate_list(path):
    with open(str(path), "r") as file:  
        content = json.load(file)

        for item in content:
            if "citation-number" in item and "title" in item:
                title = item["title"][0]

                find_title_bold = re.compile("\*(.+?)\*")
                find_title_partial_bold = re.compile("(.+?)\*")
                if find_title_bold.findall(title):
                    title = find_title_bold.findall(title)[0]
                elif find_title_partial_bold.findall(title):
                    title = find_title_partial_bold.findall(title)[0]

                yield {"title":title.strip('"').strip('*').strip('"')}
    
for annotator_i in [1,2,3]:
    for run_i in [1,2,3]:
        files_json = sorted(Path(f"search_engine_results/gpt-4o/annotator{annotator_i}/run{run_i}/").glob(f'*.json'))
        preds_annot = {path.parts[-1].replace(f".json", ""): list(generate_list(path)) for path in tqdm(files_json)}
        with open(f"preds/gpt-4o/run{run_i}/preds_annot{annotator_i}.json", "w") as fp:
            json.dump(preds_annot , fp) 