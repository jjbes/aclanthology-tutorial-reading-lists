import re
import os
import json
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def parse(path):
    with open(str(path), "r") as file:  
        content = json.load(file)

        for item in content:
            if "citation-number" in item and "title" in item:
                title = item["title"][0].replace("\\", "")

                #Find quoted articles
                if '"' in title:
                    #surrounded by quotes
                    find_quotes = re.compile('\"(.+?)\"')
                    #only title and not authors
                    find_remaining_author = re.compile('(.+?) by ')
                    if find_quotes.findall(title):
                        title = find_quotes.findall(title)[0]
                    elif find_remaining_author.findall(title):
                        title = find_remaining_author.findall(title)[0]
                    #Clean remaining titles quotes and bold
                    title = re.sub('["*]', '', title)
                #Find unquoted articles
                else:
                    #Surrounded by bold
                    find_bold = re.compile('\*\*(.+?)\*\*')
                    #partial bold (but not a section title)
                    find_trailing_bold_not_section = re.compile('([^:]+?)\*\*')
                    if find_bold.findall(title):
                        title = find_bold.findall(title)[0]
                    elif find_trailing_bold_not_section.findall(title):
                        title = find_trailing_bold_not_section.findall(title)[0]
                    #Clean remaining titles bold
                    title = re.sub('[*]', '', title)
                year = None
                if "date" in item and re.compile("^\d\d\d\d$").match(item["date"][0]):
                    year = int(item["date"][0])

                yield {"title":title.strip('"').strip('*').strip('"'), "year":year}
    
MODEL_NAME="gpt-4o"
for annotator_i in [1,2,3]:
    files_json = sorted(Path(f"results/{MODEL_NAME}/annotator{annotator_i}/").glob(f'*.json'))
    preds_annot = {path.parts[-1].replace(f".json", ""): list(parse(path)) for path in tqdm(files_json)}
    FOLDER_PATH = f"preds/{MODEL_NAME}/"
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
    with open(f"{FOLDER_PATH}/preds_annot{annotator_i}.json", "w") as fp:
        json.dump(preds_annot , fp) 