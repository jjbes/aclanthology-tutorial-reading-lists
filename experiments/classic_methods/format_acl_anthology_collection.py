from pybtex.database import parse_file
from pylatexenc.latex2text import LatexNodes2Text
import re
import json
from tqdm import tqdm

def format_bibdata(bib_data):
    formated_data = []

    for key in tqdm(list(bib_data.entries)):
        entry = bib_data.entries[key]
        formated_data.append({
            "id": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["url"]).replace("https://aclanthology.org/",""),
            "title": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["title"]),
            "abstract": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["abstract"]) if "abstract" in entry.fields else "",
            "year": int(LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["year"]))
        })
    return formated_data

def clean_string(string):
    return re.sub(r'\W+','', string).lower() 

print("Loading ACL Anthology file")
#https://aclanthology.org/anthology+abstracts.bib.gz - collected on 24.07.2024
acl_collection = parse_file('acl_anthology_dataset/anthology+abstracts.bib')

# Format content of acl anthology
print("Formatting content")
acl_references_list = format_bibdata(acl_collection)
acl_anthology_dataset = {item["id"]: item for item in acl_references_list}

print(f"Formated {len(acl_anthology_dataset)} entries")
with open('acl_anthology_dataset/acl_anthology_dataset.json', 'w') as file: 
    json.dump(acl_anthology_dataset, file) 