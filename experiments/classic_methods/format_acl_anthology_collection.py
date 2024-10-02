import json
import argparse
from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text
from pybtex.database import parse_file, BibliographyData

""" Format bibtex data to json """
def format_bibdata(bib_data:BibliographyData) -> dict:
    formated_data = []

    for key in tqdm(list(bib_data.entries), desc="Formatting BibliographyData to Json:"):
        entry = bib_data.entries[key]
        formated_data.append({
            "id": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["url"]).replace("https://aclanthology.org/",""),
            "title": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["title"]),
            "abstract": LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["abstract"]) if "abstract" in entry.fields else "",
            "year": int(LatexNodes2Text(math_mode='verbatim').latex_to_text(entry.fields["year"]))
        })
    return formated_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format ACL Dataset')
    parser.add_argument('--dataset', required=True,
                        help='path of the dataset folder')
    args = parser.parse_args()

    print("Processing ACL Anthology dataset (takes some time).")
    #https://aclanthology.org/anthology+abstracts.bib.gz - collected on 24.07.2024
    acl_collection = parse_file(f'{args.dataset}/anthology+abstracts.bib')
    acl_references_list = format_bibdata(acl_collection)
    acl_anthology_dataset = {item["id"]: item for item in acl_references_list}

    print(f"Processed {len(acl_anthology_dataset)} entries")
    with open(f'{args.dataset}/acl_anthology_dataset.json', 'w') as file: 
        json.dump(acl_anthology_dataset, file) 