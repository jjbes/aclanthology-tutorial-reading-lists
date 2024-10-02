import time
import json
import dotenv
import argparse
from tqdm import tqdm
from common import load_metadata, request_metadata, gather_paths_years, load_json

dotenv.load_dotenv()
   
"""Process all JSON files in the pathlist and fetch metadata for unique references."""
def process_references(pathlist:list, references:dict) -> dict:
    for path in tqdm(pathlist, desc="Requesting"):
        if ".proceedings.json" not in path.name:
            tutorial_metadata = list(load_json(path).values())[0]
            reading_list = tutorial_metadata.get("readingList") 

            references_ids = [
                reference for section in reading_list
                for reference in section.get("referencesIds")
                if reference and reference not in references
            ]
            if references_ids:
                response = request_metadata(
                    references_ids, 
                    fields='paperId,externalIds,title,abstract,year,publicationVenue,s2FieldsOfStudy,referenceCount,citationCount,isOpenAccess'
                )
                time.sleep(1.0) #Rate-limiting
            else:
                response = []

            for reference in response:
                if reference.get("paperId") not in references:
                    references[reference.get("paperId")] = reference
    return references

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl Semantic Scholar for the dataset metadata')
    parser.add_argument('--data', required=True,
                        help='path of the data folder')
    args = parser.parse_args()

    years = ["before_2020", "2020", "2021", "2022", "2023", "2024"]
    pathlist = gather_paths_years(f"{args.data}", years)

    print("Populating metadata")
    references = load_metadata(f'{args.data}/references_metadata.json')
    references = process_references(pathlist, references)

    with open(f'{args.data}/references_metadata.json', 'w') as f:
        json.dump(references, f, indent=2)
    print("Populated.")