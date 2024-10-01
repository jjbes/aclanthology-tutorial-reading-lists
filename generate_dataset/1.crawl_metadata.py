import time
import json
import dotenv
from tqdm import tqdm
from typing_extensions import Dict, List 
from common import load_metadata, request_metadata, gather_paths_years, load_json

dotenv.load_dotenv()
   
"""Process all JSON files in the pathlist and fetch metadata for unique references."""
def process_references(pathlist:List, references:Dict) -> Dict:
    for path in tqdm(pathlist, desc="Requesting S2"):
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
    years = ["before_2020", "2020", "2021", "2022", "2023", "2024"]
    pathlist = gather_paths_years("../data", years)

    references = load_metadata('../data/references_metadata.json')
    references = process_references(pathlist, references)

    with open('../data/references_metadata.json', 'w') as f:
        json.dump(references, f, indent=2)