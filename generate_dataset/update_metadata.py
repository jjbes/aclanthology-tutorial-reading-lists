import json
import argparse
from tqdm import tqdm
from common import load_metadata, request_metadata

""" Update specified field of all references """    
def update_references(references:dict, field:str="citationCount") -> dict:
    keys = list(references.keys())
    keys_batches = [keys[i:i + 500] for i in range(0, len(keys), 500)]#Batches of 500

    for batch in tqdm(keys_batches, desc="Requesting"):
        reponse = request_metadata(batch, fields=f"paperId,{field}")
        for reference in reponse:
            if reference["paperId"] in references:# S2 may return unwanted ids, this check prevents this behaviour
                if references[reference["paperId"]][field] != reference[field]:
                    print(f"Updated {reference['paperId']}: {references[reference['paperId']][field]} -> {reference[field]}" )
                    references[reference["paperId"]][field] = reference[field]
                
    return references


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl Semantic Scholar for the dataset metadata')
    parser.add_argument('--data', required=True,
                        help='path of the data folder')
    args = parser.parse_args()

    print("Updating metadata.") 
    references = load_metadata(f'{args.data}/references_metadata.json')
    updated_references = update_references(references, field="citationCount")

    with open(f'{args.data}/references_metadata.json', 'w') as f:
        json.dump(updated_references, f, indent=2)
    print("Updated.") 