import json
import argparse
from common import load_metadata

""" Check for references missing abstract or year and populate a file for manual annotation """
def check_missing_fields(references:dict, references_missing_metadata:dict) -> dict:
    for key, ref in references.items():
        if key not in references_missing_metadata:
            if not ref["abstract"] or not ref["year"]:
                references_missing_metadata[key] = {}
            if not ref["abstract"] and "abstract" not in references_missing_metadata[key]: 
                    references_missing_metadata[key]["abstract"] = None
            if not ref["year"] and "year" not in references_missing_metadata[key]:
                    references_missing_metadata[key]["year"] = None
                
    return references_missing_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl Semantic Scholar for the dataset metadata')
    parser.add_argument('--data', required=True,
                        help='path of the data folder')
    args = parser.parse_args()

    print("Detecting missing metadata.") 
    references = load_metadata(f'{args.data}/references_metadata.json')
    references_missing_metadata = load_metadata(f'{args.data}/references_missing_metadata.json')

    references_missing_metadata = check_missing_fields(references, references_missing_metadata)

    with open(f'{args.data}/references_missing_metadata.json', 'w') as f:
        json.dump(references_missing_metadata, f, indent=2)
    print(f"Detected. Please fill missing informations in {args.data}/references_missing_metadata.json if it is available.") 