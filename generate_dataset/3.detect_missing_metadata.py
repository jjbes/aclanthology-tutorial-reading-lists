import json
from typing import Dict

from common import load_metadata

""" Check for references missing abstract or year and populate a file for manual annotation """
def check_missing_fields(references:Dict, references_missing_metadata:Dict) -> Dict:
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
    references = load_metadata('../data/references_metadata.json')
    references_missing_metadata = load_metadata('../data/references_missing_metadata.json')

    references_missing_metadata = check_missing_fields(references, references_missing_metadata)

    with open('../data/references_missing_metadata.json', 'w') as f:
        json.dump(references_missing_metadata, f, indent=2)