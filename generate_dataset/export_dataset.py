import argparse
import pandas as pd
from pathlib import Path

from common import load_metadata, load_json, gather_paths_years, clean_string

"""Load and combine all proceedings and tutorial data from the JSON files."""
def load_proceedings_and_tutorials(pathlist: list[Path]) -> tuple[dict, dict]:
    proceedings = {}
    tutorials = {}

    for path in pathlist:
        content = load_json(path)
        if ".proceedings.json" in path.name:
            proceedings.update(content)
        else:
            tutorials.update(content)
    return proceedings, tutorials

"""Retrieve and return the reference data for a reference ID and section """
def get_reference_metadata(
        ref_id:str, 
        section:dict, 
        references_metadata:dict, 
        references_missing_metadata:dict
    ) -> dict:

    ref = references_metadata[str(ref_id)]
    return {
        "paperId": ref["paperId"],
        "corpusId": ref["externalIds"]["CorpusId"],
        "key": clean_string(ref["title"] + str(ref["year"])),
        "acl_id": ref["externalIds"]["ACL"] if "ACL" in ref["externalIds"] else "",
        "title": ref["title"],
        "authors": ref["authors"],
        "venue": ref["venue"],
        "abstract": ref["abstract"] if ref["abstract"] else references_missing_metadata[str(ref_id)]["abstract"],
        "year": ref["year"] if ref["year"] else references_missing_metadata[str(ref_id)]["year"],
        "citationCount": ref["citationCount"],
        "section": section["sectionName"],
        "subsection": section["subsectionName"]
    }

"""Generate a flat reading list with reference data for each tutorial."""   
def get_tutorial_reading_list(
        reading_list_ids:list, 
        references_metadata:dict, 
        references_missing_metadata:dict
    ) -> list:

    return [
        get_reference_metadata(ref_id, section, references_metadata, references_missing_metadata)
        for section in reading_list_ids
        for ref_id in section["referencesIds"] if ref_id
    ]

"""Retrieve and return the reference data for a tutorial """
def generate_dataset(
        proceedings:dict, 
        tutorials:dict, 
        references_metadata:dict, 
        references_missing_metadata:dict, 
        min_refs:int=3, 
        max_refs:int=20
    ) -> dict:

    dataset = {}
    for proceeding in proceedings.values():
        for tutorial_key in proceeding["tutorials"]:
            reading_list_ids = tutorials[tutorial_key]["readingList"]
            reading_list = get_tutorial_reading_list(reading_list_ids, references_metadata, references_missing_metadata)

            if min_refs <= len(reading_list) <= max_refs:
                dataset[tutorial_key] =  {
                    "id": tutorial_key,
                    "title": tutorials[tutorial_key]["title"],
                    "abstract": tutorials[tutorial_key]["abstract"],
                    "year": tutorials[tutorial_key]["year"],
                    "url": tutorials[tutorial_key]["url"],
                    "venues": ", ".join([venue["acronym"] for venue in proceeding["venues"]]),
                    "reading_list": reading_list
                }
    return dataset

"""Export a dict to a csv file"""   
def export_to_csv(
        dataset: pd.DataFrame,
        output_path:str
    ) -> None:
        pd.DataFrame.from_dict(dataset, orient='index').to_csv(f'{output_path}/reading_lists.csv', index=False)

"""Export a dict to a markdown file"""   
def export_to_md(
        dataset: pd.DataFrame,
        output_path:str
    ) -> None:
        markdown_list = []
        for tutorial in dataset.values():
            markdown_list.append(f"# {tutorial['title']}:")
            for item in tutorial["reading_list"]:
                authors = item['authors']
                authors = [author["name"].split()[-1] for author in authors]
                
                if(len(authors)) == 1:
                    authors = f"{authors[0]}"
                elif(len(authors)) == 2:
                    authors = f"{authors[0]} & {authors[1]}"
                elif(len(authors)) > 2:
                    authors = f"{authors[0]} et al."
                else:
                    authors = "Unknown authors"

                markdown_list.append(f"- [{item['title']}](https://www.semanticscholar.org/paper/{item['paperId']}) ({authors}, {item['venue']} {item['year']})")
            markdown_list.append('')
        markdown_string = "\n".join(markdown_list)
        with open(f'{output_path}/reading_lists.md', 'w') as f: f.write(markdown_string)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl Semantic Scholar for the dataset metadata')
    parser.add_argument('--data', required=True,
                        help='path of the data folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')

    args = parser.parse_args()

    years = ["before_2020", "2020", "2021", "2022", "2023", "2024"]
    pathlist = gather_paths_years(f"{args.data}", years)

    references_metadata = load_metadata(f'{args.data}/references_metadata.json')
    references_missing_metadata = load_metadata(f'{args.data}/references_missing_metadata.json')
    proceedings, tutorials = load_proceedings_and_tutorials(pathlist)

    nb_tutorials = sum(len(proceeding["tutorials"]) for proceeding in proceedings.values())
    print(f"{nb_tutorials} tutorials found in the dataset")

    dataset = generate_dataset(
        proceedings, tutorials, references_metadata, references_missing_metadata, 
        min_refs=3, max_refs=20
    )

    print("Exporting dataset as csv.") 
    export_to_csv(dataset, args.output)
    print("Exporting dataset as markdown.") 
    export_to_md(dataset, args.output)
    print("Exported.") 