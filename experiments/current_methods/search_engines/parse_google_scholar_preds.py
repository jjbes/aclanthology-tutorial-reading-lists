import json
from pathlib import Path
from bs4 import BeautifulSoup

def extract_ref_from_html(html):
    soup = BeautifulSoup(html, 'html.parser') 
    results = []
    for refs in soup.find_all("div", {"class": "gs_r gs_or gs_scl"}):
        if refs.find("h3").find("a"):
            title = refs.find("h3").find("a").text
        else:
            title = refs.find("h3").find_all('span')[3].text #Citations
        year = refs.find("div", {"class": "gs_a"}).text.split(" - ")[0][-4:]
        results.append({
            "title": title,
            "year": year
        })
    return results

def generate_list(pathlist):
    preds = {}
    for path in pathlist:
        key = path.parts[-1].replace(".html", "")
        with open(str(path)) as f:
            preds[key] = extract_ref_from_html(f)
    return preds

for annotator_i in [1,2,3]:
    preds_annot = generate_list(sorted(Path(f"results/google_scholar/annotator{annotator_i}").glob('*.html')))
    with open(f"preds/google_scholar/preds_annot{annotator_i}.json", "w") as fp:
        json.dump(preds_annot , fp)