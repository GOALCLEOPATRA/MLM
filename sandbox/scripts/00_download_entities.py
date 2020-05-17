import os
import sys
import time
import json
import argparse
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

query = """SELECT DISTINCT ?coord ?classLabel ?image
    WHERE
    {
    VALUES (?item) {(wd:WDID)}
    ?item p:P625 [ # Get coordinates of items
            psv:P625 [
                wikibase:geoLatitude ?lat ;
                wikibase:geoLongitude ?lon ;
                wikibase:geoGlobe ?globe ;
            ] ;
            ps:P625 ?coord
            ]
    FILTER ( ?globe = wd:Q2 ). # Filter for coordinates being on earth
    ?item wdt:P31 ?class .
    ?item (wdt:P18|wdt:P41|wdt:P242|wdt:P2716|wdt:P948|wdt:P3451|wdt:P1943|wdt:P94) ?image .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } # Get english labels (this make the query very slow)
    }"""

def get_results(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    # user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    user_agent = 'My User Agent 1.0'
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_wiki_links(id):
    wiki_links = {}
    for lang in ['en', 'de', 'fr', 'it', 'es', 'pl', 'ro', 'nl', 'hu', 'pt']:
        queryWiki = f"""SELECT DISTINCT ?{lang}link
                        WHERE
                        {{
                            VALUES (?item) {{(wd:{id})}}
                            ?{lang}link schema:isPartOf <https://{lang}.wikipedia.org/>; # Get item English wikipedia
                                schema:about ?item.
                        }}
                    """
        try:
            wiki_links[f'{lang}link'] = get_results(queryWiki)["results"]["bindings"][0][f'{lang}link']['value']
        except Exception as e:
            print(f'Could not get {lang}link for item {id}')
    return wiki_links

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
args = parser.parse_args()

assert args.chunk in range(1, 21)

initial_path = ROOT_PATH / f'mlm_dataset/initial/initial_{args.chunk}.json'
data = []

with open(initial_path) as json_file:
    data = json.load(json_file)

assert len(data) == len(set(data))

new_data = []
start = time.perf_counter()
for i, id in enumerate(data):
    try:
        results = get_results(query.replace('WDID', id))
        if len(results["results"]["bindings"]) == 0:
            print(f'No results for id {id}')
            continue
        coord = results["results"]["bindings"][0]['coord']['value']
        classes = []
        images = []
        for result in results["results"]["bindings"]:
            classes.append(result['classLabel']['value'])
            images.append(result['image']['value'])

        classes = list(set(classes))
        images = list(set(images))
        # check for images
        if len(images) == 0:
            print(f'No images for item {id}')
            continue
        # check for wikilinks
        wiki_links = get_wiki_links(id)
        if len(wiki_links.keys()) == 0:
            print(f'No wiki links for item {id}')
            continue
        # save data
        sample = {
            'id': id,
            'coord': coord,
            'wikilinks': wiki_links,
            'images': images,
            'classes': classes
        }
        new_data.append(sample)
    except Exception as e:
        print(f'====> Could not get results for item {id}')
        print(str(e))
        continue
    toc = time.perf_counter()
    print(f'====> Finished id {id} -- {((i+1)/len(data))*100:.2f}% -- {toc - start:0.2f}s')
end = time.perf_counter()

mlm_entites_path = ROOT_PATH / f'mlm_dataset/entities/entities_{args.chunk}.json'
with open(mlm_entites_path, 'w') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total time to extarct MLM entities: {end - start:0.4f}')
print(f'Numner of unique entities: {len(new_data)}')
print(f'Finished chunk {args.chunk}')
print(f'------------------------------------------------------')