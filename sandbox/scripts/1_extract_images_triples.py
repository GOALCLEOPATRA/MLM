'''
For pilot capitals - Extract all wikidata triples and image urls using wikidata items

Pilot test
Total wikidata items to query triples (urls): 308
Total time to query triples for all languages (en, de, fr): 744.8629
We extracted triples from 308 items
Total triples extracted: 82089

Stats (Based on pilot model results):
For 100K data we will need 68 hours to query for triples in all languages.
We can run 3 scripts one for each language and then we will need 22.5 hours for each one
'''

import os
import sys
import time
import json
import argparse
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

query = """SELECT DISTINCT ?itemLabel ?class ?classLabel ?image {
    VALUES (?item) {(wd:WDITEM)}

    ?item wdt:P31 ?class .
    ?item (wdt:P18|wdt:P41|wdt:P242|wdt:P2716|wdt:P948|wdt:P3451|wdt:P1943|wdt:P94) ?image.

    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
"""

def get_results(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    # user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    user_agent = 'My User Agent 1.0'
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--chunk', default=0, type=int, help='number of chunk')
args = parser.parse_args()

assert args.chunk in range(1, 21)

data_path = ROOT_PATH / f'mlm_dataset/unique/unique_{args.chunk}.json'
data = {}

with open(data_path) as json_file:
    data = json.load(json_file)

count_triples = 0
count_images = 0
all_triples = {}
all_images = {}

tic = time.perf_counter()
for i, id in enumerate(list(data.keys())):
    triples = []
    images = []
    try:
        print(f'Getting results for id {id}')
        # get results
        results = get_results(query.replace('WDITEM', id))
        for result in results["results"]["bindings"]:
            triples.append((result['itemLabel']['value'], result['class']['value'], result['classLabel']['value']))
            images.append(result['image']['value'])
    except:
        print(f'Problem with item {id} -- Could not get results')

    # we save example that have both triples and images
    if len(triples) > 0 and len(images) > 0:
        triples_set = list(set(triples))
        images_set = list(set(images))
        count_triples += len(triples_set)
        count_images += len(images_set)
        if id not in all_triples and id not in all_images:
            all_triples[id] = triples_set
            all_images[id] = images_set

    toc = time.perf_counter()
    print(f'====> Finished id {id} -- {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')

triples_path = ROOT_PATH / f'mlm_dataset/triples/triples_{args.chunk}.json'
with open(triples_path, 'w') as json_file:
    json.dump(all_triples, json_file, ensure_ascii=False, indent=4)

images_path = ROOT_PATH / f'mlm_dataset/images/images_{args.chunk}.json'
with open(images_path, 'w') as json_file:
    json.dump(all_images, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total wikidata items to query triples and images: {len(data)}')
print(f'Total time to query triples and images: {toc - tic:0.4f}')
print(f'We extracted triples and images from {len(all_triples.keys())} items')
print(f'Total triples extracted: {count_triples}')
print(f'Total images extracted: {count_images}')
print(f'Finished chunk {args.chunk}')
print(f'------------------------------------------------------')
