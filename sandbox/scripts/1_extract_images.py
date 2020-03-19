'''
For pilot capitals - Extract image urls using wikidata items

Pilot test
Total wikidata items to query images (urls): 308
Total time to query images for all: 91.3416s
We extracted images from 303 items (5 items had no image)

Stats (Based on pilot model results):
For 100K data we will need 8 hours to query for images
'''

import os
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from pathlib import Path
import time
ROOT_PATH = Path(os.path.dirname(__file__))

endpoint_url = "https://query.wikidata.org/sparql"

query = """SELECT DISTINCT ?image
WHERE {
  # Get images for pilot capitals
  wd:WDITEM (wdt:P18|wdt:P41|wdt:P242|wdt:P2716|wdt:P948|wdt:P3451|wdt:P1943|wdt:P94) ?image.
}"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

data_path = ROOT_PATH / 'dataset_files/pilot/capitals.json'
data = []

with open(data_path) as json_file:
    data = json.load(json_file)

data_with_images = []
tic = time.perf_counter()
for d in data:
    results = get_results(endpoint_url, query.replace('WDITEM', d['item'].rsplit('/', 1)[-1]))
    images = []
    for result in results["results"]["bindings"]:
        images.append(result['image']['value'])
    if len(images) > 0: # If the item has no image then we do not consider it
        d['images'] = images
        data_with_images.append(d)
toc = time.perf_counter()


write_path = ROOT_PATH / 'dataset_files/pilot/capitals_imgurl.json'
with open(write_path, 'w') as json_file:
    json.dump(data_with_images, json_file, ensure_ascii=False, indent=4)

print(f'Total wikidata items to query images (urls): {len(data)}')
print(f'Total time to query images for all: {toc - tic:0.4f}')
print(f'We extracted images from {len(data_with_images)} items')