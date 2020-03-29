import os
import sys
import time
import json
import argparse
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

query = """SELECT DISTINCT ?item ?coord ?enlink ?delink ?frlink
WHERE
{
  VALUES (?obj) {(wd:WDID)}
  # ?item ?p ?class ; # extract all items related to a specific item. This will be used for generating data for the full dataset
  ?item ?p ?obj ; # Items should have relation with initial entities
        p:P625 [ # Get coordinates of items
           psv:P625 [
             wikibase:geoLatitude ?lat ;
             wikibase:geoLongitude ?lon ;
             wikibase:geoGlobe ?globe ;
           ] ;
           ps:P625 ?coord
         ]
  FILTER ( ?globe = wd:Q2 ). # Filter for coordinates being on earth
  ?enlink schema:isPartOf <https://en.wikipedia.org/>; # Get item english wikipedia
          schema:about ?item.
  ?delink schema:isPartOf <https://de.wikipedia.org/>; # Get item german wikipedia
          schema:about ?item.
  ?frlink schema:isPartOf <https://fr.wikipedia.org/>; # Get item french wikipedia
          schema:about ?item.
  # SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } # Get english labels (this make the query very slow)
} LIMIT 1000"""


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

initial_path = ROOT_PATH / f'mlm_dataset/initial/initial_{args.chunk}.json'
data = []

with open(initial_path) as json_file:
    data = json.load(json_file)

new_data = []
seen_ids = []
for d in data:
	if d['item'] not in seen_ids:
		new_data.append(d)
		seen_ids.append(d['item'].rsplit('/', 1)[-1])

unique_ids = list(set([d['item'].rsplit('/', 1)[-1] for d in data]))

assert len(seen_ids) == len(unique_ids) and set(seen_ids) == set(unique_ids)

start = time.perf_counter()
for i, id in enumerate(unique_ids):
    try:
      print(f'Getting results for id {id}')
      results = get_results(query.replace('WDID', id))
      for result in results["results"]["bindings"]:
          try:
              new_id = result['item']['value'].rsplit('/', 1)[-1]
              if new_id not in seen_ids:
                  sample = {
                  'item': result['item']['value'],
                  'coord': result['coord']['value'],
                  'enlink': result['enlink']['value'],
                  'delink': result['frlink']['value'],
                  'frlink': result['delink']['value']
                  }
                  new_data.append(sample)
                  seen_ids.append(new_id)
          except:
              print(f'====> Could not save item {new_id}')
    except:
      print(f'====> Could not get results for item {id}')
    toc = time.perf_counter()
    print(f'====> Finished id {id} -- {((i+1)/len(unique_ids))*100:.2f}% -- {toc - start:0.2f}s')
end = time.perf_counter()

mlm_entites_path = ROOT_PATH / f'mlm_dataset/entities/entities_{args.chunk}.json'
with open(mlm_entites_path, 'w') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total time to extarct MLM entities: {end - start:0.4f}')
print(f'Numner of unique entities: {len(new_data)}')
print(f'Finished chunk {args.chunk}')
print(f'------------------------------------------------------')