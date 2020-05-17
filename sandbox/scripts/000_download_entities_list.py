# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import os
import sys
import time
import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

query = """# human settlement 2 hop subclass of
SELECT ?onehop ?onehopLabel ?twohop ?twohopLabel ?threehop ?threehopLabel
WHERE
{
  ?onehop wdt:P279 wd:Q486972.
  ?twohop wdt:P279 ?onehop .
  ?threehop wdt:P279 ?twohop .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
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


results = get_results(query)

all_classes = [('Q486972', 'human settlement')]

for result in results["results"]["bindings"]:
    all_classes.append((result['onehop']['value'].rsplit('/', 1)[-1], result['onehopLabel']['value']))
    all_classes.append((result['twohop']['value'].rsplit('/', 1)[-1], result['twohopLabel']['value']))
    all_classes.append((result['threehop']['value'].rsplit('/', 1)[-1], result['threehopLabel']['value']))

print(len(all_classes))
all_classes = list(set(all_classes))
print(len(all_classes))

mlm_classes_path = ROOT_PATH / f'mlm_dataset/classes.json'
with open(mlm_classes_path, 'w') as json_file:
    json.dump(all_classes, json_file, ensure_ascii=False, indent=4)

all_entities = []
start = time.perf_counter()
for i, (id, v) in enumerate(all_classes):
    print(f'Starting id {id}')
    queryEnttities = f"""SELECT DISTINCT ?item
                        WHERE
                        {{
                            ?item wdt:P31 wd:{id} .
                        }}"""
    try:
        results = get_results(queryEnttities)
        for result in results["results"]["bindings"]:
            all_entities.append({'item': result['item']['value']})
    except Exception as e:
        print(f'Failed to extarct results for id {id}: {v}')
        continue
    toc = time.perf_counter()
    print(f'====> Finished id {id} -- {((i+1)/len(all_classes))*100:.2f}% -- {toc - start:0.2f}s')
    print(f'Total entities: {len(all_entities)}')

mlm_initial_entities_path = ROOT_PATH / f'mlm_dataset/initial_query.json'
with open(mlm_initial_entities_path, 'w') as json_file:
    json.dump(all_entities, json_file, ensure_ascii=False, indent=4)

print(f'Total entities (Not unique): {len(all_entities)}')
