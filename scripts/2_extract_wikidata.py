'''
For pilot capitals - Extract all wikidata triples in all languages (en, de, fr)

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
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from pathlib import Path
import time
ROOT_PATH = Path(os.path.dirname(__file__))

endpoint_url = "https://query.wikidata.org/sparql"

query = """
    SELECT ?itemLabel ?wdLabel ?ps_Label {
        VALUES (?item) {(wd:WDITEM)}

        ?item ?p ?statement .
        ?statement ?ps ?ps_ .

        ?wd wikibase:claim ?p.
        ?wd wikibase:statementProperty ?ps.

        SERVICE wikibase:label { bd:serviceParam wikibase:language "LANG" } # Gets labels in English. We can change to de and fr.
    }
"""


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

data_with_triples = []
count_triples = 0
tic = time.perf_counter()
for d in data:
    d['wikidata'] = {
        'en': [],
        'de': [],
        'fr': []
    }
    for lang in ['en', 'de', 'fr']:
        results = get_results(endpoint_url, query.replace('WDITEM', d['item'].rsplit('/', 1)[-1]).replace('LANG', lang))
        triples = []
        for result in results["results"]["bindings"]:
            triples.append((result['itemLabel']['value'], result['wdLabel']['value'], result['ps_Label']['value']))

        d['wikidata'][lang] = triples
        count_triples += len(triples)
    data_with_triples.append(d)
toc = time.perf_counter()


write_path = ROOT_PATH / 'dataset_files/pilot/capitals_wikidata.json'
with open(write_path, 'w') as json_file:
    json.dump(data_with_triples, json_file, ensure_ascii=False, indent=4)

print(f'Total wikidata items to query triples (urls): {len(data)}')
print(f'Total time to query triples for all languages (en, de, fr): {toc - tic:0.4f}')
print(f'We extracted triples from {len(data_with_triples)} items')
print(f'Total triples extracted: {count_triples}')