'''
For pilot capitals - Extract summaries from wikipedia (en, de, fr)

Pilot Test:
Total wikidata items to query for wikipedia articles: 303
Total time to query wikipedia articles (en, de, fr) for all: 436.1962s
We extracted wikipedia articles from 303 items

Stats (Based on pilot model results):
For 100K data we will need 40 hours
to extract all articles in all three languages.
'''

import os
import sys
import time
import json
import argparse
import wikipediaapi
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--chunk', default=0, type=int, help='number of chunk')
args = parser.parse_args()

assert args.chunk in range(1, 21)

data_path = ROOT_PATH / f'mlm_dataset/unique/unique_{args.chunk}.json'
data = {}

with open(data_path) as json_file:
    data = json.load(json_file)

data_with_summaries = {}
count = 0
# we experiment getting summaries from all languages at the same time
# set wikipedia to english domain
tic = time.perf_counter() # start timer
wiki_wiki = wikipediaapi.Wikipedia('en')
for i, id in enumerate(list(data.keys())):
    print(f'Getting results for id {id}')
    try:
        en_page = wiki_wiki.page(data[id]['enlink'].rsplit('/', 1)[-1], unquote=True)
        if en_page.exists(): # we accept only datapoints that we can extract wikipedia articles
            if 'de' in en_page.langlinks and 'fr' in en_page.langlinks:
                try:
                    de_page = en_page.langlinks['de']
                    fr_page = en_page.langlinks['fr']
                    if de_page.exists() and fr_page.exists(): # we should get wiki articles in all three languages
                        data_with_summaries[id] = {
                            'en_wiki': en_page.summary,
                            'de_wiki': de_page.summary,
                            'fr_wiki': fr_page.summary
                        }
                        count += 1
                        toc = time.perf_counter()
                        print(f'====> Finished id {id} -- {((i+1)/len(list(data.keys())))*100:.2f}% -- {toc - tic:0.2f}s')
                    else:
                        print(f'---->Problem getting german or french from {en_page.title} - ID: {id}')
                except:
                    print(f'---->Problem getting german or french from {en_page.title} - ID: {id}')
            else:
                print(f'---->No german or french page for id {id}')
        else:
            print(f'---->Problem with title {data[id]["enlink"].rsplit("/", 1)[-1]} from link {data[id]["enlink"]}')
    except:
        print(f'---->Failed to get english page for id {id}')


summaries_path = ROOT_PATH / f'mlm_dataset/summaries/summaries_{args.chunk}.json'
with open(summaries_path, 'w') as json_file:
    json.dump(data_with_summaries, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total wikidata items to query for wikipedia articles (urls): {len(list(data.keys()))}')
print(f'We extracted wikipedia articles from {count} items')
print(f'Finished chunk {args.chunk}')
print(f'------------------------------------------------------')