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
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from pathlib import Path
import time
import wikipediaapi
ROOT_PATH = Path(os.path.dirname(__file__))

data_path = ROOT_PATH / 'dataset_files/pilot/capitals_imgurl.json'
data = []

with open(data_path) as json_file:
    data = json.load(json_file)

data_with_summaries = []
tic = time.perf_counter() # start timer
# we experiment getting summaries from all languages at the same time
# set wikipedia to english domain
wiki_wiki = wikipediaapi.Wikipedia('en')
for d in data:
    en_page = wiki_wiki.page(d['enlink'].rsplit('/', 1)[-1], unquote=True)
    if en_page.exists(): # we accept only datapoints that we can extract wikipedia articles
        de_page = en_page.langlinks['de']
        fr_page = en_page.langlinks['fr']
        if de_page.exists() and fr_page.exists(): # we should get wiki articles in all three languages
            d['en_wiki'] = {
                'title': en_page.title,
                'link': en_page.fullurl,
                'text': en_page.summary
            }
            d['de_wiki'] = {
                'title': de_page.title,
                'link': de_page.fullurl,
                'text': de_page.summary
            }
            d['fr_wiki'] = {
                'title': fr_page.title,
                'link': fr_page.fullurl,
                'text': fr_page.summary
            }

            # Since we have wiki titles, links and summaries delete old links
            d.pop('enlink', None)
            d.pop('delink', None)
            d.pop('frlink', None)

            data_with_summaries.append(d) # add new object to list
        else:
            print(f'Problem getting german or french from {en_page.title}')
    else:
        print(f'Problem with title {d["enlink"].rsplit("/", 1)[-1]} from link {d["enlink"]}')
toc = time.perf_counter() # end timer


write_path = ROOT_PATH / 'dataset_files/pilot/capitals_final.json'
with open(write_path, 'w') as json_file:
    json.dump(data_with_summaries, json_file, ensure_ascii=False, indent=4)

print(f'Total wikidata items to query for wikipedia articles (urls): {len(data)}')
print(f'Total time to query wikipedia articles (en, de, fr) for all: {toc - tic:0.4f}')
print(f'We extracted wikipedia articles from {len(data_with_summaries)} items')