'''
For pilot capitals - Download images and resize them to thumbnails

Pilot test
Total images downloaded: 1162
Total time to download images, resize into thumbnails and save them: 1449.2011s

Stats (Based on pilot model results):

'''

import os
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from pathlib import Path
import time
import urllib.request as req
from PIL import Image
import cairosvg
from urllib import parse
ROOT_PATH = Path(os.path.dirname(__file__))

data_path = ROOT_PATH / 'dataset_files/pilot/capitals_imgurl.json'
data = []

with open(data_path) as json_file:
    data = json.load(json_file)

data_with_images = []
image_base_path = ROOT_PATH / 'dataset_files/pilot/images/'
image_thumbnail_path = ROOT_PATH / 'dataset_files/pilot/thumbnails/'
count_images = 0
tic = time.perf_counter()
for i, d in enumerate(data):
    images_names = []
    for image in d['images']:
        # image = parse.unquote(image)
        img_name = f"{d['item'].rsplit('/', 1)[-1]}_{image.rsplit('/', 1)[-1]}"
        img_path = f'{image_base_path}/{img_name}'
        req.urlretrieve(image, img_path)
        # if '.svg' in image:
        #     img_name = img_name.replace('.svg', '.png')
        #     new_img_path = f'{image_base_path}/{img_name}'
        #     cairosvg.svg2png(url=img_path, write_to=new_img_path)
        #     img_path = new_img_path
        # thumbnail_path = f'{image_thumbnail_path}/{img_name}'
        # im = Image.open(img_path)
        # im.thumbnail((128, 128), Image.ANTIALIAS)
        # im.save(thumbnail_path)
        images_names.append(img_name)
        count_images += 1
    d.pop('images', None)
    d.pop('enlink', None)
    d.pop('delink', None)
    d.pop('frlink', None)
    d.pop('coord', None)
    d['images'] = images_names
    data_with_images.append(d)
    print(f"Finished item {i} with id {d['item'].rsplit('/', 1)[-1]}")
toc = time.perf_counter()


write_path = ROOT_PATH / 'dataset_files/pilot/capitals_images.json'
with open(write_path, 'w') as json_file:
    json.dump(data_with_images, json_file, ensure_ascii=False, indent=4)

print(f'Total images downloaded: {count_images}')
print(f'Total time to download images, resize into thumbnails and save them: {toc - tic:0.4f}')