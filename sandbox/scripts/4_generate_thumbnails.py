import os
import sys
from pathlib import Path
import time
import glob
from PIL import Image, ImageOps
import cairosvg
from urllib import parse
import numpy as np
# import cv2

ROOT_PATH = Path(os.path.dirname(__file__))
images_path = ROOT_PATH / 'dataset_files/pilot/images/'
write_path = ROOT_PATH / 'dataset_files/pilot/thumbnails/'

# get all the files from the folder
# for infile in glob.glob(f'{images_path}/*'):
#     new_name = parse.unquote(infile)
#     os.rename(infile, new_name)

# for infile in glob.glob(f'{images_path}/*.svg'):
#     try:
#         image_name = infile.rsplit('/', 1)[-1].replace('.svg', '.png')
#         cairosvg.svg2png(url=infile, write_to=f'{images_path}/{image_name}')
#     except:
#         print(f'Failed converting {infile} to PNG.')

for infile in glob.glob(f'{images_path}/*'):
    if '.svg' in infile:
        continue
    im = Image.open(infile).convert('RGB')
    # convert to thumbnail image
    fit_and_resized_image = ImageOps.fit(im, (256, 256), Image.ANTIALIAS)
    # im.thumbnail((128, 128), Image.ANTIALIAS)
    if np.array(fit_and_resized_image).shape == (256, 256, 3):
        image_name = infile.rsplit('/', 1)[-1]
        fit_and_resized_image.save(f'{write_path}/{image_name}')
        print(f'Finished image {image_name}')

print(len(glob.glob(f'{write_path}/*')))