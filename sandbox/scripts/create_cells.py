import csv
import os
import logging
import sys
import argparse
from time import time
from functools import partial
from multiprocessing import Pool
from collections import Counter
import json
import pandas as pd
import s2sphere as s2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Create Cell Partitioning')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset csv file')
    parser.add_argument('-ci', '--column_img_path', type=str, default='IMG_ID', help='column name image id / path')
    parser.add_argument('-clat', '--column_lat', type=str, default='LAT', help='column name latitude')
    parser.add_argument('-clng', '--column_lng', type=str, default='LON', help='column name longitude')

    parser.add_argument('--output', type=str, required=False, help='Path to output directory')

    parser.add_argument('--img_min', type=int, required=True, help='Minimum number of images per geographical cell')
    parser.add_argument('--img_max', type=int, required=True, help='Maximum number of images per geographical cell')

    parser.add_argument('--lvl_min', type=int, required=False, default=2, help='Minimum partitioning level')
    parser.add_argument('--lvl_max', type=int, required=False, default=30, help='Maximum partitioning level')

    args = parser.parse_args()
    return args


def _init_parallel(img, level):
    cell = create_s2_cell(img[1], img[2])
    hexid = create_cell_at_level(cell, level)
    return [*img, hexid, cell]


def init_cells(img_container_0, level):

    start = time()
    f = partial(_init_parallel, level=level)
    img_container = []
    with Pool(8) as p:
        for x in p.imap_unordered(f, img_container_0, chunksize=1000):
            img_container.append(x)
    logging.debug(f'Time multiprocessing: {time() - start:.2f}s')
    start = time()
    h = dict(Counter(list(list(zip(*img_container))[3])))
    logging.debug(f'Time creating h: {time() - start:.2f}s')

    return img_container, h


def delete_cells(img_container, h, t_min):
    del_cells = {k for k, v in h.items() if v <= t_min}
    h = {k: v for k, v in h.items() if v > t_min}
    img_container_f = []
    for img in img_container:
        hexid = img[3]
        if hexid not in del_cells:
            img_container_f.append(img)
    return img_container_f, h


def gen_subcells(img_container_0, h_0, level, t_max):
    img_container = []
    h = {}
    for img in img_container_0:
        hexid_0 = img[3]
        if h_0[hexid_0] > t_max:
            hexid = create_cell_at_level(img[4], level)
            img[3] = hexid
            try:
                h[hexid] = h[hexid] + 1
            except:
                h[hexid] = 1
        else:
            try:
                h[hexid_0] = h[hexid_0] + 1
            except:
                h[hexid_0] = 1
        img_container.append(img)
    return img_container, h


def create_s2_cell(lat, lng):
    p1 = s2.LatLng.from_degrees(lat, lng)
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def create_cell_at_level(cell, level):
    cell_parent = cell.id().parent(level)
    hexid = cell_parent.to_token()
    return hexid


# def write_output(args, img_container, h, num_images, out_p):
def write_output(args_img_min, args_img_max, img_container, h, num_images, out_p):

    if not os.path.exists(out_p):
        os.makedirs(out_p)

    fname = f"cells_{args_img_min}_{args_img_max}_images_{num_images}.csv"
    logging.info(f'Write to {os.path.join(out_p, fname)}')
    with open(os.path.join(out_p, fname), 'w', newline='') as f:
        cells_writer = csv.writer(f, delimiter=',')
        # write column names
        cells_writer.writerow(['class_label', 'hex_id', 'imgs_per_cell', 'latitude_mean', 'longitude_mean'])

        # write dict
        i = 0
        cell2class = {}
        coords_sum = {}

        # generate class ids for each hex cell id
        for k in h.keys():
            cell2class[k] = i
            coords_sum[k] = [0, 0]
            i = i + 1

        # calculate mean GPS coordinate in each cell
        for img in img_container:
            coords_sum[img[3]][0] = coords_sum[img[3]][0] + img[1]
            coords_sum[img[3]][1] = coords_sum[img[3]][1] + img[2]

        # write partitioning information
        for k, v in h.items():
            cells_writer.writerow([cell2class[k], k, v, coords_sum[k][0] / v, coords_sum[k][1] / v])
    return fname


def my_write_output(classes):

    with open("main_output", 'w') as outfile:
        json.dump(classes, outfile)



def my_func(img_container, fname):
    df = pd.read_csv('outputs/'+fname, usecols=['class_label', 'hex_id', 'imgs_per_cell','latitude_mean','longitude_mean'])
    cls_hexes = list(df.itertuples(index=False, name=None))
    n_clses = len(cls_hexes)
    img_classes = []

    for ic in img_container:
        for cls_hex in cls_hexes:
           cls = cls_hex[0]
           hexind = cls_hex[1]

           if ic[3] == hexind:
              img_classes.append(cls)

    main_output = []
    for img_class, ic in zip(img_classes, img_container):
        dic = {}
        dic['image_id'] = ic[0]
        dic['class'] = img_class
        one_hot = np.zeros([n_clses])

        for cls in range(n_clses):
            if img_class==cls:
                one_hot[cls] = 1
                break

        dic['one_hot'] = list(one_hot)
        main_output.append(dic)
    return main_output


def create_dataset(wiki_dataset):
    with open(wiki_dataset, encoding='utf-8') as json_file:
        data = json.load(json_file)
    num = len(data)

    with open('dataset.csv', 'w', newline='') as f:
        cells_writer = csv.writer(f, delimiter=',')
        # write column names
        cells_writer.writerow(['IMG_ID', 'LAT', 'LON'])

        for d, i in zip(data, range(num)):
            coord = d['coord'][6:-1].split(' ')
            lat = coord[0]
            lng = coord[1]
            img_id = d['item'].split('/')[-1]
            cells_writer.writerow([img_id, lat, lng])


def main():
    # load arguments
    # args = parse_args()
    
    args_verbose = 'store_true'
    args_dataset = 'dataset.csv'
    args_column_img_path='IMG_ID'
    args_column_lat = 'LAT'
    args_column_lng = 'LON'
    args_lvl_min = 2
    args_lvl_max = 10
    args_output = 'outputs'
    args_img_min = 0
    args_img_max = 1000
    args_dataset_name = 'pilot_wikidata'

    create_dataset(args_dataset_name)


    level = logging.INFO
    if args_verbose:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    # read dataset
    df = pd.read_csv(args_dataset, usecols=[args_column_img_path, args_column_lat, args_column_lng])
    img_container = list(df.itertuples(index=False, name=None))
    num_images = len(img_container)
    logging.info('{} images available.'.format(num_images))
    level = args_lvl_min

    # create output directory
    if not os.path.exists(args_output):
        os.makedirs(args_output)

    # initialize
    logging.info('Initialize cells of level {} ...'.format(level))
    start = time()
    img_container, h = init_cells(img_container, level)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Remove cells with |img| < t_min ...')
    start = time()
    img_container, h = delete_cells(img_container, h, args_img_min)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Create subcells ...')
    while any(v > args_img_max for v in h.values()) and level < args_lvl_max:
        level = level + 1
        logging.info('Level {}'.format(level))
        start = time()
        img_container, h = gen_subcells(img_container, h, level, args_img_max)
        logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Remove cells with |img| < t_min ...')
    start = time()
    img_container, h = delete_cells(img_container, h, args_img_min)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')
    logging.info(f'Number of images: {len(img_container)}')

    logging.info('Write output file ...')
    fname = write_output(args_img_min,args_img_max, img_container, h, num_images, args_output)

    outputs = my_func(img_container, fname)
    my_write_output(outputs)




if __name__ == '__main__':
    sys.exit(main())