from __future__ import print_function
import torch.utils.data as data
import os
import sys
import numpy as np
import h5py
import torch

class MLMLoader(data.Dataset):
    def __init__(self, data_path, partition, mismatch=0.8):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None or partition not in ['train', 'val', 'test']:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.h5f = h5py.File(os.path.join(data_path, f'{partition}_pilot.h5'), 'r')

        self.ids = self.h5f['ids']
        self.mismatch = mismatch

    def __getitem__(self, index):
        instanceId = self.ids[index]
        # we force 80 percent of them to be a mismatch
        match = np.random.uniform() > self.mismatch if self.partition == 'train' else True

        target = match and 1 or -1

        if target == 1:
            # load positive example
            coord = self.h5f[f'{instanceId}_coords'][()]
            all_img = self.h5f[f'{instanceId}_images']
            if self.partition == 'train':
                # select randomly one of the images
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                # For val and test we always take the first image
                img = all_img[0]
        else:
            # For negative examples we sample random coordinates and image
            # The wiki text will always stay correct
            # load negative examples - select random index different than current
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            # load negative examples
            rndId = self.ids[rndindex]
            coord = self.h5f[f'{rndId}_coords'][()]
            all_img = self.h5f[f'{rndId}_images']

            if self.partition == 'train':
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                img = all_img[0]

        # load wiki texts
        en_wiki = self.h5f[f'{instanceId}_enwiki'][()]
        fr_wiki = self.h5f[f'{instanceId}_frwiki'][()]
        de_wiki = self.h5f[f'{instanceId}_dewiki'][()]

        # output
        output = {
            'image': img,
            'coord': coord,
            'en_wiki': en_wiki,
            'fr_wiki': fr_wiki,
            'de_wiki': de_wiki,
            'triple': np.zeros((2048), dtype='float32'), # later we replace
            'target': target
        }

        if self.partition != 'train':
            output['id'] = instanceId

        return output

    def __len__(self):
        return len(self.ids)
