from __future__ import print_function
import torch.utils.data as data
import os
import sys
import numpy as np
import h5py

class MLMLoader(data.Dataset):
    def __init__(self, data_path, partition, mismatch=0.5):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None or partition not in ['train', 'val', 'test']:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.h5f = h5py.File(os.path.join(data_path, f'{partition}.h5'), 'r')

        self.ids = self.h5f['ids']
        self.mismatch = mismatch

    def __getitem__(self, index):
        instanceId = self.ids[index]
        # we force 50 percent of them to be a mismatch
        match = np.random.uniform() > self.mismatch if self.partition == 'train' else True

        target = match and 1 or -1

        if target == 1:  # load Pos example

            coord = self.h5f[f'{instanceId}_onehot'][()]
            all_img = self.h5f[f'{instanceId}_images'][()]
            if self.partition == 'train':
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                img = all_img[0]

            summaries = self.h5f[f'{instanceId}_summaries'][()]
            multi_wiki = summaries[np.random.choice(range(np.shape(summaries)[0]))]

        else:
            # load Neg
            all_idx = range(len(self.ids))
            coord_t = self.h5f[f'{instanceId}_onehot'][()]
            coord = self.h5f[f'{instanceId}_onehot'][()]
            # we have to be sure that we get wrong coordinates and not just random id
            while np.array_equal(coord_t, coord):
                rndCoordIndex = np.random.choice(all_idx)  # pick a random index for coordinates
                rndId = self.ids[rndCoordIndex]
                coord = self.h5f[f'{rndId}_onehot'][()]

            # neg sample of image and summary find ids that have the same coordinate class with instanceId

            its_ids = self.h5f[f'{instanceId}_cluster']
            rndCoordIndex = np.random.choice(range(len(its_ids)))
            rndId = its_ids[rndCoordIndex]

            all_img = self.h5f[f'{rndId}_images'][()]
            if self.partition == 'train':
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                img = all_img[0]
            summaries = self.h5f[f'{rndId}_summaries'][()]
            multi_wiki = summaries[np.random.choice(range(np.shape(summaries)[0]))]



        # output
        output = {
            'image': img,
            'multi_wiki': multi_wiki,
            'coord': coord,
            'target': target
        }

        if self.partition != 'train':
            output['id'] = np.argmax(coord)

        return output

    def __len__(self):
        return len(self.ids)
