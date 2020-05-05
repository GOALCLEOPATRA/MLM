from __future__ import print_function
import torch.utils.data as data
import os
import sys
import numpy as np
import h5py
import torch
# instanceId = 1002124
# old = h5py.File('C:/Users/TahmasebzadehG/PycharmProjects/MLM_2/data/h5_old/train.h5', 'r')
# ins_old = old[f'{instanceId}_images']
# new = h5py.File('C:/Users/TahmasebzadehG/PycharmProjects/MLM_2/data/train.h5', 'r')
# ins_new = new[f'{instanceId}_images']
# old_old = h5py.File('C:/Users/TahmasebzadehG/PycharmProjects/MLM_3/dataset/loc/train.h5', 'r')
# old_old_ids = old_old['ids']
# # for o in old_old_ids:
# #     print(o)
# print(ins_new, ins_old, old_old[f'{1082885}_images'])
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

            ids_of_this_cell = []
            ids_of_this_cell.append(instanceId)
            for ii in self.ids:
                if self.h5f[f'{ii}_onehot'] == self.h5f[f'{instanceId}_onehot']:
                    ids_of_this_cell.append(ii)

            all_idx = range(len(ids_of_this_cell))
           # print(len(all_idx))



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
            # 'coord': coord,
            'target': target
        }

        if self.partition != 'train':
            output['id'] = np.argmax(coord)

        return output

    def __len__(self):
        return len(self.ids)
