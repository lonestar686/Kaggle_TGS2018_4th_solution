import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_process.transform import *
import random
import pandas as pd

import cv2

def read_txt(txt):
    f = open(txt, 'r')
    lines = f.readlines()
    f.close()
    return [tmp.strip() for tmp in lines]

class SaltDataset(Dataset):
    def __init__(self, data_dir, transform, mode, image_size, fold_index, aug_list, pseudo_csv=None, pseudo_index=-1):

        # train/test
        self.train_image_path = os.path.join(data_dir, r'train/images')
        self.train_mask_path = os.path.join(data_dir, r'train/masks')
        self.test_image_path = os.path.join(data_dir, r'test/images')

        #
        self.transform = transform
        self.mode = mode
        self.image_size = image_size
        self.aug_list = aug_list

        if pseudo_csv is None:
            self.is_pseudo = False
            self.pseudo_index = -1
        else:
            self.is_pseudo = True
            self.pseudo_mask_path = r'./data_process/pseudo_mask_path'
            csv_name = os.path.split(pseudo_csv)[1].replace('.csv', '')
            self.pseudo_mask_path = os.path.join(self.pseudo_mask_path, csv_name)
            print(self.pseudo_mask_path)

            if not os.path.exists(self.pseudo_mask_path):
                os.makedirs(self.pseudo_mask_path)

            print('save the csv images to disk')
            save_csv_images(pseudo_csv, self.pseudo_mask_path)
            self.pseudo_index = pseudo_index

        print('AugList: ')
        print(self.aug_list)

        self.fold_index = None
        self.set_mode(mode, fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print('fold index set: ' + str(fold_index))

        if self.mode == 'train':

            data = pd.read_csv(r'./data_process/10fold/fold' + str(fold_index) + '_train.csv')

            self.train_list = data['fold']
            self.train_list = [tmp + '.png' for tmp in self.train_list]
            self.num_data = len(self.train_list)

            if self.is_pseudo:
                print('pseudo labeling part: ' + str(self.pseudo_index))
                self.pseudo_list = read_txt(r'./data_process/pseudo_split/pseudo_split'+str(self.pseudo_index)+'.txt')
                print(len(self.pseudo_list))

        elif self.mode == 'val':

            data = pd.read_csv(r'./data_process/10fold/fold' + str(fold_index) + '_valid.csv')

            self.val_list = data['fold']
            self.val_list = [tmp + '.png' for tmp in self.val_list]
            self.num_data = len(self.val_list)

        elif self.mode == 'test':

            self.test_list = read_txt(r'./data_process/10fold/test.txt')
            self.num_data = len(self.test_list)

        print('set dataset mode: ' + self.mode)

    def __getitem__(self, index):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        # testing
        if self.mode == 'test':
            image = cv2.imread(os.path.join(self.test_image_path, self.test_list[index]), 1)
            image_id = self.test_list[index].replace('.png', '')

            if self.image_size == 128:
                image = resize_and_pad(image, resize_size=101, factor=64)

            image = image.reshape([self.image_size, self.image_size, 3])
            image = np.transpose(image, (2, 0, 1))
            # image = image.astype(np.float32)
            # image = image.reshape([3, self.image_size, self.image_size])
            image = (image.astype(np.float32) - 127.5) / 127.5

            return image_id, torch.FloatTensor(image)

        # training/validation
        if self.mode == 'train':

            switch = 0
            # random select pseudo label images
            if self.is_pseudo:
                switch = random.randint(0, 1)

            if switch == 0:
                # Second argument is a flag which specifies the way image should be read.
                # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
                # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
                # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
                # Note Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

                # original data
                image = cv2.imread(os.path.join(self.train_image_path, self.train_list[index]), 1)
                label = cv2.imread(os.path.join(self.train_mask_path, self.train_list[index]), 0)
            else:
                # data distillation
                # reset index for pseudo label
                index = random.randint(0, len(self.pseudo_list) - 1)

                # grab a test image and its pseudo label
                image = cv2.imread(os.path.join(self.test_image_path, self.pseudo_list[index]), 1)
                label = cv2.imread(os.path.join(self.pseudo_mask_path, self.pseudo_list[index]), 0)

        if self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index]), 1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]), 0)

        # empty salt
        is_empty = False
        if np.sum(label) == 0:
            is_empty = True

        # handling image and its label
        if self.mode == 'train':
            # train
            image, label = resize_and_random_pad(image, label, resize_size=101, factor=128, limit=(-13, 13))
        else:
            # validation
            image = resize_and_pad(image, resize_size=101, factor=128)
            label = resize_and_pad(label, resize_size=101, factor=128)

        # resize to the given image size
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size))

        if self.mode == 'train':
            if 'flip_lr' in self.aug_list:
                if random.randint(0, 1) == 0:
                    image = cv2.flip(image, 1)
                    label = cv2.flip(label, 1)

        # normalization of image and its label
        # image
        image = image.reshape([self.image_size, self.image_size, 3])
        image = np.transpose(image, (2, 0, 1))
        image = (image.astype(np.float32) - 127.5) / 127.5

        # ground truth
        label = label.reshape([1, self.image_size, self.image_size])
        label = np.asarray(label).astype(np.float32) / 255.0
        # threshold
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0

        return torch.FloatTensor(image), torch.FloatTensor(label), is_empty

    def __len__(self):
        return self.num_data

def get_foldloader(image_size, batch_size, fold_index, aug_list=None, mode='train', pseudo_csv=None, pseudo_index=-1):
    """Build and return data loader."""

    # change to your path
    # on h050018
    raw_data_dir = r'/wgdisk/st0008/hzh/workspace/tgs/input'
    # my laptop
    # raw_data_dir = r'/home/hzh/MachineLearning/equinor/tgs/input'

    # pytorch dataset
    dataset = SaltDataset(raw_data_dir, None, mode, image_size, fold_index, aug_list, pseudo_csv=pseudo_csv, pseudo_index=pseudo_index)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle)
    return data_loader