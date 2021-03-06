""" prepare data """
import os
import argparse
import cv2
from utils import decode_csv

def save_train_mask(train_csv, save_dir):
    print(train_csv)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dict = decode_csv(train_csv)
    for item in dict:
        image = dict[item]*255
        cv2.imwrite(os.path.join(save_dir, item+'.png'), image)
    print('done')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='data prepare')
    # on h050018
    data_dir = r'/wgdisk/st0008/hzh/workspace/tgs/input'
    # on my laptop
    # data_dir = r'/home/hzh/MachineLearning/equinor/tgs/input'
    #
    parser.add_argument('--train_csv_path', default=os.path.join(data_dir, r'train.csv'), help='train.csv path')
    parser.add_argument('--train_mask_save_dir', default=os.path.join(data_dir, r'train_mask_try'), help='train mask save_dir')
    args = parser.parse_args()

    save_train_mask(args.train_csv_path, args.train_mask_save_dir)

