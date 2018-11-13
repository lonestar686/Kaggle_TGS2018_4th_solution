from utils import *

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
    parser.add_argument('--train_csv_path', default = r'/wgdisk/st0008/hzh/workspace/tgs/input/train.csv', help='train.csv path')
    parser.add_argument('--train_mask_save_dir', default = r'/wgdisk/st0008/hzh/workspace/tgs/input/train_mask_try', help='train mask save_dir')
    args = parser.parse_args()
    save_train_mask(args.train_csv_path, args.train_mask_save_dir)

