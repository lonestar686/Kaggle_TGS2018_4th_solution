import argparse
from data_process.data_loader import *
from data_process.transform import *
from loss.bce_losses import *
from loss.cyclic_lr import *
from loss.lovasz_losses import *

from model.model import model34_DeepSupervion,\
                        model50A_DeepSupervion,\
                        model50A_slim_DeepSupervion,\
                        model101A_DeepSupervion,\
                        model101B_DeepSupervion,\
                        model152_DeepSupervion,\
                        model154_DeepSupervion

from utils import create_submission
from loss.metric import do_kaggle_metric
import time
import datetime

class SingleModelSolver(object):
    def __init__(self, config):
        self.model_name = config.model_name
        self.model = config.model

        self.dice_weight = config.dice_weight
        self.bce_weight = config.bce_weight

        # Model hyper-parameters
        self.image_size = config.image_size

        # Hyper-parameteres
        self.g_lr = config.lr
        self.cycle_num = config.cycle_num
        self.cycle_inter = config.cycle_inter
        self.dice_bce_pretrain_epochs = config.dice_bce_pretrain_epochs

        self.batch_size = config.batch_size
        self.pretrained_model = config.pretrained_model

        #pseudo label
        self.pseudo_csv = config.pseudo_csv
        self.pseudo_split = config.pseudo_split

        # Path
        running_directory = './my_runs'
        #
        self.log_path = os.path.join(running_directory, self.model_name, config.log_path)
        self.sample_path = os.path.join(running_directory, self.model_name, config.sample_path)
        self.model_save_path = os.path.join(running_directory, self.model_name, config.model_save_path)
        self.result_path = os.path.join(running_directory, self.model_name, config.result_path)

        # Create directories if not exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        self.load_pretrained_model(config.train_fold_index)

    def build_model(self):

        if self.model == 'model_34':
            self.G = model34_DeepSupervion()

        elif self.model == 'model_50A':
            self.G = model50A_DeepSupervion()

        elif self.model == 'model_50A_slim':
            self.G = model50A_slim_DeepSupervion()

        elif self.model == 'model_101A':
            self.G = model101A_DeepSupervion()

        elif self.model == 'model_101B':
            self.G = model101B_DeepSupervion()

        elif self.model == 'model_152':
            self.G = model152_DeepSupervion()

        elif self.model == 'model_154':
            self.G = model154_DeepSupervion()

        self.g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.G.parameters()),
                                           self.g_lr, weight_decay=0.0002, momentum=0.9)
        self.print_network(self.G, 'G')
        if torch.cuda.is_available():
            self.G = torch.nn.DataParallel(self.G)
            self.G.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self, fold_index, mode=None, Cycle=None):

        if mode is None:
            if os.path.exists(os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                            '{}_G.pth'.format(self.pretrained_model))):
                self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                                                '{}_G.pth'.format(self.pretrained_model))))
                print('loaded trained G models fold: {} (step: {})..!'.format(fold_index, self.pretrained_model))

        elif mode == 'max_map':
            if Cycle is None:
                pth = os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                    'Lsoftmax_maxMap_G.pth')
            else:
                pth = os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                    'Cycle_'+str(Cycle)+'_Lsoftmax_maxMap_G.pth')

            # print(pth)

            if os.path.exists(pth):
                self.G.load_state_dict(torch.load(pth))
                print('loaded trained G models fold: {} (step: {})..!'.format(fold_index, pth))

        elif mode == 'min_loss':
            if Cycle is None:
                pth = os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                    'Lsoftmax_minValidLoss_G.pth')
            else:
                pth = os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                    'Cycle_' + str(Cycle) + '_Lsoftmax_minValidLoss_G.pth')

            if os.path.exists(pth):
                self.G.load_state_dict(torch.load(pth))
                print('loaded trained G models fold: {} (step: {})..!'.format(fold_index, pth))

    def update_lr(self, g_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def criterion(self, logits, label):
        logits = logits.squeeze(1)
        label = label.squeeze(1)
        #
        loss = lovasz_hinge(logits, label, per_image=True, ignore=None)
        return loss

    def train_fold(self, fold_index, aug_list):

        CE = torch.nn.CrossEntropyLoss()

        if not os.path.exists(os.path.join(self.model_save_path, 'fold_'+str(fold_index))):
            os.makedirs(os.path.join(self.model_save_path, 'fold_'+str(fold_index)))

        print(f'train loader to load fold {fold_index}!!!')
        data_loader = get_foldloader(self.image_size,
                                     self.batch_size,
                                     fold_index, aug_list,
                                     mode='train',
                                     pseudo_csv=self.pseudo_csv,
                                     pseudo_index=self.pseudo_split)

        print(f'val loader to load fold {fold_index}!!!')
        val_loader = get_foldloader(self.image_size, 1, fold_index, mode='val')

        iters_per_epoch = len(data_loader)

        # warm up
        for init_index in range(self.dice_bce_pretrain_epochs):

            for i, (images, labels, is_empty) in enumerate(data_loader):

                inputs = self.to_var(images)
                labels = self.to_var(labels)
                class_lbls = self.to_var(torch.LongTensor(is_empty))

                # forward
                binary_logits, no_empty_logits, final_logits = self.G(inputs)

                # loss
                bce_loss_final = mixed_dice_bce_loss(final_logits, labels, dice_weight=self.dice_weight, bce_weight=self.bce_weight)
                class_loss = CE(binary_logits, class_lbls)

                non_empty = []
                for c in range(len(is_empty)):
                    if is_empty[c] == 0:
                        non_empty.append(c)

                has_empty_nonempty = False
                if len(non_empty) * len(is_empty) > 0:
                    has_empty_nonempty = True

                # total loss
                all_loss = bce_loss_final + 0.05 * class_loss

                loss = {}
                loss['loss_seg'] = bce_loss_final.item() #bce_loss_final.data[0]
                loss['loss_classifier'] = class_loss.item() #class_loss.data[0]

                if has_empty_nonempty:
                    indices = self.to_var(torch.LongTensor(non_empty))
                    y_non_empty = torch.index_select(no_empty_logits, 0, indices)
                    mask_non_empty = torch.index_select(labels, 0, indices)

                    # loss
                    loss_no_empty = mixed_dice_bce_loss(y_non_empty, mask_non_empty, dice_weight=self.dice_weight, bce_weight=self.bce_weight)

                    all_loss += 0.50 * loss_no_empty
                    loss['loss_seg_noempty'] = loss_no_empty.item() #loss_no_empty.data[0]

                self.g_optimizer.zero_grad()
                all_loss.backward()
                self.g_optimizer.step()

                # Print out log info
                if (i+1) % 10 == 0:
                    lr = self.g_optimizer.param_groups[0]['lr']
                    log = "{} FOLD: {}, BCE+DICE pretrain Epoch [{}/{}], lr {:.4f}".format(
                        self.model_name, fold_index, init_index, 10, lr)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


        # learning rate using cosine annealing and restart
        sgdr = CosineAnnealingLR_with_Restart(self.g_optimizer,
                                              T_max=self.cycle_inter,
                                              T_mult=1,
                                              model=self.G,
                                              out_dir='../input/',
                                              take_snapshot=False,
                                              eta_min=1e-3)

        # Start training
        start_time = time.time()
        for cycle_index in range(self.cycle_num):

            print('cycle index: '+ str(cycle_index))

            valid_loss_plot = []
            max_map_plot = []

            for e in range(0, self.cycle_inter):

                sgdr.step()

                lr = self.g_optimizer.param_groups[0]['lr']
                print('change learning rate into: {:.4f}'.format(lr))

                for i, (images, labels, is_empty) in enumerate(data_loader):

                    # all images
                    inputs = self.to_var(images)
                    labels = self.to_var(labels)
                    class_lbls = self.to_var(torch.LongTensor(is_empty))

                    # forward
                    binary_logits, no_empty_logits, final_logits = self.G(inputs)

                    # loss
                    loss_final = self.criterion(final_logits, labels)
                    class_loss = CE(binary_logits, class_lbls)

                    non_empty = []
                    for c in range(len(is_empty)):
                        if is_empty[c] == 0:
                            non_empty.append(c)

                    has_empty_nonempty = False
                    if len(non_empty) * len(is_empty) > 0:
                        has_empty_nonempty = True

                    # total loss
                    all_loss = loss_final +  0.05 * class_loss

                    loss = {}
                    loss['loss_seg'] = loss_final.item() #loss_final.data[0]
                    loss['loss_classifier'] = class_loss.item() #class_loss.data[0]

                    if has_empty_nonempty:
                        indices = self.to_var(torch.LongTensor(non_empty))
                        y_non_empty = torch.index_select(no_empty_logits, 0, indices)
                        mask_non_empty = torch.index_select(labels, 0, indices)

                        # loss
                        loss_no_empty = self.criterion(y_non_empty, mask_non_empty)

                        all_loss += 0.50 * loss_no_empty
                        loss['loss_seg_noempty'] = loss_no_empty.item() #loss_no_empty.data[0]

                    # backward propagation
                    self.g_optimizer.zero_grad()
                    all_loss.backward()
                    self.g_optimizer.step()

                    # Print out log info
                    if (i+1) % self.log_step == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        lr = self.g_optimizer.param_groups[0]['lr']

                        log = "{} FOLD: {}, Cycle: {}, Elapsed [{}], Epoch [{}/{}], Iter [{}/{}], lr {:.4f}".format(
                            self.model_name, fold_index, cycle_index, elapsed, e+1, self.cycle_inter, i+1, iters_per_epoch, lr)

                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)
                        print(log)

                if e + 1 >= 20 and (e+1) % 5 == 0:

                    # validation
                    valid_loss, max_map, max_thres = self.val_TTA(fold_index, val_loader, is_load=False)

                    if len(valid_loss_plot) == 0 or valid_loss < min(valid_loss_plot):
                        print('save min valid loss model')
                        torch.save(self.G.state_dict(), os.path.join(self.model_save_path,
                                                                     'fold_' + str(fold_index),
                                                                     'Cycle_'+str(cycle_index)+'_Lsoftmax_minValidLoss_G.pth'))

                        f = open(os.path.join(self.model_save_path, 'fold_' + str(fold_index), 'Cycle_'+str(cycle_index)+'_Lsoftmax_min_valid_loss.txt'), 'w')
                        f.write(str(max_map) + ' ' + str(max_thres) + ' ' + str(valid_loss) + ' epoch: ' + str(e))
                        f.close()

                    if len(valid_loss_plot) == 0 or max_map > max(max_map_plot):
                        print('save max map model')
                        torch.save(self.G.state_dict(), os.path.join(self.model_save_path,
                                                                     'fold_' + str(fold_index),
                                                                     'Cycle_'+str(cycle_index)+'_Lsoftmax_maxMap_G.pth'))

                        f = open(os.path.join(self.model_save_path, 'fold_' + str(fold_index), 'Cycle_'+str(cycle_index)+'_Lsoftmax_max_map.txt'), 'w')
                        f.write(str(max_map) + ' ' + str(max_thres) + ' ' + str(valid_loss) + ' epoch: ' + str(e))
                        f.close()

                    valid_loss_plot.append(valid_loss)
                    max_map_plot.append(max_map)

    def val_TTA(self, fold_index, val_loader, is_load=False, mode=None, Cycle=None):

        if fold_index < 0:
            return

        if is_load:
            self.load_pretrained_model(fold_index, mode=mode, Cycle=Cycle)

        self.G.eval()

        loss = 0
        t = 0

        output_list = []
        labels_list = []

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(val_loader):

                # horizontal flip
                img1 = images.numpy()
                img2 = img1[:, :, :, ::-1]
                batch_size = img1.shape[0]
                img_all = np.concatenate([img1, img2])

                images = torch.FloatTensor(img_all)

                inputs = self.to_var(images)

                # forward
                _,_, output = self.G(inputs)

                # post processing
                output = output.data.cpu().numpy()
                mask = output[0:batch_size*2]
                output = mask[0:batch_size] + mask[batch_size:batch_size*2][:, :, :, ::-1]

                output = output / 2.0
                labels = labels.numpy()

                output = output.transpose(2, 3, 0, 1).reshape([self.image_size, self.image_size, -1])
                labels = labels.transpose(2, 3, 0, 1).reshape([self.image_size, self.image_size, -1])

                if self.image_size == 128:
                    output = center_crop(output, self.image_size, crop_size=101)
                    labels = center_crop(labels, self.image_size, crop_size=101)
                elif self.image_size == 160:
                    output = center_crop(output, self.image_size, crop_size=101)
                    labels = center_crop(labels, self.image_size, crop_size=101)
                elif self.image_size == 256:
                    output = center_crop(output, self.image_size, crop_size=202)
                    labels = center_crop(labels, self.image_size, crop_size=202)

                output = cv2.resize(output, (101, 101)).reshape([101, 101, -1])
                labels = cv2.resize(labels, (101, 101)).reshape([101, 101, -1])

                # save the results and ground truths (H, W, C)
                output_list.append(output)
                labels_list.append(labels)

                # calculate loss, BHWC->BCHW->B1HW
                output = output.transpose(2, 0, 1).reshape([batch_size, 1, 101, 101])
                labels = labels.transpose(2, 0, 1).reshape([batch_size, 1, 101, 101])

                output = torch.FloatTensor(output)
                labels = torch.FloatTensor(labels)

                labels = self.to_var(labels)
                output = self.to_var(output)

                # loss
                bce_loss = self.criterion(output, labels)
                loss += bce_loss.item() #bce_loss.data[0]

                t += 1.0

        valid_loss = loss / t

        # (H, W, C)
        output = np.concatenate(output_list, axis=2)
        labels = np.concatenate(labels_list, axis=2)

        # HWC --> CHW
        output = output.transpose(2, 0, 1)
        labels = labels.transpose(2, 0, 1)

        # brute-force way
        threshold = np.arange(-0.5, 0.5, 0.0125)
        def get_max_map(output_, labels_):
            precision_list = []
            for thres in threshold:
                precision, _, _ = do_kaggle_metric(output_, labels_, threshold=thres)
                precision = precision.mean()
                precision_list.append(precision)

            max_map = max(precision_list)
            max_index = np.argmax(np.asarray(precision_list))

            return max_map, max_index

        max_map, max_index = get_max_map(output, labels)
        print("max map: {:.4f} at thres:{:.4f}".format(max_map, threshold[max_index]))

        log = "{} FOLD: {} valid loss: {:.4f}".format(self.model_name, fold_index, valid_loss)
        print(log)

        self.G.train()

        return valid_loss, max_map, threshold[max_index]

    def predict_batch(self, model, images, thres):
        """ predict a batch of images """

        model.eval()

        with torch.no_grad():
            # TTA
            img1 = images.numpy()
            img2 = img1[:, :, :, ::-1]    # BCHW, horizonal flip
            batch_size = img1.shape[0]
            img_all = np.concatenate([img1, img2])

            images = torch.FloatTensor(img_all)

            inputs = self.to_var(images)

            # forward
            _, _, output = model(inputs)

            # average two values
            output = output.data.cpu().numpy()
            mask = output[0:batch_size * 2]
            output = mask[0:batch_size] + mask[batch_size:batch_size * 2][:, :, :, ::-1]
            output = output / 2.0

            # HWBC, C=1
            output = output.transpose(2, 3, 0, 1).reshape([self.image_size, self.image_size, batch_size])

            if self.image_size == 128:
                output = center_crop(output, self.image_size, crop_size=101)

            output = cv2.resize(output, (101, 101), cv2.INTER_CUBIC)
            output[output >= thres] = 1.0
            output[output < thres] = 0.0

            output = output.transpose(2, 0, 1)   # HWB --> BHW
            output = output.reshape([batch_size, 101, 101]).astype(np.uint8)

        return output

    def get_infer_TTA(self, fold_index, test_loader, thres):

        # self.G.eval()

        out = []
        for i, (img_ids, images) in enumerate(test_loader):

            # predict a batch of images
            output = self.predict_batch(self.G, images, thres)

            batch_size = len(output)
            for id_index in range(batch_size):
                out.append([img_ids[id_index], output[id_index].reshape([101, 101])])

            if i%1000 == 0 and i > 0:
                print(self.model_name + ' fold index: '+str(fold_index) +' '+str(i))

        return out

    def infer_fold_TTA(self, fold_index, mode='max_map', Cycle=None):
        """ Test Time Augmentation:
            take 4 data augmentations at random as well as the un-augmented original (center-cropped). We will
            then calculate predictions for all these images, take the average, and make that our final prediction.
            Note that this is only for validation set and/or test set.
        """
        print(mode)

        # for validation
        val_loader = get_foldloader(self.image_size, self.batch_size//2, fold_index, mode='val')
        # validation
        _, max_map, thres = self.val_TTA(fold_index, val_loader, is_load=True, mode=mode, Cycle=Cycle)

        if fold_index < 0:
            return

        # load test dataset
        test_loader = get_foldloader(self.image_size, self.batch_size//2, 0, mode='test')

        # prediction
        infer = self.get_infer_TTA(fold_index, test_loader, thres)

        if Cycle is None:
            name_tmp = 'fold_{}_TTA_{}{:.3f}at{:.3f}.csv'.format(fold_index, mode, max_map, thres)
        else:
            name_tmp = 'fold_{}_Cycle_{}_TTA_{}{:.3f}at{:.3f}.csv'.format(fold_index, Cycle, mode, max_map, thres)

        if not os.path.exists(os.path.join(self.result_path, 'fold_' + str(fold_index))):
            os.makedirs(os.path.join(self.result_path, 'fold_' + str(fold_index)))

        output_name = os.path.join(self.result_path, 'fold_' + str(fold_index), name_tmp)
        submission = create_submission(infer)
        submission.to_csv(output_name, index=None)

    def infer_fold_all_Cycle(self, fold_index, mode='max_map'):

        for i in range(self.cycle_num):
            self.infer_fold_TTA(fold_index, mode, Cycle=i)

def main(config, aug_list):
    # cudnn.benchmark = True

    # folds
    all_folds = None

    fold_index = config.train_fold_index
    if fold_index < 0:
        all_folds = range(10)
    else:
        all_folds = [fold_index]

    # solver
    solver = SingleModelSolver(config)
    if config.mode == 'train':
        # train
        for fold_index in all_folds:
            solver.train_fold(fold_index, aug_list)

    if config.mode == 'test':
        # test
        for fold_index in all_folds:
            solver.infer_fold_all_Cycle(fold_index)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default=0)
    parser.add_argument('--model', type=str, default='model_34')
    parser.add_argument('--model_name', type=str, default='model_34')
    # parser.add_argument('--model_name', type=str, default='model_34_pseudo_0')
    parser.add_argument('--image_size', type=int, default=128)
    #parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)

    aug_list = ['flip_lr']
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    # pseudo label
    # parser.add_argument('--pseudo_csv', type=str, default=r'0.8935_for_pseudo_label.csv')
    parser.add_argument('--pseudo_csv', type=str, default=None)
    parser.add_argument('--pseudo_split', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cycle_num', type=int, default=7)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--dice_bce_pretrain_epochs', type=int, default=10)
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--bce_weight', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=16)

    # Test settings
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--result_path', type=str, default='results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=20000)

    config = parser.parse_args()
    print(config)
    main(config, aug_list)
