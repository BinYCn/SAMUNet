import os
import sys
from tqdm import tqdm
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from SAM_UNet import SAM_UNet
from datasets.dataset_synapse import Synapse_dataset


class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                 7: 'aorta', 8: 'pancreas'}


def load_parameters(model, filename):
    assert filename.endswith(".pt") or filename.endswith('.pth')

    state_dict = torch.load(filename)
    model_weight = model.state_dict()

    for key in model_weight.keys():
        if key in state_dict.keys():
            model_weight[key] = state_dict[key]

    model.load_state_dict(model_weight)
    return model


def inference(args, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        try:
            logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (
            i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
        except:
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='data/Synapse/test_vol_h5/')
    parser.add_argument('--output', type=str, default='output_Synapse/SAM-UNet')
    parser.add_argument('--split', type=str, default='train18')  # 'train18, train9, train6'
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=512, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.005, help='segmentation network learning rate')
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--pretrain', action='store_true', help='Whether to use pretrained_weight')
    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }

    net = SAM_UNet(args.num_classes).cuda()

    if not args.pretrain:
        args.is_pretrain = False
        args.exp = dataset_name + '_' + str(args.img_size)
        snapshot_path = os.path.join(args.output, "{}".format(args.exp))
        snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
        snapshot_path += '_' + args.vit_name
        snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
        snapshot_path = snapshot_path + '_sample' + str(args.split)[5:]
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    else:
        snapshot_path = args.output + '/pretrained_' + args.split.split('train')[-1]
        os.makedirs(snapshot_path, exist_ok=True)

    if not args.pretrain:
        save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
    else:
        save_mode_path = os.path.join('pretrained_weight', args.dataset, args.split.split('train')[-1], 'best_model.pth')
        print('a')
    net = load_parameters(net, save_mode_path)

    # initialize log
    log_folder = os.path.join(snapshot_path, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(snapshot_path, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, dataset_config[dataset_name], net, test_save_path)
