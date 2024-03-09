import argparse
import os
import shutil
import SimpleITK as sitk
from einops import repeat
from medpy import metric
import h5py
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
from SAM_UNet import SAM_UNet

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path',
    type=str,
    default='data/ACDC',
    help='root dir for data')
parser.add_argument('--output', type=str, default='output_ACDC/SAM-UNet')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_ACDC', help='list dir')
parser.add_argument('--train_num', type=int,
                    default=70, help='output channel of network')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--warmup', type=bool, default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', type=bool, default=True, help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--input_size', type=int, default=512, help='The input size for training SAM model')
parser.add_argument('--dice_param', type=float, default=0.5)
parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
parser.add_argument('--pretrain', action='store_true', help='Whether to use pretrained_weight')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def load_parameters(model, filename):
    assert filename.endswith(".pt") or filename.endswith('.pth')
    state_dict = torch.load(filename)
    model_weight = model.state_dict()

    for key in model_weight.keys():
        if key in state_dict.keys():
            model_weight[key] = state_dict[key]
    model.load_state_dict(model_weight)
    return model


def test_single_volume(case, net, FLAGS, test_save_path=None):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (512 / x, 512 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input = repeat(input, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 512, y / 512), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 10))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing((1, 1, 10))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.SetSpacing((1, 1, 10))
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS, snapshot_path, test_save_path):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    net = SAM_UNet(FLAGS.num_classes).cuda()

    if not FLAGS.pretrain:
        save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
    else:
        save_mode_path = os.path.join('pretrained_weight', FLAGS.dataset, str(FLAGS.train_num), 'best_model.pth')
    net = load_parameters(net, save_mode_path)
    print("init weight from {}".format(save_mode_path))
    net.eval()
    dice_list = []
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, FLAGS, test_save_path)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        avg_dice = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
        dice_list.append(avg_dice)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    with open(snapshot_path + '/performance.txt', 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))

    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    if not FLAGS.pretrain:
        FLAGS.is_pretrain = False
        FLAGS.exp = 'ACDC' + '_' + str(FLAGS.img_size)
        snapshot_path = os.path.join(FLAGS.output, "{}".format(FLAGS.exp))
        snapshot_path = snapshot_path + '_pretrain' if FLAGS.is_pretrain else snapshot_path
        snapshot_path += '_' + FLAGS.vit_name
        snapshot_path = snapshot_path + '_epo' + str(FLAGS.max_epochs) if FLAGS.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path + '_bs' + str(FLAGS.batch_size)
        snapshot_path = snapshot_path + '_sample' + str(FLAGS.train_num)
        snapshot_path = snapshot_path + '_lr' + str(FLAGS.base_lr) if FLAGS.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_s' + str(FLAGS.seed) if FLAGS.seed != 1234 else snapshot_path
    else:
        snapshot_path = FLAGS.output + '/pretrained_' + str(FLAGS.train_num)
        os.makedirs(snapshot_path, exist_ok=True)

    if FLAGS.is_savenii:
        test_save_path = snapshot_path + '/predictions/'
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    metric = Inference(FLAGS, snapshot_path, test_save_path)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(snapshot_path + '/performance.txt', 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format((metric[0]+metric[1]+metric[2])/3))
