from PIL import Image
from dataset_share import get_loader
import torch
from torchvision import transforms
from util import Logger, save_tensor_img
from tqdm import tqdm
from torch import nn
import os
import argparse
from models.MineNet import MineNet
import time

def main(args):
    dataset_names = args.testset.split('+')
    for idataset in dataset_names:
        if idataset == 'CoCA':
            test_img_path = './data/test/CoCAtest1/img/'
            test_gt_path = './data/test/CoCAtest1/gt/'
            saved_root = os.path.join(args.save_root, 'CoCAtest1')
        elif idataset == 'CoSOD3k':
            test_img_path = './data/test/CoSOD3ktest1/img/'
            test_gt_path = './data/test/CoSOD3ktest1/gt/'
            saved_root = os.path.join(args.save_root, 'CoSOD3ktest1')
        elif idataset == 'CoSal2015':
            test_img_path = './data/test/CoSal2015test1/img/'
            test_gt_path = './data/test/CoSal2015test1/gt/'
            saved_root = os.path.join(args.save_root, 'CoSal2015test1')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(test_img_path,
                                 test_gt_path,
                                 test_gt_path,
                                 args.size,
                                 1,
                                 istrain=False,
                                 shuffle=False,
                                 num_workers=4,
                                 pin=True)

        # Init model
        device = torch.device("cuda")
        model = eval('MineNet()')
        model = model.to(device)

        minenet_dict = torch.load(os.path.join(args.param_root, 'checkpoint.pth'))

        model.coatten.load_state_dict(minenet_dict, strict = False)
        model.iife.load_state_dict(minenet_dict, strict = False)

        model.eval()
        model.set_mode('test')

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device)
            subpaths = batch[1]
            ori_sizes = batch[2]
            
            scaled_preds, edges = model(inputs)

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]),
                        exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]

                tmpfilename, tmpfileExt = os.path.splitext(subpath)
                
                ori_size = (ori_sizes[inum][0].item(),
                            ori_sizes[inum][1].item())

                res = nn.functional.interpolate(scaled_preds[inum][-1],
                                                size=ori_size,
                                                mode='bilinear',
                                                align_corners=True)
                
                save_tensor_img(res, os.path.join(saved_root, subpath))

if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='MineNet', type=str)
    parser.add_argument(
        '--testset',
        default='CoCA',
        type=str,
        help="Options: 'CoCA','CoSal2015','CoSOD3k'")
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_root',
                        default='./tmp/MineNet/',
                        type=str,
                        help='model folder')
    parser.add_argument('--save_root',
                        default='./results/',
                        type=str,
                        help='Output folder')
    args = parser.parse_args()

    main(args)
