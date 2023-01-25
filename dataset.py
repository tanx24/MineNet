import os
from PIL import Image
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random
import cv2


class CoData_Train(data.Dataset):
    def __init__(self, img_root, gt_root, edge_root, img_size, transform, max_num, jigsaw=True):
        # Path Pool
        self.img_root = img_root  # root
        self.gt_root = gt_root
        self.edge_root = edge_root
        self.dirs = os.listdir(img_root)  # all dir
        self.img_dir_paths = list(  # [img_root+dir1, ..., img_root+dir2]
            map(lambda x: os.path.join(img_root, x), self.dirs))
        self.gt_dir_paths = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(gt_root, x), self.dirs))
        self.edge_dir_paths = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(edge_root, x), self.dirs))
        self.img_name_list = [os.listdir(idir) for idir in self.img_dir_paths
                              ]  # [[name00,..., 0N],..., [M0,..., MN]]
        self.gt_name_list = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list
        ]
        self.edge_name_list = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list
        ]
        self.img_path_list = [
            list(
                map(lambda x: os.path.join(self.img_dir_paths[idx], x),
                    self.img_name_list[idx]))
            for idx in range(len(self.img_dir_paths))
        ]  # [[impath00,..., 0N],..., [M0,..., MN]]
        self.gt_path_list = [
            list(
                map(lambda x: os.path.join(self.gt_dir_paths[idx], x),
                    self.gt_name_list[idx]))
            for idx in range(len(self.gt_dir_paths))
        ]  # [[gtpath00,..., 0N],..., [M0,..., MN]]
        self.edge_path_list = [
            list(
                map(lambda x: os.path.join(self.edge_dir_paths[idx], x),
                    self.edge_name_list[idx]))
            for idx in range(len(self.edge_dir_paths))
        ]
        self.nclass = len(self.dirs)

        # Other Hyperparameters
        self.size = img_size
        self.cat_size = int(img_size * 2)
        self.sizes = [img_size, img_size]
        self.transform = transform
        self.max_num = max_num
        self.jigsaw = jigsaw

    def __getitem__(self, item):
        
        img_paths = self.img_path_list[item]
        gt_paths = self.gt_path_list[item]
        edge_paths = self.edge_path_list[item]
        num = len(img_paths)

        if num > self.max_num:
            sampled_list = random.sample(range(num), self.max_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            gt_paths = new_gt_paths
            new_edge_paths = [edge_paths[i] for i in sampled_list]
            edge_paths = new_edge_paths
            num = self.max_num

        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])
        gts = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        edges = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        distractors = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')
            edge = Image.open(edge_paths[idx]).convert('L')
            distractor = Image.fromarray(np.uint8(np.zeros(self.sizes)), mode='L')

            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))

            # Co-Data Augmentation
            if self.jigsaw and random.random() > 0.25:
                [img, gt, edge, distractor] = self.jigsaw2(img_paths[idx],
                                                     gt_paths[idx],
                                                     edge_paths[idx],
                                                     dir_item=item)

            [img, gt, edge, distractor] = self.transform(img, gt, edge, distractor)

            imgs[idx] = img
            gts[idx] = gt
            edges[idx] = edge
            distractors[idx] = distractor

        return imgs, gts, edges, subpaths, ori_sizes, distractors

    def __len__(self):
        return len(self.dirs)

    def jigsaw2(self, img_path, gt_path, edge_path, dir_item):

        obj_img = cv2.resize(cv2.imread(img_path), (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)
        obj_gt = cv2.resize(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE),
                            (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)
        obj_edge = cv2.resize(cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE),
                            (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)

        # Sample an additional foreground objects
        Candidate_list = list(range(self.nclass))
        Candidate_list.remove(dir_item)
        addi_item = random.choice(Candidate_list)
        addi_idx = random.randint(0, len(self.img_path_list[addi_item]) - 1)
        addi_img_path = self.img_path_list[addi_item][addi_idx]
        addi_gt_path = self.gt_path_list[addi_item][addi_idx]
        addi_edge_path = self.edge_path_list[addi_item][addi_idx]

        addi_img = cv2.resize(cv2.imread(addi_img_path),
                              (self.size, self.size),
                              interpolation=cv2.INTER_LINEAR)
        addi_gt = cv2.resize(cv2.imread(addi_gt_path, cv2.IMREAD_GRAYSCALE),
                             (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)
        addi_edge = cv2.resize(cv2.imread(addi_edge_path, cv2.IMREAD_GRAYSCALE),
                             (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)

        if random.random() < 0.5:
            img = np.zeros([self.size, self.cat_size, 3])
            gt = np.zeros([self.size, self.cat_size])
            edge = np.zeros([self.size, self.cat_size])
            distractor = np.zeros([self.size, self.cat_size])

            if random.random() < 0.5:
                img[0:self.size, 0:self.size] = obj_img
                img[0:self.size, self.size:2 * self.size] = addi_img
                gt[0:self.size, 0:self.size] = obj_gt
                gt[0:self.size, self.size:2 * self.size] = addi_gt * 0
                edge[0:self.size, 0:self.size] = obj_edge
                edge[0:self.size, self.size:2 * self.size] = addi_edge * 0
                distractor[0:self.size, 0:self.size] = obj_gt * 0
                distractor[0:self.size, self.size:2 * self.size] = addi_gt
            else:
                img[0:self.size, 0:self.size] = addi_img
                img[0:self.size, self.size:2 * self.size] = obj_img
                gt[0:self.size, 0:self.size] = addi_gt * 0
                gt[0:self.size, self.size:2 * self.size] = obj_gt
                edge[0:self.size, 0:self.size] = addi_edge * 0
                edge[0:self.size, self.size:2 * self.size] = obj_edge
                distractor[0:self.size, 0:self.size] = addi_gt
                distractor[0:self.size, self.size:2 * self.size] = obj_gt * 0
        else:
            img = np.zeros([self.cat_size, self.size, 3])
            gt = np.zeros([self.cat_size, self.size])
            edge = np.zeros([self.cat_size, self.size])
            distractor = np.zeros([self.cat_size, self.size])

            if random.random() < 0.5:
                img[0:self.size, 0:self.size] = obj_img
                img[self.size:2 * self.size, 0:self.size] = addi_img
                gt[0:self.size, 0:self.size] = obj_gt
                gt[self.size:2 * self.size, 0:self.size] = addi_gt * 0
                edge[0:self.size, 0:self.size] = obj_edge
                edge[self.size:2 * self.size, 0:self.size] = addi_edge * 0
                distractor[0:self.size, 0:self.size] = obj_gt * 0
                distractor[self.size:2 * self.size, 0:self.size] = addi_gt
            else:
                img[0:self.size, 0:self.size] = addi_img
                img[self.size:2 * self.size, 0:self.size] = obj_img
                gt[0:self.size, 0:self.size] = addi_gt * 0
                gt[self.size:2 * self.size, 0:self.size] = obj_gt
                edge[0:self.size, 0:self.size] = addi_edge * 0
                edge[self.size:2 * self.size, 0:self.size] = obj_edge
                distractor[0:self.size, 0:self.size] = addi_gt
                distractor[self.size:2 * self.size, 0:self.size] = obj_gt * 0

        return Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)), Image.fromarray(np.uint8(gt), mode='L'), Image.fromarray(np.uint8(edge),mode='L'), Image.fromarray(np.uint8(distractor),mode='L')


class CoData_Test(data.Dataset):
    def __init__(self, img_root, img_size):

        class_list = os.listdir(img_root)
        self.sizes = [img_size, img_size]

        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))

        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img

        return imgs, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


    def __len__(self):
        return len(self.img_dirs)

class FixedResize(object):
    def __init__(self, size):
        self.sizes = (size, size)  # size: (h, w)

    def __call__(self, img, gt, edge, distractor):
        # assert img.size == gt.size

        img = img.resize(self.sizes, Image.BILINEAR)
        gt = gt.resize(self.sizes, Image.NEAREST)
        edge = edge.resize(self.sizes, Image.NEAREST)
        distractor = distractor.resize(self.sizes, Image.NEAREST)

        return img, gt, edge, distractor


class ToTensor(object):
    def __call__(self, img, gt, edge, distractor):

        return F.to_tensor(img), F.to_tensor(gt),  F.to_tensor(edge), F.to_tensor(distractor)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt, edge, distractor):
        img = F.normalize(img, self.mean, self.std)

        return img, gt, edge, distractor


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt, edge, distractor):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
            distractor = distractor.transpose(Image.FLIP_LEFT_RIGHT)

        return img, gt, edge, distractor


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, gt, edge, distractor):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, Image.BILINEAR, self.expand, self.center)
        gt = F.rotate(gt, angle, Image.NEAREST, self.expand, self.center)
        edge = F.rotate(edge, angle, Image.NEAREST, self.expand, self.center)
        distractor = F.rotate(distractor, angle, Image.NEAREST, self.expand,
                              self.center)

        return img, gt, edge, distractor


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, edge, distractor):
        for t in self.transforms:
            img, gt, edge, distractor = t(img, gt, edge, distractor)
        return img, gt, edge, distractor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# get the dataloader (Note: without data augmentation)
def get_loader(img_root,
               gt_root,
               edge_root,
               img_size,
               batch_size,
               max_num=float('inf'),
               istrain=True,
               jigsaw=True,
               shuffle=False,
               num_workers=0,
               pin=False):
    if istrain:
        transform = Compose([
            FixedResize(img_size),
            RandomHorizontalFlip(),
            RandomRotation((-90, 90)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = CoData_Train(img_root, gt_root, edge_root, img_size, transform, max_num, jigsaw)
    else:
        dataset = CoData_Test(img_root, img_size)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader

