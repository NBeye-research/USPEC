# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import math
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
import random


class ImageFolderPercentTrain(Dataset):

    def __init__(self, root_path, args, transform, partition_tag):

        self.filepaths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_path))
        self.id2classes = {}
        self.rootpath = root_path
        self.name = os.path.basename(args.data_path)
        self.partition_tag = partition_tag
        
        ori_sum = 0
        for i, c in enumerate(self.classes):
            self.id2classes[i] = c
            if not os.path.isdir(os.path.join(root_path, c)):
                continue
            image_names = os.listdir(os.path.join(root_path, c))
            ori_image_num = len(image_names)
            ori_sum += ori_image_num

            if self.partition_tag == 'train':
                if not math.isclose(args.train_percent, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                    assert args.train_percent < 1.
                    train_num = int(args.train_percent * ori_image_num)
                    rnd = random.Random(args.seed)
                    image_names = rnd.sample(list(image_names), train_num)
                    
            for filename in image_names:
                self.filepaths.append(os.path.join(root_path, c, filename))
                self.labels.append(i)
        print('partition_tag:{}, ori number:{}, percent :{}%, left number:{}'.format(partition_tag, ori_sum, float(args.train_percent) * 100, len(self.filepaths)))
        args.metircs_logs.append('partition_tag:{}, ori number:{}, percent :{}%, left number:{}'.format(partition_tag, ori_sum, float(args.train_percent)*100, len(self.filepaths)))

        self.n_classes = len(self.classes)
        
        
        self.transform = transform
        number_str = self.counts_str()
        print(number_str)
        args.metircs_logs.append(number_str)

    def counts_str(self):
        counts = Counter(self.labels)
        parts = []
        for idx, name in enumerate(self.classes):
            cnt = counts.get(idx, 0)
            parts.append(f"{name}:{cnt}")
        return self.partition_tag + ": " + ", ".join(parts)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')
        return self.transform(img), self.labels[i]

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    # dataset = datasets.ImageFolder(root, transform=transform)
    dataset = ImageFolderPercentTrain(root, args, transform=transform, partition_tag = is_train)

    return dataset


class FewShotDataset(Dataset):
    def __init__(self, filepaths, labels, classes, transform=None):
        """
        samples: list of (image_path, label)
        transform: torchvision transform
        """
        self.filepaths = filepaths
        self.labels = labels
        self.classes = classes
        self.n_classes = len(classes)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


def build_few_shot_dataset(args):

    train_transform = build_transform('train', args)
    test_transform = build_transform('test', args)
    root_path = args.data_path
    
    classes = sorted(os.listdir(args.data_path))
    
    train_filepaths, train_labels = [],[]
    test_filepaths, test_labels = [],[]
    rnd = random.Random(args.seed)

    for i, c in enumerate(classes):
        if not os.path.isdir(os.path.join(root_path, c)):
            continue
        filepaths,labels = [],[]

        for filename in sorted(os.listdir(os.path.join(root_path, c))):
            filepaths.append(os.path.join(root_path, c, filename))
            labels.append(i)
        cate_num = len(filepaths)
        if cate_num == 0:
            continue
        train_num = int(cate_num * args.few_shot) if args.few_shot < 1 else int(args.few_shot)

        train_num = max(0, min(train_num, cate_num))


        img_files_shuffled = filepaths[:]
        rnd.shuffle(img_files_shuffled)
        cate_train_paths = img_files_shuffled[:train_num]
        cate_test_paths = img_files_shuffled[train_num:]
        
        cate_train_labels = labels[:train_num]
        cate_test_labels = labels[train_num:]

        train_filepaths.extend(cate_train_paths)
        train_labels.extend(cate_train_labels)
        test_filepaths.extend(cate_test_paths)
        test_labels.extend(cate_test_labels)
    
    dataset_train = FewShotDataset(train_filepaths, train_labels, classes, transform=train_transform)
    dataset_test = FewShotDataset(test_filepaths, test_labels, classes, transform=test_transform)

    return dataset_train, dataset_test


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    mean, std = [0.5723625, 0.34657937, 0.2374997], [0.21822436, 0.19240488, 0.17723322]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
