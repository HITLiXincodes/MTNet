import os
import random
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
from torchvision.transforms import InterpolationMode

from data.base_dataset import BaseDataset


class CelebADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.isTrain else False
        self.lr_size = opt.load_size // opt.scale_factor
        self.hr_size = opt.load_size

        self.input_img_dir = self.input_data
        self.ground_img_dir = self.ground_truth

        self.input_img_names, self.ground_img_names = self.get_img_names()

        self.aug = ComposePair([
            RandomHorizontalFlipPair(),
            Scale((1.0, 1.3), opt.load_size)
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_img_names(self):
        input_img_names, ground_img_names = [x for x in os.listdir(self.input_img_dir)],[x for x in os.listdir(self.input_img_dir)]
        #ground_img_names = [x for x in os.listdir(self.ground_img_dir)]
        
        if self.shuffle:
            indices = list(range(len(input_img_names)))
            random.shuffle(indices)
            input_img_names = [input_img_names[i] for i in indices]
            ground_img_names = [ground_img_names[i] for i in indices]
        
        return input_img_names, ground_img_names

    def __len__(self ):
        return len(self.input_img_names)
        
    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_img_dir, self.input_img_names[idx])
        ground_img_path = os.path.join(self.ground_img_dir, self.ground_img_names[idx])

        hr_img = Image.open(ground_img_path).convert('RGB')
        lr_img = Image.open(input_img_path).convert('RGB')

        hr_img, lr_img = self.aug(hr_img,lr_img)
        # downsample and upsample to get the LR image
        lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_img_up = lr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img_up)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': ground_img_path, 'LR_paths': input_img_path}

class RandomHorizontalFlipPair():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return tf.hflip(img1), tf.hflip(img2)
        return img1, img2

class Scale():
    """
    Random scale the image and pad to the same size if needed.
    ---------------
    # Args:
        factor: tuple input, max and min scale factor.
    """

    def __init__(self, factor, size):
        self.factor = factor
        rc_scale = (2 - factor[1], 1)
        self.size = (size, size)
        self.rc_scale = rc_scale
        self.ratio = (3. / 4., 4. / 3.)
        self.resize_crop = transforms.RandomResizedCrop(size, rc_scale)

    def __call__(self, img1, img2):
        scale_factor = random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]
        w, h = img1.size
        sw, sh = int(w * scale_factor), int(h * scale_factor)
        scaled_img1 = tf.resize(img1, (sh, sw), InterpolationMode.BICUBIC)
        scaled_img2 = tf.resize(img2, (sh, sw), InterpolationMode.BICUBIC)
        if sw > w:
            i, j, h, w = self.resize_crop.get_params(img1, self.rc_scale, self.ratio)
            scaled_img1 = tf.resized_crop(img1, i, j, h, w, self.size, InterpolationMode.BICUBIC)
            scaled_img2 = tf.resized_crop(img2, i, j, h, w, self.size, InterpolationMode.BICUBIC)
        elif sw < w:
            lp = (w - sw) // 2
            tp = (h - sh) // 2
            padding = (lp, tp, w - sw - lp, h - sh - tp)
            scaled_img1 = tf.pad(scaled_img1, padding)
            scaled_img2 = tf.pad(scaled_img2, padding)

        return scaled_img1, scaled_img2

class ComposePair():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2
