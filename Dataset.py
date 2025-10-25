
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import random
import os
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self,
                 input_dir='/home/ubuntu/FR_Attack/databases/arcface',
                 ground_dir='/home/ubuntu/FR_Attack/databases/arcface',
                 lr_size=16,
                 hr_size=128,
                 train=True,
                 device='cuda:0',
                 ):
        self.device = device
        self.shuffle = True if train else False
        self.input_dir = input_dir
        self.ground_dir = ground_dir
        self.lr_size = lr_size
        self.hr_size = hr_size

        self.input_imgs, self.ground_imgs = self.get_img_names()

        self.aug = ComposePair([
            RandomHorizontalFlipPair(),
            Scale((1.0, 1.3), self.hr_size)
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def get_img_names(self):

        input_imgs = [x for x in os.listdir(self.input_dir)]
        ground_imgs = [x for x in os.listdir(self.input_dir)]

        if self.shuffle:
            indices = list(range(len(input_imgs)))
            random.shuffle(indices)
            out_input_imgs = [input_imgs[i] for i in indices]
            out_ground_imgs = [ground_imgs[i] for i in indices]
        else:
            out_input_imgs = input_imgs
            out_ground_imgs = ground_imgs
        return out_input_imgs, out_ground_imgs

    def __len__(self):
        #return 32
        return len(self.input_imgs)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.input_imgs[idx])
        ground_img_path = os.path.join(self.ground_dir, self.ground_imgs[idx])

        hr_img = Image.open(ground_img_path).convert('RGB')
        lr_img = Image.open(input_img_path).convert('RGB')

        hr_img, lr_img = self.aug(hr_img, lr_img)
        # downsample and upsample to get the LR image
        lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_img_up = lr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        hr_tensor = self.to_tensor(hr_img).to(self.device)
        lr_tensor = self.to_tensor(lr_img_up).to(self.device)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': ground_img_path, 'LR_paths': input_img_path}

class RandomHorizontalFlipPair():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return tf.hflip(img1), tf.hflip(img2)
        return img1, img2


class Scale():

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

        return scaled_img1, scaled_img2

class ComposePair():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2

#dataset=MyDataset('/home/ubuntu/thumb_data/ce_test_en1_16','/home/ubuntu/thumb_data/ce_test_en2_16')
#print(dataset.__getitem__(0))
