import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
from torchvision import transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import Dataset
from utils.FID import fid
gpu_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

class thumbnailDataset(Dataset):
    def __init__(self, num, file1,file2):
        self.num = num
        self.file1 = file1
        self.filelist1 = sorted(os.listdir(self.file1))
        self.file2=file2
        self.filelist2 = sorted(os.listdir(self.file2))
        self.lpips_model = lpips.LPIPS(net='vgg')

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def calmetric(self):
        ssim, psnr,lpips, = [], [], []
        for i in range(self.num):
            ground_truth = cv2.imread(self.file1 + self.filelist1[i])
            fake = cv2.imread(self.file2 + self.filelist2[i])
            s, p , l= self.metrics(ground_truth, fake)
            ssim.append(s)
            psnr.append(p)
            lpips.append(l)
        return np.mean(ssim),np.mean(psnr),np.mean(lpips)

    def metrics(self, Ig, Io):
        ssim=SSIM(Ig, Io)
        psnr=PSNR(Ig, Io, use_y_channel=True)
        ig_tensor=preprocess_image(Ig)
        io_tensor=preprocess_image(Io)
        lpips_value = self.lpips_model(ig_tensor, io_tensor)
        return ssim, psnr, lpips_value.item()

# 图像预处理函数
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 转换为PIL图像
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])
    return transform(image).unsqueeze(0)  # 增加一个批次维度

def rgb2y_matlab(x):
    """Convert RGB image to illumination Y in Ycbcr space in matlab way.
    -------------
    # Args
        - Input: x, byte RGB image, value range [0, 255]
        - Ouput: byte gray image, value range [16, 235] 

    # Shape
        - Input: (H, W, C)
        - Output: (H, W) 
    """
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255**2 / mse) 

def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)
     
    ssim_score = compare_ssim(gt_img, noise_img, gaussian_weights=True, 
            sigma=1.5, use_sample_covariance=False)
    return ssim_score

path1='/home/ubuntu/thumb_data/ff_test_ori/'
path2='/home/ubuntu/hog/Ablation4-encry2/ff/'
model=thumbnailDataset(3000,path1,path2)
print(model.calmetric())
path = [path1, path2]
fid_result = fid(path)
print(fid_result)
'''
with open('test.txt','w') as file:
    path1='/home/ubuntu/thumb_data/ce_test_ori/'
    for i in range(160000,203000,2000):
        path2='/home/ubuntu/hog/Ablation4-encry2/result/'+str(i)+'/'
        model=thumbnailDataset(3000,path1,path2)    
        result=model.calmetric()
        file.write(str(i)+':'+str(result)+'\n')
'''