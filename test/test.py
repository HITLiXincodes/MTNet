import torch
MAE_loss = torch.nn.L1Loss()

import cv2
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

# =================== import Dataset ======================
import sys
sys.path.append('/home/ubuntu/hog/MTNet/train')
from Dataset import MyDataset
from torch.utils.data import DataLoader

test_input_dir="/home/ubuntu/thumb_data/ce_test_en1_4"
test_ground_dir='/home/ubuntu/thumb_data/ce_test_ori'

testing_dataset = MyDataset(train=False, lr_size=32, device=device, input_dir=test_input_dir,ground_dir=test_ground_dir)
test_dataloader = DataLoader(testing_dataset, batch_size=8, shuffle=True)

print('The number of testing images = %d' % len(testing_dataset))

# =================== import Mapping Network =====================
from model import MTNetModel
model_Generator = MTNetModel(device)
pretrain_path="/home/ubuntu/hog/Ablation4-encry1/checkpoints/ab16_1/iter_202000_net_G.pth"
model_Generator.load_pretrain_model(pretrain_path)
model_Generator.netG.eval()
# ========================================================

# =================== Save models ===============
import os
os.makedirs('testing_files', exist_ok=True)
os.makedirs('testing_files/Ground_images', exist_ok=True)
os.makedirs('testing_files/Generated_images', exist_ok=True)
# ========================================================
count=0
for item in test_dataloader:
    with torch.no_grad():
        # ==================forward==================
        img_SR_coarse, img_SR = model_Generator.forward(item['LR'])
        img_SR = img_SR.detach().cpu()
        img_GT = item['HR'].cpu()
        for i in range(img_SR.size(0)):
            img = img_GT[i].squeeze()
            img = (img + 1) / 2
            im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
            cv2.imwrite(f'testing_files/Ground_images/ground_{count}.jpg',
                        np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
            img = img_SR[i].squeeze()
            img = (img + 1) / 2
            im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
            cv2.imwrite(f'testing_files/Generated_images/{count}.jpg',
                        np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
            count+=1
    if count>=100:
        break





