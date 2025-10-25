import torch
MAE_loss = torch.nn.L1Loss()

import cv2
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

# =================== import Dataset ======================
from Dataset import MyDataset
from torch.utils.data import DataLoader
train_input_dir='/home/ubuntu/thumb_data/tra_en1_4'
train_ground_dir='/home/ubuntu/thumb_data/tra_ori'

test_input_dir="/home/ubuntu/thumb_data/ce_test_en1_4"
test_ground_dir='/home/ubuntu/thumb_data/ce_test_ori'

training_dataset = MyDataset(train=True, device=device, input_dir=train_input_dir,ground_dir=train_ground_dir)
train_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True)

testing_dataset = MyDataset(train=False, device=device, input_dir=test_input_dir,ground_dir=test_ground_dir)
test_dataloader = DataLoader(testing_dataset, batch_size=8, shuffle=True)

print('The number of training images = %d' % len(training_dataset))

# =================== import Mapping Network =====================
from model import MTNetModel
model_Generator = MTNetModel(device)
# ========================================================

# =================== Save models and logs ===============
import os
os.makedirs('training_files', exist_ok=True)
os.makedirs('training_files/models', exist_ok=True)
os.makedirs('training_files/Generated_images', exist_ok=True)
os.makedirs('training_files/logs_train', exist_ok=True)

with open('training_files/logs_train/generator.csv', 'w') as f:
    f.write("epoch, Pixel_loss\n")

with open('training_files/logs_train/log.txt', 'w') as f:
    pass
for item in test_dataloader:
    pass
real_image = item['HR'].cpu()
for i in range(real_image.size(0)):
    os.makedirs(f'training_files/Generated_images/{i}', exist_ok=True)
    img = real_image[i].squeeze()
    img = (img+1)/2
    im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
    cv2.imwrite(f'training_files/Generated_images/{i}/real_image.jpg',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
# ========================================================
num_epochs = 100
for epoch in range(num_epochs):
    iteration = 0
    model_Generator.netG.train()
    for item in train_dataloader:
        # ==================forward==================
        img_SR_coarse, img_SR = model_Generator.forward(item['LR'])
        # ==================backward=================
        loss = model_Generator.optimize_parameters(img_SR_coarse,img_SR,item['HR'])
        # ==================log======================
        iteration += 1
        if iteration % 200 == 0:
            with open('training_files/logs_train/log.txt', 'a') as f:
                f.write(
                    f'epoch:{epoch + 1}, \t iteration: {iteration}, \t pixel_loss:{loss.data.item()}\n')
    # ******************** Eval Genrator ********************
    model_Generator.netG.eval()
    pixel_loss_test = 0
    iteration = 0
    for item in test_dataloader:
        iteration += 1
        # ==================forward==================
        with torch.no_grad():
            img_SR_coarse, img_SR = model_Generator.forward(item['LR'])
            pixel_loss = MAE_loss(item['HR'], img_SR)
            pixel_loss_test += pixel_loss.item()

    with open('training_files/logs_train/generator.csv', 'a') as f:
        f.write(
            f"{epoch + 1}, {pixel_loss_test / iteration}\n")

    img_SR = img_SR.detach().cpu()
    for i in range(img_SR.size(0)):
        img = img_SR[i].squeeze()
        img = (img + 1) / 2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(f'training_files/Generated_images/{i}/epoch_{epoch + 1}.jpg',
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))

    # *******************************************************

    # Save model_Generator
    if epoch % 10 == 0:
        torch.save(model_Generator.netG.state_dict(), 'training_files/models/Generator_{}.pth'.format(epoch + 1))
# ========================================================





