import torch.nn as nn
import torch.optim as optim
from MTNet import MTNet
import torch

class MTNetModel():

    def __init__(self,device):

        self.netG = MTNet().to(device)
        self.criterionL1 = nn.L1Loss()
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.optimizers = [self.optimizer_G]

    def load_pretrain_model(self, pretrain_model_path):
        print('Loading pretrained model', pretrain_model_path)
        weight = torch.load(pretrain_model_path)
        self.netG.load_state_dict(weight)

    def forward(self,data):
        img_SR_coarse, img_SR = self.netG(data)
        return img_SR_coarse,img_SR

    def backward_G(self,img_SR_coarse,img_SR,img_HR):
        self.loss_Pix = self.criterionL1(img_SR, img_HR) * 1.0 + self.criterionL1(img_SR_coarse,img_HR) * 0.2
        self.loss_Pix.backward()
        return self.loss_Pix

    def optimize_parameters(self, img_SR_coarse,img_SR,img_HR):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        loss=self.backward_G(img_SR_coarse,img_SR,img_HR)
        self.optimizer_G.step()
        return loss






