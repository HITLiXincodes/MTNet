# Information Disclosure Risk of Thumbnail-Preserving Encryption
Source code of the paper titled "Information Disclosure Risk of Thumbnail-Preserving Encryption" published in IEEE Transactions on Multimedia.

**Abstract:** With the rapid growth of cloud services, the storage of images in cloud environments requires secure and effective
data encryption methods. Many thumbnail-preserving encryption
(TPE) methods have thus been proposed to balance privacy and
usability of image data. However, the exposure of thumbnail
information in TPE methods may introduce privacy leakage risk,
and a systematic evaluation of their security has not yet been
conducted. In this paper, we propose a new Mamba-Transformer
cooperation Network (MTNet) to recover the original images
from the limited exposed thumbnail information, highlighting
the information disclosure problem in TPE. Specifically, the core
model component integrates a Mamba block and a Transformer
block, which employ the powerful capabilities of the Mamba
for wide field dependency modeling and the Transformer for
effective channel interaction. Besides, the cascade architecture
incorporates an intermediate output that provides supplementary information and achieves multilevel supervision, thereby
improving the quality of the final output. Finally, to better
utilize the subtle details in different levels, we propose a multi-scale fusion module that adaptively integrates features from
various stages of the encoding process. The experimental results
achieved by our proposed MTNet reveal that the privacy risk
associated with TPE is significantly underestimated and more
robust defense mechanisms are required.

<p align='center'>  
  <img src='https://github.com/HITLiXincodes/MTNet/blob/main/whole.png' width='870'/>
</p>
<p align='center'>  
  <em>Framework of MTNet.</em>
</p>

## How to use
### Train
You can re-train the model easily by running the train.py file.
```
python train.py
```
You need to modify the relevant setting at first in the file train.py.

```
input_dir: the path that you put the encrypted images
ground_dir: the path that you put the ground truth
lr_size: the size of thumbnails
hr_size: the size of reconstructed images
```
**make sure there is no sub-dirs and all the encrypted images and ground truth share the same name**
 

### Test
You can get the reconstructed images by running the test.py file.
```
python test.py
```
You need to set the correct file paths and the model weigthts path. For more detail, please refer to the test.py.

### Pretrained-Weights
We offer the pretrained weights that can be download as follows:
| Model | Description | Download link|
|:----------:|:------------:| :----------:|
| BE-4 | BE-TPE with block size 4 |[Baidu Yun (Code:m7u9)](https://pan.baidu.com/s/146fQ2hCi8Pp4AMgXsVl91w)|
| BE-8 | BE-TPE with block size 8 |[Baidu Yun (Code:yghh)](https://pan.baidu.com/s/1OqB06QFrb2hTuh_75CKg7Q)|
| BE-16 | BE-TPE with block size 16 |[Baidu Yun (Code:95xt)](https://pan.baidu.com/s/1GXfvivBtU46RocMMfWhzCA)|
| SP-4 | SP-TPE with block size 4 |[Baidu Yun (Code:hr8i)](https://pan.baidu.com/s/1fw-oPyilNqmDfwz-E8eiHQ)|
| SP-8 | SP-TPE with block size 8 |[Baidu Yun (Code:f2hc)](https://pan.baidu.com/s/1innDndVyf2Vd9Q2VrI89qw )|
| SP-16 | SP-TPE with block size 16 |[Baidu Yun (Code:8fft)](https://pan.baidu.com/s/1lAOdVRCSS8ETR27kjmtdog)|
| JPEG-4 | JPEG-TPE with block size 4|[Baidu Yun (Code:hw3t)](https://pan.baidu.com/s/1NUQcdAEp5jTve8TPc_51DA)|
| JPEG-8 | JPEG-TPE with block size 8|[Baidu Yun (Code:xqct)](https://pan.baidu.com/s/11g1zx0rl93FLHECK_86fnQ)|
| JPEG-16 | JPEG-TPE with block size 16 |[Baidu Yun (Code:k2na)](https://pan.baidu.com/s/1ZcPj2LZVTs9TT03hku79Zg)|
## Quick Start
```
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
```
## Citation
Please cite our paper if this code helps you
```

```
