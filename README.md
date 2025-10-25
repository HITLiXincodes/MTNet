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
### Test
## Quick Start
## Citation
Please cite our paper if this code helps you
```

```
