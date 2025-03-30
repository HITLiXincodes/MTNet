

import kornia
import os
from torch.nn import functional as F
from mamba_ssm import Mamba
import torch
import torch.nn as nn
import numbers
from functools import partial
from einops import rearrange
from torchsummary import summary

gpu_id = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class shallow(BaseNetwork):
    def __init__(self, output_channel):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=8, mode='bicubic')
        self.layer_up = nn.Sequential(
            nn.Conv2d(3, output_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, data):
        #up = self.up(data)
        up1 = self.layer_up(data)
        return up1

#net=shallow(32).cuda()
#summary(net,(3,16,16))

class ChannelAttention(BaseNetwork):

    def __init__(self, num_feat, squeeze_factor=4):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class Mamba_Block(BaseNetwork):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mamba = Mamba(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, input_dim)
        )

    def forward(self, x):
        b, c = x.shape[:2]
        length = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_embed = x.reshape(b, c, length).transpose(-1, -2)
        x_norm = self.norm(x_embed)
        x_mamba = self.mamba(x_norm) + x_embed
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(b, c, *img_dims)
        return out

# net=Mamba_Block(16,32).cuda()
# summary(net,(16,128,128))
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x 

#net=TransformerBlock(128,4,2.66,False,'WithBias').cuda()
#summary(net,(128,128,128))

class AFDU(BaseNetwork):
    def __init__(self, input):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.cab = ChannelAttention(input, squeeze_factor=2)
        self.deduction = nn.Conv2d(input, input // 2, kernel_size=1, stride=1)
        self.expansion = nn.Conv2d(input // 2, input, kernel_size=1, stride=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input * 2, input, kernel_size=1, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input, input, groups=input, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input, input, kernel_size=1, stride=1)
        )

    def forward(self, fea):
        thumb = self.deduction(fea)
        thumb = self.act(thumb)
        exp = self.expansion(thumb)
        concat = torch.concat([exp, fea], dim=1)
        conv1 = self.conv1(concat)
        conv1 = self.act(conv1)
        attention = self.cab(conv1)
        conv3 = self.conv3(attention)
        return conv3 + fea

# net=AFDU(8).cuda()
# summary(net,(8,128,128))

class Vision_Block(BaseNetwork):
    def __init__(self, input_dim):
        super().__init__()
        self.act=nn.LeakyReLU(0.2)
        self.afdu1=AFDU(input_dim)
        self.mamba=Mamba_Block(input_dim,input_dim*2)
        self.transformer=TransformerBlock(input_dim)
        self.afdu2=AFDU(input_dim)

    def forward(self, x):
        identity=x
        x=self.afdu1(x)
        x=self.act(x)
        x=self.mamba(x)+identity
        identity2=x
        x=self.afdu2(x)
        x=self.act(x)
        x=self.transformer(x)
        x=x+identity2
        return x

#net = Vision_Block(32).cuda()
#summary(net, (32, 128, 128))

class downsample(BaseNetwork):
    def __init__(self, input, output):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, fea):
        conv1=self.conv1(fea)
        conv1=self.act(conv1)
        conv2=self.conv2(conv1)
        conv2=self.act(conv2)
        return conv2

#net=downsample(32).cuda()
#summary(net,(32,128,128))

class upsample(BaseNetwork):
    def __init__(self, input, output):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.ConvTranspose2d(input, output, kernel_size=6, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, fea):
        conv1=self.conv1(fea)
        conv1=self.act(conv1)
        conv2=self.conv2(conv1)
        conv2=self.act(conv2)
        return conv2

class fu_32(BaseNetwork):
    def __init__(self, input1, input2, input3, output):
        super().__init__()

        self.ca_main=ChannelAttention(output)
        self.norm_main = LayerNorm(output,'WithBias')

        self.down128=nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=4),
            nn.Conv2d(input1*16, input1*16, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down64=nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(input2*4, input2*4, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down32=nn.Sequential(
            nn.Conv2d(input3, input3, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )

        self.con=nn.Sequential(
            nn.Conv2d(input1*16+input2*4+input3+output,output,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )
        self.ca=ChannelAttention(output)
        self.norm = LayerNorm(output,'WithBias')

        self.deal=nn.Conv2d(output,output,kernel_size=1,stride=1)

    def forward(self, ski1, ski2, ski3, infea):
        fea_main = self.norm_main(infea)
        fea_main = self.ca_main(fea_main)+infea

        ski1=self.down128(ski1)
        ski2=self.down64(ski2)
        ski3=self.down32(ski3)
        fea_path = torch.cat([ski1,ski2,ski3,infea],dim=1)
        fea_path = self.con(fea_path)
        fea_path = self.norm(fea_path)
        fea_path = self.ca(fea_path)+infea

        out = fea_main*torch.sigmoid(fea_path)
        out = self.deal(out)+infea

        return out


#net=fu_32(32,64,128,128).cuda()
#summary(net,[(32,128,128),(64,64,64),(128,32,32),(128,32,32)])

class fu_64(BaseNetwork):
    def __init__(self, input1, input2, input3, output):
        super().__init__()

        self.ca_main=ChannelAttention(output)
        self.norm_main = LayerNorm(output,'WithBias')

        self.down128=nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(input1*4, input1*4, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down64=nn.Sequential(
            nn.Conv2d(input2, output, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down32=nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(input3//4,input3//4,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )

        self.con=nn.Sequential(
            nn.Conv2d(input1*4+input2+input3//4+output,output,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )
        self.ca=ChannelAttention(output)
        self.norm = LayerNorm(output,'WithBias')

        self.deal=nn.Conv2d(output,output,kernel_size=1,stride=1)

    def forward(self, ski1, ski2, ski3, infea):
        fea_main = self.norm_main(infea)
        fea_main = self.ca_main(fea_main)+infea

        ski1=self.down128(ski1)
        ski2=self.down64(ski2)
        ski3=self.down32(ski3)
        fea_path = torch.cat([ski1,ski2,ski3,infea],dim=1)
        fea_path = self.con(fea_path)
        fea_path = self.norm(fea_path)
        fea_path = self.ca(fea_path)+infea

        out = fea_main*torch.sigmoid(fea_path)
        out = self.deal(out)+infea

        return out

#net=fu_64(32,64,128,64).cuda()
#summary(net,[(32,128,128),(64,64,64),(128,32,32),(64,64,64)])

class fu_128(BaseNetwork):
    def __init__(self, input1, input2, input3, output):
        super().__init__()

        self.ca_main=ChannelAttention(output)
        self.norm_main = LayerNorm(output,'WithBias')

        self.down128=nn.Sequential(
            nn.Conv2d(input1, input1, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down64=nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(input2//4,input2//4,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )
        self.down32=nn.Sequential(
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(input3//16,input3//16,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )

        self.con=nn.Sequential(
            nn.Conv2d(input1+input2//4+input3//16+output,output,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )
        self.ca=ChannelAttention(output)
        self.norm = LayerNorm(output,'WithBias')

        self.deal=nn.Conv2d(output,output,kernel_size=1,stride=1)

    def forward(self, ski1, ski2, ski3, infea):
        fea_main = self.norm_main(infea)
        fea_main = self.ca_main(fea_main)+infea

        ski1=self.down128(ski1)
        ski2=self.down64(ski2)
        ski3=self.down32(ski3)
        fea_path = torch.cat([ski1,ski2,ski3,infea],dim=1)
        fea_path = self.con(fea_path)
        fea_path = self.norm(fea_path)
        fea_path = self.ca(fea_path)+infea

        out = fea_main*torch.sigmoid(fea_path)
        out = self.deal(out)+infea

        return out

#net=fu_128(32,64,128,32).cuda()
#summary(net,[(32,128,128),(64,64,64),(128,32,32),(32,128,128)])

class Encoder(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            Vision_Block(32)
        )
        self.down2 = nn.Sequential(
            downsample(32,64),
            Vision_Block(64),
            Vision_Block(64),
            Vision_Block(64)
        )
        self.down3 = nn.Sequential(
            downsample(64,128),
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128)
        )

        self.middle = nn.Sequential(
            downsample(128,256),
            Vision_Block(256),
            Vision_Block(256),
            Vision_Block(256),
            Vision_Block(256),
        )

    def forward(self, data):
        down1 = self.down1(data)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        middle = self.middle(down3)
        
        return down1,down2,down3,middle
'''
class key():
    def __init__(self, block_size):
        super().__init__()
        self.blocksize = block_size
        self.ave = nn.AvgPool2d(kernel_size=block_size)
        self.up = nn.UpsamplingNearest2d(scale_factor=block_size)
        self.max = nn.MaxPool2d(kernel_size=block_size)

    def forward(self, data, eps=1e-5):
        b, c, h, w = data.shape
        mean = self.ave(data)
        mean2 = self.up(mean)
        minus = data - mean2 + eps
        abs = torch.abs(minus)
        real_max = self.max(abs)
        result = self.up(real_max)

        symbols = minus / result
        pos = symbols == torch.ones(b, c, h, w).cuda()
        neg = symbols == ((torch.ones(b, c, h, w)).cuda() * -1)
        temp = torch.zeros(b, c, h, w).cuda()
        poss = torch.where(pos, symbols, temp)
        negg = torch.where(neg, symbols, temp)
        real_symbol = poss + negg
        return data-((real_symbol * result) + mean2) * (real_symbol ** 2)
'''
class Partial_Decoder(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.up1 = upsample(256,128)

        self.vision1 = nn.Sequential(
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128)
        )

        self.up2 = upsample(128,64)  
        self.vision2 = nn.Sequential(
            Vision_Block(64),
            Vision_Block(64),
            Vision_Block(64)
        )

        self.up3 = upsample(64,32)
        self.vision3 = nn.Sequential(
            Vision_Block(32)
        )

        self.fu32=fu_32(32,64,128,128)
        self.fu64=fu_64(32,64,128,64)
        self.fu128=fu_128(32,64,128,32)

    def forward(self, down1,down2,down3,middle):

        up1 = self.up1(middle)
        up1 = self.fu32(down1,down2,down3,up1)
        up1 = self.vision1(up1)

        up2 = self.up2(up1)
        up2 = self.fu64(down1,down2,down3,up2)
        up2 = self.vision2(up2)
        
        up3 = self.up3(up2)
        up3 = self.fu128(down1,down2,down3,up3)
        up3 = self.vision3(up3)

        return up3

class Cascade(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.act=nn.LeakyReLU(0.2)
        self.up = Partial_Decoder()
        self.down1 = downsample(32,64)
        self.down2 = downsample(64,128)
        self.down3 = downsample(128,256)

        self.con_out=nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1)
        self.con_in=nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)

        self.vision1 = nn.Sequential(
            Vision_Block(32)
        )
        self.vision2 = nn.Sequential(
            Vision_Block(64),
            Vision_Block(64),
            Vision_Block(64)
        )

        self.vision3 = nn.Sequential(
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128)
        )

        self.vision4 = nn.Sequential(
            Vision_Block(256),
            Vision_Block(256),
            Vision_Block(256),
            Vision_Block(256),
        )

    def forward(self, down1,down2,down3,fea):
        fea1 = self.up(down1,down2,down3,fea)
        c_out=self.con_out(fea1)
        fea1 = self.con_in(c_out)        
        fea1 = self.vision1(fea1)

        fea2 = self.down1(fea1)
        fea2 = self.vision2(fea2)

        fea3 = self.down2(fea2)
        fea3 = self.vision3(fea3)

        fea4 = self.down3(fea3)
        fea4 = self.vision4(fea4)
        return c_out,fea1,fea2,fea3,fea4

#net=Cascade().cuda()
#summary(net,[(256,16,16)])

class Middle_Block(BaseNetwork):
    def __init__(self, input_dim):
        super().__init__()
        self.vision21=Vision_Block(input_dim)
        self.vision22=Vision_Block(input_dim)
        self.con_2 = nn.Sequential(
            nn.Conv2d(2*input_dim,input_dim,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x1,x2):

        f2=self.con_2(torch.cat([x1,x2],dim=1))
        f21x=self.vision21(x1)+f2
        f22x=self.vision22(x2)+f2
        f21x=torch.sigmoid(f21x)
        f22x=torch.sigmoid(f22x)

        return (f21x@f22x)+f2

class Decoder(BaseNetwork):
    def __init__(self):
        super().__init__()

        self.con = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2)
        )
        self.up1 = upsample(256,128)

        self.vision1 = nn.Sequential(
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128),
            Vision_Block(128)
        )

        self.up2 = upsample(128,64)
            
        self.vision2 = nn.Sequential(
            Vision_Block(64),
            Vision_Block(64),
            Vision_Block(64)
        )

        self.up3 = upsample(64,32)

        self.vision3 = nn.Sequential(
            Vision_Block(32)
        )
        
        self.cascade=Cascade()
        self.middle=Middle_Block(256)

        self.fu32=fu_32(32,64,128,128)
        self.fu64=fu_64(32,64,128,64)
        self.fu128=fu_128(32,64,128,32)

        self.ski32=fu_32(32,64,128,128)
        self.ski64=fu_64(32,64,128,64)
        self.ski128=fu_128(32,64,128,32)

    def forward(self, down1,down2,down3,middle):

        mid_out,ski1,ski2,ski3,enter = self.cascade(down1,down2,down3,middle)

        up1 = self.up1(self.middle(middle,enter))
        up11 = self.fu32(down1,down2,down3,up1)
        up12 = self.ski32(ski1,ski2,ski3,up1)
        up1 = self.vision1(up11+up12)
        
        up2 = self.up2(up1)
        up21 = self.fu64(down1,down2,down3,up2)
        up22 = self.ski64(ski1,ski2,ski3,up2)
        up2 = self.vision2(up21+up22)
        
        up3 = self.up3(up2)
        up31 = self.ski128(ski1,ski2,ski3,up3)
        up32 = self.fu128(down1,down2,down3,up3)
        up3 = self.vision3(up31+up31)
        
        return mid_out,up3
        
class reconstruction(BaseNetwork):
    def __init__(self, input_channel):
        super(reconstruction, self).__init__()
        self.layer1=nn.Conv2d(input_channel,3,kernel_size=3,stride=1, padding=1)
    def forward(self, data):
        out1=self.layer1(data)
        return out1


class CTCNet(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.shallow = shallow(32)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.recon = reconstruction(32)
    def forward(self, data):
        up= self.shallow(data)
        down1,down2,down3,middle = self.encoder(up)
        out1,deep = self.decoder(down1,down2,down3,middle)
        out = self.recon(deep)
        return data+out1,data+out

#net=CTCNet().cuda()
#summary(net,(3,128,128))

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'weights/'
