import torch
import torch.nn as nn
import math
#from nets.ConvNext import Block
import torch.nn.functional as F



class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2], 
                    x[..., 1::2, ::2], 
                    x[..., ::2, 1::2], 
                    x[..., 1::2, 1::2]
                ], 1
            )
        )
###############################################################################
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
#####################################################################################
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)
"""
class VoVGSCSP(nn.Module):
    # VoV-GSCSP https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))
"""


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = nn.Identity()
    def forward(self, x):
        return self.conv_lighting(x)


#####################################################################################

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, phi, pretrained):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        #-----------------------------------------------#

        #-----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        #-----------------------------------------------#
        self.stem       = Focus(3, base_channels, k=3)
        
        #-----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels * 2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        
        #-----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #                   在这里引出有效特征层80, 80, 256
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        #-----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #                   在这里引出有效特征层40, 40, 512
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        
        #-----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )
        if pretrained:
            url = {
                's' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
                'm' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
                'l' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
                'x' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", url.split('/')[-1])

    #    self.eca_block1 = eca_block(64)
     #   self.eca_block2 = eca_block(128)
    #    self.eca_block3 = eca_block(256)
     #   self.eca_block4 = eca_block(512)
     #   self.eca_block5 = eca_block(1024)
   #     self.Texture_Enhance_v1=Texture_Enhance_v1(64,1)

#        self.GAM_Attention1 = GAM_Attention(32,32)
 #       self.GAM_Attention2 = GAM_Attention(64,64)
  #      self.GAM_Attention3 = GAM_Attention(128,128)
   #     self.GAM_Attention4 = GAM_Attention(256,256)
   #     self.GAM_Attention5 = GAM_Attention(512,512)
 #       self.GhostModule1 = GhostModule(64, 64, 1)
 #       self.GhostModule2 = GhostModule(128, 128, 1)
  #      self.GhostModule3 = GhostModule(256, 256, 1)
  #      self.GhostModule4 = GhostModule(512, 512, 1)

    def forward(self, x):
        x = self.stem(x)


     #   W1 = self.Block1(x)

   #     P1 = self.GAM_Attention1(x)
        x = self.dark2(x)
      #  P1=self.Texture_Enhance_v1(x)
    #    W2 = self.Block2(x)

  #      P2 = self.GAM_Attention2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)

       # W3 = self.Block3(x)

  #      P3 = self.GAM_Attention3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)

      #  W4 = self.Block4(x)

  #      P4 = self.GAM_Attention4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
 #       P5 = self.GAM_Attention5(x)
        feat3 = x
        return feat1, feat2, feat3

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        y = self.avg_pool(x0)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x0 * y.expand_as(x0)


class RFCAConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv(c1, 9 *c1, k=1, g=c1)
        self.group_conv2 = Conv(c1, 9 *c1, k=3, g=c1)
        self.group_conv3 = Conv(c1, 9 *c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 9 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c1, k=3, s=3)
        self.CA = CAConv(c1, c2, kernel_size, stride)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))
        # g1 =  torch.cat([group1, group2, group3], dim=1)

        g2 = self.group_conv(x)

        out1 = g2 * group1.expand_as(g2)
        out2 = g2 * group2.expand_as(g2)
        out3 = g2 * group3.expand_as(g2)

        out = sum([out1, out2, out3])
        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = channels // 9

        # 重塑并转置特征图以将通道数分成3x3个子通道并扩展高度和宽度
        out = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
        out = self.convDown(out)
        out = self.CA(out)
        return out
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 48)
    b, c, h, w = x.shape
    net = GAM_Attention(in_channels=c, out_channels=c)
    y = net(x)

