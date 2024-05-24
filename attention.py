import torch.nn as nn
import torch
import numpy as np
from einops import rearrange

class P2I_CrossAttention(nn.Module):
    """
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_dim: 输入的通道数
        out_dim: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """

    def __init__(self, in_dim, out_dim):
        super(P2I_CrossAttention, self).__init__()
        self.query_conv = nn.Linear(in_dim, out_dim)
        self.key_conv = nn.Linear(in_dim, out_dim)
        self.value_conv = nn.Linear(in_dim, out_dim)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat0, feat1):
        '''
        :param feat0: query
        :param feat1: key & value
        :return: attentional value
        '''
        batch_size, C, width, height = feat0.size()
        _, C1, N = feat1.size()
        assert C == C1
        # size = B X (W * H) × C
        proj_query = self.query_conv(rearrange(feat0, 'b c w h -> b (w h) c'))

        # size = B X C x N
        proj_key = self.key_conv(rearrange(feat1, 'b c n -> b n c'))

        # [B, (W * H), N]
        energy = torch.bmm(proj_query, rearrange(proj_key, 'b n c -> b c n'))  # transpose check

        # 对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy) / C ** .5  # B X (W * H) X N

        proj_value = self.value_conv(rearrange(feat1, 'b c n -> b n c'))  # B X N X C
        out = self.out_conv(torch.bmm(attention, proj_value))  # B X (W * H) X C
        out = rearrange(out, 'b (w h) c -> b c w h', w=width, h=height) # B X C X W X H
        out = out + feat0 # B X C X W X H
        
        return out


class I2P_CrossAttention(nn.Module):
    """
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_dim: 输入的通道数
        out_dim: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """

    def __init__(self, in_dim, out_dim):
        super(I2P_CrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim
        self.query_conv = nn.Linear(in_dim, out_dim)
        self.key_conv = nn.Linear(in_dim, out_dim)
        self.value_conv = nn.Linear(in_dim, out_dim)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat0, feat1):
        '''
        :param feat0: query size:[B * C * N]
        :param feat1: key & value size:[B * C * W * H]
        :return: attentional value
        '''
        _, C1, N = feat0.size()
        batch_size, C, width, height = feat1.size()
        assert C1 == C
        # size = B x N x C
        proj_query = self.query_conv(rearrange(feat0, 'b c n -> b n c'))

        proj_key = self.key_conv(rearrange(feat1, 'b c w h -> b (w h) c'))
        # B X N x (W * H)
        energy = torch.bmm(proj_query, rearrange(proj_key, 'b wh c -> b c wh'))  # transpose check

        # 对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy) / C ** .5  # B X N X (W * H)

        proj_value = self.value_conv(rearrange(feat1, 'b c w h -> b (w h) c'))  # B X (W * H) X C
        out = self.out_conv(torch.bmm(attention, proj_value))  # B X N X C
        # out = out.view(batch_size, C, width, height)  # B X C X N
        out = rearrange(out, 'b n c -> b c n') + feat0
        # # 跨连，Gamma是需要学习的参数
        # out = self.gamma * out + x  # B X C X W X H
        return out


class SelfAttention(nn.Module):
    """
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_dim: 输入的通道数
        out_dim: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """

    def __init__(self, in_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        # proj_query中的第i行表示第i个像素位置上所有通道的值。size = B X N × C1 (4 * 400 * 4)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        # proj_key中的第j行表示第j个像素位置上所有通道的值，size = B X C1 x N (4, 4, 400)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # Energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j行点乘得到
        # energy中第(i,j)位置的元素是指输入特征图第j个元素对第i个元素的影响，
        # 从而实现全局上下文任意两个元素的依赖关系 (4, 400, 400)
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        # 对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy)  # B X N X N (4, 400, 400)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N (4, 16, 400)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B X C X N (4, 16, 400)
        out = out.view(m_batchsize, C, width, height)  # B X C X W X H (4, 16, 20, 20)

        # # 跨连，Gamma是需要学习的参数
        # out = self.gamma * out + x  # B X C X W X H
        return out, attention


if __name__ == '__main__':
    x = torch.randn(size=(4, 16, 20, 20))
    self_atten_spatial = SelfAttention(16, 4)
    y = self_atten_spatial(x)
    print('y.size:', y[0].size())

'''
y.size: torch.Size([4, 16, 20, 20])
'''
