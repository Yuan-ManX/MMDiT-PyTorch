from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, reduce, rearrange
from einops.layers.torch import Rearrange


# helper functions

def exists(v):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        v: 需要检查的值。

    返回:
        bool: 如果值存在（不为 None），则返回 True；否则返回 False。
    """
    return v is not None


def softclamp(t, value):
    """
    对张量进行软裁剪（soft clamping），确保张量的值不超过指定的值。

    参数:
        t (Tensor): 输入张量。
        value (float): 裁剪的上限值。

    返回:
        Tensor: 裁剪后的张量。
    """
    return (t / value).tanh() * value


# 使用偏函数定义一个不带偏置参数的线性层
Linear = partial(nn.Linear, bias = False)


# class

class AdaptiveAttention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        num_adaptive_weights = 1, # num_adaptive_weights 为 1 时，相当于常规的自注意力，无门控
        softclamp = False,
        softclamp_value = 50.,
    ):
        """
        this idea was inspired by adaptive convs from gigagan https://arxiv.org/abs/2303.05511

        ein notation:
        b - batch
        n - sequence
        h - heads
        d - feature dimension
        w - adaptive weight
        """
        """
        自适应注意力模块，灵感来源于 GigaGAN 中的自适应卷积。

        参数:
            dim (int): 输入数据的维度。
            dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
            heads (int, 可选): 注意力头的数量。默认值为 8。
            num_adaptive_weights (int, 可选): 自适应权重的数量。默认值为 1（相当于常规的自注意力，无门控）。
            softclamp (bool, 可选): 是否使用软裁剪。默认值为 False。
            softclamp_value (float, 可选): 软裁剪的值。默认值为 50.0。
        """

        super().__init__()
        assert num_adaptive_weights >= 1

        # 判断是否使用门控
        has_gating = num_adaptive_weights > 1
        self.has_gating = has_gating
        self.num_adaptive_weights = num_adaptive_weights

        # 计算内部维度
        dim_inner = dim_head * heads
        # 计算缩放因子
        scale = dim_head ** -0.5

        self.scale = scale
        self.softclamp = softclamp
        self.softclamp_value = softclamp_value

        # 定义查询、键和值（QKV）的线性变换层
        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * num_adaptive_weights * 3), # 线性变换
            Rearrange('b n (qkv h d w) -> qkv b h n d w', qkv = 3, h = heads, w = num_adaptive_weights) # 重塑张量形状
        )

        if has_gating:
            # 如果使用门控，定义门控线性变换层
            self.to_gates = nn.Sequential(
                Linear(dim, num_adaptive_weights * heads),
                Rearrange('b n (h w) -> b h n 1 w', w = num_adaptive_weights),
                nn.Softmax(dim = -1)
            )
        
        # 定义输出权重的可学习参数
        self.to_out_weights = nn.Parameter(torch.randn(heads, dim_head, dim * num_adaptive_weights))

    def forward(
        self,
        x,
        mask = None
    ):
        """
        前向传播函数，实现自适应注意力机制。

        参数:
            x (Tensor): 输入张量。
            mask (Optional[Tensor], 可选): 输入的掩码张量。默认值为 None。

        返回:
            Tensor: 输出张量。
        """
        # 判断是否使用门控
        has_gating = self.has_gating

        # 计算查询、键和值（QKV）
        qkv = self.to_qkv(x)

        # token dependent choosing of which weight
        # 根据输入张量选择不同的权重
        if has_gating:
            # 计算门控权重
            gates = self.to_gates(x)
            # 应用门控权重
            qkv = reduce(qkv * gates, '... w -> ...', 'sum')
        else:
            # 如果不使用门控，重塑张量形状
            qkv = rearrange(qkv, '... 1 -> ...')

        # usual self attention logic
        # 常规的自注意力逻辑
        # 解包查询、键和值
        q, k, v = qkv

        # 缩放查询
        q = q * self.scale
        # 计算相似度矩阵
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if self.softclamp:
            # 应用软裁剪
            sim = softclamp(sim, self.softclamp_value)

        if exists(mask):
            # 重塑掩码张量形状
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim = -1)

        # 计算输出
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # again, adaptive weight on the outward projection
        # with gates from above
        # 再次应用自适应权重到输出投影
        # 应用输出权重
        out = einsum(out, self.to_out_weights, 'b h n d, h d e -> b h n e')

        if has_gating:
            # 重塑输出张量形状
            out = rearrange(out, '... (d w) -> ... d w', w = self.num_adaptive_weights)
            # 应用门控权重
            out = reduce(out * gates, '... w -> ...', 'sum')
        else:
            # 重塑输出张量形状
            out = rearrange(out, 'b h n d -> b n (h d)')

        # 返回输出张量
        return out



# MMDiT for image、audio and text
# 多模态：图像、音频和文本

if __name__ == '__main__':

    # 创建一个自适应注意力模块实例
    adaptive_attn = AdaptiveAttention(
        dim = 512,                # 输入数据的维度
        num_adaptive_weights = 4  # 自适应权重的数量
    )

    # 生成随机文本标记，形状为 (1, 256, 512)
    text_tokens = torch.randn(1, 256, 512)

    # 生成随机图像标记，形状为 (1, 1024, 512)
    image_tokens = torch.randn(1, 1024, 512)

    # 生成随机音频标记，形状为 (1, 128, 512)
    audio_tokens = torch.randn(1, 128, 512)

    # 将文本、图像和音频标记在倒数第二个维度上进行拼接
    # 拼接后的形状为 (1, 1408, 512)
    tokens = torch.cat((text_tokens, image_tokens, audio_tokens), dim = -2)

    # 将拼接后的标记输入到自适应注意力模块中
    out = adaptive_attn(tokens)

    # 输出结果
    print(out.shape)  # 输出形状为 (1, 1408, 512)
