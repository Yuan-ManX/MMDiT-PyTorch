from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import RMSNorm, FeedForward


# helpers

def exists(v):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        v: 需要检查的值。

    返回:
        bool: 如果值存在（不为 None），则返回 True；否则返回 False。
    """
    return v is not None


def default(v, d):
    """
    如果值存在，则返回该值；否则，返回默认值。

    参数:
        v: 需要检查的可选值。
        d: 默认值。

    返回:
        Any: 如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


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


# rmsnorm
# RMSNorm 实现

class MultiHeadRMSNorm(Module):
    """
    多头RMS归一化层。
    RMSNorm 是一种归一化方法，它通过计算输入张量的均方根值来进行归一化。
    这里实现的是多头版本，每个头有独立的归一化参数。

    参数:
        dim (int): 输入数据的维度。
        heads (int, 可选): 注意力头的数量。默认值为 1。
    """
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5  # 计算缩放因子，通常为维度的平方根
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))  # 初始化可学习的参数 gamma，形状为 (heads, 1, dim)


    def forward(self, x):
        """
        前向传播函数，执行多头 RMS 归一化。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, ..., dim)。

        返回:
            Tensor: 归一化后的张量。
        """
        # 对输入张量进行归一化
        # F.normalize 函数对最后一个维度（即 dim 维度）进行归一化，使其均值为 0，方差为 1
        # 应用缩放因子 gamma 和 scale
        # gamma 的形状为 (heads, 1, dim)，因此会在 heads 和 dim 维度上进行广播
        # 返回最终的归一化结果
        return F.normalize(x, dim = -1) * self.gamma * self.scale


# attention
# 注意力模块

class JointAttention(Module):
    """
    联合注意力模块，用于处理多个输入序列的注意力计算。

    参数:
        dim (int): 输出的特征维度。
        dim_inputs (Tuple[int, ...]): 每个输入序列的维度。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        qk_rmsnorm (bool, 可选): 是否对查询 (Q) 和键 (K) 进行 RMS 归一化。默认值为 False。
        flash (bool, 可选): 是否使用 FlashAttention 优化注意力计算。默认值为 False。
        softclamp (bool, 可选): 是否对注意力分数进行软裁剪。默认值为 False。
        softclamp_value (float, 可选): 软裁剪的阈值。默认值为 50.0。
        attend_kwargs (dict, 可选): 传递给 Attend 模块的其他关键字参数。默认值为空字典。
    """
    def __init__(
        self,
        *,
        dim,
        dim_inputs: Tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """
        """ 
        einstein notation 解释:
        
        b - batch (批次)
        h - heads (注意力头)
        n - sequence (序列)
        d - feature dimension (特征维度)
        """

        # 计算内部维度，即每个头的总维度
        dim_inner = dim_head * heads

        # 获取输入序列的数量
        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        # 为每个输入序列创建一个线性变换层，用于生成查询 (Q)、键 (K) 和值 (V)
        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        # 重塑张量形状，将 Q、K、V 分离到不同的维度
        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        # 创建 Attend 模块，用于计算注意力
        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        # 重塑张量形状，将多头注意力结果合并
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 为每个输入序列创建一个线性变换层，用于生成输出
        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        # 是否对 Q 和 K 进行 RMS 归一化
        self.qk_rmsnorm = qk_rmsnorm
        # 初始化 Q 的 RMS 归一化层
        self.q_rmsnorms = (None,) * num_inputs
        # 初始化 K 的 RMS 归一化层
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            # 如果需要对 Q 和 K 进行 RMS 归一化，则创建相应的归一化层
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        # 注册一个虚拟缓冲区，用于存储一个常量张量，不会在模型保存时被保存
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: Tuple[Tensor], # 输入的元组，包含多个输入张量
        masks: Tuple[Tensor | None] | None = None  # 输入的掩码元组，包含多个掩码张量或 None
    ):

        device = self.dummy.device

        # 确保输入的数量与初始化时的数量一致
        assert len(inputs) == self.num_inputs

        # 如果没有提供掩码，则为每个输入分配一个 None
        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens
        # 对每个模态（modality）分别进行 QKV 投影
        # 同时处理掩码，假设 None 表示关注所有 token

        # 用于存储所有模态的 QKV
        all_qkvs = []
        # 用于存储所有模态的掩码
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):
            # 对每个输入张量 x 进行 QKV 线性变换
            qkv = to_qkv(x)
            # 重塑张量形状，将 QKV 分离到不同的维度
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality
            # 可选的 QK RMS 归一化，每个模态分别进行
            if self.qk_rmsnorm:
                q, k, v = qkv
                # 对 Q 和 K 进行 RMS 归一化
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                # 重新堆叠 QKV
                qkv = torch.stack((q, k, v))
            # 将处理后的 QKV 添加到列表中
            all_qkvs.append(qkv)

            # handle mask per modality
            # 处理每个模态的掩码
            if not exists(mask):
                # 如果没有提供掩码，则创建一个全为 True 的布尔张量，表示关注所有 token
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)
            # 将处理后的掩码添加到列表中
            all_masks.append(mask)

        # combine all qkv and masks
        # 合并所有 QKV 和掩码
        # 打包 QKV
        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        # 打包掩码
        all_masks, _ = pack(all_masks, 'b *')

        # attention
        # 注意力计算
        # 解包 QKV
        q, k, v = all_qkvs
        # 调用 Attend 模块进行注意力计算
        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection
        # 合并多头注意力结果，然后根据模态分离出来进行组合头部的投影
        # 合并多头注意力结果
        outs = self.merge_heads(outs)
        # 拆包
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality
        # 对每个模态分别进行组合头部的投影
        # 用于存储每个模态的输出
        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out) # 对输出进行线性变换
            all_outs.append(out) # 将处理后的输出添加到列表中

        # 返回所有模态的输出
        return tuple(all_outs)


# class

class MMDiTBlock(Module):
    """
    多模态多头注意力块（MMDiTBlock），用于处理文本和图像的多模态数据。

    参数:
        dim_joint_attn (int): 联合注意力层的维度。
        dim_text (int): 文本模态的维度。
        dim_image (int): 图像模态的维度。
        dim_cond (int, 可选): 条件模态的维度。如果存在，则使用条件信息。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        qk_rmsnorm (bool, 可选): 是否对查询 (Q) 和键 (K) 进行 RMS 归一化。默认值为 False。
        flash_attn (bool, 可选): 是否使用 FlashAttention 优化注意力计算。默认值为 False。
        ff_kwargs (dict, 可选): 传递给前馈网络（FeedForward）的其他关键字参数。默认值为空字典。
    """
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_text,
        dim_image,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # handle optional time conditioning
        # 处理可选的时间条件信息

        # 判断是否存在条件信息
        has_cond = exists(dim_cond)
        # 保存是否存在条件信息的状态
        self.has_cond = has_cond

        if has_cond:
            # 定义 gamma 和 beta 的维度
            dim_gammas = (
                *((dim_text,) * 4), # 文本模态的 gamma 维度
                *((dim_image,) * 4) # 图像模态的 gamma 维度
            )

            dim_betas = (
                *((dim_text,) * 2),  # 文本模态的 beta 维度
                *((dim_image,) * 2), # 图像模态的 beta 维度
            )

            # 组合 gamma 和 beta 的维度
            self.cond_dims = (*dim_gammas, *dim_betas)

            # 定义线性变换层，将条件信息映射到 gamma 和 beta
            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            # 定义序列处理层，包括重塑形状、SiLU 激活函数和线性变换
            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'), # 重塑张量形状
                nn.SiLU(), # SiLU 激活函数
                to_cond_linear # 线性变换
            )

            # 初始化线性变换层的权重和偏置
            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms
        # 处理自适应归一化层
        
        # 定义文本模态的自适应归一化层
        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        # 定义文本模态的前馈网络的自适应归一化层
        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        # attention and feedforward
        # 定义注意力机制和前馈网络

        # 定义联合注意力层
        self.joint_attn = JointAttention(
            dim = dim_joint_attn,  # 联合注意力层的维度
            dim_inputs = (dim_text, dim_image),  # 输入模态的维度
            dim_head = dim_head,  # 每个注意力头的维度
            heads = heads,  # 注意力头的数量
            flash = flash_attn  # 是否使用 FlashAttention 优化
        )

        # 定义文本模态的前馈网络
        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        # 定义图像模态的前馈网络
        self.image_ff = FeedForward(dim_image, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        skip_feedforward_text_tokens = True
    ):
        """
        前向传播方法，用于处理文本和图像的多模态输入。

        参数:
            text_tokens (Tensor): 文本输入张量。
            image_tokens (Tensor): 图像输入张量。
            text_mask (Tensor, 可选): 文本掩码张量，用于掩码注意力机制。
            time_cond (Tensor, 可选): 时间条件张量，用于条件信息。
            skip_feedforward_text_tokens (bool, 可选): 是否跳过文本前馈网络。默认值为 True。
        """
        # 确保条件信息的一致性
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        # 如果存在条件信息，则进行条件信息的处理
        if self.has_cond:
            # 将时间条件信息输入到条件处理层
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                image_pre_attn_beta,
                image_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm
        # 处理注意力机制的自适应层归一化
        # 保存文本和图像输入的残差
        text_tokens_residual = text_tokens
        image_tokens_residual = image_tokens

        # 对文本和图像输入进行层归一化
        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        # 如果存在条件信息，则对文本和图像输入进行条件缩放和偏置
        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

        # attention
        # 进行联合注意力计算
        text_tokens, image_tokens = self.joint_attn(
            inputs = (text_tokens, image_tokens),  # 输入文本和图像张量
            masks = (text_mask, None)  # 文本掩码和图像掩码（图像掩码为 None）
        )

        # condition attention output
        # 对注意力机制的输出进行条件处理

        if self.has_cond:
            # 文本注意力输出进行条件缩放
            text_tokens = text_tokens * text_post_attn_gamma  
            # 图像注意力输出进行条件缩放
            image_tokens = image_tokens * image_post_attn_gamma  

        # add attention residual
        # 添加注意力机制的残差

        # 文本注意力输出添加残差
        text_tokens = text_tokens + text_tokens_residual  
        # 图像注意力输出添加残差
        image_tokens = image_tokens + image_tokens_residual  

        # handle feedforward adaptive layernorm
        # 处理前馈网络的自适应层归一化

        # 保存文本前馈网络的残差
        text_tokens_residual = text_tokens  
        # 保存图像前馈网络的残差
        image_tokens_residual = image_tokens  

        # 对文本注意力输出进行层归一化
        text_tokens = self.text_attn_layernorm(text_tokens)
        # 对图像注意力输出进行层归一化
        image_tokens = self.image_attn_layernorm(image_tokens)

        # 如果存在条件信息，则对文本和图像输入进行条件缩放和偏置
        if self.has_cond:
            # 文本前馈输入进行条件缩放和偏置
            text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta
            # 图像前馈输入进行条件缩放和偏置
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

        # images feedforward
        # 进行图像前馈网络计算

        # 图像前馈网络处理
        image_tokens = self.image_ff(image_tokens)

        # images condition feedforward output
        # 对图像前馈网络的输出进行条件处理

        if self.has_cond:
            # 图像前馈输出进行条件缩放
            image_tokens = image_tokens * image_post_ff_gamma

        # images feedforward residual
        # 添加图像前馈网络的残差

        # 图像前馈输出添加残差
        image_tokens = image_tokens + image_tokens_residual

        # early return, for last block in mmdit
        # 如果需要跳过文本前馈网络，则提前返回
        if skip_feedforward_text_tokens:
            return text_tokens, image_tokens

        # text feedforward
        # 进行文本前馈网络计算

        # 文本前馈网络处理
        text_tokens = self.text_ff(text_tokens)

        # text condition feedforward output
        # 对文本前馈网络的输出进行条件处理
        if self.has_cond:
            # 文本前馈输出进行条件缩放
            text_tokens = text_tokens * text_post_ff_gamma

        # text feedforward residual
        # 添加文本前馈网络的残差

        # 文本前馈输出添加残差
        text_tokens = text_tokens + text_tokens_residual

        # 返回处理后的文本和图像张量
        return text_tokens, image_tokens


# MMDiT Transformer  - simply many blocks

class MMDiT(Module):
    """
    多模态多头注意力Transformer（MMDiT），由多个MMDiTBlock组成，用于处理文本和图像的多模态数据。

    参数:
        depth (int): Transformer的深度，即MMDiTBlock的数量。
        dim_image (int): 图像模态的维度。
        num_register_tokens (int, 可选): 注册标记的数量。默认值为0。
        final_norm (bool, 可选): 是否在最后应用归一化层。默认值为True。
        **block_kwargs: 传递给MMDiTBlock的其他关键字参数。
    """
    def __init__(
        self,
        *,
        depth,
        dim_image,
        num_register_tokens = 0,
        final_norm = True,
        **block_kwargs
    ):
        super().__init__()
        # 判断是否存在注册标记
        self.has_register_tokens = num_register_tokens > 0
        # 定义注册标记参数，并初始化为标准正态分布
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_image))
        nn.init.normal_(self.register_tokens, std = 0.02)

        # 定义一个模块列表，用于存储MMDiTBlock
        self.blocks = ModuleList([])

        for _ in range(depth):
            # 创建MMDiTBlock，并添加到模块列表中
            block = MMDiTBlock(
                dim_image = dim_image,
                **block_kwargs
            )

            self.blocks.append(block)

        # 定义归一化层，如果final_norm为True，则使用RMSNorm，否则使用恒等变换
        self.norm = RMSNorm(dim_image) if final_norm else nn.Identity()

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        should_skip_last_feedforward = True
    ):
        """
        前向传播方法，用于处理文本和图像的多模态输入。

        参数:
            text_tokens (Tensor): 文本输入张量。
            image_tokens (Tensor): 图像输入张量。
            text_mask (Tensor, 可选): 文本掩码张量，用于掩码注意力机制。
            time_cond (Tensor, 可选): 时间条件张量，用于条件信息。
            should_skip_last_feedforward (bool, 可选): 是否跳过最后一个块的文本前馈网络。默认值为True。
        """
        # 如果存在注册tokens，则将注册tokens与图像tokens合并
        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = image_tokens.shape[0])
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')

        # 遍历所有MMDiTBlock
        for ind, block in enumerate(self.blocks):
            # 判断是否是最后一个块
            is_last = ind == (len(self.blocks) - 1)

            # 调用MMDiTBlock的前向传播方法
            text_tokens, image_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                text_mask = text_mask,
                skip_feedforward_text_tokens = is_last and should_skip_last_feedforward
            )

        # 如果存在注册tokens，则将注册tokens从图像tokens中分离出来
        if self.has_register_tokens:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        # 对图像tokens应用归一化层
        image_tokens = self.norm(image_tokens)

        # 返回处理后的文本和图像tokens
        return text_tokens, image_tokens
