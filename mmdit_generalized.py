from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import RMSNorm, FeedForward

from mmdit import JointAttention


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
    如果值存在，则返回该值；否则返回默认值。

    参数:
        v: 需要检查的可选值。
        d: 默认值。

    返回:
        Any: 如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


# adaptive layernorm
# aim for clarity in generalized version
# 自适应层归一化

class AdaptiveLayerNorm(Module):
    """
    自适应层归一化（AdaptiveLayerNorm）模块。
    该模块在标准层归一化（LayerNorm）的基础上，增加了基于条件信息的自适应参数调整。

    参数:
        dim (int): 输入数据的特征维度。
        dim_cond (int, 可选): 条件信息的特征维度。如果提供，则使用条件信息进行自适应调整。
    """
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        # 判断是否存在条件信息
        has_cond = exists(dim_cond)
        # 保存是否存在条件信息的状态
        self.has_cond = has_cond

        # 定义标准层归一化层
        # 如果存在条件信息，则不进行参数归一化（elementwise_affine=False）
        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
        if has_cond:
            # 定义一个线性层，将条件信息映射到 gamma 和 beta
            cond_linear = nn.Linear(dim_cond, dim * 2)

            # 定义序列处理层，包括重塑形状、SiLU 激活函数和线性变换
            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),  # 重塑张量形状
                nn.SiLU(),  # SiLU 激活函数
                cond_linear  # 线性变换
            )

            # 初始化线性层的权重为0
            nn.init.zeros_(cond_linear.weight)
            
            # 初始化线性层的偏置，前 dim 个值设为1，其余设为0
            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        """
        前向传播方法，执行自适应层归一化。

        参数:
            x (Tensor): 输入张量。
            cond (Optional[Tensor], 可选): 条件信息张量。如果 dim_cond 在初始化时被设置，则必须提供条件信息；否则，不应提供。

        返回:
            Tensor: 归一化后的张量。
        """
        # 确保条件信息的一致性
        assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        # 执行标准层归一化
        x = self.ln(x)

        if self.has_cond:
            # 将条件信息输入到条件处理层，并将输出拆分为 gamma 和 beta
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            # 应用 gamma 和 beta 进行自适应调整
            x = x * gamma + beta

        return x


# class

class MMDiTBlock(Module):
    """
    多模态多头注意力块（MMDiTBlock），用于处理多个模态（如文本和图像）的数据。

    参数:
        dim_joint_attn (int): 联合注意力层的维度。
        dim_modalities (Tuple[int, ...]): 每个模态的维度。
        dim_cond (int, 可选): 条件模态的维度。如果存在，则使用条件信息。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        qk_rmsnorm (bool, 可选): 是否对查询 (Q) 和键 (K) 进行 RMS 归一化。默认值为 False。
        flash_attn (bool, 可选): 是否使用 FlashAttention 优化注意力计算。默认值为 False。
        softclamp (bool, 可选): 是否对注意力分数进行软裁剪。默认值为 False。
        softclamp_value (float, 可选): 软裁剪的阈值。默认值为 50.0。
        ff_kwargs (dict, 可选): 传递给前馈网络（FeedForward）的其他关键字参数。默认值为空字典。
    """
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_modalities: Tuple[int, ...],
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        softclamp = False,
        softclamp_value = 50.,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        # 计算模态的数量
        self.num_modalities = len(dim_modalities)
        # 保存每个模态的维度
        self.dim_modalities = dim_modalities

        # handle optional time conditioning
        # 处理可选的时间条件信息

        # 判断是否存在条件信息
        has_cond = exists(dim_cond)
        # 保存是否存在条件信息的状态
        self.has_cond = has_cond

        if has_cond:
            # 定义线性层，将条件信息映射到每个模态的 gamma 和 beta
            cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)

            # 定义序列处理层，包括重塑形状、SiLU 激活函数和线性变换
            self.to_post_branch_gammas = nn.Sequential(
                Rearrange('b d -> b 1 d'), # 重塑张量形状
                nn.SiLU(), # SiLU 激活函数
                cond_linear # 线性变换
            )

            # 初始化线性层的权重和偏置
            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias, 1.)

        # joint modality attention
        # 联合模态注意力

        # 为每个模态定义自适应层归一化层
        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.attn_layernorms = ModuleList(attention_layernorms)

        # 定义联合注意力层
        self.joint_attn = JointAttention(
            dim = dim_joint_attn,  # 联合注意力层的维度
            dim_inputs = dim_modalities,  # 输入模态的维度
            dim_head = dim_head,  # 每个注意力头的维度
            heads = heads,  # 注意力头的数量
            flash = flash_attn,  # 是否使用 FlashAttention 优化
            softclamp = softclamp,  # 是否对注意力分数进行软裁剪
            softclamp_value = softclamp_value,  # 软裁剪的阈值
        )

        # feedforwards
        # 前馈网络

        # 为每个模态定义自适应层归一化层
        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        # 定义前馈网络
        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):
        """
        前向传播方法，用于处理多个模态的输入。

        参数:
            modality_tokens (Tuple[Tensor, ...]): 输入的多个模态的标记张量。
            modality_masks (Tuple[Tensor | None, ...] | None, 可选): 输入的多个模态的掩码张量或 None。
            time_cond (Tensor, 可选): 时间条件张量，用于条件信息。

        返回:
            Tuple[Tensor, ...]: 处理后的多个模态的标记张量。
        """
        # 确保输入的模态数量与初始化时一致
        assert len(modality_tokens) == self.num_modalities
        # 确保条件信息的一致性
        assert not (exists(time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        # 初始化层归一化参数
        ln_kwargs = dict()

        if self.has_cond:
            # 如果存在条件信息，则将其作为层归一化的参数
            ln_kwargs = dict(cond = time_cond)

            # 将时间条件信息输入到条件处理层
            gammas = self.to_post_branch_gammas(time_cond)
            # 将输出拆分为注意力 gamma 和前馈网络 gamma
            attn_gammas, ff_gammas = gammas.chunk(2, dim = -1)

        # attention layernorms
        # 注意力层的层归一化

        # 保存输入的模态标记作为残差
        modality_tokens_residual = modality_tokens
        # 对每个模态的标记进行层归一化
        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

        # attention
        # 注意力计算

        # 进行联合注意力计算
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # post attention gammas
        # 后注意力 gamma 处理

        if self.has_cond:
            # 将注意力 gamma 拆分为每个模态的 gamma
            attn_gammas = attn_gammas.split(self.dim_modalities, dim = -1)
            # 应用注意力 gamma 进行缩放
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, attn_gammas)]

        # add attention residual
        # 添加注意力残差

        # 添加残差
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # handle feedforward adaptive layernorm
        # 处理前馈网络的自适应层归一化

        # 保存前馈网络的输入作为残差
        modality_tokens_residual = modality_tokens

        # 对每个模态的标记进行层归一化
        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

        # 进行前馈网络计算
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas
        # 后前馈网络 gamma 处理

        if self.has_cond:
            # 将前馈网络 gamma 拆分为每个模态的 gamma
            ff_gammas = ff_gammas.split(self.dim_modalities, dim = -1)
            # 应用前馈网络 gamma 进行缩放
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, ff_gammas)]

        # add feedforward residual
        # 添加前馈网络残差

        # 添加残差
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # 返回处理后的模态标记
        return modality_tokens


# mm dit transformer - simply many blocks

class MMDiT(Module):
    """
    多模态多头注意力Transformer（MMDiT），由多个MMDiTBlock组成，用于处理多个模态（如文本和图像）的数据。

    参数:
        depth (int): Transformer的深度，即MMDiTBlock的数量。
        dim_modalities (Tuple[int, ...]): 每个模态的维度。
        final_norms (bool, 可选): 是否在最后应用归一化层。默认值为True。
        **block_kwargs: 传递给MMDiTBlock的其他关键字参数。
    """
    def __init__(
        self,
        *,
        depth,
        dim_modalities,
        final_norms = True,
        **block_kwargs
    ):
        super().__init__()
        # 创建多个MMDiTBlock，并添加到模块列表中
        blocks = [MMDiTBlock(dim_modalities = dim_modalities, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        # 为每个模态创建归一化层，并添加到模块列表中
        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):
        """
        前向传播方法，用于处理多个模态的输入。

        参数:
            modality_tokens (Tuple[Tensor, ...]): 输入的多个模态的标记张量。
            modality_masks (Tuple[Tensor | None, ...] | None, 可选): 输入的多个模态的掩码张量或None。
            time_cond (Tensor, 可选): 时间条件张量，用于条件信息。

        返回:
            Tuple[Tensor, ...]: 处理后的多个模态的标记张量。
        """
        # 遍历所有MMDiTBlock，对每个块进行前向传播
        for block in self.blocks:
            modality_tokens = block(
                time_cond = time_cond,
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        # 对每个模态的标记应用归一化层
        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        # 返回处理后的多个模态的标记张量
        return tuple(modality_tokens)
