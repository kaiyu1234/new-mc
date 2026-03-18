#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.nn import functional as F
import time
from typing import Union, Optional  # <--- 添加 Optional
import os                            # <--- 添加 import os

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore
import json


class MCMark_WatermarkCode(AbstractWatermarkCode):
    # [FIX] 添加一个类变量作为缓存
    _cached_shuffle = None
    _cached_path = None

    def __init__(self, shuffle: LongTensor, split_k: BoolTensor):
        self.shuffle = shuffle
        self.split_k = split_k
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, list[torch.Generator]],
        vocab_size: int,
        split_num: int,
        shuffle_file_path: Optional[str] = None
    ):
        # 1. 尝试设置默认路径
        if shuffle_file_path is None and os.path.exists("semantic_shuffle_opt.pt"):
            shuffle_file_path = "semantic_shuffle_opt.pt"

        # 2. 检查缓存：如果路径没变且缓存已有，直接使用缓存
        loaded_shuffle = None
        if shuffle_file_path:
            if cls._cached_shuffle is not None and cls._cached_path == shuffle_file_path:
                # 命中缓存！直接使用，不打印日志，不读硬盘
                loaded_shuffle = cls._cached_shuffle
            elif os.path.exists(shuffle_file_path):
                # 未命中缓存，从硬盘加载
                try:
                    print(f"[MCMark] Loading semantic shuffle from {shuffle_file_path}...") # 这句话只会出现一次了
                    loaded_shuffle = torch.load(shuffle_file_path, map_location="cpu")
                    if len(loaded_shuffle) != vocab_size:
                        print(f"[MCMark] Warning: Size mismatch. Random fallback.")
                        loaded_shuffle = None
                    else:
                        # 存入缓存
                        cls._cached_shuffle = loaded_shuffle
                        cls._cached_path = shuffle_file_path
                except Exception as e:
                    print(f"[MCMark] Error loading file: {e}")
                    loaded_shuffle = None

        # 3. 构造 Shuffle Tensor (这部分保持原样)
        if isinstance(rng, list):
            batch_size = len(rng)
            if loaded_shuffle is not None:
                # 使用缓存的 shuffle，扩展到 batch 维度
                shuffle = loaded_shuffle.to(rng[0].device).unsqueeze(0).expand(batch_size, -1)
            else:
                shuffle = torch.stack(
                    [
                        torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                        for i in range(batch_size)
                    ]
                )
            
            split_k = torch.cat(
                [
                    torch.randint(
                        low=0,
                        high=split_num,
                        size=(1,),
                        dtype=torch.long,
                        generator=rng[i],
                        device=rng[i].device,
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            if loaded_shuffle is not None:
                shuffle = loaded_shuffle.to(rng.device)
            else:
                shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
                
            split_k = torch.randint(
                low=0,
                high=split_num,
                size=(1,),
                dtype=torch.long,
                device=rng.device,
                generator=rng,
            )
            
        return cls(shuffle, split_k)

    # @classmethod
    # def from_random(
    #     cls,
    #     rng: Union[torch.Generator, list[torch.Generator]],
    #     vocab_size: int,
    #     split_num: int,
    # ):
    #     if isinstance(rng, list):
    #         batch_size = len(rng)
    #         shuffle = torch.stack(
    #             [
    #                 torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
    #                 for i in range(batch_size)
    #             ]
    #         )
    #         split_k = torch.cat(
    #             [
    #                 torch.randint(
    #                     low=0,
    #                     high=split_num,
    #                     size=(1,),
    #                     dtype=torch.long,
    #                     generator=rng[i],
    #                     device=rng[i].device,
    #                 )
    #                 for i in range(batch_size)
    #             ],
    #             dim=0,
    #         )
    #     else:
    #         shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
    #         split_k = torch.randint(
    #             low=0,
    #             high=split_num,
    #             size=(1,),
    #             dtype=torch.long,
    #             device=rng.device,
    #             generator=rng,
    #         )
    #     return cls(shuffle, split_k)

        


# class MC_Reweight(AbstractReweight):
#     watermark_code_type = MCMark_WatermarkCode

#     def __init__(self, n: float):
#         self.n = n

#     def __repr__(self):

#         return f"MC_Reweight(n={self.n})"

#     def reweight_logits(
#         self, code: AbstractWatermarkCode, p_logits: FloatTensor
#     ) -> FloatTensor:
#         """
#         \textbf{$\gamma$-reweight:}
#         Let the watermark code space $\mathcal{E}$ be the set of all bijective function between symbol set $\Sigma$ and a set of number $[\abs{\Sigma}]=\{1,\dots,\abs{\Sigma}\}$, where $\abs{\Sigma}$ is the size of symbol set $\Sigma$.
#         Essentially, any watermark code $E$ is an indexing function for symbol set $\Sigma$, and also assign an order on $\Sigma$. Let $P_E$ be the uniform probability on $\mathcal{E}$, it would be easy to sample a watermark code $E$ by randomly shuffle the symbol list.

#         Assume the original distribution is $P_T(t)\in\Delta_\Sigma,\forall t\in\Sigma$.
#         We interpret watermark code $E:\Sigma\to[\abs{\Sigma}]$ as a indexing function and we introduce parameter $\gamma$ to control the strength of watermark.
#         % Use the hash of
#         % $E$ as a pseudorandom number seed and sample a random permutation $\sigma:\Sigma\to N$.
#         Then we construct auxiliary functions
#         % $F_I(i)=P_{t\sim P_T}(E(t)\leq i),$
#         $F_I(i)=\sum_{t\in\Sigma} \mathbf{1}(E(t)\leq i) P_T(t),$
#         $F_S(s)=\begin{cases}(1-\gamma)s & s\leq\frac{1}{2}\\-\gamma+(1+\gamma)s ~~~& s>\frac{1}{2}\end{cases},$
#         $F_{I'}(i)=F_S(F_I(i)).$
#         The new distribution is given by $P_{T'}(t)=F_{I'}(E(t))-F_{I'}(E(t)-1)$.
#         """

#         def set_nan_to_zero(x):
#             x[torch.isnan(x)] = 0
#             return x

#         start = time.time()
#         # s_ means shuffled
#         s_logits = torch.gather(p_logits, -1, code.shuffle)
#         s_probs = torch.softmax(s_logits, dim=-1)
#         bsz, vocab_size = s_logits.shape

#         splits = []
#         if self.n == vocab_size:
#             splits = [[i] for i in range(self.n)]
#         elif vocab_size % self.n == 0:
#             splits = (
#                 torch.arange(start=0, end=vocab_size)
#                 .reshape(self.n, vocab_size // self.n)
#                 .to(p_logits.device)
#             )
#         else:
#             for n_idx in range(self.n):
#                 splits.append(
#                     list(
#                         range(
#                             round(vocab_size * n_idx / self.n),
#                             round(vocab_size * (n_idx + 1) / self.n),
#                         )
#                     )
#                 )

#         split_k = code.split_k.to(s_logits.device)

#         split_sums = []
#         if self.n == vocab_size:
#             split_sums = s_probs
#         elif vocab_size % self.n == 0:
#             split_sums = s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
#         else:
#             for n_idx in range(self.n):
#                 cur_split = splits[n_idx]
#                 split_sums.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))

#             split_sums = torch.cat(split_sums, dim=-1)  # [bsz,n]
#         scales = torch.minimum(
#             self.n * torch.ones_like(split_sums).to(s_probs.device), 1 / split_sums
#         )  # [bsz,n]

#         overflow_scales = (
#             self.n * split_sums - 1
#         ) / split_sums  # [bsz,n] note: might be negative or nan
#         overflow_scales = set_nan_to_zero(overflow_scales)
#         overflow_scales[overflow_scales < 0] = 0  # [bsz,n]

#         target_scales = scales[range(bsz), split_k]  # [bsz]
#         target_sums = split_sums[range(bsz), split_k]  # [bsz]

#         remain_sums = 1 - target_scales * target_sums  # [bsz]
#         overflow_sums = (overflow_scales * split_sums).sum(dim=-1)  # [bsz]
#         fill_scale = remain_sums / overflow_sums  # [bsz]
#         fill_scale = set_nan_to_zero(fill_scale)  # [bsz]

#         split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(
#             bsz, 1
#         ) == split_k.view(-1, 1).repeat(1, self.n)
#         final_scale = torch.where(
#             split_mask,
#             target_scales.view(-1, 1).repeat(1, self.n),
#             fill_scale.view(-1, 1) * overflow_scales,
#         )  # [bsz,n]

#         reweighted_s_probs = torch.zeros_like(s_probs).to(s_logits.device)

#         if self.n == vocab_size:
#             reweighted_s_probs = final_scale * s_probs
#         elif vocab_size % self.n == 0:
#             reweighted_s_probs = (
#                 final_scale.view(bsz, self.n, 1)
#                 .expand((-1, -1, vocab_size // self.n))
#                 .reshape(bsz, vocab_size)
#                 * s_probs
#             )
#         else:
#             for n_idx in range(self.n):
#                 cur_split = splits[n_idx]
#                 reweighted_s_probs[:, cur_split] = (
#                     final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split]
#                 )

#         reweighted_s_probs[reweighted_s_probs < 0] = 0

#         reweighted_s_logits = torch.log(reweighted_s_probs)
#         reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

#         return reweighted_logits


# class MC_Reweight(AbstractReweight):
#     watermark_code_type = MCMark_WatermarkCode

#     # [NEW] 在初始化中引入 entropy_threshold (熵阈值)
#     # 当熵大于该阈值时，施加 100% 的水印强度；越小则强度越弱。
#     # 阈值通常可以设为 3.0 ~ 5.0，视词表大小和模型而定。
#     def __init__(self, n: float, entropy_threshold: float = 4.0):
#         self.n = n
#         self.entropy_threshold = entropy_threshold

#     def __repr__(self):
#         return f"MC_Reweight(n={self.n}, entropy_threshold={self.entropy_threshold})"

#     def reweight_logits(
#         self, code: AbstractWatermarkCode, p_logits: FloatTensor
#     ) -> FloatTensor:
#         def set_nan_to_zero(x):
#             x[torch.isnan(x)] = 0
#             return x

#         start = time.time()
#         # s_ means shuffled
#         s_logits = torch.gather(p_logits, -1, code.shuffle)
#         s_probs = torch.softmax(s_logits, dim=-1)
#         bsz, vocab_size = s_logits.shape

#         # ==============================================================
#         # [NEW] 1. 计算当前预测分布的香农熵 (Shannon Entropy)
#         # ==============================================================
#         # 为了防止 log(0) 导致 NaN，加上 1e-9
#         entropy = -torch.sum(s_probs * torch.log(s_probs + 1e-9), dim=-1) # shape: [bsz]
        
#         # 计算动态强度系数 alpha
#         # 截断在 [0, 1] 之间。熵越小，alpha 越接近 0；熵大于阈值，alpha 为 1。
#         alpha = torch.clamp(entropy / self.entropy_threshold, min=0.0, max=1.0).view(bsz, 1) # shape: [bsz, 1]
#         # ==============================================================

#         splits = []
#         if self.n == vocab_size:
#             splits = [[i] for i in range(self.n)]
#         elif vocab_size % self.n == 0:
#             splits = (
#                 torch.arange(start=0, end=vocab_size)
#                 .reshape(self.n, vocab_size // self.n)
#                 .to(p_logits.device)
#             )
#         else:
#             for n_idx in range(self.n):
#                 splits.append(
#                     list(
#                         range(
#                             round(vocab_size * n_idx / self.n),
#                             round(vocab_size * (n_idx + 1) / self.n),
#                         )
#                     )
#                 )

#         split_k = code.split_k.to(s_logits.device)

#         split_sums = []
#         if self.n == vocab_size:
#             split_sums = s_probs
#         elif vocab_size % self.n == 0:
#             split_sums = s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
#         else:
#             for n_idx in range(self.n):
#                 cur_split = splits[n_idx]
#                 split_sums.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))

#             split_sums = torch.cat(split_sums, dim=-1)  # [bsz,n]
            
#         scales = torch.minimum(
#             self.n * torch.ones_like(split_sums).to(s_probs.device), 1 / split_sums
#         )  # [bsz,n]

#         overflow_scales = (
#             self.n * split_sums - 1
#         ) / split_sums  # [bsz,n] note: might be negative or nan
#         overflow_scales = set_nan_to_zero(overflow_scales)
#         overflow_scales[overflow_scales < 0] = 0  # [bsz,n]

#         target_scales = scales[range(bsz), split_k]  # [bsz]
#         target_sums = split_sums[range(bsz), split_k]  # [bsz]

#         remain_sums = 1 - target_scales * target_sums  # [bsz]
#         overflow_sums = (overflow_scales * split_sums).sum(dim=-1)  # [bsz]
#         fill_scale = remain_sums / overflow_sums  # [bsz]
#         fill_scale = set_nan_to_zero(fill_scale)  # [bsz]

#         split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(
#             bsz, 1
#         ) == split_k.view(-1, 1).repeat(1, self.n)
#         final_scale = torch.where(
#             split_mask,
#             target_scales.view(-1, 1).repeat(1, self.n),
#             fill_scale.view(-1, 1) * overflow_scales,
#         )  # [bsz,n]

#         reweighted_s_probs = torch.zeros_like(s_probs).to(s_logits.device)

#         if self.n == vocab_size:
#             reweighted_s_probs = final_scale * s_probs
#         elif vocab_size % self.n == 0:
#             reweighted_s_probs = (
#                 final_scale.view(bsz, self.n, 1)
#                 .expand((-1, -1, vocab_size // self.n))
#                 .reshape(bsz, vocab_size)
#                 * s_probs
#             )
#         else:
#             for n_idx in range(self.n):
#                 cur_split = splits[n_idx]
#                 reweighted_s_probs[:, cur_split] = (
#                     final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split]
#                 )

#         reweighted_s_probs[reweighted_s_probs < 0] = 0

#         # ==============================================================
#         # [NEW] 2. 结合自适应权重 alpha，平滑插值原始概率与水印概率
#         # ==============================================================
#         # 当 alpha 趋于 0（低熵），使用原始 s_probs；
#         # 当 alpha 趋于 1（高熵），完全使用 reweighted_s_probs。
#         adaptive_s_probs = (1.0 - alpha) * s_probs + alpha * reweighted_s_probs
        
#         # 为了防止数值下溢导致 log(0)，进行极小值截断
#         adaptive_s_probs = torch.clamp(adaptive_s_probs, min=1e-9)
        
#         # 转回 Logit 空间
#         reweighted_s_logits = torch.log(adaptive_s_probs)
#         # ==============================================================

#         reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

#         return reweighted_logits

# class MC_Reweight(AbstractReweight):
#     watermark_code_type = MCMark_WatermarkCode

#     def __init__(self, n: float, entropy_threshold: float = 4.0, num_iters: int = 3):
#         self.n = n
#         self.entropy_threshold = entropy_threshold
#         self.num_iters = num_iters # [NEW] 迭代次数

#     def __repr__(self):
#         return f"MC_Reweight(n={self.n}, entropy_threshold={self.entropy_threshold}, num_iters={self.num_iters})"

#     def reweight_logits(
#         self, code: AbstractWatermarkCode, p_logits: FloatTensor
#     ) -> FloatTensor:
#         def set_nan_to_zero(x):
#             x[torch.isnan(x)] = 0
#             return x

#         start = time.time()
#         # s_ means shuffled
#         s_logits = torch.gather(p_logits, -1, code.shuffle)
#         original_s_probs = torch.softmax(s_logits, dim=-1)
#         bsz, vocab_size = s_logits.shape

#         # ==============================================================
#         # 1. 基于原始分布计算香农熵和自适应权重 alpha
#         # ==============================================================
#         entropy = -torch.sum(original_s_probs * torch.log(original_s_probs + 1e-9), dim=-1)
#         # alpha 在整个迭代过程中保持固定，因为它是当前上下文的固有属性
#         alpha = torch.clamp(entropy / self.entropy_threshold, min=0.0, max=1.0).view(bsz, 1)

#         # 准备分桶索引
#         splits = []
#         if self.n == vocab_size:
#             splits = [[i] for i in range(self.n)]
#         elif vocab_size % self.n == 0:
#             splits = (
#                 torch.arange(start=0, end=vocab_size)
#                 .reshape(self.n, vocab_size // self.n)
#                 .to(p_logits.device)
#             )
#         else:
#             for n_idx in range(self.n):
#                 splits.append(
#                     list(
#                         range(
#                             round(vocab_size * n_idx / self.n),
#                             round(vocab_size * (n_idx + 1) / self.n),
#                         )
#                     )
#                 )

#         split_k = code.split_k.to(s_logits.device)

#         # ==============================================================
#         # 2. 迭代式重加权 (Iterative Reweighting)
#         # ==============================================================
#         current_s_probs = original_s_probs.clone()

#         for _ in range(self.num_iters):
#             # 2.1 基于 *当前* 的概率分布计算桶概率和 (split_sums)
#             split_sums = []
#             if self.n == vocab_size:
#                 split_sums = current_s_probs
#             elif vocab_size % self.n == 0:
#                 split_sums = current_s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
#             else:
#                 for n_idx in range(self.n):
#                     cur_split = splits[n_idx]
#                     split_sums.append(current_s_probs[:, cur_split].sum(dim=-1, keepdim=True))
#                 split_sums = torch.cat(split_sums, dim=-1)  # [bsz, n]
                
#             # 2.2 计算目标桶的缩放比例和其他桶的压缩比例
#             scales = torch.minimum(
#                 self.n * torch.ones_like(split_sums).to(current_s_probs.device), 1 / split_sums
#             )  # [bsz, n]

#             overflow_scales = (self.n * split_sums - 1) / split_sums
#             overflow_scales = set_nan_to_zero(overflow_scales)
#             overflow_scales[overflow_scales < 0] = 0

#             target_scales = scales[range(bsz), split_k]
#             target_sums = split_sums[range(bsz), split_k]

#             remain_sums = 1 - target_scales * target_sums
#             overflow_sums = (overflow_scales * split_sums).sum(dim=-1)
#             fill_scale = remain_sums / overflow_sums
#             fill_scale = set_nan_to_zero(fill_scale)

#             split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(bsz, 1) == split_k.view(-1, 1).repeat(1, self.n)
#             final_scale = torch.where(
#                 split_mask,
#                 target_scales.view(-1, 1).repeat(1, self.n),
#                 fill_scale.view(-1, 1) * overflow_scales,
#             )

#             # 2.3 生成本轮的水印概率分布
#             step_watermarked_probs = torch.zeros_like(current_s_probs).to(s_logits.device)

#             if self.n == vocab_size:
#                 step_watermarked_probs = final_scale * current_s_probs
#             elif vocab_size % self.n == 0:
#                 step_watermarked_probs = (
#                     final_scale.view(bsz, self.n, 1)
#                     .expand((-1, -1, vocab_size // self.n))
#                     .reshape(bsz, vocab_size)
#                     * current_s_probs
#                 )
#             else:
#                 for n_idx in range(self.n):
#                     cur_split = splits[n_idx]
#                     step_watermarked_probs[:, cur_split] = (
#                         final_scale[:, n_idx].view(-1, 1) * current_s_probs[:, cur_split]
#                     )

#             step_watermarked_probs[step_watermarked_probs < 0] = 0

#             # 2.4 融合: P_next = (1 - alpha) * P_current + alpha * P_watermark
#             current_s_probs = (1.0 - alpha) * current_s_probs + alpha * step_watermarked_probs
            
#             # [关键] 每次融合后重新归一化，防止浮点数精度截断导致概率总和不为 1
#             current_s_probs = current_s_probs / current_s_probs.sum(dim=-1, keepdim=True)
#             current_s_probs = torch.clamp(current_s_probs, min=1e-9)

#         # ==============================================================
#         # 3. 映射回 Logits 空间并还原顺序
#         # ==============================================================
#         reweighted_s_logits = torch.log(current_s_probs)
#         reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

#         return reweighted_logits

class MC_Reweight(AbstractReweight):
    watermark_code_type = MCMark_WatermarkCode

    # [优化参数] 提高 entropy_threshold 以适应真实大模型，增加 max_alpha 保护上限
    def __init__(self, n: float, entropy_threshold: float = 6.0, num_iters: int = 3, max_alpha: float = 0.85):
        self.n = n
        self.entropy_threshold = entropy_threshold
        self.num_iters = num_iters
        self.max_alpha = max_alpha

    def __repr__(self):
        return f"MC_Reweight(n={self.n}, entropy_threshold={self.entropy_threshold}, num_iters={self.num_iters})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        def set_nan_to_zero(x):
            x[torch.isnan(x)] = 0
            return x

        # s_ means shuffled
        s_logits = torch.gather(p_logits, -1, code.shuffle)
        
        # [关键] 必须使用高温平滑一下极端的原始 logit，防止 log(0)
        original_s_probs = torch.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        # ==============================================================
        # 1. 计算香农熵和绝对受限的自适应权重 alpha
        # ==============================================================
        entropy = -torch.sum(original_s_probs * torch.log(original_s_probs + 1e-9), dim=-1)
        
        # 强制设置 alpha 上限为 max_alpha (如 0.85)，永远保留至少 15% 的原始分布！
        alpha = torch.clamp(entropy / self.entropy_threshold, min=0.0, max=self.max_alpha).view(bsz, 1)

        splits = []
        if self.n == vocab_size:
            splits = [[i] for i in range(self.n)]
        elif vocab_size % self.n == 0:
            splits = (
                torch.arange(start=0, end=vocab_size)
                .reshape(self.n, vocab_size // self.n)
                .to(p_logits.device)
            )
        else:
            for n_idx in range(self.n):
                splits.append(
                    list(
                        range(
                            round(vocab_size * n_idx / self.n),
                            round(vocab_size * (n_idx + 1) / self.n),
                        )
                    )
                )

        split_k = code.split_k.to(s_logits.device)

        # ==============================================================
        # 2. 锚定式迭代重加权 (Anchored Iterative Reweighting)
        # ==============================================================
        current_s_probs = original_s_probs.clone()

        for step in range(self.num_iters):
            # 基于 current_s_probs 计算当前的缩放比例
            split_sums = []
            if self.n == vocab_size:
                split_sums = current_s_probs
            elif vocab_size % self.n == 0:
                split_sums = current_s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
            else:
                for n_idx in range(self.n):
                    cur_split = splits[n_idx]
                    split_sums.append(current_s_probs[:, cur_split].sum(dim=-1, keepdim=True))
                split_sums = torch.cat(split_sums, dim=-1)  # [bsz, n]
                
            # 计算缩放因子
            scales = torch.minimum(
                self.n * torch.ones_like(split_sums).to(current_s_probs.device), 1 / (split_sums + 1e-9)
            )

            overflow_scales = (self.n * split_sums - 1) / (split_sums + 1e-9)
            overflow_scales = set_nan_to_zero(overflow_scales)
            overflow_scales[overflow_scales < 0] = 0

            target_scales = scales[range(bsz), split_k]
            target_sums = split_sums[range(bsz), split_k]

            remain_sums = 1 - target_scales * target_sums
            overflow_sums = (overflow_scales * split_sums).sum(dim=-1)
            fill_scale = remain_sums / (overflow_sums + 1e-9)
            fill_scale = set_nan_to_zero(fill_scale)

            split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(bsz, 1) == split_k.view(-1, 1).repeat(1, self.n)
            final_scale = torch.where(
                split_mask,
                target_scales.view(-1, 1).repeat(1, self.n),
                fill_scale.view(-1, 1) * overflow_scales,
            )

            # 施加重加权
            step_watermarked_probs = torch.zeros_like(current_s_probs).to(s_logits.device)

            if self.n == vocab_size:
                step_watermarked_probs = final_scale * current_s_probs
            elif vocab_size % self.n == 0:
                step_watermarked_probs = (
                    final_scale.view(bsz, self.n, 1)
                    .expand((-1, -1, vocab_size // self.n))
                    .reshape(bsz, vocab_size)
                    * current_s_probs
                )
            else:
                for n_idx in range(self.n):
                    cur_split = splits[n_idx]
                    step_watermarked_probs[:, cur_split] = (
                        final_scale[:, n_idx].view(-1, 1) * current_s_probs[:, cur_split]
                    )

            step_watermarked_probs[step_watermarked_probs < 0] = 0

            # ==========================================================
            # [核心修复] 锚定融合：永远与 original_s_probs 融合
            # ==========================================================
            # 这样保证了非目标桶的低概率词在 K 次迭代后，其下限至少为 (1-alpha) * P_original，而不是 0！
            current_s_probs = (1.0 - alpha) * original_s_probs + alpha * step_watermarked_probs
            
            # 重新归一化并截断防溢出
            current_s_probs = current_s_probs / (current_s_probs.sum(dim=-1, keepdim=True) + 1e-9)
            current_s_probs = torch.clamp(current_s_probs, min=1e-12)

        # ==============================================================
        # 3. 映射回 Logits 空间
        # ==============================================================
        reweighted_s_logits = torch.log(current_s_probs)
        reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

        return reweighted_logits

# class MC_Reweight(AbstractReweight):
#     watermark_code_type = MCMark_WatermarkCode

#     def __init__(self, n: float, entropy_threshold: float = 5.0, num_iters: int = 3, max_alpha: float = 0.85, barren_threshold: float = 0.05):
#         self.n = n
#         self.entropy_threshold = entropy_threshold
#         self.num_iters = num_iters
#         self.max_alpha = max_alpha
#         # [NEW] 荒漠桶的概率阈值 (低于此值的水印强度将被衰减)
#         self.barren_threshold = barren_threshold

#     def __repr__(self):
#         return f"MC_Reweight(n={self.n}, entropy_threshold={self.entropy_threshold}, num_iters={self.num_iters}, barren_threshold={self.barren_threshold})"

#     def reweight_logits(
#         self, code: AbstractWatermarkCode, p_logits: FloatTensor
#     ) -> FloatTensor:
#         def set_nan_to_zero(x):
#             x[torch.isnan(x)] = 0
#             return x

#         s_logits = torch.gather(p_logits, -1, code.shuffle)
#         original_s_probs = torch.softmax(s_logits, dim=-1)
#         bsz, vocab_size = s_logits.shape

#         # ==============================================================
#         # Gate 1: 熵感知门 (Entropy Gate)
#         # ==============================================================
#         entropy = -torch.sum(original_s_probs * torch.log(original_s_probs + 1e-9), dim=-1)
#         entropy_alpha = torch.clamp(entropy / self.entropy_threshold, min=0.0, max=self.max_alpha).view(bsz, 1)

#         # 准备分桶逻辑
#         splits = []
#         if self.n == vocab_size:
#             splits = [[i] for i in range(self.n)]
#         elif vocab_size % self.n == 0:
#             splits = (torch.arange(start=0, end=vocab_size).reshape(self.n, vocab_size // self.n).to(p_logits.device))
#         else:
#             for n_idx in range(self.n):
#                 splits.append(list(range(round(vocab_size * n_idx / self.n), round(vocab_size * (n_idx + 1) / self.n))))

#         split_k = code.split_k.to(s_logits.device)

#         # 计算初始的 split_sums
#         initial_split_sums = []
#         if self.n == vocab_size:
#             initial_split_sums = original_s_probs
#         elif vocab_size % self.n == 0:
#             initial_split_sums = original_s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
#         else:
#             for n_idx in range(self.n):
#                 cur_split = splits[n_idx]
#                 initial_split_sums.append(original_s_probs[:, cur_split].sum(dim=-1, keepdim=True))
#             initial_split_sums = torch.cat(initial_split_sums, dim=-1)

#         # ==============================================================
#         # Gate 2: 目标桶活性门 (Viability Gate) - [核心创新点]
#         # ==============================================================
#         # 获取目标桶在原始分布下的总概率
#         target_sums_initial = initial_split_sums[range(bsz), split_k].view(bsz, 1)
        
#         # 如果目标桶的概率低于 barren_threshold，viability_factor 将小于 1，甚至趋于 0
#         viability_factor = torch.clamp(target_sums_initial / self.barren_threshold, min=0.0, max=1.0)
        
#         # 最终的水印强度由 熵 和 桶活性 共同决定！
#         final_alpha = entropy_alpha * viability_factor
#         # ==============================================================

#         current_s_probs = original_s_probs.clone()

#         for step in range(self.num_iters):
#             # 基于 current_s_probs 计算当前的缩放比例
#             split_sums = []
#             if self.n == vocab_size:
#                 split_sums = current_s_probs
#             elif vocab_size % self.n == 0:
#                 split_sums = current_s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
#             else:
#                 for n_idx in range(self.n):
#                     cur_split = splits[n_idx]
#                     split_sums.append(current_s_probs[:, cur_split].sum(dim=-1, keepdim=True))
#                 split_sums = torch.cat(split_sums, dim=-1)
                
#             scales = torch.minimum(
#                 self.n * torch.ones_like(split_sums).to(current_s_probs.device), 1 / (split_sums + 1e-9)
#             )

#             overflow_scales = (self.n * split_sums - 1) / (split_sums + 1e-9)
#             overflow_scales = set_nan_to_zero(overflow_scales)
#             overflow_scales[overflow_scales < 0] = 0

#             target_scales = scales[range(bsz), split_k]
#             target_sums = split_sums[range(bsz), split_k]

#             remain_sums = 1 - target_scales * target_sums
#             overflow_sums = (overflow_scales * split_sums).sum(dim=-1)
#             fill_scale = remain_sums / (overflow_sums + 1e-9)
#             fill_scale = set_nan_to_zero(fill_scale)

#             split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(bsz, 1) == split_k.view(-1, 1).repeat(1, self.n)
#             final_scale = torch.where(
#                 split_mask,
#                 target_scales.view(-1, 1).repeat(1, self.n),
#                 fill_scale.view(-1, 1) * overflow_scales,
#             )

#             step_watermarked_probs = torch.zeros_like(current_s_probs).to(s_logits.device)

#             if self.n == vocab_size:
#                 step_watermarked_probs = final_scale * current_s_probs
#             elif vocab_size % self.n == 0:
#                 step_watermarked_probs = (
#                     final_scale.view(bsz, self.n, 1)
#                     .expand((-1, -1, vocab_size // self.n))
#                     .reshape(bsz, vocab_size)
#                     * current_s_probs
#                 )
#             else:
#                 for n_idx in range(self.n):
#                     cur_split = splits[n_idx]
#                     step_watermarked_probs[:, cur_split] = (
#                         final_scale[:, n_idx].view(-1, 1) * current_s_probs[:, cur_split]
#                     )

#             step_watermarked_probs[step_watermarked_probs < 0] = 0

#             # 融合时使用双重门控计算出的 final_alpha
#             current_s_probs = (1.0 - final_alpha) * original_s_probs + final_alpha * step_watermarked_probs
            
#             current_s_probs = current_s_probs / (current_s_probs.sum(dim=-1, keepdim=True) + 1e-9)
#             current_s_probs = torch.clamp(current_s_probs, min=1e-12)

#         reweighted_s_logits = torch.log(current_s_probs)
#         reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

#         return reweighted_logits