#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from torch import FloatTensor, LongTensor
from .base import AbstractContextCodeExtractor
import torch
import os
from transformers import PreTrainedTokenizer

def get_boundary_ids(tokenizer: PreTrainedTokenizer) -> list[int]:
    """
    根据传入的 Tokenizer 动态获取句号、问号、叹号和换行符的 Token ID。
    """
    punctuation_list = [".", "?", "!", "\n", "。", "？", "！"] # 兼容了中文标点
    boundary_ids = []
    
    for p in punctuation_list:
        # add_special_tokens=False 防止引入额外的 BOS/EOS token
        tokens = tokenizer.encode(p, add_special_tokens=False)
        boundary_ids.extend(tokens)
        
    # 去重并返回
    return list(set(boundary_ids))

@dataclass
class All_ContextCodeExtractor(AbstractContextCodeExtractor):
    def extract(self, context: LongTensor) -> any:
        return context.detach().cpu().numpy().tobytes()


@dataclass
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    """Extracts the last n tokens in the context"""

    n: int

    def extract(self, context: LongTensor) -> any:
        return context[-self.n :].detach().cpu().numpy().tobytes()


@dataclass
class SemanticRobustContextCodeExtractor(AbstractContextCodeExtractor):
    """
    语义鲁棒的上下文提取器 (优化版)。
    使用类级缓存和延迟加载，避免在多进程 Queue 中传输巨大的 Tensor，
    解决 FileNotFoundError 和资源共享问题。
    """
    n: int
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    # [关键修改] 使用类变量作为缓存，所有实例共享，且不会被 pickle 传输
    _class_cache = {}

    def __post_init__(self):
        # 初始化时只检查文件是否存在，不加载数据，保持对象轻量
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found. Will degrade to raw tokens.")

    def _get_unshuffle_map(self):
        """
        延迟加载逻辑：只有在真正需要数据时（在 GPU Worker 内部）才加载。
        """
        # 检查缓存是否已有该文件的数据
        if self.shuffle_file_path not in self._class_cache:
            if os.path.exists(self.shuffle_file_path):
                try:
                    # 每个进程（GPU Worker）只会打印一次加载信息
                    print(f"[SemanticContext] Loading {self.shuffle_file_path} into process cache...")
                    shuffle = torch.load(self.shuffle_file_path, map_location="cpu")
                    # 计算 argsort 并存入缓存
                    self._class_cache[self.shuffle_file_path] = torch.argsort(shuffle)
                except Exception as e:
                    print(f"[SemanticContext] Error loading file: {e}")
                    self._class_cache[self.shuffle_file_path] = None
            else:
                self._class_cache[self.shuffle_file_path] = None
        
        return self._class_cache[self.shuffle_file_path]

    def extract(self, context: LongTensor) -> bytes:
        # 获取上下文的最后 n 个 token
        raw_ids = context[-self.n :].detach().cpu()
        
        if len(raw_ids) == 0:
            return b""

        # [关键修改] 从缓存获取映射表，而不是从实例变量
        unshuffle = self._get_unshuffle_map()

        # 如果加载失败或文件不存在，退化为普通 ID 模式
        if unshuffle is None:
            return raw_ids.numpy().tobytes()

        # 映射: TokenID -> Semantic Rank
        max_id = unshuffle.size(0)
        valid_mask = raw_ids < max_id
        
        ranks = raw_ids.clone()
        if valid_mask.all():
            ranks = unshuffle[raw_ids]
        else:
            ranks[valid_mask] = unshuffle[raw_ids[valid_mask]]

        # 量化/分桶
        buckets = ranks // self.bucket_size
        
        return buckets.numpy().tobytes()


@dataclass
class SentenceBoundarySemanticExtractor(AbstractContextCodeExtractor):
    """
    方案一：基于“逻辑块边界”（句子级）的上下文提取器。
    通过识别句号、问号、回车等边界Token，提取【上一个完整句子】的语义MinHash。
    优势：只要上一句话的语义骨架存在，当前句子的生成和检测就能100%对齐，彻底免疫当前句内的乱序和增删。
    """
    # 默认包含常用模型的句号和换行ID，请在实例化时根据具体Tokenizer传入
    boundary_ids: list[int] = field(default_factory=lambda: [13, 2, 29889]) 
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    _class_cache = {}

    def __post_init__(self):
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found.")
        self.hash_seeds = [15485863]
        self._boundary_tensor = None

    def _get_unshuffle_map(self):
        if self.shuffle_file_path not in self._class_cache:
            if os.path.exists(self.shuffle_file_path):
                try:
                    shuffle = torch.load(self.shuffle_file_path, map_location="cpu")
                    self._class_cache[self.shuffle_file_path] = torch.argsort(shuffle)
                except Exception as e:
                    self._class_cache[self.shuffle_file_path] = None
            else:
                self._class_cache[self.shuffle_file_path] = None
        return self._class_cache[self.shuffle_file_path]

    def _hash_bucket(self, bucket_val: int, seed: int) -> int:
        val = (bucket_val ^ seed) * 2654435761
        return val % (2**32)

    def extract(self, context: LongTensor) -> bytes:
        if len(context) == 0:
            return b""

        # 1. 初始化边界 Tensor，确保在正确的设备上
        if self._boundary_tensor is None or self._boundary_tensor.device != context.device:
            self._boundary_tensor = torch.tensor(self.boundary_ids, dtype=torch.long, device=context.device)

        # 2. 寻找 Context 中所有标点符号（句子边界）的位置
        is_boundary = torch.isin(context, self._boundary_tensor)
        boundary_indices = torch.nonzero(is_boundary, as_tuple=True)[0]

        # 3. 核心逻辑：截取“上一个完整句子”的 Token
        if len(boundary_indices) == 0:
            # 还没写完第一句话，退化为全局 MinHash
            target_ids = context
        elif len(boundary_indices) == 1:
            # 正在写第二句话，上一句话是从开头到第一个标点
            target_ids = context[:boundary_indices[0]]
        else:
            # 正在写第三句话及以后，提取倒数第二个标点到倒数第一个标点之间的词
            start_idx = boundary_indices[-2] + 1
            end_idx = boundary_indices[-1]
            target_ids = context[start_idx:end_idx]

            # 增强鲁棒性：如果提取的句子太短（如遇到连续标点 "..." ），则向前合并一句
            if len(target_ids) < 3 and len(boundary_indices) >= 3:
                start_idx = boundary_indices[-3] + 1
                target_ids = context[start_idx:end_idx]

        # 兜底防护：如果由于异常导致target_ids为空，使用末尾窗口
        if len(target_ids) == 0:
            target_ids = context[-10:]

        # 4. 计算上一个句子的语义 MinHash
        raw_ids = target_ids.detach().cpu()
        unshuffle = self._get_unshuffle_map()

        if unshuffle is None:
            w_min = min([self._hash_bucket(int(x), self.hash_seeds[0]) for x in torch.unique(raw_ids)] + [0])
            return w_min.to_bytes(4, byteorder='little', signed=False)

        # 映射到语义分桶
        max_id = unshuffle.size(0)
        valid_mask = raw_ids < max_id
        ranks = raw_ids.clone()
        if valid_mask.all():
            ranks = unshuffle[raw_ids]
        else:
            ranks[valid_mask] = unshuffle[raw_ids[valid_mask]]

        buckets = ranks // self.bucket_size
        unique_buckets = torch.unique(buckets).numpy()

        if len(unique_buckets) > 0:
            min_h = min(self._hash_bucket(int(b), self.hash_seeds[0]) for b in unique_buckets)
        else:
            min_h = 0

        return min_h.to_bytes(4, byteorder='little', signed=False)
