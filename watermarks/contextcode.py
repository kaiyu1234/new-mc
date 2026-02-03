#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from torch import FloatTensor, LongTensor
from .base import AbstractContextCodeExtractor
import torch
import os

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