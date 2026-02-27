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
class HybridSemanticContextCodeExtractor(AbstractContextCodeExtractor):
    """
    混合语义上下文提取器：解决MinHash导致上下文静止的问题。
    将长窗口的“无序语义锚点(MinHash)”与极短窗口的“局部语义(上一个Token)”拼接。
    既保证了Context Code在每一步都会正常变化（恢复 Baseline，防止水印被 Mask），
    又通过全局锚点增强了对 DeepSeek/Dipper 句式改写和长句调整的抗性。
    """
    window_size: int = 20  # 长窗口：捕获整个句子的核心语义锚点
    n_local: int = 1       # 短窗口：只看最后 1 个词的语义，提供必需的步进动态性
    num_hashes: int = 1    # 混合模式下 1 个 Hash 就够了，降低碰撞率
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    _class_cache = {}

    def __post_init__(self):
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found.")
        self.hash_seeds = [15485863, 32452843]

    def _get_unshuffle_map(self):
        if self.shuffle_file_path not in self._class_cache:
            if os.path.exists(self.shuffle_file_path):
                try:
                    print(f"[HybridContext] Loading {self.shuffle_file_path} into cache...")
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

        # 1. 提取长窗口(锚点) 和 短窗口(动态特征) 的原始 ID
        raw_ids_window = context[-self.window_size :].detach().cpu()
        raw_ids_local = context[-self.n_local :].detach().cpu()

        unshuffle = self._get_unshuffle_map()

        if unshuffle is None:
            # 文件不存在时的降级处理
            w_min = min([self._hash_bucket(int(x), self.hash_seeds[0]) for x in torch.unique(raw_ids_window)] + [0])
            return w_min.to_bytes(4, byteorder='little') + raw_ids_local.numpy().tobytes()

        # 2. 映射分桶的通用函数
        def get_buckets(raw_ids):
            max_id = unshuffle.size(0)
            valid_mask = raw_ids < max_id
            ranks = raw_ids.clone()
            if valid_mask.all():
                ranks = unshuffle[raw_ids]
            else:
                ranks[valid_mask] = unshuffle[raw_ids[valid_mask]]
            return ranks // self.bucket_size

        window_buckets = get_buckets(raw_ids_window)
        local_buckets = get_buckets(raw_ids_local)

        # 3. 计算全局锚点 (无序 MinHash)
        unique_buckets = torch.unique(window_buckets).numpy()
        anchor_bytes = b""
        if len(unique_buckets) > 0:
            min_h = min(self._hash_bucket(int(b), self.hash_seeds[0]) for b in unique_buckets)
        else:
            min_h = 0
        anchor_bytes = min_h.to_bytes(4, byteorder='little', signed=False)

        # 4. 提取局部动态特征 (保证每步不同，躲避 cc_history 拦截)
        local_bytes = local_buckets.numpy().tobytes()

        # 5. 拼接：哪怕结构被重排，只要上一个词是个同义词且大窗口锚点不变，Hash就能对齐
        return anchor_bytes + local_bytes

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
class RobustMinHashContextCodeExtractor(AbstractContextCodeExtractor):
    """
    终极版：纯净的无序语义 MinHash 提取器。
    彻底免疫 Dipper/DeepSeek 的句式重排和同义词替换。
    必须配合 ignore_history=True 使用。
    """
    window_size: int = 30  # 扩大到 30，覆盖完整的长句或跨句语义
    num_hashes: int = 1    # 仅用 1 个哈希，最大化重写前后的对齐/碰撞概率
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    _class_cache = {}

    def __post_init__(self):
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found.")
        self.hash_seeds = [15485863]

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

        # 仅截取大窗口，完全丢弃局部顺序
        raw_ids = context[-self.window_size :].detach().cpu()
        unshuffle = self._get_unshuffle_map()

        if unshuffle is None:
            w_min = min([self._hash_bucket(int(x), self.hash_seeds[0]) for x in torch.unique(raw_ids)] + [0])
            return w_min.to_bytes(4, byteorder='little', signed=False)

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


@dataclass
class AnchorSemanticContextCodeExtractor(AbstractContextCodeExtractor):
    """
    终极版：基于语义锚点（Semantic Anchors）的抗深层改写提取器。
    完美自适应 Dipper 的文本增删和乱序，并保持高频的密钥更新以维持文本质量。
    """
    anchor_freq: int = 4  # 约 25% 的词会被选为锚点 (1/4)
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    _class_cache = {}

    def __post_init__(self):
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found. Fallback to raw ID anchors.")

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

    def _is_anchor(self, bucket: int) -> bool:
        # 用一个固定的哈希算法来决定某个语义桶是否是“锚点”
        val = (bucket ^ 15485863) * 2654435761
        return (val % (2**32)) % self.anchor_freq == 0

    def extract(self, context: LongTensor) -> bytes:
        if len(context) == 0:
            return b""

        unshuffle = self._get_unshuffle_map()
        
        # 为了效率和防止极端情况，最多往回扫描 50 个 Token 寻找锚点
        search_window = context[-50:].detach().cpu()
        
        if unshuffle is None:
            # 文件不存在时的降级逻辑
            for i in range(len(search_window)-1, -1, -1):
                token_id = int(search_window[i])
                if self._is_anchor(token_id):
                    return token_id.to_bytes(4, byteorder='little', signed=False)
            return int(search_window[-1]).to_bytes(4, byteorder='little', signed=False)

        max_id = unshuffle.size(0)
        
        # 从当前词往前倒序扫描，寻找最近的锚点
        for i in range(len(search_window)-1, -1, -1):
            raw_id = search_window[i]
            
            # 获取 Rank 和 Bucket
            if raw_id < max_id:
                rank = unshuffle[raw_id]
            else:
                rank = raw_id
                
            bucket = int(rank // self.bucket_size)
            
            # 找到锚点，立即将其作为 Context Code 锁定并返回
            if self._is_anchor(bucket):
                return bucket.to_bytes(4, byteorder='little', signed=False)
        
        # 如果 50 个词内都没遇到锚点（极小概率），直接用上一个词兜底
        last_id = search_window[-1]
        rank = unshuffle[last_id] if last_id < max_id else last_id
        bucket = int(rank // self.bucket_size)
        return bucket.to_bytes(4, byteorder='little', signed=False)


@dataclass
class FastRecoverySemanticExtractor(AbstractContextCodeExtractor):
    """
    极速自愈的微型语义词袋提取器。
    原理：仅取最近 2 个 Token（n=2）的语义桶并排序。
    优势：
    1. Context 高频变化，彻底解决词汇饥饿，保证生成质量；
    2. Sort 排序无视 Dipper 的局部语序打乱；
    3. n=2 保证遇到增删攻击时，最多只错乱 2 个词，随后瞬间自愈恢复对齐！
    """
    n: int = 2  # 黄金数字：兼顾局部上下文熵值与极速自愈能力
    bucket_size: int = 64
    shuffle_file_path: str = "semantic_shuffle_opt.pt"
    
    _class_cache = {}

    def __post_init__(self):
        if not os.path.exists(self.shuffle_file_path):
            print(f"[Warning] {self.shuffle_file_path} not found.")

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

    def extract(self, context: LongTensor) -> bytes:
        if len(context) == 0:
            return b""
        
        # 仅截取最后 n=2 个 Token，保证极速自愈
        raw_ids = context[-self.n :].detach().cpu()
        unshuffle = self._get_unshuffle_map()

        if unshuffle is None:
            # 降级：直接对 Token ID 排序（防语序打乱）
            sorted_ids = torch.sort(raw_ids)[0]
            return sorted_ids.numpy().tobytes()

        # 映射到语义排名
        max_id = unshuffle.size(0)
        valid_mask = raw_ids < max_id
        
        ranks = raw_ids.clone()
        if valid_mask.all():
            ranks = unshuffle[raw_ids]
        else:
            ranks[valid_mask] = unshuffle[raw_ids[valid_mask]]

        # 划分语义桶
        buckets = ranks // self.bucket_size
        
        # 【致胜核心】：对这 2 个桶进行排序。
        # Dipper 把 "The big" 改成 "large The"，依然完美匹配！
        sorted_buckets = torch.sort(buckets)[0]
        
        return sorted_buckets.numpy().tobytes()