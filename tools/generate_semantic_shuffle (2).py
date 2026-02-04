# tools/generate_semantic_shuffle.py

import torch
import argparse
import os
import numpy as np
from transformers import AutoModel
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import math
import warnings

# 忽略 K-Means 在某些极端情况下的警告
warnings.filterwarnings("ignore")

def generate_semantic_shuffle(freq_file, model_name, split_num, save_path, high_freq_ratio=0.1, softness=0.2):
    # 1. 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading model: {model_name}...")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vocab_size = embeddings.shape[0]
    print(f"Vocab Size: {vocab_size}")

    # --- Step 1: 频率分层 (基于真实词频或 Norm) ---
    sorted_indices = None
    if freq_file and os.path.exists(freq_file):
        print(f"Loading real token frequencies from {freq_file}...")
        counts = torch.load(freq_file, map_location="cpu").numpy()
        
        # 尺寸对齐
        if len(counts) != vocab_size:
            if len(counts) < vocab_size:
                padding = np.zeros(vocab_size - len(counts))
                counts = np.concatenate([counts, padding])
            else:
                counts = counts[:vocab_size]
        sorted_indices = np.argsort(-counts) # 降序
    else:
        print("[Warning] No frequency file provided! Falling back to Embedding Norm.")
        norms = np.linalg.norm(embeddings, axis=1)
        sorted_indices = np.argsort(norms) # Norm 小的通常是高频
    
    cutoff = int(vocab_size * high_freq_ratio)
    high_freq_indices = sorted_indices[:cutoff].copy()
    np.random.shuffle(high_freq_indices)
    
    semantic_indices = sorted_indices[cutoff:].copy()
    n_semantic = len(semantic_indices)
    
    print(f"High Freq Tokens: {len(high_freq_indices)}")
    print(f"Semantic Tokens:  {n_semantic}")

    # --- Step 2: 语义聚类 (L2归一化 + 鲁棒递归拆分) ---
    print("Clustering semantic tokens...")
    semantic_embeddings = embeddings[semantic_indices]
    
    # [关键] L2 归一化：解决长尾词聚成一团的问题
    print("Normalizing embeddings for spherical clustering...")
    emb_norms = np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
    semantic_embeddings_norm = semantic_embeddings / (emb_norms + 1e-10)
    
    # 建立 ID 映射 (Global Token ID -> Local Matrix Row Index)
    id_to_local_idx = {original_id: row_idx for row_idx, original_id in enumerate(semantic_indices)}
    
    # 初始聚类 (数量少一点，后续靠分裂)
    initial_k = split_num * 10
    print(f"Running initial MiniBatchKMeans (k={initial_k})...")
    kmeans = MiniBatchKMeans(n_clusters=initial_k, random_state=42, batch_size=256, n_init="auto")
    cluster_labels = kmeans.fit_predict(semantic_embeddings_norm)
    
    # 构建初始簇
    clusters = {i: [] for i in range(initial_k)}
    for idx, label in zip(semantic_indices, cluster_labels):
        clusters[label].append(idx)
        
    # [关键] 递归拆分逻辑 (带防死锁)
    # 设定最大允许簇大小 (例如平均值的 1.2 倍)
    avg_cluster_size = n_semantic / (split_num * 10) 
    max_cluster_size = int(max(avg_cluster_size * 1.5, 100)) # 至少 100
    print(f"Recursive Splitting: Enforcing max cluster size <= {max_cluster_size}")

    final_clusters = {}
    next_cluster_id = 0
    
    # 队列项: (member_list, depth)
    process_queue = [(clusters[i], 0) for i in range(initial_k) if len(clusters[i]) > 0]
    
    loop_count = 0
    while process_queue:
        loop_count += 1
        if loop_count % 50 == 0:
            print(f"Processing queue... Remaining: {len(process_queue)} clusters")
            
        current_members, depth = process_queue.pop(0)
        
        # 终止条件1: 簇足够小
        if len(current_members) <= max_cluster_size:
            final_clusters[next_cluster_id] = current_members
            next_cluster_id += 1
            continue
            
        # 终止条件2: 递归太深 (防死锁)，强制接受或随机打散
        if depth > 5:
            # 强制随机切分 (既然 K-Means 分不开，说明语义极度相似，随机分也不影响)
            mid = len(current_members) // 2
            grp1 = current_members[:mid]
            grp2 = current_members[mid:]
            process_queue.insert(0, (grp1, depth + 1))
            process_queue.insert(0, (grp2, depth + 1))
            continue

        # 尝试 K-Means 拆分
        local_indices = [id_to_local_idx[tid] for tid in current_members]
        sub_embeddings = semantic_embeddings_norm[local_indices]
        
        # 决定拆成几份
        split_k = 2 
        if len(current_members) > max_cluster_size * 2:
            split_k = 3 # 加速大簇拆分
            
        try:
            sub_kmeans = MiniBatchKMeans(n_clusters=split_k, random_state=42 + depth, n_init=3, batch_size=256)
            sub_labels = sub_kmeans.fit_predict(sub_embeddings)
            
            sub_groups = [[] for _ in range(split_k)]
            for i, label in enumerate(sub_labels):
                sub_groups[label].append(current_members[i])
            
            # 检查拆分是否有效 (防止出现 [N, 0] 的无效拆分)
            lens = [len(g) for g in sub_groups]
            if min(lens) == 0:
                raise ValueError("K-Means produced empty cluster")
                
            # 将新组放回队列头
            for grp in sub_groups:
                process_queue.insert(0, (grp, depth + 1))
                
        except Exception:
            # 如果 K-Means 失败 (例如数据太少或全是一样的)，回退到强制随机切分
            mid = len(current_members) // 2
            process_queue.insert(0, (current_members[:mid], depth + 1))
            process_queue.insert(0, (current_members[mid:], depth + 1))

    clusters = final_clusters
    n_clusters = len(clusters)
    
    # 打印最终统计
    sizes = [len(c) for c in clusters.values()]
    print("-" * 30)
    print(f"Final Cluster Count: {n_clusters}")
    print(f"Max Cluster Size: {max(sizes)}")
    print(f"Min Cluster Size: {min(sizes)}")
    print(f"Avg Cluster Size: {sum(sizes)/n_clusters:.2f}")
    print("-" * 30)

    # --- Step 3: 贪心分配 (Greedy Bin Packing) ---
    print("Allocating clusters to channels (Greedy Balance)...")
    
    cluster_info = [(cid, len(clusters[cid])) for cid in clusters]
    cluster_info.sort(key=lambda x: x[1], reverse=True) # 从大到小
    
    channel_buckets = [[] for _ in range(split_num)]
    channel_loads = [0] * split_num
    
    # 先填高频词
    hf_chunk = len(high_freq_indices) // split_num
    for i in range(split_num):
        start = i * hf_chunk
        end = (i + 1) * hf_chunk if i != split_num - 1 else len(high_freq_indices)
        chunk = high_freq_indices[start:end]
        channel_buckets[i].extend(chunk)
        channel_loads[i] += len(chunk)
        
    # 再填语义簇
    # 记录每个簇被分到了哪个通道，以便处理 Softness
    cluster_to_channel = {} 
    
    for cid, size in cluster_info:
        # 找当前最轻的通道
        target_ch = np.argmin(channel_loads)
        channel_buckets[target_ch].extend(clusters[cid]) # 暂存，稍后统一处理softness比较麻烦，不如现在直接记录映射
        channel_loads[target_ch] += size
        cluster_to_channel[cid] = target_ch

    print(f"Base Channel Loads (before softness): {channel_loads}")

    # --- Step 4: 应用 Softness 并生成最终 Tensor ---
    # 重新构建 bucket，这次带随机性
    final_channel_buckets = [[] for _ in range(split_num)]
    
    # 4.1 高频词 (保持原样)
    hf_idx = 0
    for i in range(split_num):
        start = i * hf_chunk
        end = (i + 1) * hf_chunk if i != split_num - 1 else len(high_freq_indices)
        final_channel_buckets[i].extend(high_freq_indices[start:end])

    # 4.2 语义词 (应用 Softness)
    for cid, main_channel in cluster_to_channel.items():
        tokens = clusters[cid]
        for token in tokens:
            if np.random.rand() < softness:
                # 逃逸：随机去任意通道
                dest = np.random.randint(0, split_num)
            else:
                # 归队
                dest = main_channel
            final_channel_buckets[dest].append(token)
            
    # --- Step 5: 最终强制平衡 (Global Rebalance) ---
    print("Executing Final Rebalancing...")
    target_size = vocab_size // split_num
    
    global_overflow = []
    
    # 削峰
    for i in range(split_num):
        if len(final_channel_buckets[i]) > target_size:
            keep = final_channel_buckets[i][:target_size]
            overflow = final_channel_buckets[i][target_size:]
            final_channel_buckets[i] = keep
            global_overflow.extend(overflow)
            
    # 填谷
    for i in range(split_num):
        needed = target_size - len(final_channel_buckets[i])
        if needed > 0:
            if len(global_overflow) >= needed:
                final_channel_buckets[i].extend(global_overflow[:needed])
                global_overflow = global_overflow[needed:]
            else:
                # 溢出池不够了，从前面已经满的通道借（或者自我复制）
                # 这里简单处理：自我复制填充 (极少发生)
                if len(final_channel_buckets[i]) > 0:
                    final_channel_buckets[i].extend(final_channel_buckets[i][:needed])
                    
    # 处理剩余 (丢给最后一个)
    if global_overflow:
         final_channel_buckets[-1].extend(global_overflow)

    # --- Step 6: 保存 ---
    final_flat = []
    final_counts = [len(b) for b in final_channel_buckets]
    print(f"Final Channel Counts: {final_counts}")
    
    for b in final_channel_buckets:
        final_flat.extend(b)
        
    shuffle_tensor = torch.tensor(final_flat, dtype=torch.long)
    
    # 尺寸最终修正
    if len(shuffle_tensor) > vocab_size:
        shuffle_tensor = shuffle_tensor[:vocab_size]
    elif len(shuffle_tensor) < vocab_size:
        diff = vocab_size - len(shuffle_tensor)
        shuffle_tensor = torch.cat([shuffle_tensor, torch.zeros(diff, dtype=torch.long)])
    
    print(f"Saving to {save_path}...")
    torch.save(shuffle_tensor, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq_file", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split_num", type=int, default=4)
    parser.add_argument("--output", type=str, default="semantic_shuffle.pt")
    parser.add_argument("--high_freq_ratio", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=0.2)
    args = parser.parse_args()
    
    generate_semantic_shuffle(args.freq_file, args.model, args.split_num, args.output, args.high_freq_ratio, args.softness)