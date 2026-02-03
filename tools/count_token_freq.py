import torch
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def count_frequencies(model_name, save_path, dataset_name="wikitext", dataset_config="wikitext-103-raw-v1"):
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    print(f"Loading dataset: {dataset_name} ({dataset_config})...")
    # 使用 streaming=True 以免下载整个巨大的数据集
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    # 初始化计数器 (使用 int64 防止溢出)
    counts = np.zeros(vocab_size + 10000, dtype=np.int64) # 多给一点空间防止特殊token溢出

    print("Counting tokens (processing 500k samples)...")
    # 只需要扫描一部分数据即可得到足够准确的分布（例如 50万行）
    max_samples = 500000 
    
    for i, sample in tqdm(enumerate(dataset), total=max_samples):
        if i >= max_samples:
            break
        text = sample['text']
        if not text.strip():
            continue
            
        # 只需要 input_ids
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        if tokens:
            # 批量更新计数
            token_arr = np.array(tokens)
            # 过滤掉超出 vocab_size 的异常值（如果有）
            token_arr = token_arr[token_arr < len(counts)]
            np.add.at(counts, token_arr, 1)

    # 截断回真实的 vocab_size
    # 注意：某些模型的 vocab_size 属性可能比实际 embedding 层小，最好以 tokenizer.vocab_size 为准
    # 或者为了安全，保留到实际统计到的最大 ID
    real_counts = counts[:vocab_size]
    
    print(f"Saving frequencies to {save_path}...")
    torch.save(torch.from_numpy(real_counts), save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--output", type=str, default="token_counts.pt")
    args = parser.parse_args()
    
    count_frequencies(args.model, args.output)