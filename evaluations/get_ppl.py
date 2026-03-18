# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import torch
# import json
# import argparse
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# @torch.no_grad()
# def calculate_ppl(file_path, model_name, device="cuda:0"):
#     print(f"Loading Oracle Model for PPL Evaluation: {model_name}...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16, 
#         device_map=device
#     )
#     model.eval()

#     # 读取生成的 jsonl 文件
#     data = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 data.append(json.loads(line))

#     total_nll = 0.0
#     total_tokens = 0
#     all_ppls = []

#     print(f"Calculating PPL for {len(data)} generated texts...")
#     for item in tqdm(data):
#         # 假设生成阶段把模型输出存放在了 "output" 或 "display_output" 字段
#         # 根据你 common.py 中的逻辑，"output" 是包含特殊 token 的，而 "display_output" 是纯文本
#         # 计算 PPL 通常使用模型生成的纯文本
#         if isinstance(item.get("display_output"), list):
#             text = item["display_output"][0] 
#         elif isinstance(item.get("output"), list):
#             text = item["output"][0]
#         else:
#             text = item.get("output", "")

#         if not text.strip():
#             continue

#         # 编码文本
#         encodings = tokenizer(text, return_tensors="pt").to(device)
#         input_ids = encodings.input_ids
#         seq_len = input_ids.size(1)

#         # 忽略过短的文本
#         if seq_len < 2:
#             continue

#         # 将 labels 设为 input_ids，HuggingFace 的 CausalLM 会自动进行 shift 并计算 CrossEntropyLoss
#         outputs = model(input_ids, labels=input_ids)
#         loss = outputs.loss  # 这里的 loss 就是整个序列的平均负对数似然 (NLL)

#         # 记录 Token 级别的累积（用于计算整个数据集的宏观 PPL）
#         total_nll += loss.item() * seq_len
#         total_tokens += seq_len

#         # 记录单条文本的 PPL（用于统计方差或绘制分布图）
#         seq_ppl = torch.exp(loss).item()
#         all_ppls.append(seq_ppl)

#     # 1. 计算语料库级别的整体 PPL (Corpus-level PPL，更常用且更稳定)
#     corpus_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    
#     # 2. 计算句子级别的平均 PPL (Sentence-level PPL)
#     avg_sentence_ppl = sum(all_ppls) / len(all_ppls) if all_ppls else 0

#     print("-" * 50)
#     print(f"Evaluation File: {file_path}")
#     print(f"Total Valid Tokens: {total_tokens}")
#     print(f"Corpus-level PPL: {corpus_ppl:.4f}")
#     print(f"Average Sentence-level PPL: {avg_sentence_ppl:.4f}")
#     print("-" * 50)

#     return corpus_ppl, avg_sentence_ppl


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Calculate PPL for watermarked texts.")
#     parser.add_argument("--file", type=str, required=True, help="Path to the generated file.")
#     parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Oracle model to calculate PPL.")
#     parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
    
#     args = parser.parse_args()
#     calculate_ppl(args.file, args.model, args.device)

import json
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(file_path, model_name="meta-llama/Llama-2-7b-chat-hf", device="cuda"):
    """
    计算给定 JSON-lines 文件中文本的困惑度。
    """
    print(f"正在加载模型和分词器: {model_name}")
    # 建议使用与你生成文本或评估一致的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16
    )
    model.eval()

    # 1. 读取水印文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 使用 display_output 作为计算的文本，因为它去除了特殊的 prompt token
                text = data.get("display_output", data.get("output", ""))
                if text:
                    texts.append(text)

    print(f"成功加载 {len(texts)} 条数据。开始计算困惑度 (Perplexity)...")
    
    ppls = []
    total_loss = 0.0
    total_length = 0

    # 2. 逐条计算 PPL
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating PPL"):
            # 将文本进行分词
            encodings = tokenizer(text, return_tensors="pt").to(device)
            input_ids = encodings.input_ids
            
            # 过滤掉过短的文本
            if input_ids.size(1) < 2:
                continue
                
            # Causal LM 在传入 labels=input_ids 时，会自动计算序列的 Next-Token Prediction 损失
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 单句的困惑度 = exp(loss)
            ppl = torch.exp(loss).item()
            ppls.append(ppl)
            
            # 累加用于计算全局（Token 级别）平均困惑度
            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)

    # 3. 计算并输出统计结果
    avg_sentence_ppl = sum(ppls) / len(ppls) if ppls else 0
    global_token_ppl = math.exp(total_loss / total_length) if total_length > 0 else 0
    
    print("-" * 50)
    print(f"评估完成！基于模型: {model_name}")
    print(f"句子级别的平均困惑度 (Average Sentence PPL): {avg_sentence_ppl:.4f}")
    print(f"全局 Token 级别的困惑度 (Global Token PPL): {global_token_ppl:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    # 文件路径指向你上传的水印生成文件
    input_file = "./results/c4_subset/Mistral_7B_Instruct_v0.3/mcmarkDP/text_generation.txt"
    # autodl-tmp/results/c4_subset/Mistral_7B_Instruct_v0.3/mcmark/text_generation.txt
    # 结合你 common.py 文件中的模型设置，这里默认使用了 Llama-2-7b-chat-hf
    # autodl-tmp/results/c4_subset/Mistral_7B_Instruct_v0.3/mcmarkDP/score.txt
    # 如果你在生成时使用的是 OPT 或其他 Llama 版本，请对应替换模型名称
    calculate_perplexity(input_file, model_name="mistralai/Mistral-7B-Instruct-v0.3")