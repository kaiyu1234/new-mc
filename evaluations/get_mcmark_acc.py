import json
import os
import numpy as np
from collections import defaultdict
import math
import re
import argparse

# ==========================================
# 恢复原始的理论 FPR (p-value) 计算逻辑
# ==========================================
def get_split_fpr(n, m, split_num):
    res = 0
    # 防止因为 DTW 导致 m > n 的极端情况抛出异常
    m = min(m, n)
    for i in range(m, n + 1):
        res += math.comb(n, i) * ((split_num - 1) ** (n - i))
    res /= split_num**n
    return res

def extract_n_value(text):
    # 去掉了正则末尾的 \) ，使其能兼容 "MC_Reweight(n=20, entropy_threshold=4.0)"
    pattern = re.search(r"MC_Reweight\(n=(\d+)", text)
    if pattern:
        return int(pattern.group(1))
    raise NotImplementedError

def get_save_path(score_path, fpr_thres, len_limit):
    save_dir = "./results"
    info_path = "/".join(score_path.split("/")[-4:-1])
    file_name = (
        score_path.split("/")[-1].split(".")[0]
        + "_mcmark_eval_results_"
        + str(fpr_thres).replace(".", "_")
        + "_"
        + str(len_limit)
        + ".txt"
    )
    return os.path.join(save_dir, info_path, file_name)

def calculate_empirical_thresholds(baseline_score_path, fpr_thres, len_limit):
    """
    计算经验阈值 (防作弊机制)
    """
    if not os.path.exists(baseline_score_path):
        print(f"[Warning] 未找到 Baseline 文件 {baseline_score_path}，将仅使用理论 p-value 进行评估。")
        return {}
        
    with open(baseline_score_path, "r") as f:
        lines = f.readlines()
        
    baseline_scores_dict = defaultdict(list)
    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]
        wp = res_dict["watermark_processor"]
        if cur_len < len_limit or "MC_Reweight" not in wp:
            continue
        raw_score = res_dict["raw_scores"]
        cur_n = extract_n_value(wp)
        baseline_scores_dict[cur_n].append(raw_score)
        
    empirical_thresholds = {}
    percentile = (1.0 - fpr_thres) * 100
    
    print("-" * 50)
    print(">>> 经验阈值校准结果 (Empirical Calibration) <<<")
    for cur_n, scores in baseline_scores_dict.items():
        if not scores:
            continue
        threshold = np.percentile(scores, percentile)
        empirical_thresholds[cur_n] = threshold
        print(f"n={cur_n:<4} | Baseline 样本: {len(scores):<4} | 动态阈值(FPR={fpr_thres}): {threshold:.2f}")
    print("-" * 50)
    return empirical_thresholds

def generate_result(score_path, baseline_score_path, save_path, fpr_thres, len_limit):
    empirical_thresholds = calculate_empirical_thresholds(baseline_score_path, fpr_thres, len_limit)
    
    with open(score_path, "r") as f:
        lines = f.readlines()

    tot_cnt = defaultdict(int)
    acc_cnt_empirical = defaultdict(int)
    acc_cnt_theory = defaultdict(int)
    fpr_list = defaultdict(list)
    score_list = defaultdict(list)

    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]
        wp = res_dict["watermark_processor"]

        if cur_len < len_limit:
            continue
        if "MC_Reweight" not in wp:
            continue

        raw_score = int(res_dict["raw_scores"]) 
        cur_len = int(cur_len)
        cur_n = extract_n_value(wp)
        
        # 1. 理论计算
        cur_fpr = get_split_fpr(cur_len, raw_score, split_num=cur_n)
        fpr_list[cur_n].append(cur_fpr)
        if cur_fpr <= fpr_thres:
            acc_cnt_theory[cur_n] += 1
            
        # 2. 经验计算
        if cur_n in empirical_thresholds:
            if raw_score >= empirical_thresholds[cur_n]:
                acc_cnt_empirical[cur_n] += 1

        score_list[cur_n].append(raw_score)
        tot_cnt[cur_n] += 1

    lines_to_write = []
    for wp_n in sorted(tot_cnt.keys()):
        # ⬇️ 这里严格保证是 7 行
        lines_to_write.append("-" * 80 + "\n")                                                                       # 第 1 行
        lines_to_write.append(f"MC_Reweight(n={wp_n})\n")                                                            # 第 2 行
        lines_to_write.append(f"Total cnt: {tot_cnt[wp_n]}\n")                                                       # 第 3 行
        
        median_p = np.median(fpr_list[wp_n])
        lines_to_write.append(f"Median p-value: {median_p:.4e}\n")                                                   # 第 4 行
        lines_to_write.append(f"TPR (Theory) @FPR={fpr_thres}: {acc_cnt_theory[wp_n]/tot_cnt[wp_n]:.4f}\n")          # 第 5 行
        
        if wp_n in empirical_thresholds:
            emp_tpr = acc_cnt_empirical[wp_n]/tot_cnt[wp_n]
            lines_to_write.append(f"Threshold (Empirical) @FPR={fpr_thres}: {empirical_thresholds[wp_n]:.2f}\n")     # 第 6 行
            lines_to_write.append(f"TPR (Empirical) @FPR={fpr_thres}: {emp_tpr:.4f}\n")                              # 第 7 行
        else:
            lines_to_write.append(f"Threshold (Empirical) @FPR={fpr_thres}: N/A\n")                                  # 第 6 行
            lines_to_write.append(f"TPR (Empirical) @FPR={fpr_thres}: N/A\n")                                        # 第 7 行

    with open(save_path, "w") as f:
        f.writelines(lines_to_write)

def get_lines(score_path, baseline_score_path, fpr_thres, len_limit):
    save_path = get_save_path(score_path, fpr_thres, len_limit)
    save_dir = "/".join(save_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)
    
    generate_result(score_path, baseline_score_path, save_path, fpr_thres, len_limit)

    with open(save_path, "r") as f:
        lines = f.readlines()
    return lines

def get_result_dict(score_path, baseline_score_path, fpr_thres, len_limit):
    lines = get_lines(score_path, baseline_score_path, fpr_thres, len_limit)

    line_num = len(lines)
    assert line_num % 7 == 0  # 现在严格对应 7 行了
    res_num = line_num // 7
    res_dict = {}
    
    for res_idx in range(res_num):
        # 按行号提取对应数据
        cur_n = extract_n_value(lines[res_idx * 7 + 1].strip())
        median_p = float((lines[res_idx * 7 + 3].strip()).split(":")[-1])
        tpr_theory = float((lines[res_idx * 7 + 4].strip()).split(":")[-1])
        
        thres_str = (lines[res_idx * 7 + 5].strip()).split(":")[-1].strip()
        tpr_emp_str = (lines[res_idx * 7 + 6].strip()).split(":")[-1].strip()
        
        tpr_emp = float(tpr_emp_str) if tpr_emp_str != "N/A" else "N/A"
        
        res_dict[f"MCMark(l={cur_n})"] = {
            "median_p": median_p, 
            "tpr_theory": tpr_theory, 
            "threshold": thres_str,
            "tpr_empirical": tpr_emp
        }
    return res_dict

def print_results(res_dict):
    print("\n>>> 最终评估结果 (Evaluation Results) <<<")
    for key in sorted(res_dict.keys()):
        print(f"{key}: {res_dict[key]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--baseline_score_path", type=str, default="")
    parser.add_argument("--fpr_thres", type=float, default=0.01)
    args = parser.parse_args()
    
    res_dict = get_result_dict(args.score_path, args.baseline_score_path, args.fpr_thres, len_limit=300)
    print_results(res_dict)

if __name__ == "__main__":
    main()