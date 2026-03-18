import json
import os
import numpy as np
from collections import defaultdict
import math
import re
import numpy as np
import argparse
# 新增导入
from sklearn.metrics import roc_auc_score

def get_fpr(n, m):
    res = 0
    for i in range(m, n + 1):
        res += math.comb(n, i) * (2 ** (n - i))
    res /= 3**n
    return res


def get_split_fpr(n, m, split_num):
    res = 0
    for i in range(m, n + 1):
        res += math.comb(n, i) * ((split_num - 1) ** (n - i))
    res /= split_num**n
    return res


def extract_n_value(text):
    pattern = re.search(r"MC_Reweight\(n=(\d+)\)", text)
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


def generate_result(score_path, save_path, fpr_thres, len_limit):
    with open(score_path, "r") as f:
        lines = f.readlines()

    tot_cnt = defaultdict(int)
    acc_cnt = defaultdict(int)
    fpr_list = defaultdict(list)

    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]

        wp = res_dict["watermark_processor"]

        if cur_len < len_limit:
            continue
        if "MC_Reweight" not in wp:
            continue

        raw_score = res_dict["raw_scores"]
        cur_n = extract_n_value(wp)

        cur_fpr = get_split_fpr(int(cur_len), int(raw_score), split_num=cur_n)
        fpr_list[cur_n].append(cur_fpr)

        if cur_fpr <= fpr_thres:
            acc_cnt[cur_n] += 1
        tot_cnt[cur_n] += 1

    lines = []
    for wp in sorted(tot_cnt.keys()):



        # 1. 获取带有水印文本的 p-value (正样本)
        pos_p_vals = fpr_list[wp]
        
        # 2. 模拟人类文本的 p-value (负样本)，理论上服从 0~1 的均匀分布
        # 为了让 AUC 计算更精确稳定，我们生成比正样本多 100 倍的负样本
        neg_p_vals = np.random.uniform(0, 1, len(pos_p_vals) * 100)
        
        # 3. 拼接真实标签 (正样本标为 1，负样本标为 0)
        y_true = [1] * len(pos_p_vals) + [0] * len(neg_p_vals)
        
        # 4. 拼接得分。注意：由于 p-value 越小越说明是水印，
        # 而 sklearn 默认分数越大越是正类，因此我们加上负号作为得分
        y_scores = [-p for p in pos_p_vals] + [-p for p in neg_p_vals]
        
        # 5. 计算标准的 ROC 曲线下面积 AUC
        auc_value = roc_auc_score(y_true, y_scores)



        
        lines.append("-" * 80 + "\n")
        lines.append(f"MC_Reweight(n={wp})\n")
        lines.append(f"Total cnt: {tot_cnt[wp]}\n")
        lines.append(f"Median p-value: {np.median(fpr_list[wp])}\n")
        lines.append(f"TPR@FPR={fpr_thres}: {acc_cnt[wp]/tot_cnt[wp]}\n")
        # 新增将 AUC 写入到输出文件
        lines.append(f"AUC: {auc_value:.4f}\n")
    # return lines

    with open(save_path, "w") as f:
        f.writelines(lines)


def get_lines(score_path, fpr_thres, len_limit):
    save_path = get_save_path(score_path, fpr_thres, len_limit)

    if not os.path.exists(save_path):
        save_dir = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)
        generate_result(score_path, save_path, fpr_thres, len_limit)

    with open(save_path, "r") as f:
        lines = f.readlines()
    return lines


def get_result_dict(score_path, fpr_thres, len_limit):
    lines = get_lines(score_path, fpr_thres, len_limit)

    line_num = len(lines)


    assert line_num % 6 == 0
    res_num = line_num // 6
    res_dict = {}
    for res_idx in range(res_num):
        cur_n = extract_n_value(lines[res_idx * 6 + 1].strip())
        median_p = float((lines[res_idx * 6 + 3].strip()).split(":")[-1])
        tpr = float((lines[res_idx * 6 + 4].strip()).split(":")[-1])
        
        # 解析第 6 行的 AUC 值
        auc = float((lines[res_idx * 6 + 5].strip()).split(":")[-1])
        
        # 将 auc 加入返回字典
        res_dict[f"MCMark(l={cur_n})"] = {"median_p": median_p, "tpr": tpr, "auc": auc}
    return res_dict
    # assert line_num % 5 == 0
    # res_num = line_num // 5
    # res_dict = {}
    # for res_idx in range(res_num):
    #     cur_n = extract_n_value(lines[res_idx * 5 + 1].strip())
    #     median_p = float((lines[res_idx * 5 + 3].strip()).split(":")[-1])
    #     tpr = float((lines[res_idx * 5 + 4].strip()).split(":")[-1])
    #     res_dict[f"MCMark(l={cur_n})"] = {"median_p": median_p, "tpr": tpr}
    # return res_dict


def print_results(res_dict):
    for key in sorted(res_dict.keys()):
        print(f"{key}: {res_dict[key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--fpr_thres", type=float)
    args = parser.parse_args()
    # res_dict = get_result_dict(args.score_path, args.fpr_thres, len_limit=510)
    res_dict = get_result_dict(args.score_path, args.fpr_thres, len_limit=350)
    print_results(res_dict)


if __name__ == "__main__":
    main()
