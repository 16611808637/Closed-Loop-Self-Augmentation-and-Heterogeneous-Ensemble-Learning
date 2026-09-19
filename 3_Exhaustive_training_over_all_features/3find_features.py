import os
import csv
from typing import Dict, List, Tuple, Set
# Store: feature group name list + corresponding F1_Score score
FeatureGroup = Tuple[List[str], float]
MIN_FEATURE_COUNT = 1   # 最少特征数阈值

def process_single_file(file_path: str) -> Dict[str, FeatureGroup]:
    """
    Process single log file: parse 'Used column names:' and 'F1 Score:'
    Extract all feature list + corresponding F1_Score score, NO threshold filter
    Tolerate extra metrics after F1_Score line, skip entry if parsing fails
    Support unicode special characters such as δ γ μ
    """
    feature_groups = {}
    current_features: List[str] = []
    current_acc: float = -1.0
    # Try utf‑8 first, fallback to gbk for special characters
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        # Parse feature column line: Used column names:
        if line.startswith('Used column names: ['):
            content = line.split('[')[-1].replace("]", "").strip()
            if content:
                feat_list = [f.strip(" '") for f in content.split(",")]
                current_features = feat_list
        # Extract F1_Score value, tolerate trailing extra metrics
        elif 'F1 Score: ' in line:
            acc_part = line.split('F1 Score: ')[-1].strip()
            # Truncate by | or whitespace, take leading numeric segment
            acc_part = acc_part.split("|")[0].split()[0].strip()
            try:
                current_acc = float(acc_part)
            except ValueError:
                # Parse failed, reset state and skip this group
                current_features = []
                current_acc = -1.0
                continue
            # ===== 保留所有解析成功的特征组，不做过滤 =====
            if current_features:
                feature_groups[f"group_{idx}"] = (current_features.copy(), current_acc)
                current_features = []
                current_acc = -1.0
    return feature_groups


def pick_best_with_pm(feature_dict: Dict[str, FeatureGroup]) -> Tuple[List[str], float]:
    """
    统一筛选规则：必须包含"Pm"特征 + 特征数≥MIN_FEATURE_COUNT，选取F1 Score最高的组合
    适用于所有模型（KNN/CATboost/LightGBM/RF/XGBoost）
    """
    candidates = []
    for feats, score in feature_dict.values():
        # 核心筛选条件：同时满足含Pm + 特征数≥8
        if "Pm" in feats and len(feats) >= MIN_FEATURE_COUNT:
            candidates.append((feats, score))
    if not candidates:
        raise ValueError(
            f"No valid feature group found! Requirement: contain 'Pm' + at least {MIN_FEATURE_COUNT} features"
        )
    # 按F1 Score降序排序，取Top1
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def main():
    out_file = "out.txt"
    # File paths and corresponding model names for 5 models
    file_infos = [
        (r"D:\Papers_related_materials\second_paper\Exhaustive_training_over_all_features\random_training_features_KNN\KNN_training_log.txt", "KNN"),
        (r"D:\Papers_related_materials\second_paper\Exhaustive_training_over_all_features\random_training_features_CATboost\CatBoost_fast_log.txt", "CATboost"),
        (r"D:\Papers_related_materials\second_paper\Exhaustive_training_over_all_features\random_training_features_LightGBM\LGBM_fast_log.txt", "LightGBM"),
        (r"D:\Papers_related_materials\second_paper\Exhaustive_training_over_all_features\random_training_features_RF\RF_training_log.txt", "RF"),
        (r"D:\Papers_related_materials\second_paper\Exhaustive_training_over_all_features\random_training_features_XGBoost\XGB_training_log.txt", "XGBoost")
    ]
    best_combination = []
    sum_f1 = 0.0
    try:
        for file_path, model_name in file_infos:
            print(f"\nProcessing file: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            feature_group_dict = process_single_file(file_path)
            print(f"Processing complete, extracted {len(feature_group_dict)} parsed feature groups (no F1 filter)")
            
            # 所有模型统一使用相同的筛选规则
            best_feats, best_score = pick_best_with_pm(feature_group_dict)
            print(f"----> {model_name} selected best group (contain 'Pm', >= {MIN_FEATURE_COUNT} features), F1={best_score:.4f}")
            
            best_combination.append((model_name, best_feats, best_score))
            sum_f1 += best_score
        # Output result
        output_lines = []
        line_sep = "=" * 80
        output_lines.append(line_sep)
        output_lines.append("Optimal Feature Combination Result")
        output_lines.append(line_sep)
        # 更新规则说明，与新逻辑一致
        output_lines.append(f"Uniform Rule for all models: must contain 'Pm' feature + at least {MIN_FEATURE_COUNT} features, select max F1 Score")
        output_lines.append(f"Sum of F1_Score scores across 5 models: {sum_f1:.4f}")
        output_lines.append("")
        output_lines.append("Selected feature group and F1_Score for each model:")
        for model_name, feat_list, acc_score in best_combination:
            feat_str = ", ".join(feat_list)
            output_lines.append(f" {model_name}: [{feat_str}] (F1 Score: {acc_score:.4f}, Feature count:{len(feat_list)})")
            print(f"[DEBUG] {model_name} feature list: {feat_list}, feature_num:{len(feat_list)}")
        output_lines.append(line_sep)
        print("\n" + "\n".join(output_lines))
        # Write output text file with utf‑8 encoding
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nText result saved to: {os.path.abspath(out_file)}")
    except Exception as e:
        err_msg = f"\nError: {e}"
        print(err_msg)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(err_msg)


if __name__ == "__main__":
    main()
