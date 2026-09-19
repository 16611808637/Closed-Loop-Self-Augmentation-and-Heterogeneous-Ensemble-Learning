import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, recall_score, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import random

warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_JOBS = 8
CHUNK_SIZE = 4
STEP_SIZE = 1
CV_FOLDS = 10
N_RANDOM_REPLACES = 10  # 每个窗口替换评估10次取均值

# 文件路径
TRAIN_LGBM_PATH = "train1.csv"
TEST_LGBM_PATH = "test1.csv"
CANDIDATE_LGBM_PATH = "data.csv"
OUTPUT_FILE = "aug_lgbm.csv"
LOG_FILE = "aug_lgbm_log.txt"

# ====================== 模型超参数 ======================
# LightGBM最优参数
LGBM_BEST_PARAMS = {
    'random_state': RANDOM_SEED,
    'verbosity': -1,
    'n_estimators': 250,
    'max_depth': 8,
    'colsample_bytree': 0.8876,
    'class_weight': 'balanced',
    'min_child_samples': 15,
    'min_child_weight': 0.001,
    'learning_rate': 0.285,
    'reg_alpha': 0.2701,
    'reg_lambda': 3.2274,
    'n_jobs': 1
}

# ====================== 基础工具函数 ======================
def cv_eval_metrics(X, y, clf, clf_params, n_splits=CV_FOLDS, seed=RANDOM_SEED):
    """分层10折交叉验证，返回Accuracy, Macro-F1, Macro-Recall均值"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_list, mf1_list, mrecall_list = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = clf(**clf_params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        acc_list.append(accuracy_score(y_val, y_pred))
        mf1_list.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        mrecall_list.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(acc_list)), float(np.mean(mf1_list)), float(np.mean(mrecall_list))

def load_train_test(train_path, test_path):
    """读取csv，最后一列为标签"""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.iloc[:, :-1].copy()
    y_train = df_train.iloc[:, -1].copy()
    X_test = df_test.iloc[:, :-1].copy()
    y_test = df_test.iloc[:, -1].copy()
    return X_train, y_train, X_test, y_test

def calc_baseline_metrics(X_tr, y_tr, clf, clf_params):
    """计算基线CV指标"""
    return cv_eval_metrics(X_tr, y_tr, clf, clf_params, n_splits=CV_FOLDS, seed=RANDOM_SEED)

def replace_and_eval_chunk(X_base, y_base, chunk_df, label2idx, clf, clf_params):
    """同类别无放回替换，重复N_RANDOM_REPLACES次，返回平均指标；标签不足返回None"""
    chunk_X = chunk_df.iloc[:, :-1].copy().reset_index(drop=True)
    chunk_y = chunk_df.iloc[:, -1].copy().reset_index(drop=True)
    label_counts = chunk_y.value_counts().to_dict()
    # 校验标签是否存在、样本数量是否足够
    for lbl, cnt in label_counts.items():
        if lbl not in label2idx or len(label2idx[lbl]) < cnt:
            return None
    repeat_acc, repeat_mf1, repeat_mrec = [], [], []
    for _ in range(N_RANDOM_REPLACES):
        label_sampled = {}
        for lbl, cnt in label_counts.items():
            label_sampled[lbl] = np.random.choice(label2idx[lbl], size=cnt, replace=False)
        replace_idx = np.empty(CHUNK_SIZE, dtype=int)
        label_ptr = {lbl: 0 for lbl in label_counts}
        for i in range(CHUNK_SIZE):
            lbl = chunk_y.iloc[i]
            replace_idx[i] = label_sampled[lbl][label_ptr[lbl]]
            label_ptr[lbl] += 1
        X_mod = X_base.copy()
        y_mod = y_base.copy()
        X_mod.iloc[replace_idx, :] = chunk_X.values
        y_mod.iloc[replace_idx] = chunk_y.values
        acc, mf1, mrec = cv_eval_metrics(X_mod, y_mod, clf, clf_params, n_splits=CV_FOLDS, seed=RANDOM_SEED)
        repeat_acc.append(acc)
        repeat_mf1.append(mf1)
        repeat_mrec.append(mrec)
    mean_acc = float(np.mean(repeat_acc))
    mean_mf1 = float(np.mean(repeat_mf1))
    mean_mrec = float(np.mean(repeat_mrec))
    return mean_acc, mean_mf1, mean_mrec

def process_lgbm_chunk(
        chunk_idx, start, end,
        chunk_lgbm_data,
        X_base_lgbm, y_base_lgbm, label2idx_lgbm, base_lgbm_metrics,
        col_names_lgbm
):
    """LightGBM 单模型评估窗口
    返回窗口评估结果，包含是否通过阈值、窗口覆盖行
    """
    res = {
        "chunk_idx": chunk_idx,
        "start": start,
        "end": end,
        "window_rows": list(range(start, end)),
        "lgbm_pass": False,
        "all_improved": False,
        "skip_reason": None
    }
    chunk_lgbm_data.columns = col_names_lgbm
    lgbm_eval = replace_and_eval_chunk(X_base_lgbm, y_base_lgbm, chunk_lgbm_data, label2idx_lgbm,
                                       LGBMClassifier, LGBM_BEST_PARAMS)
    b_lgbm_acc, b_lgbm_mf1, b_lgbm_mrec = base_lgbm_metrics
    if lgbm_eval is None:
        res["skip_reason"] = "LGBM label missing/insufficient"
        return res
    lgbm_aug_acc, lgbm_aug_mf1, lgbm_aug_mrec = lgbm_eval
    # 阈值判断
    cond_acc_lgbm = (lgbm_aug_acc - b_lgbm_acc) > 0.048
    cond_mrec_lgbm = (lgbm_aug_mrec - b_lgbm_mrec) > 0.039
    cond_mf1_lgbm = (lgbm_aug_mf1 - b_lgbm_mf1) > 0.039
    lgbm_all_improved = cond_acc_lgbm and cond_mrec_lgbm and cond_mf1_lgbm

    res["lgbm_pass"] = lgbm_all_improved
    res["lgbm_acc"] = lgbm_aug_acc
    res["lgbm_mf1"] = lgbm_aug_mf1
    res["lgbm_mrec"] = lgbm_aug_mrec
    res["all_improved"] = lgbm_all_improved
    return res

def _init_aug_file(path, col_names):
    pd.DataFrame(columns=col_names).to_csv(path, index=False)

def run_lgbm_pipeline(
        cand_lgbm_path,
        save_lgbm_path, log_path,
        X_base_lgbm, y_base_lgbm, label2idx_lgbm, base_lgbm_metrics,
        col_names_lgbm,
        pipeline_tag="LGBM Pipeline"
):
    print("\n" + "=" * 80)
    print(f"🚀 [{pipeline_tag}] LightGBM 单模型筛选【单轮，窗口达标即保存窗口全部4行】")
    print(f"   LGBM候选: {cand_lgbm_path}")
    print(f"   LGBM输出: {save_lgbm_path}")
    print(f"   Log: {log_path}")
    print(f"   规则：窗口指标全部超过阈值，直接保存该窗口内全部4行，全局去重")
    print("=" * 80)

    df_cand_lgbm = pd.read_csv(cand_lgbm_path).reset_index(drop=True)
    # 候选集先随机打乱
    df_cand_lgbm = df_cand_lgbm.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"✅ 候选数据集data.csv已完成行随机打乱，seed={RANDOM_SEED}")
    df_cand_lgbm.columns = col_names_lgbm
    total_rows = df_cand_lgbm.shape[0]
    print(f"候选数据行数: {total_rows}")

    if total_rows < CHUNK_SIZE:
        print(f"⚠️ 候选数据不足窗口大小 {CHUNK_SIZE}，直接跳过")
        _init_aug_file(save_lgbm_path, col_names_lgbm)
        with open(log_path, "w", encoding="utf-8") as fp:
            fp.write(f"[{pipeline_tag}] 候选行数不足窗口，跳过\n")
        return 0

    _init_aug_file(save_lgbm_path, col_names_lgbm)
    tasks = []
    chunk_idx = 0
    for start in range(0, total_rows - CHUNK_SIZE + 1, STEP_SIZE):
        end = start + CHUNK_SIZE
        chunk_idx += 1
        chunk_lgbm = df_cand_lgbm.iloc[start:end, :].copy()
        tasks.append((
            chunk_idx, start, end,
            chunk_lgbm,
            X_base_lgbm, y_base_lgbm, label2idx_lgbm, base_lgbm_metrics,
            col_names_lgbm
        ))
    num_chunks = len(tasks)
    print(f"滑动窗口总数: {num_chunks}, Workers={N_JOBS}")
    print(f"\n[{pipeline_tag}] Step2: 并行窗口评估: 每个窗口评估{N_RANDOM_REPLACES}次取均值")
    results = joblib.Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        joblib.delayed(process_lgbm_chunk)(*t) for t in tasks
    )
    results.sort(key=lambda r: r["chunk_idx"])
    skip_count = sum(1 for r in results if r.get("skip_reason"))
    print(f"\n[{pipeline_tag}] Step3: 筛选达标窗口，收集窗口内全部行 | 跳过窗口数: {skip_count}/{num_chunks}")
    print("-" * 80)

    saved_row_indices = set()
    for r in results:
        if r["all_improved"]:
            # 当前窗口达标，保存窗口内全部4行
            for row_id in r["window_rows"]:
                if row_id not in saved_row_indices:
                    saved_row_indices.add(row_id)
                    row_data = df_cand_lgbm.iloc[[row_id], :].copy()
                    row_data.to_csv(save_lgbm_path, mode="a", header=False, index=False)
                    print(f"✅ Win{r['chunk_idx']} pass, save row[{row_id}]")
        else:
            print(f"❌ Win{r['chunk_idx']} fail")

    saved_count = len(saved_row_indices)
    # 写入日志
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("=" * 80 + "\n")
        log_fp.write(f"[{pipeline_tag}] LightGBM Single Filter | 单轮筛选，窗口达标即保存窗口所有行\n")
        log_fp.write(f"Rule: per chunk {N_RANDOM_REPLACES} random same-label replace, 10-fold CV\n")
        log_fp.write(f"Threshold: Acc>+0.03, MacroRecall>+0.02, MacroF1>+0.017\n")
        bl_acc, bl_mf1, bl_mrec = base_lgbm_metrics
        log_fp.write(f"LGBM Baseline CV: Acc={bl_acc:.6f}, MF1={bl_mf1:.6f}, MRec={bl_mrec:.6f}\n")
        log_fp.write(f"Total candidate rows:{total_rows}, windows:{num_chunks}, skipped:{skip_count}\n\n")
        log_fp.write("==== WINDOW DETAIL ====\n")
        for r in results:
            flag = "✅" if r["all_improved"] else "❌"
            sk = r.get("skip_reason", "")
            lgbm_ok = r["lgbm_pass"]
            line = f"{flag} Win{r['chunk_idx']}[{r['start']}:{r['end']}] rows:{r['window_rows']}, LGBM:{lgbm_ok}, Skip:{sk}\n"
            log_fp.write(line)
        log_fp.write(f"\nSaved unique rows = {saved_count}\n")

    print(f"\n[{pipeline_tag}] Summary:")
    if os.path.exists(save_lgbm_path):
        dfl = pd.read_csv(save_lgbm_path)
        print(f"  LGBM aug file shape: {dfl.shape}")
    print(f"  Saved rows count: {saved_count}")
    print(f"  Log saved to {log_path}")
    return saved_count

if __name__ == "__main__":
    print("=" * 80)
    print("LightGBM 单模型数据增强筛选流水线【单轮版本】")
    print("机制：滑动窗口(4行)，窗口指标全部超过阈值，保存窗口内全部4行，全局去重")
    print(f"窗口大小={CHUNK_SIZE}, 步长={STEP_SIZE}, 每个窗口评估重复{N_RANDOM_REPLACES}次")
    print("=" * 80)

    # 加载 LGBM 训练集
    X_lgbm_base, y_lgbm_base, _, _ = load_train_test(TRAIN_LGBM_PATH, TEST_LGBM_PATH)
    col_names_lgbm = list(X_lgbm_base.columns) + [y_lgbm_base.name]
    print(f"\nLGBM训练集 X={X_lgbm_base.shape}, y={y_lgbm_base.shape}")
    print("\nLGBM标签分布：")
    print(y_lgbm_base.value_counts().to_string())

    # 构建标签索引字典
    label2idx_lgbm = {}
    for lab in np.unique(y_lgbm_base):
        label2idx_lgbm[lab] = np.where(y_lgbm_base.values == lab)[0]

    # 计算 LightGBM 基线
    base_lgbm_acc, base_lgbm_mf1, base_lgbm_mrec = calc_baseline_metrics(
        X_lgbm_base, y_lgbm_base, LGBMClassifier, LGBM_BEST_PARAMS
    )
    base_lgbm_metrics = (base_lgbm_acc, base_lgbm_mf1, base_lgbm_mrec)
    print(f"\n【LGBM Baseline CV】Acc={base_lgbm_acc:.6f}, MF1={base_lgbm_mf1:.6f}, MRec={base_lgbm_mrec:.6f}")

    saved_cnt = run_lgbm_pipeline(
        cand_lgbm_path=CANDIDATE_LGBM_PATH,
        save_lgbm_path=OUTPUT_FILE,
        log_path=LOG_FILE,
        X_base_lgbm=X_lgbm_base, y_base_lgbm=y_lgbm_base,
        label2idx_lgbm=label2idx_lgbm, base_lgbm_metrics=base_lgbm_metrics,
        col_names_lgbm=col_names_lgbm,
        pipeline_tag="SingleRound LGBM filter data.csv"
    )

    # 最终汇总
    print("\n" + "=" * 80)
    print("📊 LightGBM 单模型筛选 单轮结果汇总")
    if os.path.exists(OUTPUT_FILE):
        dl = pd.read_csv(OUTPUT_FILE)
        print(f"Output | {OUTPUT_FILE:<24} shape={dl.shape} | log: {LOG_FILE}")
    else:
        print(f"Output | {OUTPUT_FILE} NOT GENERATED")
    print("=" * 80)
    print("✅ Pipeline Finished")
