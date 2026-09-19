import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
import os
import random
warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
N_JOBS = 8
CHUNK_SIZE = 4
STEP_SIZE = 1
CV_FOLDS = 10
REPEAT_TIMES = 10
# 移除迭代相关 MAX_ROUND、MIN_HIT_COUNT、SHUFFLED_TEMP_PATH
TRAIN_PATH = "train1.csv"
TEST_PATH = "test1.csv"
CANDIDATE_PATH_1 = "data.csv"
OUTPUT_FILE = "augmentation_xgb.csv"
LOG_FILE = "aug_log_xgb.txt"

# ====================== ⭐ 固定最优超参数（XGBoost） ======================
XGB_BEST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'n_jobs': 1,
    'random_state': 20,
    'verbosity': 0
}
# ⭐ 全局标签编码器（在main中fit一次，全程复用）
LABEL_ENCODER = LabelEncoder()

def cv_eval_metrics(X, y_raw, le, clf_params, n_splits=CV_FOLDS, seed=RANDOM_SEED):
    """分层10折交叉验证，返回Accuracy, Macro-F1, Macro-Recall均值"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_list, mf1_list, mrecall_list = [], [], []
    for train_idx, val_idx in skf.split(X, y_raw):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_raw, y_val_raw = y_raw.iloc[train_idx], y_raw.iloc[val_idx]
        y_tr = le.transform(y_tr_raw)
        y_val = le.transform(y_val_raw)
        clf = xgb.XGBClassifier(**clf_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        acc_list.append(accuracy_score(y_val, y_pred))
        mf1_list.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        mrecall_list.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(acc_list)), float(np.mean(mf1_list)), float(np.mean(mrecall_list))

def load_train_test(train_path, test_path):
    """读取带表头的训练、测试csv，最后一列为字符串标签"""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.iloc[:, :-1].copy()
    y_train = df_train.iloc[:, -1].copy()
    X_test = df_test.iloc[:, :-1].copy()
    y_test = df_test.iloc[:, -1].copy()
    return X_train, y_train, X_test, y_test

def calc_baseline_metrics(X_tr, y_tr, le):
    """基线：原始训练集，10折CV得到指标"""
    return cv_eval_metrics(X_tr, y_tr, le, XGB_BEST_PARAMS, n_splits=CV_FOLDS, seed=RANDOM_SEED)

def replace_and_eval_chunk(X_base, y_base, chunk_df, label2idx, le, chunk_idx):
    """同类别标签替换 × REPEAT_TIMES 次取均值"""
    chunk_X = chunk_df.iloc[:, :-1].copy().reset_index(drop=True)
    chunk_y = chunk_df.iloc[:, -1].copy().reset_index(drop=True)
    label_counts = chunk_y.value_counts().to_dict()
    for lbl in label_counts:
        if lbl not in label2idx or len(label2idx[lbl]) == 0:
            return None
    acc_list, mf1_list, mrec_list = [], [], []
    for rep in range(REPEAT_TIMES):
        rep_seed = RANDOM_SEED + chunk_idx * 100000 + rep
        rng = np.random.default_rng(rep_seed)
        valid = True
        for lbl, cnt in label_counts.items():
            if len(label2idx[lbl]) < cnt:
                valid = False
                break
        if not valid:
            continue
        label_sampled = {}
        for lbl, cnt in label_counts.items():
            label_sampled[lbl] = rng.choice(label2idx[lbl], size=cnt, replace=False)
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
        acc, mf1, mrec = cv_eval_metrics(X_mod, y_mod, le, XGB_BEST_PARAMS,
                                         n_splits=CV_FOLDS, seed=RANDOM_SEED)
        acc_list.append(acc)
        mf1_list.append(mf1)
        mrec_list.append(mrec)
    if len(acc_list) == 0:
        return None
    return (
        float(np.mean(acc_list)),
        float(np.mean(mf1_list)),
        float(np.mean(mrec_list))
    )

def process_chunk(chunk_idx, start, end, chunk_data,
                  X_base, y_base, label2idx,
                  base_metrics, col_names, le):
    """单窗口评估，多进程worker"""
    c = chunk_data.copy()
    c.columns = col_names
    results = {
        "chunk_idx": chunk_idx,
        "start": start,
        "end": end,
        "window_rows": list(range(start, end))
    }
    base_acc, base_mf1, base_mrec = base_metrics
    eval_result = replace_and_eval_chunk(X_base, y_base, c, label2idx, le, chunk_idx)
    if eval_result is None:
        results.update({
            "acc_aug": base_acc, "mf1_aug": base_mf1, "mrec_aug": base_mrec,
            "cond_acc": False, "cond_mf1": False, "cond_mrec": False,
            "all_improved": False, "skip_reason": "label_missing_or_insufficient"
        })
        return results
    aug_acc, aug_mf1, aug_mrec = eval_result
    cond_acc = (aug_acc - base_acc) > 0.035
    cond_mrec = (aug_mrec - base_mrec) > 0.03
    cond_mf1 = (aug_mf1 - base_mf1) > 0.035
    all_improved = cond_acc and cond_mrec and cond_mf1
    results.update({
        "acc_aug": aug_acc, "mf1_aug": aug_mf1, "mrec_aug": aug_mrec,
        "cond_acc": cond_acc, "cond_mf1": cond_mf1, "cond_mrec": cond_mrec,
        "all_improved": all_improved
    })
    return results

def _init_aug_file(path, col_names):
    pd.DataFrame(columns=col_names).to_csv(path, index=False)

def run_augmentation_pipeline(candidate_path, aug_save_path, log_file,
                              X_base, y_base, label2idx,
                              base_metrics, col_names, le,
                              pipeline_tag="Pipeline",
                              has_header=True):
    print("\n" + "=" * 70)
    print(f"🚀 [{pipeline_tag}] 开始执行【单轮，窗口达标直接保存窗口全部行】")
    print(f"   候选数据: {candidate_path}")
    print(f"   输出文件: {aug_save_path}")
    print(f"   日志文件: {log_file}")
    print(f"   保存规则: 窗口满足全部阈值条件，直接保存窗口内所有样本，全局去重")
    print("=" * 70)
    if has_header:
        data_full = pd.read_csv(candidate_path, header=0)
        data_full.columns = col_names
        # ========= 修改点1：读取data.csv后立刻随机打乱 =========
        data_full = data_full.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print("✅ data.csv 已完成行随机打乱")
    else:
        data_full = pd.read_csv(candidate_path, header=None)
        data_full.columns = col_names

    total_rows = data_full.shape[0]
    print(f"候选数据行数: {total_rows}")
    if total_rows < CHUNK_SIZE:
        print(f"⚠️ 候选数据不足一个窗口 ({total_rows}<{CHUNK_SIZE})，跳过。")
        _init_aug_file(aug_save_path, col_names)
        with open(log_file, "w", encoding="utf-8") as fp:
            fp.write(f"[{pipeline_tag}] 候选数据不足窗口大小，跳过\n")
        return 0
    _init_aug_file(aug_save_path, col_names)
    tasks = []
    chunk_idx = 0
    for start in range(0, total_rows - CHUNK_SIZE + 1, STEP_SIZE):
        end = start + CHUNK_SIZE
        chunk_idx += 1
        tasks.append((
            chunk_idx, start, end,
            data_full.iloc[start:end, :].copy(),
            X_base, y_base, label2idx,
            base_metrics, col_names, le
        ))
    num_chunks = len(tasks)
    print(f"滑动窗口数: {num_chunks}, Workers={N_JOBS}")
    print(f"\n[{pipeline_tag}] Step2: 并行评估窗口（同类别无放回替换×{REPEAT_TIMES}次均值）+ {CV_FOLDS}折CV")
    results = joblib.Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        joblib.delayed(process_chunk)(*t) for t in tasks
    )
    results.sort(key=lambda r: r["chunk_idx"])
    skip_count = sum(1 for r in results if r.get("skip_reason"))
    print(f"\n[{pipeline_tag}] Step3: 窗口达标直接保存窗口内全部行，集合去重 | 跳过窗口: {skip_count}/{num_chunks}")
    print("-" * 70)
    saved_row_set = set()
    saved_count = 0
    passed_window_count = 0
    for r in results:
        if r["all_improved"]:
            passed_window_count += 1
            # ========= 修改点2：窗口满足阈值，直接保存窗口所有行，集合去重避免重复写入 =========
            for row_id in r["window_rows"]:
                if row_id not in saved_row_set:
                    save_df = data_full.iloc[[row_id], :].copy()
                    save_df.columns = col_names
                    save_df.to_csv(aug_save_path, mode="a", header=False, index=False)
                    saved_row_set.add(row_id)
                    saved_count += 1
                    print(f"✅ Win{r['chunk_idx']} pass, save row[{row_id}]")
        else:
            print(f"❌ Win{r['chunk_idx']} fail")

    # 写日志
    with open(log_file, "w", encoding="utf-8") as log_fp:
        log_fp.write("=" * 70 + "\n")
        log_fp.write(f"[{pipeline_tag}]\n")
        log_fp.write(f"Model: XGBClassifier\n")
        log_fp.write(f"Protocol: SAME-LABEL no-replacement × {REPEAT_TIMES} repeats avg + {CV_FOLDS} fold CV\n")
        log_fp.write(f"Criteria: Acc-base>0.02 AND Macro-Recall-base>0.01 AND Macro-F1-base>0.02\n")
        log_fp.write(f"Sliding window: chunk={CHUNK_SIZE}, step={STEP_SIZE}\n")
        log_fp.write(f"Save rule: window meet all thresholds, save all rows inside window, unique\n\n")
        b_acc, b_mf1, b_mrec = base_metrics
        log_fp.write(f"Baseline(CV): Acc={b_acc:.6f}, Macro‑F1={b_mf1:.6f}, Macro‑Recall={b_mrec:.6f}\n")
        log_fp.write(f"Candidate rows: {total_rows}, windows: {num_chunks}, skipped: {skip_count}\n")
        log_fp.write(f"Passed windows: {passed_window_count}/{num_chunks}\n\n")
        log_fp.write("==== WINDOW DETAILS ====\n")
        for r in results:
            flag = "✅" if r["all_improved"] else "❌"
            skip_info = f" [SKIP:{r.get('skip_reason', '')}]" if r.get("skip_reason") else ""
            line = (f"{flag} Win{r['chunk_idx']} [{r['start']}:{r['end']}] "
                    f"rows:{r['window_rows']} "
                    f"Acc={r['acc_aug']:.4f}({r['cond_acc']}) "
                    f"MF1={r['mf1_aug']:.4f}({r['cond_mf1']}) "
                    f"MRec={r['mrec_aug']:.4f}({r['cond_mrec']}){skip_info}\n")
            log_fp.write(line)
        log_fp.write(f"\nSummary: saved unique rows = {saved_count}\n")
    print(f"\n[{pipeline_tag}] Summary:")
    if os.path.exists(aug_save_path):
        df_out = pd.read_csv(aug_save_path)
        print(f"  Output shape: {df_out.shape} -> {aug_save_path}")
    print(f"  Saved unique candidate rows: {saved_count}")
    print(f"  Log: {log_file}")
    return saved_count

if __name__ == "__main__":
    print("=" * 70)
    print(f"任务：XGBoost滑动窗口筛选【单轮版本】")
    print(f"✅ 同类别无放回标签匹配替换模式 × {REPEAT_TIMES}次取均值，支持字符串标签")
    print(f"窗口：每组{CHUNK_SIZE}行，步长{STEP_SIZE}")
    print(f"判定：Acc>0.02, Recall>0.01, MF1>0.02")
    print(f"保存：窗口满足全部阈值，直接保存窗口内全部样本，自动去重")
    print("=" * 70)
    X_base, y_base, X_test, y_test = load_train_test(TRAIN_PATH, TEST_PATH)
    col_names = list(X_base.columns) + [y_base.name]
    print(f"原始训练集 shape: X={X_base.shape}, y={y_base.shape}")
    print(f"训练集标签分布:\n{y_base.value_counts().to_string()}")
    # ========= 全局LabelEncoder，仅用原始训练集拟合一次 =========
    LABEL_ENCODER.fit(y_base)
    print(f"\n✅ LabelEncoder 类别映射：")
    for cls, code in zip(LABEL_ENCODER.classes_, range(len(LABEL_ENCODER.classes_))):
        print(f"   {cls} → {code}")
    # ========= 预构建标签->行索引字典（字符串label作为key） =========
    label2idx = {}
    for lab in np.unique(y_base):
        label2idx[lab] = np.where(y_base.values == lab)[0]
    print("\n✅ 标签索引预构建完成：")
    for lab, idx_arr in label2idx.items():
        print(f"   Label {lab}: {len(idx_arr)} samples")
    # ---------- 基线 ----------
    base_acc, base_mf1, base_mrec = calc_baseline_metrics(X_base, y_base, LABEL_ENCODER)
    base_metrics = (base_acc, base_mf1, base_mrec)
    print(f"\n【Baseline（10折CV）】Acc={base_acc:.6f}, MF1={base_mf1:.6f}, MRec={base_mrec:.6f}")

    # 只执行一轮，删除迭代循环
    saved_cnt = run_augmentation_pipeline(
        candidate_path=CANDIDATE_PATH_1,
        aug_save_path=OUTPUT_FILE,
        log_file=LOG_FILE,
        X_base=X_base, y_base=y_base, label2idx=label2idx,
        base_metrics=base_metrics, col_names=col_names, le=LABEL_ENCODER,
        pipeline_tag="SingleRound XGB filter data.csv",
        has_header=True
    )

    # ========= Final Summary =========
    print("\n" + "=" * 70)
    print("📊 Final Summary")
    if os.path.exists(OUTPUT_FILE):
        d = pd.read_csv(OUTPUT_FILE)
        print(f"  Output | {OUTPUT_FILE:<22} shape={d.shape} | log:{LOG_FILE}")
    else:
        print(f"  Output | {OUTPUT_FILE} NOT GENERATED")
    print("=" * 70)
    print("Done 🎉")
