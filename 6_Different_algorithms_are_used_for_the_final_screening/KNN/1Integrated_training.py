import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import random
warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
N_JOBS = 8
CHUNK_SIZE = 3
STEP_SIZE = 1
OVERLAP = CHUNK_SIZE - STEP_SIZE
TRAIN_PATH = "train1.csv"
TEST_PATH = "test1.csv"
CANDIDATE_PATH_1 = "data.csv"
SHUFFLED_TEMP_PATH = "aug_temp_shuffled.csv"
CV_FOLDS = 10
REPEAT_TIMES = 10  # 同标签替换重复10次，取均值
# ====================== ⭐ 固定最优超参数（KNN） ======================
KNN_BEST_PARAMS = {
    'n_neighbors': 6,
    'weights': 'distance',
    'metric': 'minkowski',
    'p': 3,
    'leaf_size': 24,
    'n_jobs': 1
}
def cv_eval_metrics(X, y, clf_params, n_splits=CV_FOLDS, seed=RANDOM_SEED):
    """分层10折交叉验证，返回Accuracy, Macro-F1, Macro-Recall均值"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_list, mf1_list, mrecall_list = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clf = KNeighborsClassifier(**clf_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        acc_list.append(accuracy_score(y_val, y_pred))
        mf1_list.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        mrecall_list.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(acc_list)), float(np.mean(mf1_list)), float(np.mean(mrecall_list))
def load_train_test(train_path, test_path):
    """读取带表头的训练、测试csv，最后一列为标签"""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.iloc[:, :-1].copy()
    y_train = df_train.iloc[:, -1].copy()
    X_test = df_test.iloc[:, :-1].copy()
    y_test = df_test.iloc[:, -1].copy()
    return X_train, y_train, X_test, y_test
def calc_baseline_metrics(X_tr, y_tr):
    """基线：原始训练集，10折CV得到指标"""
    return cv_eval_metrics(X_tr, y_tr, KNN_BEST_PARAMS, n_splits=CV_FOLDS, seed=RANDOM_SEED)
def replace_and_eval_chunk(X_base, y_base, chunk_df, label2idx):
    """
    同类别标签替换，重复REPEAT_TIMES次，取10次指标均值再返回
    - 每次：按标签分组，组内无放回抽样，保证3个替换位置互不相同
    - 重复REPEAT_TIMES轮替换评估，对acc/mf1/mrec分别求平均作为该chunk最终指标
    - 若某标签在训练集中不存在或数量不足，返回 None
    """
    chunk_X = chunk_df.iloc[:, :-1].copy().reset_index(drop=True)
    chunk_y = chunk_df.iloc[:, -1].copy().reset_index(drop=True)
    # ---- Step 1: 检查所有标签是否存在且数量充足 ----
    label_counts = chunk_y.value_counts().to_dict()
    for lbl, cnt in label_counts.items():
        if lbl not in label2idx or len(label2idx[lbl]) < cnt:
            return None  # 标签缺失或不足，无法执行有效替换
    # ---- Step2: 重复REPEAT_TIMES次替换+评估，记录每轮指标 ----
    repeat_acc = []
    repeat_mf1 = []
    repeat_mrec = []
    for _ in range(REPEAT_TIMES):
        # 按标签分组无放回抽样
        label_sampled = {}
        for lbl, cnt in label_counts.items():
            label_sampled[lbl] = np.random.choice(label2idx[lbl], size=cnt, replace=False)
        # 按chunk原始顺序映射替换索引
        replace_idx = np.empty(CHUNK_SIZE, dtype=int)
        label_ptr = {lbl:0 for lbl in label_counts}
        for i in range(CHUNK_SIZE):
            lbl = chunk_y.iloc[i]
            replace_idx[i] = label_sampled[lbl][label_ptr[lbl]]
            label_ptr[lbl] += 1
        # 执行替换
        X_mod = X_base.copy()
        y_mod = y_base.copy()
        X_mod.iloc[replace_idx, :] = chunk_X.values
        y_mod.iloc[replace_idx] = chunk_y.values
        # 计算本次替换后的CV指标
        acc, mf1, mrec = cv_eval_metrics(X_mod, y_mod, KNN_BEST_PARAMS, n_splits=CV_FOLDS, seed=RANDOM_SEED)
        repeat_acc.append(acc)
        repeat_mf1.append(mf1)
        repeat_mrec.append(mrec)
    # 10次结果取均值作为该窗口最终指标
    mean_acc = float(np.mean(repeat_acc))
    mean_mf1 = float(np.mean(repeat_mf1))
    mean_mrec = float(np.mean(repeat_mrec))
    return mean_acc, mean_mf1, mean_mrec
def process_chunk(chunk_idx, start, end, chunk_data,
                  X_base, y_base, label2idx,
                  base_metrics, col_names):
    """单窗口评估，多进程worker"""
    c = chunk_data.copy()
    c.columns = col_names
    results = {"chunk_idx": chunk_idx, "start": start, "end": end}
    base_acc, base_mf1, base_mrec = base_metrics
    eval_result = replace_and_eval_chunk(X_base, y_base, c, label2idx)
    # 容错：标签不匹配时标记为失败
    if eval_result is None:
        results.update({
            "acc_aug": base_acc, "mf1_aug": base_mf1, "mrec_aug": base_mrec,
            "cond_acc": False, "cond_mf1": False, "cond_mrec": False,
            "all_improved": False, "skip_reason": "label_missing_or_insufficient"
        })
        return results
    aug_acc, aug_mf1, aug_mrec = eval_result
    cond_acc = (aug_acc - base_acc) > 0.02
    cond_mrec = (aug_mrec - base_mrec) > 0.015
    cond_mf1 = (aug_mf1 - base_mf1) > 0.015
    all_improved = cond_acc and cond_mrec and cond_mf1
    results.update({
        "acc_aug": aug_acc, "mf1_aug": aug_mf1, "mrec_aug": aug_mrec,
        "cond_acc": cond_acc, "cond_mf1": cond_mf1, "cond_mrec": cond_mrec,
        "all_improved": all_improved
    })
    return results
def _init_aug_file(path, col_names):
    pd.DataFrame(columns=col_names).to_csv(path, index=False)
def dedupe_and_shuffle(df, random_seed=RANDOM_SEED):
    """去重（按整行）后打乱行的顺序，返回新DataFrame"""
    before = df.shape[0]
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    after = df_dedup.shape[0]
    print(f"  [去重] {before} 行 -> {after} 行 (去除 {before - after} 行重复)")
    df_shuffled = df_dedup.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df_shuffled
def run_augmentation_pipeline(candidate_path, aug_save_path, log_file,
                              X_base, y_base, label2idx,
                              base_metrics, col_names,
                              pipeline_tag="Pipeline",
                              has_header=True):
    print("\n" + "=" * 70)
    print(f"🚀 [{pipeline_tag}] 开始执行 | 每个窗口重复替换{REPEAT_TIMES}次取均值")
    print(f"   候选数据: {candidate_path}")
    print(f"   输出文件: {aug_save_path}")
    print(f"   日志文件: {log_file}")
    print("=" * 70)
    if has_header:
        data_full = pd.read_csv(candidate_path, header=0)
        data_full = data_full.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"✅ 原始候选集data.csv已完成行随机打乱，seed={RANDOM_SEED}")
        data_full.columns = col_names
        data_full = data_full.reset_index(drop=True)
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
            base_metrics, col_names
        ))
    num_chunks = len(tasks)
    print(f"滑动窗口数: {num_chunks}, Workers={N_JOBS}")
    print(f"\n[{pipeline_tag}] Step2: 并行评估窗口（同类别无放回替换，每窗口重复{REPEAT_TIMES}次取均值）+ {CV_FOLDS}折CV")
    results = joblib.Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        joblib.delayed(process_chunk)(*t) for t in tasks
    )
    results.sort(key=lambda r: r["chunk_idx"])
    # 统计跳过数
    skip_count = sum(1 for r in results if r.get("skip_reason"))
    print(f"\n[{pipeline_tag}] Step3: 筛选窗口 | 跳过窗口: {skip_count}/{num_chunks}")
    print("-" * 70)
    saved_count = 0
    saved_row_indices = set()
    # ========= 修改点：只要当前窗口all_improved，直接保存窗口中间行，不再判断相邻窗口 =========
    for r in results:
        if r["all_improved"]:
            save_row_idx = r["start"] + 1
            if save_row_idx not in saved_row_indices:
                save_df = data_full.iloc[[save_row_idx], :].copy()
                save_df.columns = col_names
                save_df.to_csv(aug_save_path, mode="a", header=False, index=False)
                saved_row_indices.add(save_row_idx)
                saved_count += 1
                print(f"✅ Chunk{r['chunk_idx']} pass: save row[{save_row_idx}]")
        else:
            print(f"❌ Chunk{r['chunk_idx']} fail")
    # 写日志
    with open(log_file, "w", encoding="utf-8") as log_fp:
        log_fp.write("=" * 70 + "\n")
        log_fp.write(f"[{pipeline_tag}]\n")
        log_fp.write(f"Model: KNeighborsClassifier\n")
        log_fp.write(f"Protocol: SAME-LABEL no-replacement + Repeat {REPEAT_TIMES} times per chunk + {CV_FOLDS} fold CV\n")
        log_fp.write(f"Criteria: Acc>0.03 AND Macro-Recall>0.02 AND Macro-F1>0.02\n")
        log_fp.write(f"Sliding window: chunk={CHUNK_SIZE}, step={STEP_SIZE}\n")
        log_fp.write(f"Save rule: single window pass → save window's middle single row\n\n")
        b_acc, b_mf1, b_mrec = base_metrics
        log_fp.write(f"Baseline(CV): Acc={b_acc:.6f}, Macro‑F1={b_mf1:.6f}, Macro‑Recall={b_mrec:.6f}\n")
        log_fp.write(f"Candidate rows: {total_rows}, windows: {num_chunks}, skipped: {skip_count}\n\n")
        log_fp.write("==== WINDOW DETAILS ====\n")
        for r in results:
            flag = "✅" if r["all_improved"] else "❌"
            skip_info = f" [SKIP:{r.get('skip_reason','')}]" if r.get("skip_reason") else ""
            line = (f"{flag} Win{r['chunk_idx']} [{r['start']}:{r['end']}] "
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
    print(f"任务：滑动窗口单轮筛选（KNN），**仅执行一轮，无迭代**")
    print(f"✅ 同类别无放回标签匹配替换；每个窗口随机替换{REPEAT_TIMES}次，取指标均值后与基线对比")
    print(f"窗口：每组3行，步长1")
    print(f"判定：Acc提升>0.03, Recall提升>0.02, MF1提升>0.02；**单个窗口满足阈值即保存中间行**")
    print("=" * 70)
    X_base, y_base, X_test, y_test = load_train_test(TRAIN_PATH, TEST_PATH)
    col_names = list(X_base.columns) + [y_base.name]
    print(f"原始训练集 shape: X={X_base.shape}, y={y_base.shape}")
    print(f"训练集标签分布:\n{y_base.value_counts().to_string()}")
    # ========= 预构建标签->行索引字典 =========
    label2idx = {}
    for lab in np.unique(y_base):
        label2idx[lab] = np.where(y_base.values == lab)[0]
    print("\n✅ 标签索引预构建完成：")
    for lab, idx_arr in label2idx.items():
        print(f"   Label {lab}: {len(idx_arr)} samples")
    # ---------- 基线 ----------
    base_acc, base_mf1, base_mrec = calc_baseline_metrics(X_base, y_base)
    base_metrics = (base_acc, base_mf1, base_mrec)
    print(f"\n【Baseline（10折CV）】Acc={base_acc:.6f}, MF1={base_mf1:.6f}, MRec={base_mrec:.6f}")
    aug_file = "augmentation1.csv"
    log_file = "aug_log_round1.txt"
    saved_cnt = run_augmentation_pipeline(
        candidate_path=CANDIDATE_PATH_1,
        aug_save_path=aug_file,
        log_file=log_file,
        X_base=X_base, y_base=y_base, label2idx=label2idx,
        base_metrics=base_metrics, col_names=col_names,
        pipeline_tag="SingleRound data.csv → augmentation1.csv",
        has_header=True
    )
    # ========= Final Summary =========
    print("\n" + "=" * 70)
    print("📊 Final Summary")
    if os.path.exists(aug_file):
        d = pd.read_csv(aug_file)
        print(f"  Output | {aug_file:<22} shape={d.shape} | log:{log_file}")
    else:
        print(f"  Output | {aug_file:<22} NOT GENERATED   | log:{log_file}")
    print("=" * 70)
    print("Done 🎉")
