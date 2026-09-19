import pandas as pd
import itertools
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
import multiprocessing
from multiprocessing import Lock, Manager
import json
import os
# ==========================================
# 1. Configuration Section (keep original, only add PM_COL_INDEX)
# ==========================================
MAX_FEATURES = 9
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
OUTPUT_LOG = 'CatBoost_fast_log.txt'
TASK_STATE_FILE = "task_done_cat.json"
FIXED_COLS = 0
# Pm feature is the first column, index 0
PM_COL_INDEX = 0
# ========== Resume breakpoint parameters (修改：从8个特征开始，从第1个组合开始) ==========
RESUME_N = 1                 
LAST_FINISHED_COMBO_ID = 1  
# combination is 0-indexed, combo_idx = Combination ID - 1
start_combo_idx = LAST_FINISHED_COMBO_ID
# Parallel settings (keep original)
NUM_WORKERS = 32
CAT_N_THREADS = 1   # CatBoost internal threads set to 1, managed by external process pool
# Global variables for worker processes
global_X_train = None
global_X_test = None
global_y_train = None
global_y_test = None
global_lock = None
global_done_dict = None
# ==========================================
# 3. Fixed Model Hyperparameters (keep original, thread_count reserved)
# ==========================================
params = {
    'random_state': 20,
    'verbose': 0,
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 1,
    'auto_class_weights': 'Balanced',
    'min_data_in_leaf': 1,
    'thread_count': CAT_N_THREADS
}
def load_task_status():
    if os.path.exists(TASK_STATE_FILE):
        with open(TASK_STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()
def save_task_status(done_set):
    with open(TASK_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_set), f)
# ======================
# Worker initialization, inject dataset, lock and shared dict
# ======================
def init_worker(X_train, X_test, y_train, y_test, shared_lock, shared_done_dict):
    global global_X_train, global_X_test, global_y_train, global_y_test
    global global_lock, global_done_dict
    global_X_train = X_train
    global_X_test = X_test
    global_y_train = y_train
    global_y_test = y_test
    global_lock = shared_lock
    global_done_dict = shared_done_dict
# ======================
# Single feature combination training task
# ======================
def single_combo_task(task_key, n, combo_idx, selected_optional_tuple, fixed_cols):
    with global_lock:
        if task_key in global_done_dict:
            return {"skip": True, "n": n, "cid": combo_idx}
        global_done_dict[task_key] = True
    # Generate final column list: fixed columns + selected optional columns
    final_cols = fixed_cols + list(selected_optional_tuple)
    X_train = global_X_train.iloc[:, final_cols]
    X_test = global_X_test.iloc[:, final_cols]
    # Train model
    model = CatBoostClassifier(**params)
    model.fit(X_train, global_y_train)
    y_pred = model.predict(X_test)
    # Calculate metrics
    acc = accuracy_score(global_y_test, y_pred)
    f1 = f1_score(global_y_test, y_pred, average='weighted')
    combo_display_id = combo_idx + 1
    # Generate log message
    log_msg = (
        f"[Total Features: {n} | Combination ID: {combo_display_id}]\n"
        f"Used column indexes: {final_cols}\n"
        f"Used column names: {list(X_train.columns)}\n"
        f"Test Accuracy: {acc:.4f}\n"
        f"Test Weighted F1 Score: {f1:.4f}\n"
        f"{'-' * 40}\n"
    )
    # Write log with lock
    with global_lock:
        with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
            f.write(log_msg)
    return {
        "skip": False,
        "n": n,
        "cid": combo_display_id,
        "acc": acc,
        "f1": f1
    }
# Top-level wrapper function to avoid lambda pickle error
def task_wrapper(args):
    return single_combo_task(*args)
def main():
    start_total = time.time()
    # ==========================================
    # 2. Load Dataset (load only once)
    # ==========================================
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    total_feature_cols = train_data.shape[1] - 1  # Exclude label column
    # Validate max feature count
    if MAX_FEATURES > total_feature_cols:
        raise ValueError(f"Dataset only has {total_feature_cols} feature columns")
    # Load completed tasks for resume
    done_tasks = load_task_status()
    print(f"✅ Number of completed tasks in task state file: {len(done_tasks)}")
    # Build task list for execution
    task_list = []
    # ========== 修改：特征循环从8开始，到MAX_FEATURES结束 ==========
    for n in range(1, MAX_FEATURES + 1):
        # ======================
        # Core rule implementation
        # ======================
        if n == 1:
            # Rule 1: 1 feature -> exclude Pm, select 1 from non-Pm columns
            fixed_cols = []
            optional_cols = list(range(PM_COL_INDEX + 1, total_feature_cols))
            k = 1  # Number of optional columns to select
        else:
            # Rule 2: ≥2 features -> must include Pm, select remaining from non-Pm columns
            fixed_cols = [PM_COL_INDEX]  # Force include Pm column (index 0)
            optional_cols = list(range(PM_COL_INDEX + 1, total_feature_cols))
            k = n - 1  # Number of optional columns to select
        
        # Validate optional column availability
        if k <= 0:
            print(f"Skip n={n}, k={k} invalid")
            continue
        if k > len(optional_cols):
            print(f"Warning: Need to select {k} optional columns, but only {len(optional_cols)} available, skip n={n}.")
            continue
        
        # Generate all combinations for current n
        all_combos = list(itertools.combinations(optional_cols, k))
        # Resume breakpoint logic
        if n < RESUME_N:
            continue
        if n == RESUME_N:
            iterate_from = start_combo_idx
        else:
            iterate_from = 0
        # Add tasks to list (skip completed ones)
        for combo_idx, selected_optional_tuple in enumerate(all_combos[iterate_from:], start=iterate_from):
            task_key = f"{n}_{combo_idx}"
            if task_key in done_tasks:
                continue
            task_list.append((task_key, n, combo_idx, selected_optional_tuple, fixed_cols))
    # Print task summary
    print(f"📋 Total number of tasks to be executed this time: {len(task_list)}")
    if len(task_list) == 0:
        print("🎉 All traversal completed!")
        return
    # Initialize shared resources for multiprocessing
    mgr = Manager()
    shared_lock = mgr.Lock()
    shared_done_dict = mgr.dict()
    # Load completed tasks into shared dict
    for tk in done_tasks:
        shared_done_dict[tk] = True
    # Full feature dataset for worker initialization
    X_train_all = train_data
    X_test_all = test_data
    # Start process pool
    with multiprocessing.Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        initargs=(X_train_all, X_test_all, y_train, y_test, shared_lock, shared_done_dict)
    ) as pool:
        # Execute tasks with unordered imap
        for res in pool.imap_unordered(task_wrapper, task_list):
            if res["skip"]:
                continue
            print(f"Running | n={res['n']}, ComboID={res['cid']}, Acc={res['acc']:.4f}, F1={res['f1']:.4f}")
    # Save final completed tasks
    final_done = set(shared_done_dict.keys())
    save_task_status(final_done)
    # Calculate total running time
    end_total = time.time()
    # Write final log
    with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
        f.write(f"\nAll tasks finished! Total running time (resume part): {end_total - start_total:.2f} seconds\n")
    print(f"\n✅ Resume part all completed! Time elapsed: {end_total - start_total:.2f}s")
if __name__ == "__main__":
    main()
