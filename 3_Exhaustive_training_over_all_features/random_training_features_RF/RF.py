import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import re
import multiprocessing
from multiprocessing import Lock, Manager
import json
import os

# ==========================================
# 1. Configuration Section (Keep exactly the same as the original code)
# ==========================================
MAX_FEATURES = 9
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
OUTPUT_LOG = 'RF_training_log.txt'
TASK_STATE_FILE = "task_done_rf.json"
FIXED_COLS = 1  # Fixed first column (Pm) as mandatory feature for all combinations
# Parallel settings, slurm allocates 32 cores
NUM_WORKERS = 32
RF_N_JOBS = 1  # Disable RF internal threads, use external process pool
# Global variables for child processes
global_X_train = None
global_X_test = None
global_y_train = None
global_y_test = None
global_lock = None
global_done_dict = None

# ==========================================
# 3. Fixed Hyperparameter Settings
# ==========================================
rf_params = {
    'n_estimators': 400,
    'max_depth': 15,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': 20,
    'n_jobs': RF_N_JOBS
}

# ==========================================
# ✅ Breakpoint auto-parse function: Find the last completed position from the log
# ==========================================
def get_breakpoint(log_file):
    pat = re.compile(r'\[Total Features:\s*(\d+)\s*\|\s*Combination ID:\s*(\d+)\]')
    last_n = None
    last_cid = None
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    last_n = int(m.group(1))
                    last_cid = int(m.group(2))
    except FileNotFoundError:
        pass
    return last_n, last_cid

def load_task_status():
    if os.path.exists(TASK_STATE_FILE):
        with open(TASK_STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_task_status(done_set):
    with open(TASK_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_set), f)

# ======================
# Worker initialization: Inject dataset, lock and shared dict
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
    # Fixed columns (including mandatory Pm) + selected optional columns
    final_cols = list(range(fixed_cols)) + list(selected_optional_tuple)
    X_train = global_X_train.iloc[:, final_cols]
    X_test = global_X_test.iloc[:, final_cols]
    # Train model
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, global_y_train)
    best_model = model
    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(global_y_test, y_pred)
    f1 = f1_score(global_y_test, y_pred, average='weighted')
    # Model filename (save logic reserved but commented out)
    model_name = f'RF_Top{n}_Combo{combo_idx}.pkl'
    # joblib.dump(best_model, model_name)
    # Generate log message
    log_msg = (
        f"[Total Features: {n} | Combination ID: {combo_idx}]\n"
        f"Used column names: {list(X_train.columns)}\n"
        f"Test Accuracy: {acc:.4f}\n"
        f"Test Weighted F1 Score: {f1:.4f}\n"
        f"Model filename: {model_name}\n\n"
    )
    # Write log with lock
    with global_lock:
        with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
            f.write(log_msg)
    return {
        "skip": False,
        "n": n,
        "cid": combo_idx,
        "acc": acc,
        "f1": f1
    }

# Top-level wrapper function to avoid lambda pickle error
def task_wrapper(args):
    return single_combo_task(*args)

def main():
    # ==========================================
    # 2. Load Dataset
    # ==========================================
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    total_feature_cols = train_data.shape[1] - 1
    if MAX_FEATURES > total_feature_cols:
        raise ValueError(f"Error: Dataset only has {total_feature_cols} feature columns, but target feature number is {MAX_FEATURES}.")
    # Optional columns: all columns after the fixed Pm column
    optional_col_indices = list(range(FIXED_COLS, total_feature_cols))
    # Get breakpoint info
    break_n, break_cid = get_breakpoint(OUTPUT_LOG)
    done_tasks = load_task_status()
    print(f"✅ Breakpoint detected: Total Features={break_n}, Last Completed Combination ID={break_cid}")
    print(f"✅ Number of completed tasks in state file: {len(done_tasks)}")
    # Build full task list
    task_list = []
    for n in range(1, MAX_FEATURES + 1):
        # Number of optional columns to select (fixed Pm is already included)
        k = n - FIXED_COLS
        if k > len(optional_col_indices):
            continue
        combo_gen = itertools.combinations(optional_col_indices, k)
        combo_idx = 0
        for selected_optional_tuple in combo_gen:
            combo_idx += 1
            task_key = f"{n}_{combo_idx}"
            # Skip tasks completed before breakpoint
            skip_by_log = False
            if break_n is not None and break_cid is not None:
                if n < break_n or (n == break_n and combo_idx <= break_cid):
                    skip_by_log = True
            # Skip if already completed
            if skip_by_log or task_key in done_tasks:
                continue
            task_list.append((task_key, n, combo_idx, selected_optional_tuple, FIXED_COLS))
    print(f"📋 Total number of tasks to execute this run: {len(task_list)}")
    if len(task_list) == 0:
        print("🎉 All combinations have been traversed!")
        return
    # Initialize shared resources
    mgr = Manager()
    shared_lock = mgr.Lock()
    shared_done_dict = mgr.dict()
    for tk in done_tasks:
        shared_done_dict[tk] = True
    # Start parallel pool
    X_train_all = train_data
    X_test_all = test_data
    with multiprocessing.Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        initargs=(X_train_all, X_test_all, y_train, y_test, shared_lock, shared_done_dict)
    ) as pool:
        for res in pool.imap_unordered(task_wrapper, task_list):
            if res["skip"]:
                continue
            print(f"▶ n={res['n']}, ComboID={res['cid']}, Acc={res['acc']:.4f}, F1={res['f1']:.4f}")
    # Save final task status
    final_done = set(shared_done_dict.keys())
    save_task_status(final_done)
    print("\n🎉 All combinations have been traversed!")

if __name__ == "__main__":
    main()
