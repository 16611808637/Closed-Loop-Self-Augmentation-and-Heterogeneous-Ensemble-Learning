# -*- coding: utf-8 -*-
import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ==========================================
# 1. Configuration Section
# ==========================================
np.random.seed(42)
MAX_FEATURES = 9
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
OUTPUT_LOG = 'KNN_training_log.txt'
PM_COL_INDEX = 0
FIXED_COLS = 0

knn_params = {
    "n_neighbors": 15,
    "weights": "distance",
    "metric": "euclidean",
    "p": 2
}

# ==========================================
# 2. Load Dataset
# ==========================================
print("Loading data...")
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]
total_feature_cols = train_data.shape[1] - 1

if MAX_FEATURES > total_feature_cols:
    raise ValueError(f"Error: Dataset only contains {total_feature_cols} feature columns, but MAX_FEATURES is set to {MAX_FEATURES}.")

print(f"Pm feature column index: {PM_COL_INDEX}")
print(f"Total feature columns: {total_feature_cols}, index range: 0 to {total_feature_cols-1}")

best_result = {"acc": -1, "combo": None, "cols": None}

# ==========================================
# 3. Main Loop
# Rule: n=1 -> without Pm; n>=2 -> must include Pm
# ==========================================
with open(OUTPUT_LOG, 'w', encoding="utf-8") as f_log:
    f_log.write("Training Log: Traverse all feature combinations (KNN)\n")
    f_log.write("="*50 + "\n\n")

    for n in range(1, MAX_FEATURES + 1):
        if n == 1:
            fixed_cols = []
            optional_cols = list(range(PM_COL_INDEX + 1, total_feature_cols))
            k = 1
        else:
            fixed_cols = [PM_COL_INDEX]
            optional_cols = list(range(PM_COL_INDEX + 1, total_feature_cols))
            k = n - 1

        if k <= 0:
            print(f"Skip n={n}, k={k} invalid")
            continue
        if k > len(optional_cols):
            print(f"Warning: Need to select {k} optional columns, but only {len(optional_cols)} available, skip n={n}.")
            continue

        print(f"\n{'='*60}")
        print(f">>> Target total feature count: {n}, fixed columns: {fixed_cols}, pick {k} optional columns")
        print(f"{'='*60}")

        for combo_idx, selected_optional_tuple in enumerate(itertools.combinations(optional_cols, k)):
            final_cols = fixed_cols + list(selected_optional_tuple)

            print(f"\n>>> Training combination [{combo_idx + 1}] ...")
            print(f"Applied column indices: {final_cols}")
            print(f"Applied column names: {list(train_data.columns[final_cols])}")

            X_train_raw = train_data.iloc[:, final_cols]
            X_test_raw = test_data.iloc[:, final_cols]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            knn = KNeighborsClassifier(**knn_params)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            if acc > best_result["acc"]:
                best_result["acc"] = acc
                best_result["combo"] = combo_idx + 1
                best_result["cols"] = final_cols

            log_msg = (
                f"[Total Features: {n} | Combination ID: {combo_idx + 1}]\n"
                f"Used column indices: {final_cols}\n"
                f"Used column names: {list(X_train_raw.columns)}\n"
                f"Hyperparameters: {knn_params}\n"
                f"Test Accuracy: {acc:.4f}\n"
                f"Test Weighted F1 Score: {f1:.4f}\n"
                f"{'-'*40}\n"
            )
            f_log.write(log_msg)
            print(f"Training completed! Accuracy: {acc:.4f}")

    best_info = (
        f"\n===== BEST RESULT =====\n"
        f"Best Acc: {best_result['acc']:.4f}, Best Combination ID: {best_result['combo']}\n"
        f"Best Feature index list: {best_result['cols']}\n"
        f"Best Feature names: {list(train_data.columns[best_result['cols']])}\n"
    )
    f_log.write(best_info)
    print(best_info)

print("\nAll KNN training tasks finished.")
