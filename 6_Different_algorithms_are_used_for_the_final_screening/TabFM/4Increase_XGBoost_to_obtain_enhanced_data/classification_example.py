# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TabFM classification: read csv train/test, output prediction to csv."""
import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> pd.DataFrame:
    """Load csv dataset, train classification model, predict external test, return result dataframe."""
    if model is None:
        # Option A: JAX Backend (default)
        model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
        # Option B: PyTorch Backend
        # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")

    # 2. Initialize scikit-learn compatible classifier
    clf = tabfm.TabFMClassifier(model=model)

    # 3. Read training csv: train_test.csv, last column is label y, others are features X
    df_train_all = pd.read_csv("train1.csv")
    X_train = df_train_all.iloc[:, :-1]  # all columns except last as features
    y_train = df_train_all.iloc[:, -1]   # last column as label

    # 4. Read external test dataset
    X_test = pd.read_csv("external_test.csv")

    # 5. Fit classifier
    clf.fit(X_train, y_train)

    # 6. Predict probabilities
    probs = clf.predict_proba(X_test)

    # Build result dataframe: keep original test features + append probability columns
    result_df = X_test.copy()
    # get class labels from fitted classifier
    class_names = clf.classes_
    for idx, cls_name in enumerate(class_names):
        result_df[f"prob_{cls_name}"] = probs[:, idx]
    
    # also output predicted label
    pred_labels = clf.predict(X_test)
    result_df["pred_label"] = pred_labels

    return result_df


if __name__ == "__main__":
    print("Running TabFM classification model... (Note: compilation and model execution may take a few minutes on first run)")
    predictions_df = run_example()
    # save result
    predictions_df.to_csv("test_result.csv", index=False)
    print("Prediction saved to test_result.csv")
    print(predictions_df)
