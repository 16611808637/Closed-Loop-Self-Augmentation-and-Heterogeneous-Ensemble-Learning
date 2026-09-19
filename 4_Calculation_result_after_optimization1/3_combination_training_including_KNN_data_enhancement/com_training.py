"""
Five models parallel training script
- KNN       ← train1.csv
- CatBoost  ← train2.csv
- LightGBM  ← train3.csv
- RF        ← train4.csv
- XGBoost   ← train5.csv
Each model runs in independent process, limit 2 CPU cores, no GPU
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from multiprocessing import Process
warnings.filterwarnings('ignore')

# ====================== Common utility functions ======================
def set_cpu_limit(n_cores):
    """Limit available CPU cores for current process (Linux only; Windows use n_jobs)"""
    try:
        os.sched_setaffinity(0, set(range(n_cores)))
    except AttributeError:
        pass

def split_train_test(csv_path, test_ratio=0.2, random_state=42):
    """Read CSV, last column is label, split 8:2 stratified"""
    from sklearn.model_selection import train_test_split
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# ====================== KNN training function ======================
def train_knn():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from bayes_opt import BayesianOptimization
    import warnings
    warnings.filterwarnings('ignore')

    try:
        os.sched_setaffinity(0, {0, 1})
    except AttributeError:
        pass

    CSV_PATH = 'train1.csv'
    MODEL_SAVE_PATH = "knn.pkl"
    SCALER_SAVE_PATH = "knn_scaler.pkl"
    OUTPUT_RESULT = "knn_result_out.txt"
    RANDOM_SEED = 42
    OPTIM_ITER = 100
    INIT_POINTS = 10
    N_SPLITS = 5

    PBounds = {
        'n_neighbors': (2, 40),
        'p': (1, 5),
        'weights_flag': (0, 1),
        'leaf_size': (10, 60)
    }

    print("[KNN] Loading data...")
    X_train_full, X_test, y_train_full, y_test = split_train_test(CSV_PATH, random_state=RANDOM_SEED)

    def knn_evaluation(n_neighbors, p, weights_flag, leaf_size):
        n = int(round(n_neighbors))
        p_val = int(round(p))
        weights = 'uniform' if int(round(weights_flag)) == 0 else 'distance'
        leaf_size_val = int(round(leaf_size))
        knn_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=n,
                weights=weights,
                metric='minkowski',
                p=p_val,
                leaf_size=leaf_size_val,
                n_jobs=2
            ))
        ])
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(
            estimator=knn_pipeline,
            X=X_train_full,
            y=y_train_full,
            cv=skf,
            scoring="f1_macro",
            n_jobs=1
        )
        return cv_scores.mean()

    print("[KNN] Start Bayesian optimization...")
    optimizer = BayesianOptimization(
        f=lambda n_neighbors, p, weights_flag, leaf_size: knn_evaluation(n_neighbors, p, weights_flag, leaf_size),
        pbounds=PBounds,
        random_state=RANDOM_SEED,
        verbose=1
    )
    optimizer.maximize(init_points=INIT_POINTS, n_iter=OPTIM_ITER)
    best_params_raw = optimizer.max['params']
    best_knn_params = {
        "n_neighbors": int(round(best_params_raw['n_neighbors'])),
        "weights": 'uniform' if int(round(best_params_raw['weights_flag'])) == 0 else 'distance',
        "metric": "minkowski",
        "p": int(round(best_params_raw['p'])),
        "leaf_size": int(round(best_params_raw['leaf_size'])),
        "n_jobs": 2
    }
    print(f"[KNN] Best params: {best_knn_params}")
    print(f"[KNN] Best CV Macro-F1: {optimizer.max['target']:.4f}")

    final_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**best_knn_params))
    ])
    final_pipeline.fit(X_train_full, y_train_full)
    final_scaler = final_pipeline.named_steps['scaler']
    final_knn = final_pipeline.named_steps['knn']
    y_pred = final_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    confusion_mat = confusion_matrix(y_test, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.diag(confusion_mat) / confusion_mat.sum(axis=1)
    per_class_recall = np.nan_to_num(per_class_recall, nan=0.0)
    class_labels = sorted(np.unique(y_test))
    class_report = classification_report(y_test, y_pred)

    print(f"[KNN] Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        f.write("====== Bayesian Optimized KNN Final Evaluation (5‑fold CV no leakage) ======\n")
        f.write(f"Best hyper‑parameters: {best_knn_params}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1‑Score: {weighted_f1:.4f}\n")
        f.write(f"Macro F1‑Score: {macro_f1:.4f}\n\n")
        f.write("====== Confusion Matrix ======\n")
        f.write(str(confusion_mat) + "\n\n")
        f.write("====== Per‑class Recall ======\n")
        for idx, label in enumerate(class_labels):
            f.write(f"Class {label}: Recall = {per_class_recall[idx]:.4f}\n")
        f.write("\n====== Sklearn Classification Report ======\n")
        f.write(class_report)

    joblib.dump(final_knn, MODEL_SAVE_PATH)
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    print(f"[KNN] Training finished, model saved: {MODEL_SAVE_PATH}")

# ====================== CatBoost training function ======================
def train_catboost():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import StratifiedKFold
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import warnings
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        os.sched_setaffinity(0, {2, 3})
    except AttributeError:
        pass

    CSV_PATH = 'train2.csv'
    MODEL_SAVE_PATH = "catboost.pkl"
    OUTPUT_RESULT = "catboost_bayes_opt_result.txt"
    RANDOM_SEED = 20
    N_TRIALS = 60
    N_SPLITS = 5

    print("[CatBoost] Loading data...")
    X_train_full, X_test, y_train_full, y_test = split_train_test(CSV_PATH, random_state=RANDOM_SEED)

    fixed_params = {
        'random_state': RANDOM_SEED,
        'verbose': 0,
        'thread_count': 2,
        'auto_class_weights': 'Balanced',
        'bootstrap_type': 'Bernoulli'
    }

    def objective(trial):
        search_params = {
            'iterations': trial.suggest_int("iterations", 200, 1000, step=50),
            'depth': trial.suggest_int("depth", 3, 6),
            'learning_rate': trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 10, 40),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            'random_strength': trial.suggest_float("random_strength", 0.1, 2.0)
        }
        params = {**fixed_params, **search_params}
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            X_train_fold = X_train_full.iloc[train_idx]
            X_val_fold = X_train_full.iloc[val_idx]
            y_train_fold = y_train_full.iloc[train_idx]
            y_val_fold = y_train_full.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=50,
                verbose=0
            )
            y_pred_fold = model.predict(X_val_fold).ravel()
            fold_macro_f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
            fold_scores.append(fold_macro_f1)
            trial.report(fold_macro_f1, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(fold_scores)

    print("[CatBoost] Start Optuna optimization...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    best_params.update(fixed_params)
    print(f"[CatBoost] Best params: {best_params}")
    print(f"[CatBoost] Best CV Macro‑F1: {study.best_value:.4f}")

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train_full, y_train_full)
    y_pred = final_model.predict(X_test).ravel()

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_recall = cm.diagonal() / cm.sum(axis=1)
    class_recall = np.nan_to_num(class_recall, nan=0.0)
    labels = sorted(list(set(y_test)))

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        f.write("====== CatBoost‑Bayesian Optimized Final Evaluation (5‑fold CV no leakage) ======\n")
        f.write(f"Best hyper‑parameters: {best_params}\n\n")
        f.write(f"5‑fold CV mean Macro‑F1: {study.best_value:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1‑score: {f1_weighted:.4f}\n")
        f.write(f"Macro F1‑score: {f1_macro:.4f}\n\n")
        f.write("====== Confusion Matrix ======\n")
        f.write(str(cm) + "\n\n")
        f.write("====== Per‑class Recall ======\n")
        for idx, label in enumerate(labels):
            f.write(f"Class {label} : Recall = {class_recall[idx]:.4f}\n")
        f.write("\n====== Sklearn Classification Report ======\n")
        report = classification_report(y_test, y_pred)
        f.write(report)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"[CatBoost] Accuracy: {acc:.4f}, Macro‑F1: {f1_macro:.4f}")
    print(f"[CatBoost] Training finished, model saved: {MODEL_SAVE_PATH}")

# ====================== LightGBM training function ======================
def train_lightgbm():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from bayes_opt import BayesianOptimization
    import warnings
    warnings.filterwarnings('ignore')

    try:
        os.sched_setaffinity(0, {4, 5})
    except AttributeError:
        pass

    CSV_PATH = 'train3.csv'
    MODEL_SAVE_PATH = "lgbm.pkl"
    OUTPUT_RESULT = "lgbm_result.txt"
    RANDOM_SEED = 20

    print("[LightGBM] Loading data...")
    X_train_full, X_test, y_train_full, y_test = split_train_test(CSV_PATH, random_state=RANDOM_SEED)

    def lgb_eval(max_depth, colsample_bytree, min_child_samples, learning_rate, n_estimators, reg_alpha, reg_lambda):
        params = {
            "random_state": RANDOM_SEED,
            "verbosity": -1,
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "colsample_bytree": colsample_bytree,
            "class_weight": "balanced",
            "min_child_samples": int(min_child_samples),
            "min_child_weight": 0.001,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_jobs": 2
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X_train_full, y_train_full, cv=skf, scoring="f1_macro", n_jobs=1)
        return scores.mean()

    pbounds = {
        "max_depth": (3, 10),
        "colsample_bytree": (0.3, 0.9),
        "min_child_samples": (5, 30),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 400),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5)
    }

    print("[LightGBM] Start Bayesian optimization...")
    optimizer = BayesianOptimization(
        f=lgb_eval,
        pbounds=pbounds,
        random_state=RANDOM_SEED,
        verbose=2
    )
    optimizer.maximize(init_points=10, n_iter=60)
    best_params_raw = optimizer.max['params']
    best_params = {
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_estimators": int(best_params_raw["n_estimators"]),
        "max_depth": int(best_params_raw["max_depth"]),
        "colsample_bytree": best_params_raw["colsample_bytree"],
        "class_weight": "balanced",
        "min_child_samples": int(best_params_raw["min_child_samples"]),
        "min_child_weight": 0.001,
        "learning_rate": best_params_raw["learning_rate"],
        "reg_alpha": best_params_raw["reg_alpha"],
        "reg_lambda": best_params_raw["reg_lambda"],
        "n_jobs": 2
    }
    print(f"[LightGBM] Best params: {best_params}")
    print(f"[LightGBM] Best CV Macro‑F1: {optimizer.max['target']:.4f}")

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train_full, y_train_full)
    y_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_recall = cm.diagonal() / cm.sum(axis=1)
    class_recall = np.nan_to_num(class_recall, nan=0.0)
    labels = sorted(list(set(y_test)))

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        f.write("====== LightGBM‑Bayesian‑Opt Final Evaluation (5‑fold CV tuning) ======\n")
        f.write(f"Best parameters: {best_params}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted‑F1: {f1_weighted:.4f}\n")
        f.write(f"Macro‑F1: {f1_macro:.4f}\n\n")
        f.write("====== Confusion Matrix ======\n")
        f.write(str(cm) + "\n\n")
        f.write("====== Per‑class Recall ======\n")
        for idx, label in enumerate(labels):
            f.write(f"Class {label} : Recall = {class_recall[idx]:.4f}\n")
        f.write("\n====== Classification Report ======\n")
        report = classification_report(y_test, y_pred)
        f.write(report)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"[LightGBM] Accuracy: {acc:.4f}, Macro‑F1: {f1_macro:.4f}")
    print(f"[LightGBM] Training finished, model saved: {MODEL_SAVE_PATH}")

# ====================== RandomForest training function ======================
def train_rf():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import optuna
    import warnings
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        os.sched_setaffinity(0, {6, 7})
    except AttributeError:
        pass

    CSV_PATH = 'train4.csv'
    MODEL_SAVE_PATH = "RF.pkl"
    OUTPUT_RESULT = "RF_result.txt"
    RANDOM_SEED = 20

    print("[RF] Loading data...")
    X_train, X_test, y_train, y_test = split_train_test(CSV_PATH, random_state=RANDOM_SEED)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 200, 600),
            'max_depth': trial.suggest_int("max_depth", 3, 6),
            'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            'min_samples_split': trial.suggest_int("min_samples_split", 4, 14),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 7),
            'random_state': RANDOM_SEED,
            'n_jobs': 1,
            'oob_score': False
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_macro_f1 = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr_fold, X_val_fold = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr_fold, y_val_fold = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            class_counts = Counter(y_tr_fold)
            n_samples_fold = len(y_tr_fold)
            n_classes_fold = len(class_counts)
            sqrt_weights = {}
            for cls, cnt in class_counts.items():
                sqrt_weights[cls] = float(np.sqrt(n_samples_fold / (n_classes_fold * cnt)))
            params['class_weight'] = sqrt_weights
            model = RandomForestClassifier(**params)
            model.fit(X_tr_fold, y_tr_fold)
            y_val_pred = model.predict(X_val_fold)
            fold_f1 = f1_score(y_val_fold, y_val_pred, average="macro")
            fold_macro_f1.append(fold_f1)
        return np.mean(fold_macro_f1)

    print("[RF] Start Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=70)
    best_params = study.best_params
    best_params['random_state'] = RANDOM_SEED
    best_params['n_jobs'] = 2

    class_counts_full = Counter(y_train)
    n_samples_full = len(y_train)
    n_classes_full = len(class_counts_full)
    sqrt_weights_full = {}
    for cls, cnt in class_counts_full.items():
        sqrt_weights_full[cls] = float(np.sqrt(n_samples_full / (n_classes_full * cnt)))
    best_params['class_weight'] = sqrt_weights_full

    print(f"[RF] Best params: {best_params}")
    print(f"[RF] Best CV Macro‑F1: {study.best_value:.4f}")

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_recall = cm.diagonal() / cm.sum(axis=1)
    class_recall = np.nan_to_num(class_recall, nan=0.0)
    labels = sorted(list(set(y_test)))

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        f.write("====== RandomForest‑Optuna Best Model Evaluation ======\n")
        f.write(f"Best hyper‑parameters: {best_params}\n")
        f.write(f"5‑fold CV Macro‑F1: {study.best_value:.4f}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1‑score: {f1_weighted:.4f}\n")
        f.write(f"Macro F1‑score: {f1_macro:.4f}\n\n")
        f.write("====== Confusion Matrix ======\n")
        f.write(str(cm) + "\n\n")
        f.write("====== Per‑class Recall ======\n")
        for idx, label in enumerate(labels):
            f.write(f"Class {label} : Recall = {class_recall[idx]:.4f}\n")
        f.write("\n====== Sklearn Classification Report ======\n")
        report = classification_report(y_test, y_pred)
        f.write(report)

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[RF] Accuracy: {acc:.4f}, Macro‑F1: {f1_macro:.4f}")
    print(f"[RF] CV Macro‑F1: {study.best_value:.4f}, Gap={study.best_value-f1_macro:.4f}")
    print(f"[RF] Training finished, model saved: {MODEL_SAVE_PATH}")

# ====================== XGBoost training function ======================
def train_xgboost():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
    from bayes_opt import BayesianOptimization
    import warnings
    warnings.filterwarnings('ignore')

    try:
        os.sched_setaffinity(0, {8, 9})
    except AttributeError:
        pass

    CSV_PATH = 'train5.csv'
    MODEL_SAVE_PATH = "xgb.pkl"
    OUTPUT_RESULT = "XGB.out"
    RANDOM_SEED = 20
    LABEL_ENCODER_SAVE_PATH = "label_encoder_xgb.pkl"

    print("[XGBoost] Loading data...")
    data = pd.read_csv(CSV_PATH)
    from sklearn.model_selection import train_test_split
    X = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_SEED, stratify=y_raw
    )
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    X_train = X_train_raw
    X_test = X_test_raw
    joblib.dump(le, LABEL_ENCODER_SAVE_PATH)
    print(f"[XGBoost] Label encoder saved: {LABEL_ENCODER_SAVE_PATH}")

    def xgb_cv_macro_f1(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight):
        n_estimators_int = int(n_estimators)
        max_depth_int = int(max_depth)
        min_child_weight_int = int(min_child_weight)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators_int,
            max_depth=max_depth_int,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight_int,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=2
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=kf, scoring='f1_macro', n_jobs=1
        )
        return cv_scores.mean()

    param_bounds = {
        'n_estimators': (100, 500),
        'max_depth': (4, 14),
        'learning_rate': (0.005, 0.02),
        'subsample': (0.4, 1.0),
        'colsample_bytree': (0.4, 1.0),
        'gamma': (0.0, 0.05),
        'min_child_weight': (1, 5)
    }

    print("[XGBoost] Start Bayesian optimization...")
    optimizer = BayesianOptimization(
        f=xgb_cv_macro_f1,
        pbounds=param_bounds,
        random_state=RANDOM_SEED,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=100)
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params.update({
        'random_state': RANDOM_SEED,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'n_jobs': 2
    })
    print(f"[XGBoost] Best params: {best_params}")
    print(f"[XGBoost] Best CV Macro‑F1: {optimizer.max['target']:.4f}")

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_recall = test_conf_matrix.diagonal() / test_conf_matrix.sum(axis=1)
    class_recall = np.nan_to_num(class_recall, nan=0.0)
    labels = sorted(list(set(y_test)))

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        f.write("XGBoost Model Training and Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Bayesian optimized 5‑fold CV Macro‑F1: {optimizer.max['target']:.4f}\n")
        f.write("Best hyper‑parameters:\n")
        for param_name, param_value in best_params.items():
            f.write(f"  {param_name}: {param_value}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Macro‑F1: {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted‑F1: {test_weighted_f1:.4f}\n\n")
        f.write("Test Confusion Matrix:\n")
        f.write(str(test_conf_matrix))
        f.write("\n\n")
        f.write("====== Per‑class Recall ======\n")
        for idx, label in enumerate(labels):
            f.write(f"Class {label}: Recall = {class_recall[idx]:.4f}\n")
        f.write("\n====== Full Classification Report ======\n")
        report = classification_report(y_test, y_test_pred)
        f.write(report)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"[XGBoost] Accuracy: {acc:.4f}, Macro‑F1: {test_macro_f1:.4f}")
    print(f"[XGBoost] Training finished, model saved: {MODEL_SAVE_PATH}")

# ====================== Main entry: start 5 parallel processes ======================
if __name__ == "__main__":
    print("=" * 70)
    print("  Five‑model parallel training start (5 processes × 2 cores = total 10 cores)")
    print("=" * 70)
    # Mapping
    # KNN      ← train1.csv  (core 0,1)
    # CatBoost ← train2.csv  (core 2,3)
    # LightGBM ← train3.csv  (core 4,5)
    # RF       ← train4.csv  (core 6,7)
    # XGBoost  ← train5.csv  (core 8,9)
    processes = [
        Process(target=train_knn,     name="KNN-Process"),
        Process(target=train_catboost, name="CatBoost-Process"),
        Process(target=train_lightgbm, name="LightGBM-Process"),
        Process(target=train_rf,      name="RF-Process"),
        Process(target=train_xgboost,  name="XGBoost-Process"),
    ]
    for p in processes:
        p.start()
        print(f"  ✅ Process [{p.name}] started (PID: {p.pid})")
    for p in processes:
        p.join()
        print(f"  🏁 Process [{p.name}] finished")
    print("\n" + "=" * 70)
    print("  🎉 All five models training completed!")
    print("=" * 70)
    print("Output file list:")
    print("  📦 knn.pkl, knn_scaler.pkl, knn_result_out.txt")
    print("  📦 catboost.pkl, catboost_bayes_opt_result.txt")
    print("  📦 lgbm.pkl, lgbm_result.txt")
    print("  📦 RF.pkl, RF_result.txt")
    print("  📦 xgb.pkl, label_encoder_xgb.pkl, XGB.out")
