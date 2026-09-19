import pandas as pd
import numpy as np
import joblib
# ------------------------------------------------------------------------------
# 1. 配置信息（输入输出分离）
# ------------------------------------------------------------------------------
INPUT_CSV  = "external_test.csv"       # 输入文件，已包含全部特征列
OUTPUT_CSV = "phase_predict_result.csv"# 输出结果文件
CHUNK_SIZE = 100000                    # 分块大小，避免大文件内存溢出
# 模型文件根路径
BASE_PATH = r"D:\Papers_related_materials\second_paper\Calculation_result_after_optimization1\Single_model\combination_training_LightGBM2"
# 各模型路径
RF_MODEL_PATH    = BASE_PATH + r"\RF.pkl"
XGB_MODEL_PATH   = BASE_PATH + r"\xgb.pkl"
KNN_MODEL_PATH   = BASE_PATH + r"\knn.pkl"
CAT_MODEL_PATH   = BASE_PATH + r"\catboost.pkl"
LGB_MODEL_PATH   = BASE_PATH + r"\lgbm.pkl"
# KNN标准化器路径（必须保留，KNN预测前必须做标准化）
KNN_SCALER_PATH  = BASE_PATH + r"\knn_scaler.pkl"
# XGB标签编码器路径
XGB_ENCODER_PATH = BASE_PATH + r"\label_encoder_xgb.pkl"

# ======================【修改点：按类别定义模型权重】======================
# 类别顺序：BCC, BCC+FCC, IM, FCC
# 字典结构：{类别名称: [RF权重, XGB权重, KNN权重, CAT权重, LGB权重]}
CLASS_WEIGHTS = {
    "BCC":      [0, 0, 1, 0, 0],
    "BCC+FCC":  [0, 0, 0.5, 0, 0.5],
    "IM":       [0, 0, 0.52, 0, 0.48],
    "FCC":      [0.5, 0, 0, 0, 0.5]
}
MODEL_NAMES = ["rf", "xgb", "knn", "cat", "lgb"]
# =========================================================================

# ------------------------------------------------------------------------------
# 【更新】各模型新特征列表
# ------------------------------------------------------------------------------
KNN_FEATURES = ["Pm", "ρ", "α", "ad-e/n", "δ-e/n", "δ-Ec"]
CAT_FEATURES = ["Pm", "ΔS", "ad-e/n", "ad-HD", "ad-G", "δ-χ", "δ-EWF", "δ-HD", "δ-G"]
LGB_FEATURES = ["Pm", "e/a", "EWF", "ad-e/n", "δ-e/n", "δ-ρ", "δ-Ec", "δ-G", "δ-υ"]
RF_FEATURES = ["Pm", "ΔS", "EWF", "ad-e/n", "δ-VEC", "δ-EWF", "δ-G", "δ-G/B"]
XGB_FEATURES = ["Pm", "Ω", "e/a", "EWF", "G", "ad-e/n", "ad-G", "δ-e/n", "δ-ρ"]
# 所有特征合集（用于前置校验，避免输入文件缺列报错）
ALL_REQUIRED_FEATURES = set(KNN_FEATURES + CAT_FEATURES + LGB_FEATURES + RF_FEATURES + XGB_FEATURES)
# ------------------------------------------------------------------------------
# 3. 加载模型、编码器与标准化器
# ------------------------------------------------------------------------------
print("🔽 正在加载所有模型...")
rf_model  = joblib.load(RF_MODEL_PATH)
# 修复旧版本RF模型在新版sklearn的monotonic_cst属性缺失报错
for tree in rf_model.estimators_:
    if not hasattr(tree, "monotonic_cst"):
        tree.monotonic_cst = None
xgb_model = joblib.load(XGB_MODEL_PATH)
KNN_model = joblib.load(KNN_MODEL_PATH)
cat_model = joblib.load(CAT_MODEL_PATH)
lgb_model = joblib.load(LGB_MODEL_PATH)
knn_scaler = joblib.load(KNN_SCALER_PATH)
le_xgb = joblib.load(XGB_ENCODER_PATH)
print("✅ 所有模型与编码器加载完成\n")
# ------------------------------------------------------------------------------
# 4. 统一类别顺序，从编码器读取XGB真实相名称
# ------------------------------------------------------------------------------
rf_classes  = [str(cls) for cls in rf_model.classes_]
knn_classes = [str(cls) for cls in KNN_model.classes_]
cat_classes = [str(cls) for cls in cat_model.classes_]
lgb_classes = [str(cls) for cls in lgb_model.classes_]
# XGB类别取编码器内的原始相名称，而非模型输出的数字
xgb_phase_classes = [str(c) for c in le_xgb.classes_]
# 合并所有相类别，去重排序
all_classes = sorted(set(rf_classes + knn_classes + cat_classes + lgb_classes + xgb_phase_classes))
print(f"📊 统一类别顺序: {all_classes}")
print(f"📊 按类别差异化权重配置: {CLASS_WEIGHTS}\n")
# ------------------------------------------------------------------------------
# 5. 分块处理预测核心逻辑
# ------------------------------------------------------------------------------
def predict_chunk(df_chunk):
    """处理单个数据块，返回预测结果与各类别概率"""
    # 前置校验：检查输入文件是否包含所有需要的特征列
    missing_features = ALL_REQUIRED_FEATURES - set(df_chunk.columns)
    if missing_features:
        raise ValueError(f"输入文件缺少必要特征列: {missing_features}")
    
    # 提取各模型输入特征
    X_rf  = df_chunk[RF_FEATURES].copy()
    X_xgb = df_chunk[XGB_FEATURES].copy()
    X_KNN_raw = df_chunk[KNN_FEATURES].copy()
    X_cat = df_chunk[CAT_FEATURES].copy()
    X_lgb = df_chunk[LGB_FEATURES].copy()
    
    # KNN必须做标准化（和训练时保持一致）
    X_KNN = knn_scaler.transform(X_KNN_raw)
    
    # step1: 各模型输出预测概率
    rf_proba  = rf_model.predict_proba(X_rf)
    xgb_proba_raw = xgb_model.predict_proba(X_xgb)
    knn_proba = KNN_model.predict_proba(X_KNN)
    cat_proba = cat_model.predict_proba(X_cat)
    lgb_proba = lgb_model.predict_proba(X_lgb)
    
    # step2: 对齐RF,KNN,CAT,LGB概率到统一类别顺序
    def reorder_other(proba, model_classes, model_name):
        reordered = np.zeros((len(df_chunk), len(all_classes)))
        for i, cls in enumerate(model_classes):
            cls_str = str(cls)
            if cls_str in all_classes:
                cls_idx = all_classes.index(cls_str)
                reordered[:, cls_idx] = proba[:, i]
            else:
                print(f"⚠️  模型[{model_name}] 输出了未在统一类别中的标签: {cls_str}，已忽略")
        return reordered
    
    rf_proba_reordered  = reorder_other(rf_proba, rf_classes,  "RF")
    knn_proba_reordered = reorder_other(knn_proba, knn_classes, "KNN")
    cat_proba_reordered = reorder_other(cat_proba, cat_classes, "CAT")
    lgb_proba_reordered = reorder_other(lgb_proba, lgb_classes, "LGB")
    
    # --------XGB概率映射修正(核心)--------
    xgb_proba_reordered = np.zeros((len(df_chunk), len(all_classes)))
    for xgb_index, phase_name in enumerate(xgb_phase_classes):
        pos = all_classes.index(phase_name)
        xgb_proba_reordered[:, pos] = xgb_proba_raw[:, xgb_index]
    
    # =====================【核心修改：逐类别加权】=====================
    # 初始化总概率矩阵
    total_proba = np.zeros_like(rf_proba_reordered)
    # 遍历每个类别，使用该类别专属权重计算加权概率
    for cls_name, w_list in CLASS_WEIGHTS.items():
        if cls_name not in all_classes:
            print(f"⚠️ 类别 {cls_name} 不在all_classes，跳过")
            continue
        c_idx = all_classes.index(cls_name)
        # w_list顺序：RF, XGB, KNN, CAT, LGB
        w_rf, w_xgb, w_knn, w_cat, w_lgb = w_list
        total_proba[:, c_idx] = (
            rf_proba_reordered[:, c_idx] * w_rf +
            xgb_proba_reordered[:, c_idx] * w_xgb +
            knn_proba_reordered[:, c_idx] * w_knn +
            cat_proba_reordered[:, c_idx] * w_cat +
            lgb_proba_reordered[:, c_idx] * w_lgb
        )
    # =================================================================
    
    # step4: 取概率最大的类别作为最终预测结果
    final_pred_idx = np.argmax(total_proba, axis=1)
    final_pred = np.array([all_classes[idx] for idx in final_pred_idx])
    
    # step5: 各模型单独预测结果
    pred_rf  = [str(x) for x in rf_model.predict(X_rf).ravel()]
    pred_knn = [str(x) for x in KNN_model.predict(X_KNN).ravel()]
    pred_cat = [str(x) for x in cat_model.predict(X_cat).ravel()]
    pred_lgb = [str(x) for x in lgb_model.predict(X_lgb).ravel()]
    # XGB数字标签反变换回相名称
    xgb_pred_num = xgb_model.predict(X_xgb).ravel()
    pred_xgb = le_xgb.inverse_transform(xgb_pred_num)
    
    # 组装结果DataFrame
    result_df = df_chunk.copy()
    result_df["pred_ensemble"] = final_pred
    result_df["pred_RF"]  = pred_rf
    result_df["pred_XGB"] = pred_xgb
    result_df["pred_KNN"] = pred_knn
    result_df["pred_CAT"] = pred_cat
    result_df["pred_LGB"] = pred_lgb
    # 每个类别的加权概率（列名格式：prob_类别名）
    for i, cls in enumerate(all_classes):
        result_df[f"prob_{cls}"] = total_proba[:, i]
    
    return result_df
# ------------------------------------------------------------------------------
# 6. 分块读取、处理、写入文件
# ------------------------------------------------------------------------------
first_write = True
for chunk_idx, df_chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)):
    print(f"📦 正在处理第 {chunk_idx + 1} 批数据，行数：{len(df_chunk)}")
    # 处理当前块
    result_df = predict_chunk(df_chunk)
    # 写入输出CSV（分块追加，避免内存溢出）
    result_df.to_csv(
        OUTPUT_CSV,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8"
    )
    first_write = False
    print(f"✅ 第 {chunk_idx + 1} 批处理完成\n")
print("🎉🎉🎉 全部处理完成！")
print(f"结果保存在：{OUTPUT_CSV}")
print(f"输出文件包含：原始特征 + 各模型单独预测结果 + 集成最终预测结果 + 所有类别的差异化加权概率")
