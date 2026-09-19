import pandas as pd
import numpy as np
import joblib
import math
# ------------------------------------------------------------------------------
# 1. 配置信息（输入输出分离）
# ------------------------------------------------------------------------------
INPUT_CSV  = "MCA_phase_predict.csv"    # 输入文件：前15列为元素成分，最后一列为Pm值
OUTPUT_CSV = "phase_predict_result.csv" # 输出结果文件
CHUNK_SIZE = 100000                     # 分块大小，避免大文件内存溢出
# 模型文件根路径
BASE_PATH = r"D:\Papers_related_materials\second_paper\github\4_Calculation_result_after_optimization1\1_combination_training"
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

# ======================【修改：按类别差异化权重】======================
# 权重顺序：[RF, XGB, KNN, CAT, LGB]
CLASS_WEIGHTS = {
    "BCC":      [0, 0.51, 0, 0, 0.49],
    "BCC+FCC":  [0, 0, 0.52, 0, 0.48],
    "IM":       [0, 0.49, 0, 0.51, 0],
    "FCC":      [0, 0, 0, 1, 0]
}
MODEL_NAMES = ["rf", "xgb", "knn", "cat", "lgb"]
# =====================================================================

# ------------------------------------------------------------------------------
# 2. 元素基础参数与特征计算函数（完全复刻原逻辑）
# ------------------------------------------------------------------------------
# 15 个元素顺序
ELEMENTS = [
    "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Mo", "Ti", "W",
    "Nb", "Ta", "V", "Mn", "Hf", "Zr"
]
# 全部元素基础参数，与excel脚本完全一致
ELEMENT_PARAMS = {
    "r_vec": [3, 9, 6, 11, 8, 10, 6, 4, 6, 5, 5, 5, 7, 4, 4],
    "e_n": [0.2308, 0.3333, 0.25, 0.3793, 0.3077, 0.3571, 0.1429, 0.1818, 0.0808, 0.122, 0.0685, 0.2174, 0.28, 0.0556, 0.1],
    "radius": [143.17, 125.1, 124.91, 127.8, 124.12, 124.59, 136.26, 146.15, 136.7, 142.9, 143, 131.6, 135, 157.75, 160.25],
    "density": [2.7, 8.84, 7.19, 8.94, 7.88, 8.91, 10.23, 4.5, 19.41, 8.58, 16.68, 6.12, 7.47, 13.28, 6.51],
    "upsilon": [0.336, 0.405, 0.21, 0.346, 0.29, 0.304, 0.298, 0.406, 0.282, 0.39, 0.339, 0.356, 0.24, 0.385, 0.418],
    "EWF": [4.28, 5, 4.5, 4.65, 4.5, 5.15, 4.6, 4.33, 4.55, 4.3, 4.25, 4.3, 4.1, 3.9, 4.05],
    "alpha": [23.6, 13.8, 6.2, 16.7, 11.8, 13.3, 5.1, 10.2, 4.5, 7.3, 6.5, 8.3, 21.7, 6.3, 5.8],
    "atomic_m": [26.982, 58.933, 51.996, 63.546, 55.845, 58.693, 95.95, 47.867, 183.84, 92.906, 180.95, 50.942, 54.938, 178.49, 91.224],
    "Tm": [933, 1768, 2180, 1358, 1811, 1728, 2896, 1941, 3695, 2750, 3290, 2183, 1519, 2506, 2128],
    "hardness": [15, 253, 90, 50, 150, 65, 230, 60, 310, 80, 100, 141, 500, 173, 150],
    "young": [70, 209, 279, 130, 211, 200, 329, 116, 411, 105, 186, 128, 198, 78, 68],
    "atomic_number": [13, 27, 24, 29, 26, 28, 42, 22, 74, 41, 73, 23, 25, 72, 40],
    "electronegativity": [1.61, 1.88, 1.66, 1.9, 1.83, 1.91, 2.16, 1.54, 2.36, 1.6, 1.5, 1.63, 1.55, 1.3, 1.33],
    "itinerant_electrons": [3, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2],
    "cohesive_energy": [3.39, 4.39, 4.1, 3.49, 4.28, 4.44, 6.82, 4.85, 8.9, 7.57, 8.1, 5.31, 2.92, 6.44, 6.25],
    "ABE": [33.7, 63.7, 54.8, 47.7, 58.7, 65.1, 70, 44.2, 89.2, 67.2, 71.7, 61.6, 38, 45.4, 43.3],
    "SHA": [1.79, 7.88, 8.62, 4.02, 8.85, 7.26, 9.17, 5.23, 13.38, 7.01, 8.22, 7.66, 6.64, 5.26, 3.94],
    "SLHF": [1.08, 1.95, 2.88, 1.92, 2.18, 2.6, 2.98, 1.64, 3.72, 2.48, 2.26, 2.11, 1.98, 1.76, 1.37],
    "bulk_modulus": [79.3, 199.6, 159.3, 137.7, 166.7, 184.3, 265, 115.9, 307.7, 173, 191.3, 157.1, 124, 117.2, 112.6],
    "shear_modulus": [29.2, 40.4, 114.5, 47.3, 81.5, 83.2, 123.7, 23.3, 157, 41.1, 69.1, 49.9, 78, 29.3, 19.6],
    "G_B": [0.37, 0.2, 0.72, 0.34, 0.49, 0.45, 0.47, 0.2, 0.51, 0.24, 0.36, 0.32, 0.63, 0.25, 0.17]
}
ABE_RAW = ELEMENT_PARAMS["ABE"]
# 混合焓计算函数
def calc_enthalpy(c):
    return 0.0004 * (
        c[0]*c[1]*(-17.11) + c[0]*c[2]*(-9.321) + c[0]*c[3]*(-7.149) + c[0]*c[4]*(-10.4) + c[0]*c[5]*(-20.67)
        + c[0]*c[6]*(-4.981) + c[0]*c[7]*(-29.12) + c[0]*c[8]*(-1.939) + c[0]*c[9]*(-17.89) + c[0]*c[10]*(-18.84)
        + c[0]*c[11]*(-15.71) + c[0]*c[12]*(-18.02) + c[0]*c[13]*(-36.44) + c[0]*c[14]*(-41.1)
        + c[1]*c[2]*(-4.383) + c[1]*c[3]*(6.321) + c[1]*c[4]*(-0.559) + c[1]*c[5]*(-0.2176) + c[1]*c[6]*(-4.6)
        + c[1]*c[7]*(-25.93) + c[1]*c[8]*(-1.317) + c[1]*c[9]*(-22.41) + c[1]*c[10]*(-21.85) + c[1]*c[11]*(-13.36)
        + c[1]*c[12]*(-5.077) + c[1]*c[13]*(-30.5) + c[1]*c[14]*(-35.14)
        + c[2]*c[3]*(12.36) + c[2]*c[4]*(-1.447) + c[2]*c[5]*(-6.546) + c[2]*c[6]*(0.3607) + c[2]*c[7]*(-6.945)
        + c[2]*c[8]*(0.9148) + c[2]*c[9]*(-6.675) + c[2]*c[10]*(-6.235) + c[2]*c[11]*(-1.905) + c[2]*c[12]*(2.102)
        + c[2]*c[13]*(-8.251) + c[2]*c[14]*(-10.66)
        + c[3]*c[4]*(12.82) + c[3]*c[5]*(3.481) + c[3]*c[6]*(17.55) + c[3]*c[7]*(-8.279) + c[3]*c[8]*(21.22)
        + c[3]*c[9]*(2.403) + c[3]*c[10]*(1.73) + c[3]*c[11]*(4.812) + c[3]*c[12]*(3.712) + c[3]*c[13]*(-14.97)
        + c[3]*c[14]*(-19.96)
        + c[4]*c[5]*(-1.527) + c[4]*c[6]*(-1.882) + c[4]*c[7]*(-15.55) + c[4]*c[8]*(-0.0452) + c[4]*c[9]*(-14.48)
        + c[4]*c[10]*(-13.84) + c[4]*c[11]*(-6.901) + c[4]*c[12]*(0.2259) + c[4]*c[13]*(-18.19) + c[4]*c[14]*(-21.69)
        + c[5]*c[6]*(-6.868) + c[5]*c[7]*(-31.58) + c[5]*c[8]*(-2.938) + c[5]*c[9]*(-27.21) + c[5]*c[10]*(-26.64)
        + c[5]*c[11]*(-17.19) + c[5]*c[12]*(-8.009) + c[5]*c[13]*(-36.97) + c[5]*c[14]*(-42.14)
        + c[6]*c[7]*(-3.531) + c[6]*c[8]*(-0.21) + c[6]*c[9]*(-5.495) + c[6]*c[10]*(-4.786) + c[6]*c[11]*(-0.0287)
        + c[6]*c[12]*(4.675) + c[6]*c[13]*(-3.69) + c[6]*c[14]*(-5.789)
        + c[7]*c[8]*(-5.583) + c[7]*c[9]*(1.96) + c[7]*c[10]*(1.374) + c[7]*c[11]*(-1.595) + c[7]*c[12]*(-7.611)
        + c[7]*c[13]*(0.1584) + c[7]*c[14]*(-0.2271)
        + c[8]*c[9]*(-8.1) + c[8]*c[10]*(-7.161) + c[8]*c[11]*(-7.865) + c[8]*c[12]*(5.981) + c[8]*c[13]*(-5.961)
        + c[8]*c[14]*(-8.476)
        + c[9]*c[10]*(0.0265) + c[9]*c[11]*(-0.9903) + c[9]*c[12]*(-3.467) + c[9]*c[13]*(3.754) + c[9]*c[14]*(3.732)
        + c[10]*c[11]*(-0.964) + c[10]*c[12]*(-3.564) + c[10]*c[13]*(2.792) + c[10]*c[14]*(2.575)
        + c[11]*c[12]*(-0.697) + c[11]*c[13]*(-1.985) + c[11]*c[14]*(-3.374)
        + c[12]*c[13]*(-10.61) + c[12]*c[14]*(-13.59)
        + c[13]*c[14]*(-0.2139)
    )
# 通用计算公式
# 【公式说明】c 为百分比成分（如 5 代表 5 at.%），内部乘 0.01 转为摩尔分数。
#   avg   = Σ(c_i * p_i) * 0.01
#   ad    = Σ(|p_i - avg| * c_i) * 0.01
#   delta = sqrt( Σ(c_i * (1 - p_i/avg)^2) * 0.01 )
# 若上述公式已变更，请在此函数内替换。
def calc_avg_ad_delta(c_frac_arr, prop_list):
    c = np.array(c_frac_arr, dtype=float)
    p = np.array(prop_list, dtype=float)
    avg = np.sum(c * p) * 0.01
    ad = np.sum(np.abs(p - avg) * c) * 0.01
    if abs(avg) < 1e-12:
        delta = 0.0
    else:
        term = ((1.0 - p / avg) ** 2) * c
        delta = math.sqrt(np.sum(term) * 0.01)
    return avg, ad, delta
# 全部特征计算函数，Pm由外部传入
def compute_all_features(comp, Pm_input, params):
    c = np.array(comp, dtype=float)
    c_pct = c * 0.01
    Pm = Pm_input
    # 混合熵
    delta_S = 0.0
    for xi in c_pct:
        if xi > 1e-12:
            delta_S += - 8.314 * xi * math.log(xi)
    # 基础热力学量
    delta_H = calc_enthalpy(c)
    Tm_avg, _, _ = calc_avg_ad_delta(c, params["Tm"])
    if abs(delta_H) < 1e-12:
        Omega = 0.0
    else:
        Omega = (Tm_avg * delta_S / abs(delta_H)) * 0.001
    # 遍历所有属性计算
    VEC, ad_VEC, delta_VEC = calc_avg_ad_delta(c, params["r_vec"])
    e_a, ad_e_a, delta_e_a = calc_avg_ad_delta(c, params["itinerant_electrons"])
    e_n, ad_e_n, delta_e_n = calc_avg_ad_delta(c, params["e_n"])
    r_avg, ad_r, delta_r = calc_avg_ad_delta(c, params["radius"])
    rho, ad_rho, delta_rho = calc_avg_ad_delta(c, params["density"])
    ups, ad_ups, delta_ups = calc_avg_ad_delta(c, params["upsilon"])
    ewf, ad_ewf, delta_ewf = calc_avg_ad_delta(c, params["EWF"])
    alpha, ad_alpha, delta_alpha = calc_avg_ad_delta(c, params["alpha"])
    HD, ad_HD, delta_HD = calc_avg_ad_delta(c, params["hardness"])
    Ec, ad_Ec, delta_Ec = calc_avg_ad_delta(c, params["cohesive_energy"])
    chi, ad_chi, delta_chi = calc_avg_ad_delta(c, params["electronegativity"])
    B, ad_B, delta_B = calc_avg_ad_delta(c, params["bulk_modulus"])
    G, ad_G, delta_G = calc_avg_ad_delta(c, params["shear_modulus"])
    G_B, ad_G_B, delta_G_B = calc_avg_ad_delta(c, params["G_B"])
    SLHF, ad_SLHF, delta_SLHF = calc_avg_ad_delta(c, params["SLHF"])
    Tm, ad_Tm, delta_Tm = calc_avg_ad_delta(c, params["Tm"])
    # ABE系列
    ABE_arr = np.array(ABE_RAW, dtype=float)
    ABE = np.sum(c * ABE_arr) * 0.01
    ad_ABE = np.sum(np.abs(ABE_arr - ABE) * c) * 0.01
    if abs(ABE) < 1e-12:
        delta_ABE = 0.0
    else:
        abe_term = ((1.0 - ABE_arr / ABE) ** 2) * c
        delta_ABE = math.sqrt(np.sum(abe_term) * 0.01)
    # 返回特征字典
    # 【v2 变更】补充 "α": alpha —— 原版本遗漏了热膨胀系数平均值本身，
    # 新 KNN 特征集 [Pm, ρ, α, ad-e/n, δ-e/n, δ-Ec] 需要该字段。
    feat = {
        "Pm": Pm,
        "ΔS": delta_S,
        "ABE": ABE,
        "ad-ABE": ad_ABE,
        "δ-ABE": delta_ABE,
        "ΔH": delta_H,
        "Ω": Omega,
        "e/a": e_a,
        "EWF": ewf,
        "ρ": rho,
        "HD": HD,
        "B": B,
        "G": G,
        "G/B": G_B,
        "α": alpha,            # ← v2 新增
        "ad-e/n": ad_e_n,
        "ad-r": ad_r,
        "ad-α": ad_alpha,
        "ad-HD": ad_HD,
        "ad-G": ad_G,
        "δ-VEC": delta_VEC,
        "δ-ρ": delta_rho,
        "δ-e/a": delta_e_a,
        "δ-Ec": delta_Ec,
        "δ-υ": delta_ups,
        "δ-e/n": delta_e_n,
        "δ-G": delta_G,
        "δ-G/B": delta_G_B,
        "δ-Tm": delta_Tm,
        "δ-HD": delta_HD,
        "δ-χ": delta_chi,
        "δ-EWF": delta_ewf,
        "δ-SLHF": delta_SLHF,
        "δ-α": delta_alpha
    }
    return feat
# ------------------------------------------------------------------------------
# 3. 各模型使用的特征列表（v2 已按新最优特征子集更新）
# ------------------------------------------------------------------------------
# KNN: F1=0.8229, 6 features
KNN_FEATURES = ["Pm", "ρ", "α", "ad-e/n", "δ-e/n", "δ-Ec"]
# CATboost: F1=0.8626, 9 features
CAT_FEATURES = ["Pm", "ΔS", "ad-e/n", "ad-HD", "ad-G", "δ-χ", "δ-EWF", "δ-HD", "δ-G"]
# LightGBM: F1=0.8783, 9 features
LGB_FEATURES = ["Pm", "e/a", "EWF", "ad-e/n", "δ-e/n", "δ-ρ", "δ-Ec", "δ-G", "δ-υ"]
# RF: F1=0.8471, 8 features
RF_FEATURES = ["Pm", "ΔS", "EWF", "ad-e/n", "δ-VEC", "δ-EWF", "δ-G", "δ-G/B"]
# XGBoost: F1=0.8605, 9 features
XGB_FEATURES = ["Pm", "Ω", "e/a", "EWF", "G", "ad-e/n", "ad-G", "δ-e/n", "δ-ρ"]
# 所有需要输出的特征列表，严格按照模型顺序排列（v2 已同步更新）
ALL_OUTPUT_FEATURES = [
    # KNN (6)
    "Pm", "ρ", "α", "ad-e/n", "δ-e/n", "δ-Ec",
    # CATboost (9)
    "Pm", "ΔS", "ad-e/n", "ad-HD", "ad-G", "δ-χ", "δ-EWF", "δ-HD", "δ-G",
    # LightGBM (9)
    "Pm", "e/a", "EWF", "ad-e/n", "δ-e/n", "δ-ρ", "δ-Ec", "δ-G", "δ-υ",
    # RF (8)
    "Pm", "ΔS", "EWF", "ad-e/n", "δ-VEC", "δ-EWF", "δ-G", "δ-G/B",
    # XGBoost (9)
    "Pm", "Ω", "e/a", "EWF", "G", "ad-e/n", "ad-G", "δ-e/n", "δ-ρ"
]
# ------------------------------------------------------------------------------
# 4. 加载模型、编码器与标准化器
# ------------------------------------------------------------------------------
print("🔽 正在加载所有模型...")
rf_model = joblib.load(RF_MODEL_PATH)
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
# 5. 统一类别顺序，从编码器读取XGB真实相名称
# ------------------------------------------------------------------------------
rf_classes  = [str(cls) for cls in rf_model.classes_]
knn_classes = [str(cls) for cls in KNN_model.classes_]
cat_classes = [str(cls) for cls in cat_model.classes_]
lgb_classes = [str(cls) for cls in lgb_model.classes_]
# XGB类别取编码器内的原始相名称
xgb_phase_classes = [str(c) for c in le_xgb.classes_]
# 合并所有相类别，去重排序
all_classes = sorted(set(rf_classes + knn_classes + cat_classes + lgb_classes + xgb_phase_classes))
print(f"📊 统一类别顺序: {all_classes}")
print(f"📊 按类别差异化权重配置: {CLASS_WEIGHTS}\n")
# ------------------------------------------------------------------------------
# 6. 预测核心逻辑
# ------------------------------------------------------------------------------
def predict_chunk(df_chunk, all_features):
    """处理单个数据块，返回预测结果与各类别概率"""
    # 构造各模型输入DataFrame（v2: 使用更新后的特征列表）
    X_rf       = pd.DataFrame([{f: d[f] for f in RF_FEATURES}  for d in all_features])
    X_xgb      = pd.DataFrame([{f: d[f] for f in XGB_FEATURES} for d in all_features])
    X_KNN_raw  = pd.DataFrame([{f: d[f] for f in KNN_FEATURES} for d in all_features])
    X_cat      = pd.DataFrame([{f: d[f] for f in CAT_FEATURES} for d in all_features])
    X_lgb      = pd.DataFrame([{f: d[f] for f in LGB_FEATURES} for d in all_features])
    # KNN标准化（和训练时保持一致）
    X_KNN = knn_scaler.transform(X_KNN_raw)
    # step1: 各模型输出预测概率
    rf_proba       = rf_model.predict_proba(X_rf)
    xgb_proba_raw  = xgb_model.predict_proba(X_xgb)
    knn_proba      = KNN_model.predict_proba(X_KNN)
    cat_proba      = cat_model.predict_proba(X_cat)
    lgb_proba      = lgb_model.predict_proba(X_lgb)
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
    rf_proba_reordered  = reorder_other(rf_proba,  rf_classes,  "RF")
    knn_proba_reordered = reorder_other(knn_proba, knn_classes, "KNN")
    cat_proba_reordered = reorder_other(cat_proba, cat_classes, "CAT")
    lgb_proba_reordered = reorder_other(lgb_proba, lgb_classes, "LGB")
    # XGB概率映射修正
    xgb_proba_reordered = np.zeros((len(df_chunk), len(all_classes)))
    for xgb_index, phase_name in enumerate(xgb_phase_classes):
        pos = all_classes.index(phase_name)
        xgb_proba_reordered[:, pos] = xgb_proba_raw[:, xgb_index]

    # =====================【核心修改：逐类别加权】=====================
    total_proba = np.zeros_like(rf_proba_reordered)
    for cls_name, w_list in CLASS_WEIGHTS.items():
        if cls_name not in all_classes:
            print(f"⚠️ 类别 {cls_name} 不在all_classes，跳过")
            continue
        c_idx = all_classes.index(cls_name)
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
    # 组装结果DataFrame（保留原始输入列+预测结果+概率）
    result_df = df_chunk.copy()
    result_df["pred_ensemble"] = final_pred
    result_df["pred_RF"]  = pred_rf
    result_df["pred_XGB"] = pred_xgb
    result_df["pred_KNN"] = pred_knn
    result_df["pred_CAT"] = pred_cat
    result_df["pred_LGB"] = pred_lgb
    # 每个类别的加权概率
    for i, cls in enumerate(all_classes):
        result_df[f"prob_{cls}"] = total_proba[:, i]
    # 按指定顺序添加所有计算的特征列到末尾
    feature_df = pd.DataFrame.from_records(all_features)
    feature_df = feature_df[ALL_OUTPUT_FEATURES].reset_index(drop=True)
    result_df = pd.concat([result_df.reset_index(drop=True), feature_df], axis=1)
    return result_df
# ------------------------------------------------------------------------------
# 7. 分块读取、计算特征、预测、写入文件
# ------------------------------------------------------------------------------
first_write = True
for chunk_idx, df_chunk in enumerate(
    pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, dtype=object)
):
    print(f"📦 正在处理第 {chunk_idx + 1} 批数据，行数：{len(df_chunk)}")
    # 提取前15列元素成分，最后一列Pm值
    comps = df_chunk.iloc[:, :15].values.astype(float)
    pm_inputs = df_chunk.iloc[:, -1].astype(float).astype(int).values
    # 批量计算所有样本的特征
    all_features = [
        compute_all_features(comp, pm, ELEMENT_PARAMS)
        for comp, pm in zip(comps, pm_inputs)
    ]
    # 预测
    result_df = predict_chunk(df_chunk, all_features)
    # 写入输出CSV（分块追加，避免内存溢出）
    result_df.to_csv(
        OUTPUT_CSV,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8",
        float_format='%.4f'
    )
    first_write = False
    print(f"✅ 第 {chunk_idx + 1} 批处理完成\n")
print("🎉🎉🎉 全部处理完成！")
print(f"结果保存在：{OUTPUT_CSV}")
print("输出文件包含：原始输入列 + 各模型单独预测结果 + 集成最终预测结果 "
      "+ 所有类别的加权概率 + 按指定顺序排列的全部计算特征")
