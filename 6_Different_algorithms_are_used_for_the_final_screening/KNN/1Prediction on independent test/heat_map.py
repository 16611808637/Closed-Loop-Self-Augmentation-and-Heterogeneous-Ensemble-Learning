import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
# ---------------------- 配置参数 ----------------------
csv_file = "phase_predict_result.csv"
labels = ['BCC', 'BCC+FCC', 'IM', 'FCC']
n_class = len(labels)
true_col_idx = 15
pred_col_indices = [83, 84, 85, 86, 87, 88]
model_names = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5", "Model_6"]
# ---------------------- 全局字体设置：Times New Roman ----------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 50
plt.rcParams['axes.unicode_minus'] = False
def plot_one_confusion_matrix(y_true, y_pred, save_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # ---------------------- 计算各类别召回率 + 宏F1 ----------------------
    class_acc = []
    for i in range(n_class):
        tp = cm[i, i]
        total_true = np.sum(cm[i, :])
        acc_i = tp / total_true if total_true != 0 else 0.0
        class_acc.append(acc_i)
    y_true_expand = []
    y_pred_expand = []
    for i in range(n_class):
        for j in range(n_class):
            cnt = cm[i, j]
            y_true_expand += [i] * cnt
            y_pred_expand += [j] * cnt
    macro_f1 = f1_score(y_true_expand, y_pred_expand, average='macro')
    # ---------------------- 绘制混淆矩阵 ----------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    # ====================== 边框与网格线设置 ======================
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    x_ticks = np.arange(n_class + 1) - 0.5
    y_ticks = np.arange(n_class + 1) - 0.5
    ax.hlines(y_ticks, xmin=x_ticks.min(), xmax=x_ticks.max(), linewidth=3, color='white', zorder=1)
    ax.vlines(x_ticks, ymin=y_ticks.min(), ymax=y_ticks.max(), linewidth=3, color='white', zorder=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=36, color='#006BAC')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=36, color='#006BAC')
    ax.set_xlabel('Predicted label', fontweight='bold')
    # 关闭原生y轴标签，消除重复的True label
    ax.set_ylabel("")
    # ========= 手动绘制True label（只保留这一行） =========
    ax.text(-2.6, n_class/2 - 0.5, 'True label', fontweight='bold', fontsize=50,
            rotation=90, va='center', ha='center')
    # ========== 左侧 recall 列：表头 + 每类数字（全部居中 ha='center'） ==========
    recall_x = -2.15
    ax.text(recall_x, -0.3, '  Recall:', fontsize=36, color='black',
            weight='bold', va='center', ha='center')
    for i in range(n_class):
        acc_text = f'{class_acc[i]:.2f}'
        ax.text(recall_x, i, acc_text, fontsize=36, color='black',
                weight='bold', va='center', ha='center')
    info_text = f'Macro-F1 = {macro_f1:.4f}'
    ax.text(-0.5, -0.18, info_text, transform=ax.transAxes, ha='left',
            fontsize=36, fontweight='bold')
    # =====================================================================
    plt.tight_layout()
    plt.subplots_adjust(left=0.22)
    fig.canvas.draw()
    # ========== 右侧色条精确缩小控制 ==========
    cbar = disp.im_.colorbar
    original_left, original_bottom, original_width, original_height = cbar.ax.get_position().bounds
    new_width = original_width / 1.5
    new_height = original_height / 1.5
    new_left = original_left + (original_width - new_width) / 2
    new_bottom = original_bottom + (original_height - new_height) / 2
    cbar.ax.set_position([new_left, new_bottom, new_width, new_height])
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图片已保存 {save_name} | Macro-F1: {macro_f1:.4f}")
if __name__ == "__main__":
    df = pd.read_csv(csv_file)
    y_true_all = df.iloc[:, true_col_idx].values
    for idx, col in enumerate(pred_col_indices):
        y_pred_model = df.iloc[:, col].values
        plot_one_confusion_matrix(
            y_true_all,
            y_pred_model,
            save_name=f"confusion_matrix_{model_names[idx]}.png"
        )
