import pandas as pd
# 配置
INPUT_FILE = 'phase_predict_result.csv'
OUTPUT_FILE = 'filtered_results3.csv'
# TARGET_COLS: [22, 23, 24, 25]，对应top数量依次为 2500, 2500, 5000, 5000
TARGET_COLS = [22, 23, 24, 25]
TOP_N_LIST = [2500, 5000, 6000, 1600]

# 第一步：只读取目标列，找出每列对应topN的行索引（取并集）
print("正在读取目标列，查找各列TopN行索引...")
df_cols = pd.read_csv(INPUT_FILE, usecols=TARGET_COLS)
top_indices = set()

for col, top_n in zip(df_cols.columns, TOP_N_LIST):
    top_idx = df_cols[col].nlargest(top_n).index
    top_indices.update(top_idx)
    print(f"  列 {col}: 取Top{top_n}, 累计并集行数 {len(top_indices)}")

sorted_indices = sorted(top_indices)
target_set = set(sorted_indices)
print(f"\n并集共 {len(sorted_indices)} 行，开始筛选输出...")
del df_cols

# 第二步：分块读取全列数据，只保留目标行
print("正在分块筛选并写出结果...")
first_chunk = True
kept = 0
for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):
    # 计算当前 chunk 在原始文件中的全局行索引
    start = chunk.index[0]
    end = chunk.index[-1]
    # 找出当前 chunk 中需要保留的行
    mask = [i in target_set for i in range(start, end + 1)]
    filtered = chunk[mask]
    if len(filtered) > 0:
        filtered.to_csv(OUTPUT_FILE, mode='a', index=False, header=first_chunk)
        kept += len(filtered)
        first_chunk = False

print(f"\n完成！已输出 {kept} 行到 {OUTPUT_FILE}")
