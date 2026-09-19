import csv
import random
from itertools import combinations, product

# -------------------------- 配置项 --------------------------
# 15个核心元素，严格按照需求顺序排列
elements = [
    'Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni', 'Mo', 'Ti', 'W', 'Nb', 'Ta', 'V', 'Mn', 'Hf', 'Zr'
]
# 非0元素数量范围：4~6个
non_zero_counts = [4, 5, 6]
# 元素含量步长、最小值、最大值（5~40，步长5）
step = 5
min_value = 5
max_value = 40
# 成分总和约束（总和=100）
total_sum = 100
# 输出文件名（直接保存在当前文件夹）
output_file = "MCA_phase_predict.csv"

# 随机种子（可选，保证Pm可复现；不需要可删掉这行）
random.seed(42)

# -------------------------- 预计算数值解 --------------------------
solutions = {}
for k in non_zero_counts:
    target_sum = total_sum // step
    min_int = min_value // step
    max_int = max_value // step

    valid_values = []
    for int_values in product(range(min_int, max_int + 1), repeat=k):
        if sum(int_values) == target_sum:
            actual_values = [v * step for v in int_values]
            valid_values.append(actual_values)
    solutions[k] = valid_values
    print(f"非0元素数量k={k}时，符合条件的数值解数量: {len(valid_values)}")

# -------------------------- 生成CSV文件 --------------------------
# 表头：15个元素 + 第16列 Pm
columns = elements + ['Pm']
element_to_index = {elem: idx for idx, elem in enumerate(elements)}

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(columns)

    total_rows = 0
    for k in non_zero_counts:
        for element_comb in combinations(elements, k):
            for value_comb in solutions[k]:
                row = [0] * len(elements)
                for i in range(k):
                    elem = element_comb[i]
                    val = value_comb[i]
                    row[element_to_index[elem]] = val
                # 第16列 Pm：随机填入 1~4 的整数
                row.append(random.randint(1, 4))
                writer.writerow(row)
                total_rows += 1
                if total_rows % 100000 == 0:
                    print(f"已生成 {total_rows} 行数据")

print(f"\n✅ 生成完成！总数据行数: {total_rows}")
print(f"✅ 文件已保存到当前文件夹：{output_file}")
