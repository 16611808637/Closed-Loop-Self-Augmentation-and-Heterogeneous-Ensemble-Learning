import pandas as pd

df = pd.read_csv("aug_lgbm_round1.csv", header=0)
col = df.columns[9]         

print("原始分类分布：")
print(df[col].value_counts())

TARGET_N =1523
samples = []
for label, g in df.groupby(col):
    n = min(TARGET_N, len(g))      
    samples.append(g.sample(n=n, random_state=42))

sampled = pd.concat(samples, ignore_index=True)

# 保留原始表头保存
sampled.to_csv("augmentation.csv", index=False)

print(f"\n共 {len(sampled)} 行，抽样后分布：")
print(sampled[col].value_counts())
