# list_instances.py
import pandas as pd
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

# 转换为DataFrame便于分析
df = pd.DataFrame(dataset)

# 计算问题描述长度作为复杂度的一个指标
df['description_length'] = df['problem_statement'].apply(len)

# 按描述长度排序（通常较短的描述对应较简单的任务）
df_sorted = df.sort_values('description_length')

# 打印前10个最短描述的实例
for i, row in df_sorted.head(10).iterrows():
    print(f"ID: {row['instance_id']}, Length: {row['description_length']}")
    print(f"Repo: {row['repo']}")
    print('-' * 50)
