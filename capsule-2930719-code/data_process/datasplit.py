import pandas as pd
import csv




df = pd.read_csv("data/smiles_with_multi_labels.csv")

min_count = 3

# 统计组合出现次数
combo_counts = df['label'].value_counts()

# 标记是否为稀有组合（小于 min_count 的）
df['is_rare'] = df['label'].apply(lambda x: combo_counts[x] < min_count)

# 先划分数据
rare_df = df[df['is_rare']]
common_df = df[~df['is_rare']]
rare_df.to_csv("rare_df.csv",index=False)


# 划分训练/验证集
from sklearn.model_selection import train_test_split
common_train, common_val = train_test_split(common_df, test_size=0.2, random_state=42)

# 合并结果
train_df = pd.concat([rare_df, common_train], ignore_index=True)
val_df = common_val.reset_index(drop=True)

# 可保存结果
train_df.to_csv('threeLabel_train_test.csv', index=False)
val_df.to_csv('threeLabel_validate.csv', index=False)
def label_level(smiles_label):
    coarse_label_list = set()
    fine_label_list = set()
    fine2_label_list = set()
    label_list = smiles_label.split(":")
    for i in label_list:
        label = list(map(int, i.strip("[]").split(",")))
        coarse_label_list.add(label[0])
        fine_label_list.add(label[1])
        fine2_label_list.add(label[2])
    return coarse_label_list,fine_label_list,fine2_label_list
df = pd.read_csv("threeLabel_validate.csv",header=None,names=["smiles","label","it"])
df = df.drop(columns=['it'])
df[["coarse_label","fine_label","fine2_label"]] = df["label"].apply(lambda x: pd.Series(label_level(x)))


df.to_csv("threeLabel_validate2.csv",index = False)



