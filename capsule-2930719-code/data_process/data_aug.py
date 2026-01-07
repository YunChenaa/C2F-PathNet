
import csv
from torch_geometric.utils import from_smiles
import pandas as pd
import random
from rdkit import Chem
import torch
import torch.nn as nn
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np





def randomize_smiles(smiles):
    """
    对输入的 SMILES 字符串进行随机枚举，生成一个随机 SMILES。
    如果分子解析失败则返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 利用 doRandom=True 生成随机枚举的 SMILES
    return Chem.MolToSmiles(mol, doRandom=True)


# 1. 读取 CSV 数据（假设文件名为 "data.csv"）
df = pd.read_csv("threeLabel_train_test.csv")  # CSV 中必须包含 'smiles' 和 'label' 两列

# 2. 统计各类别的样本数量，并找出最大样本数作为目标平衡数量
class_counts = df['label'].value_counts()
max_count = class_counts.max()
# print("各类别样本数：")
# print(class_counts)
# print("目标样本数（平衡后每个类别）：", max_count)

# 3. 将原始数据加入增强数据集列表
augmented_data = df.to_dict(orient="records")


def get_item_by_smiles(smiles, data, item):
    # 遍历数据，找到匹配的name并返回对应的age
    for record in data:
        if record['smiles'] == smiles:
            return record[item]
    return None  # 如果没有找到对应的name
# 对每个类别进行数据增强，补充样本数不足的类别
for label, count in class_counts.items():
    random_num = random.randint(0,20)
    num_to_generate = min(int(max_count - 1.5*count - random_num ),10*count) # 需要额外生成的样本数

    # 获取该类别下所有 SMILES 字符串
    samples = df[df['label'] == label]
    smiles_list = samples['smiles'].tolist()

    print(f"类别 {label} 需要额外生成 {num_to_generate} 个样本。")

    # 生成缺失数量的增强样本
    for _ in range(num_to_generate):
            # 随机选取一个已有的 SMILES
            original_smiles = random.choice(smiles_list)
            new_smiles = randomize_smiles(original_smiles)
            # 如果生成成功，则加入增强数据集
            if new_smiles is not None:
                (augmented_data.append
                 ({"smiles": new_smiles,
                   "label": label}))
            else:
                print(f"警告：SMILES '{original_smiles}' 无法被解析，跳过该样本。")


# 4. 转换为 DataFrame 并打乱顺序（防止模型训练时出现顺序偏差）
aug_df = pd.DataFrame(augmented_data)
aug_df = aug_df.sample(frac=1).reset_index(drop=True)

# 保存增强后的数据集到新的 CSV 文件
aug_df.to_csv("threeLabel_train_test_augmentation.csv", index=False)


