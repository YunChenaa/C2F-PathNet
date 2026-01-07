from collections import defaultdict


import torch
import csv

label_coarse_dict = defaultdict(list)
label_finetofine2_dict = defaultdict(list)
with open("/data/smiles_with_hierarchical_labels.csv", "r", newline='',encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        label = list(map(int,row[1].strip("[]").split(",")))
        if label[1] not in label_coarse_dict[label[0]]:
            label_coarse_dict[label[0]].append(label[1])
        if label[2] not in label_finetofine2_dict[label[1]]:
            label_finetofine2_dict[label[1]].append(label[2])


n_level1 = 11

n_level2 = 170
# 三级标签数量
n_level3 = 707

# 初始化路径矩阵
path_matrix_coarse = torch.zeros(n_level1, n_level2)
path_matrix_fine = torch.zeros(n_level2, n_level3)

# 填充路径矩阵
for level1, level2_list in label_coarse_dict.items():
    for level2 in level2_list:
        path_matrix_coarse[level1, level2] = 1

for level2, level3_list in label_finetofine2_dict.items():
    for level3 in level3_list:
        path_matrix_fine[level2, level3] = 1

total_coarse =  n_level1 + n_level2
total_fine =  n_level3 + n_level2
adj_matrix_coarse = torch.zeros((total_coarse, total_coarse))
adj_matrix_fine = torch.zeros((total_fine, total_fine))

# 构建连接：coarse i 和 fine j → 设置 adj[i][j+11] = adj[j+11][i] = 1
for i in range(n_level1):
    for j in range(n_level2):
        if path_matrix_coarse[i][j] == 1:
            coarse_idx = i
            fine_idx = j + n_level1
            adj_matrix_coarse[coarse_idx][fine_idx] = 1
            adj_matrix_coarse[fine_idx][coarse_idx] = 1  # 可选：是否对称（无向图）


for i in range(n_level2):
    for j in range(n_level3):
        if path_matrix_fine[i][j] == 1:
            coarse_idx = i
            fine_idx = j + n_level2
            adj_matrix_fine[coarse_idx][fine_idx] = 1
            adj_matrix_fine[fine_idx][coarse_idx] = 1  # 可选：是否对称（无向图）


