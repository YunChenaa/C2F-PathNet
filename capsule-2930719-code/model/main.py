import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import torch.nn.functional as F
from MLML_MP import MTGNN
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from skmultilearn.model_selection import IterativeStratification
from guide_edge_info import path_matrix_coarse,path_matrix_fine





batch_size=64

class MolDataset(InMemoryDataset):
    def __init__(self, root, graph_data, transform=None, pre_transform=None):
        self.graph_data = graph_data  # 存储 SMILES
      # DeepChem 特征提取
        super(MolDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # 读取存储数据
    def processed_file_names(self):
        return ["mol_data.pt"]

def convert_to_binary_torch(labels, num_classes):
    """
    将类别索引列表转换为 one-hot 二进制编码（PyTorch 版本）
    :param labels: list，类别索引，例如 [3, 7, 150]
    :param num_classes: 总类别数
    :return: torch.Tensor, 长度为 num_classes 的 one-hot 形式
    """
    binary_encoding = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        binary_encoding[label] = 1  # 由于类别从1开始，需要减1
    return binary_encoding



device = torch.device('cuda')


def train(train_loader,model,optimizer,a,b,c,d,e):

    criterion_coarse = nn.BCEWithLogitsLoss()
    criterion_fine = nn.BCEWithLogitsLoss()
    criterion_fine2 = nn.BCEWithLogitsLoss()

    model.train()
    total_loss = 0
    epoch_loss_coarse = 0
    epoch_loss_fine = 0
    epoch_loss_fine2 = 0
    total_samples = 0
    for data in train_loader:
        bs = data.size(0)
        data = data.to(device)
        optimizer.zero_grad()

        # 前向传播
        coarse_pre_logit,fine_pre_logit,fine2_pre_logit = model(data)

        p_coarse = torch.sigmoid(coarse_pre_logit)  # [B, C1]
        p_fine = torch.sigmoid(fine_pre_logit)  # [B, C2]
        p_fine2 = torch.sigmoid(fine2_pre_logit)  # [B, C2]

        # 2. coarse_to_fine 是 [C1, C2]，我们想得到：fine 的 coarse 支持程度
        # 即计算每个 fine 类的 coarse 激活强度（batch-wise）
        support_coarse = torch.matmul(p_coarse, path_matrix_coarse)  # [B, C2]
        support_fine = torch.matmul(p_fine, path_matrix_fine)  # [B, C2]

        # 3. 希望 p_fine <= support，即 fine 不应超过其 coarse 的强度
        # 用 hinge-like 惩罚：ReLU(p_fine - support)
        consistency_coarse = F.relu(p_fine - support_coarse)
        consistency_fine = F.relu(p_fine2 - support_fine)

        # 4. 平均 consistency 作为正则项
        consistency_loss_coarse = consistency_coarse.mean()
        consistency_loss_fine = consistency_fine.mean()

        loss1 = criterion_coarse(coarse_pre_logit, data.coarse_label)
        loss2 = criterion_fine(fine_pre_logit, data.fine_label)
        loss3 = criterion_fine2(fine2_pre_logit, data.fine2_label)




        total_loss = a*loss1 + b*loss2 + c*loss3 + d*consistency_loss_coarse + e*consistency_loss_fine


        # 反向传播
        total_loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss_coarse += loss1.item() * bs
        epoch_loss_fine += loss2.item() * bs
        epoch_loss_fine2 += loss3.item() * bs
        total_samples += bs

    avg_epoch_loss_coarse = epoch_loss_coarse / total_samples
    avg_epoch_loss_fine = epoch_loss_fine / total_samples
    avg_epoch_loss_fine2 = epoch_loss_fine2 / total_samples
    return total_loss.item(),avg_epoch_loss_coarse,avg_epoch_loss_fine,avg_epoch_loss_fine2


# --------------------------
# 测试代码
# --------------------------
def test(test_loader,model):
   # test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    # accs = [0, 0]
    all_true_coarse_labels = []
    all_true_fine_labels = []
    all_true_fine2_labels = []
    all_pred_coarse_labels = []
    all_pred_fine_labels = []
    all_pred_fine2_labels = []

    epoch_loss_coarse = 0
    epoch_loss_fine = 0
    epoch_loss_fine2 = 0
    # total_samples = 0
    with (torch.no_grad()):
        for data in test_loader:
            data = data.to(device)
            bs = data.size(0)

            coarse_pre,fine_pre,fine2_pre = model(data)

            # # 真# 实标# # # # # # ## # # 签# 值
            coarse_labels = data.coarse_label.cpu().numpy()

            fine_labels = data.fine_label.cpu().numpy()

            fine2_labels = data.fine2_label.cpu().numpy()

            loss1 = criterion(coarse_pre, data.coarse_label)
            loss2 = criterion(fine_pre, data.fine_label)
            loss3 = criterion(fine2_pre, data.fine2_label)

            epoch_loss_coarse += loss1.item() * bs
            epoch_loss_fine += loss2.item() * bs
            epoch_loss_fine2 += loss3.item() * bs

            coarse_preds = (torch.sigmoid(coarse_pre)>0.5).cpu().numpy()
            fine_preds = (torch.sigmoid(fine_pre)>0.5).cpu().numpy()
            fine2_preds = (torch.sigmoid(fine2_pre)>0.5).cpu().numpy()

            #将batch加入总列表
            all_true_coarse_labels.extend(coarse_labels)
            all_pred_coarse_labels.extend(coarse_preds)
            all_true_fine_labels.extend(fine_labels)
            all_pred_fine_labels.extend(fine_preds)
            all_true_fine2_labels.extend(fine2_labels)
            all_pred_fine2_labels.extend(fine2_preds)

    true_coarse_labels = np.array(all_true_coarse_labels)
    pred_coarse_labels = np.array(all_pred_coarse_labels)
    true_fine_labels = np.array(all_true_fine_labels)
    pred_fine_labels = np.array(all_pred_fine_labels)
    true_fine2_labels = np.array(all_true_fine2_labels)
    pred_fine2_labels = np.array(all_pred_fine2_labels)

    micro_precision_coarse = precision_score(true_coarse_labels, pred_coarse_labels, average='micro')
    micro_recall_coarse = recall_score(true_coarse_labels, pred_coarse_labels, average='micro')
    micro_f1_coarse = f1_score(true_coarse_labels, pred_coarse_labels, average='micro')

    micro_precision_fine = precision_score(true_fine_labels, pred_fine_labels, average='micro')
    micro_recall_fine = recall_score(true_fine_labels, pred_fine_labels, average='micro')
    micro_f1_fine = f1_score(true_fine_labels, pred_fine_labels, average='micro')

    micro_precision_fine2 = precision_score(true_fine2_labels, pred_fine2_labels, average='micro')
    micro_recall_fine2 = recall_score(true_fine2_labels, pred_fine2_labels, average='micro')
    micro_f1_fine2 = f1_score(true_fine2_labels, pred_fine2_labels, average='micro')
    print("---------------------------")
    print(f'micro_precision_coarse:{micro_precision_coarse:.4f}')
    print(f'micro_recall_coarse:{micro_recall_coarse:.4f}')
    print(f'micro_f1_coarse: {micro_f1_coarse:.4f}')
    print("---------------------------")
    print(f'micro_precision_fine: {micro_precision_fine:.4f}')
    print(f'micro_recall_fine: {micro_recall_fine:.4f}')
    print(f'micro_f1_fine: {micro_f1_fine:.4f}')
    print("---------------------------")
    print(f'micro_precision_fine2: {micro_precision_fine2:.4f}')
    print(f'micro_recall_fine2: {micro_recall_fine2:.4f}')
    print(f'micro_f1_fine2: {micro_f1_fine2:.4f}')
    print("---------------------------")
    return micro_f1_coarse

from rdkit import Chem

def is_same_molecule(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return False  # 无法解析，视为不同分子

    can1 = Chem.MolToSmiles(mol1, canonical=True)
    can2 = Chem.MolToSmiles(mol2, canonical=True)

    return can1 == can2
path_matrix_coarse = path_matrix_coarse.to("cuda")
path_matrix_fine = path_matrix_fine.to("cuda")

from scipy.sparse import csr_matrix


# --------------------------
# 执行训练
#--------------------------

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
#
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


dataset_train_dev = MolDataset(root="/data/data_treeLabel_aug/",graph_data = None)
dataset_test = MolDataset(root="/data/data_treeLabel_test/",graph_data = None)






if __name__ == "__main__":

    labels = np.array([data.fine_label[0] for data in dataset_train_dev])


    time_list = []
    # 转换为 CSR 矩阵
    y_csr = csr_matrix(labels)
    n_splits = 5
    stratifier = IterativeStratification(n_splits=n_splits, order=1)

    folds = list(stratifier.split(np.zeros(y_csr.shape[0]), y_csr))
    # os.makedirs("data_fold", exist_ok=True)
    # os.makedirs("data_output", exist_ok=True)
    # with open("data_fold/folds.pkl2", "wb") as f:
    #     pickle.dump(folds, f)

    # 后续使用
    # with open("folds.pkl", "rb") as f:
    #     folds = pickle.load(f)

    for fold_idx, (train_index, dev_index) in enumerate(folds):


        train_subset = Subset(dataset_train_dev, train_index)
        dev_subset = Subset(dataset_train_dev, dev_index)



        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        validate_subset = DataLoader(dev_subset, batch_size=64, shuffle=False)
        test_subset = DataLoader(dataset_test, batch_size=64, shuffle=False)


        model = MTGNN()
        # model.load_state_dict(torch.load("data_output_remove_5/threeLabel_model_epoch_1_100.pth"),strict=False)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

        # 每组参数训练几个 epoch（可调整）
        for epoch in range(1, 10):#此处一般为400，因为比较耗时，这里只用10次用于可运行测试

            loss, _, _, _ = train(train_loader, model, optimizer, 0.5, 0.5, 0.5, 0.2, 0.2)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            test(validate_subset, model)
            test(test_subset, model)
            # sys.exit()


            if epoch % 100 == 0:
                os.makedirs("data_output_remove_aug", exist_ok=True)
                torch.save(model.state_dict(), f"/result/data_output/threeLabel_model_epoch_{fold_idx + 3}_{epoch}.pth")
        break#正式训练课去掉
