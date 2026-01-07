import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch_geometric.nn import GATv2Conv, global_mean_pool, SAGEConv, TopKPooling, global_max_pool


# weight = torch.tensor(weight_matrix,requires_grad=True,device="cuda")
# weight = weight*5


def predict_smiles(smiles_list, model, tokenizer):
    inputs = tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    predictions = (probs >= 0.5).astype(int)  # 采用阈值 0.5
    return logits, predictions, probs


top_k = 0.8


# def top_K_edge(attention_weights,data):
#
#     return topk_edge_indices
def convert_to_binary_torch(labels, num_classes):
    """
    将类别索引列表转换为 one-hot 二进制编码（PyTorch 版本）
    :param labels: list，类别索引，例如 [3, 7, 150]
    :param num_classes: 总类别数
    :return: torch.Tensor, 长度为 num_classes 的 one-hot 形式
    """
    binary_encoding = torch.zeros(num_classes, dtype=torch.float32, device="cuda")
    for label in labels:
        binary_encoding[label] = 1  # 由于类别从1开始，需要减1
    return binary_encoding


# from transformers import AutoTokenizer, AutoModel
hidden_states_dim = 768


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)






from guide_edge_info import adj_matrix_fine, adj_matrix_coarse




# --------------------------
# 模型定义
# --------------------------
class WeightMatrix(nn.Module):
    def __init__(self, coarse_num, fine_num):
        super(WeightMatrix, self).__init__()
        # 创建一个 3x3 的可学习权重矩阵
        self.weight_matrix = nn.Parameter(torch.zeros(coarse_num, fine_num, requires_grad=True)).float()

    def forward(self, x):
        weight_matrix = F.softmax(self.weight_matrix, dim=1)

        x = torch.matmul(x, weight_matrix)

        return x


class SAGEGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.float()

        y = self.conv(x, edge_index)
        return x + y


class GATGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GATv2Conv(in_channels, out_channels, edge_dim=11)

    def forward(self, x, edge_index, edge_attr):
        x = x.float()

        y = self.conv(x, edge_index, edge_attr)
        return x + y


class GCNLayer(nn.Module):
    def __init__(self, adj_matrix, in_feats, out_feats):
        super().__init__()
        self.adj = adj_matrix.to("cuda")  # 预先计算好的 torch.tensor 类型邻接矩阵 A
        self.W = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        # 简单一阶传播：AXW
        A_hat = self.adj + torch.eye(self.adj.size(0)).to(x.device)
        D_hat = torch.diag(torch.sum(A_hat, dim=1) ** -0.5)
        x = torch.matmul(D_hat @ A_hat @ D_hat, x)
        x = x.t()
        return self.W(x)


class SharedGNN(nn.Module):
    """共享特征提取的基座GNN"""

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(30, 64, add_self_loops=False, edge_dim=11)
        self.conv2 = GATv2Conv(64, 256, edge_dim=11)
        self.pool1 = TopKPooling(256, ratio=0.8)
        self.conv3 = SAGEGNN(256, 256)

        self.top_k = 0.8


    def compute_edge_importance(self, edge_index, attention_weights):
        """
        直接使用 GAT 计算出的注意力权重作为边的重要性
        """
        return attention_weights[1].squeeze()


    def forward(self, data):
        x, edge_index, edge_attr, batch, smiles = data.x, data.edge_index, data.edge_attr, data.batch, data.smiles

        x = x.float()

        x = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x)
        x = F.leaky_relu(x)

        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        # data.edge_index = edge_index
        # data = data.sort(sort_by_row=False)

        # x = F.leaky_relu(self.conv3(x, edge_index))

        x = F.leaky_relu(self.conv3(x,edge_index))

        return x, edge_index, edge_attr, batch, smiles


class TaskBranchFine(nn.Module):
    """单个任务分支"""

    def __init__(self, batch_index):
        super().__init__()
        self.batch_index = batch_index
        self.conv1 = GATGNN(256, 256)
        self.conv2 = GATGNN(256, 256)

        self.line1 = torch.nn.Linear(512, 256)
        self.line2 = torch.nn.Linear(256, 170)
        self.bn = torch.nn.BatchNorm1d(256)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.top_k = 0.8

        # self.smiles_text_process = SmilesTextProcess(170)

    # (
    def forward(self, x, edge_index, edge_attr, batch, smiles):
        # y = self.smiles_text_process(smiles)

        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))

        x1 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x2 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        # x = F.leaky_relu(self.conv3(x, edge_index,edge_attr))
        # x = global_mean_pool(x, batch=batch)
        x = x1 + x2
        # x = self.line1(x)
        x = F.leaky_relu(self.line1(x))
        x = self.dropout(x)
        x = self.line2(x)

        return x


class TaskBranchFine2(nn.Module):
    """单个任务分支"""

    def __init__(self, batch_index):
        super().__init__()
        self.batch_index = batch_index
        self.conv1 = GATv2Conv(256, 512, edge_dim=11)
        self.conv2 = GATGNN(512, 512)

        self.line1 = torch.nn.Linear(1024, 512)
        self.line2 = torch.nn.Linear(512, 707)
        self.bn = torch.nn.BatchNorm1d(256)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.top_k = 0.8

        # self.smiles_text_process = SmilesTextProcess(170)

    # (
    def forward(self, x, edge_index, edge_attr, batch, smiles):
        # y = self.smiles_text_process(smiles)

        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))

        x1 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x2 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        # x = F.leaky_relu(self.conv3(x, edge_index,edge_attr))
        # x = global_mean_pool(x, batch=batch)
        x = x1 + x2

        x = F.leaky_relu(self.line1(x))
        x = self.dropout(x)
        x = self.line2(x)

        return x


class TaskBranchCoarse(nn.Module):
    """单个任务分支"""

    def __init__(self, batch_index):
        super().__init__()
        self.batch_index = batch_index
        self.conv1 = GATGNN(256, 256)
        self.conv2 = GATGNN(256, 256)

        self.line1 = torch.nn.Linear(512, 256)
        self.line2 = torch.nn.Linear(256, 64)
        self.line3 = torch.nn.Linear(64, 11)
        self.bn = torch.nn.BatchNorm1d(256)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.act3 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)
        # self.smiles_text_process = SmilesTextProcess(170)

    def forward(self, x, edge_index, edge_attr, batch, smiles):
        # y = SmilesTextProcess(smiles)
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))

        x1 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x2 = torch.cat([global_mean_pool(x, batch=batch), global_max_pool(x, batch=batch)], dim=-1)
        # x = F.leaky_relu(self.conv3(x, edge_index,edge_attr))
        # x = global_mean_pool(x, batch=batch)
        x = x1 + x2

        x = F.leaky_relu(self.line1(x))

        x = self.dropout1(x)
        x = F.leaky_relu(self.line2(x))

        x = self.dropout2(x)
        x = self.line3(x)

        return x


class MTGNN(nn.Module):
    """多任务图神经网络"""

    def __init__(self):  # 三个任务的输出维度
        super().__init__()
        # 共享基座

        self.shared_gnn = SharedGNN()
        # self.weight = weight
        # self.liner1 = torch.nn.Linear(256,245)
        # self.liner2 = torch.nn.Linear(256,170)
        # 任务分支
        self.branch_coarse = TaskBranchCoarse(64)
        self.branch_fine = TaskBranchFine(64)
        self.branch_fine2 = TaskBranchFine2(64)

        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.gcn1 = GCNLayer(adj_matrix_coarse, 181, 170)
        self.gcn2 = GCNLayer(adj_matrix_fine, 877, 707)

    def forward(self, data):
        # 共享特征提取
        # x,edge_index,smiles = data.x,data.edge_index,data.smiles
        # x = x.float()
        # coarse_logits,coarse_pre_transfer,coarse_probs_transfer = predict_smiles(smiles,model_coarse,tokenizer_coarse)
        # fine_logits,fine_pre_transfer,fine_probs_transfer = predict_smiles(smiles,model_fine,tokenizer_fine)

        x_shared, edge_index, edge_attr, batch, smiles = self.shared_gnn(data)

        # x1,attn_weights = self.attention1(x_shared,x_shared,x_shared)
        # x2,attn_weights = self.attention2(x_shared,x_shared,x_shared)
        # # 各分支预测

        out_coarse = self.branch_coarse(x_shared, edge_index, edge_attr, batch, smiles)

        out_fine = self.branch_fine(x_shared, edge_index, edge_attr, batch, smiles)
        out_fine = torch.cat([out_fine, out_coarse], dim=-1)
        out_fine = self.gcn1(out_fine.t())

        out_fine2 = self.branch_fine2(x_shared, edge_index, edge_attr, batch, smiles)
        out_fine2 = torch.cat([out_fine2, out_fine], dim=-1)
        out_fine2 = self.gcn2(out_fine2.t())


        return out_coarse, out_fine, out_fine2

