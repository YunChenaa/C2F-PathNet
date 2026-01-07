import sys
from collections import defaultdict

import networkx as nx
import numpy
from deepchem.feat import MolGraphConvFeaturizer

from torch import dtype
from torch_geometric.data import InMemoryDataset, Data
import torch
from rdkit import Chem
import csv
import re
import numpy as np


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index
def convert_to_binary_torch(labels, num_classes):

    binary_encoding = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        binary_encoding[label] = 1  # 由于类别从1开始，需要减1
    return binary_encoding

graph_data = []
with open("data/smiles_with_multi_labels.csv", "r", newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:

        single_data = dict()
        single_data["coarse_label"] = []
        single_data["fine_label"] = []
        single_data["fine2_label"] = []

        single_data["smiles"] = row[0]
        label_all = list(map(int, re.findall(r"\d+", row[1])))
        for i in range(0,len(label_all),3):
            single_data["coarse_label"].append(label_all[i])
            single_data["fine_label"].append(label_all[i+1])
            single_data["fine2_label"].append(label_all[i+2])






        graph_data.append(single_data)

class MolDataset(InMemoryDataset):
    def __init__(self, root, graph_data, transform=None, pre_transform=None):
        self.graph_data = graph_data  # 存储 SMILES
        self.featurizer = MolGraphConvFeaturizer(use_edges=True)  # DeepChem 特征提取
        super(MolDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # 读取存储数据

    # @property
    def processed_file_names(self):
        return ["mol_data.pt"]  # 存储文件名

    def process(self):

        data_list = []
        count =0
        for i,sample in enumerate(self.graph_data):  # 遍历 SMILES 数据
            mol = Chem.MolFromSmiles(sample["smiles"])  # RDKit 解析 SMILES
            # _,feature_x,edge_index = smile_to_graph(sample["smiles"])
            # if len(edge_index) == 0:
            #     print("edge_index 格式错误或为空:", edge_index)
            #
            #     edge_index_tensor = torch.tensor(edge_index,dtype=torch.long)
            #
            # else:
            #     edge_index_tensor = torch.tensor(edge_index,dtype=torch.long).transpose(1, 0)
            # try:
            graph = self.featurizer.featurize(mol)[0]  # 生成 GraphData


            # except Exception as e:
            #     print(sample["smiles"])
            #     continue

            coarse_label = convert_to_binary_torch(sample["coarse_label"], 11)
            fine_label = convert_to_binary_torch(sample["fine_label"], 170)
            fine2_label = convert_to_binary_torch(sample["fine2_label"], 707)
            smiles = sample["smiles"]
            # if isinstance(graph,numpy.ndarray):
            #     continue
            # print(i,graph)

            try:
                data = Data(
                    x=torch.tensor(graph.node_features, dtype=torch.float),
                    edge_index=torch.tensor(graph.edge_index, dtype=torch.int64),
                    edge_attr=torch.tensor(graph.edge_features, dtype=torch.float) if graph.edge_features is not None else None,
                    coarse_label= torch.tensor(coarse_label,dtype=torch.float).unsqueeze(0),
                    fine_label=torch.tensor(fine_label, dtype=torch.float).unsqueeze(0),
                    fine2_label=torch.tensor(fine2_label, dtype=torch.float).unsqueeze(0),

                    smiles= smiles
                )
            except Exception as e:
                print(f"❌ Error processing molecule {i}: {e}")
                count += 1
                continue

            data_list.append(data)
        print("-------num of remove------")
        print(count)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  # 存储数据



dataset = MolDataset(root="data/data_treeLabel_aug/", graph_data=graph_data)
dataset.process()




