import sys

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,label_ranking_loss,roc_auc_score,matthews_corrcoef,hamming_loss

from main import MolDataset
from MLML_MP import MTGNN
from torch_geometric.loader import DataLoader


def one_error(y_pred,y_true):
    # 计算 Top-1 预测的标签
    # 计算 Top-1 预测的标签
    top_1_pred = np.argmax(y_pred, axis=1)

    # 计算 One-Error
    one_error = np.mean([1 if y_true[i, top_1_pred[i]] == 0 else 0 for i in range(len(y_true))])
    return  one_error
def accuracy_score(label,predict):
    label = np.array(label)
    predict = np.array(predict)
    equal_ratio = np.mean(label == predict)
    return equal_ratio
def MCC(y_pred,y_true):
    mcc_per_label = [matthews_corrcoef(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    mcc_macro = np.mean(mcc_per_label)
    return mcc_macro
def coverage(y_true,y_pred):
    coverage = 0
    for i in range(len(y_true)):
        # 找到真实标签的索引
        true_labels = np.where(y_true[i] == 1)[0]
        # 对预测分数进行排序 (从高到低)
        sorted_indices = np.argsort(-y_pred[i])
        # 找到真实标签中的最高排名
        max_rank = max([np.where(sorted_indices == j)[0][0] for j in true_labels])
        coverage += max_rank

    coverage /= len(y_true)

    return coverage


def top_k_success_rate(y_true, y_pred, k=3):
    # 初始化成功率列表
    success_rates = []

    # 遍历每个样本
    for i in range(len(y_true)):
        # 获取真实标签索引
        true_labels = set(np.where(y_true[i] == 1)[0])

        # 取概率值最高的 k 个标签索引
        top_k_labels = set(np.argsort(y_pred[i])[-k:])

        # 计算覆盖率
        intersection = true_labels & top_k_labels
        success_rate = len(intersection) / len(true_labels)
        success_rates.append(success_rate)

    # 返回每个样本的成功率，以及总体平均成功率
    return np.mean(success_rates), success_rates


def predict_smiles(smiles_list,model,tokenizer):
    inputs = tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    predictions = (probs >= 0.5).astype(int)  # 采用阈值 0.5
    return logits,predictions,probs

da_val = MolDataset(root="data/data_treeLabel_test",graph_data = None)

validate_loader = DataLoader(da_val, batch_size=64,shuffle=False)
device = torch.device('cuda')



# for a in dataset[100:200]:
#     a= a.fine2_label.squeeze()
#     indices = torch.where(a == 1)[0]
#     print(indices)
# sys.exit()
# model = MTGNN().to(device)
# model.load_state_dict(torch.load("C:/Users/陈韵/PycharmProjects/pythonProject1/data/weights/fine_tune_model2.2_epoch.pth"))
# checkpoint = torch.load("fine_tune2_model_epoch_10.pth", weights_only=True)

# 加载模型权重
model = MTGNN().to(device)
# model.load_state_dict(torch.load("C:/Users/陈韵/PycharmProjects/pythonProject1/data/weights/fine_tune_no_condition_model2.3_epoch.pth"))

checkpoint = torch.load("data_output_remove_aug/threeLabel_model_epoch_2_300.pth", weights_only=True)
model.load_state_dict(checkpoint)

# model.load_state_dict(torch.load("C:/Users/陈韵/PycharmProjects/pythonProject1/data/weights/fine_tune_no_condition_model2.3_epoch.pth", weights_only=True))
# model.train()  # 切换到训练模式
def test():
    model.eval()
    # accs = [0, 0]
    all_true_coarse_labels = []
    all_true_fine_labels = []
    all_true_fine2_labels = []
    all_pred_coarse_labels = []
    all_pred_fine_labels = []
    all_pred_fine2_labels = []

    with torch.no_grad():
        for data in validate_loader:
            data = data.to(device)
            coarse_pre,fine_pre,fine2_pre= model(data)
            smiles_lsit = list(data.smiles)

            # 真实标签值
            coarse_labels = data.coarse_label.cpu().numpy()

            fine_labels = data.fine_label.cpu().numpy()

            fine2_labels = data.fine2_label.cpu().numpy()

            # logits_coarse,coarse_pre_transfer, coarse_probs_transfer = predict_smiles(data.smiles, model_coarse, tokenizer_coarse)
            # logits_pre,fine_pre_transfer, fine_probs_transfer = predict_smiles(data.smiles, model_fine, tokenizer_fine)

            coarse_pre = torch.sigmoid(coarse_pre)
            fine_pre = torch.sigmoid(fine_pre)
            fine2_pre = torch.sigmoid(fine2_pre)

            # coarse_pre = torch.min(coarse_pre, torch.tensor(coarse_probs_transfer).to("cuda").detach())
            # fine2_pre = torch.max(fine2_pre, torch.tensor(fine_probs_transfer).to("cuda").detach())





            # topk_values_coarse, topk_indices_coarse = torch.topk(coarse_pre, k=3)
            # topk_values_fine, topk_indices_fine = torch.topk(fine_pre, k=5)
            # topk_values_fine2, topk_indices_fine2 = torch.topk(fine2_pre, k=10)
            #
            # # 创建一个全 0 的张量，并将前 5 大的位置设为 1
            # result_coarse = torch.zeros_like(coarse_pre)
            # for i in range(len(topk_indices_coarse)):
            #     result_coarse[i][topk_indices_coarse[i]] = 1
            #
            # result_fine = torch.zeros_like(fine_pre)
            # for i in range(len(topk_indices_fine)):
            #     result_fine[i][topk_indices_fine[i]] = 1

            # result_fine2 = torch.zeros_like(fine2_pre)
            # for i in range(len(topk_indices_fine2)):
            #     result_fine2[i][topk_indices_fine2[i]] = 1

            pred_coarse = (coarse_pre > 0.5)
            pred_fine = (fine_pre > 0.5)
            pred_fine2 = (fine2_pre > 0.5)


            coarse_preds = coarse_pre.cpu().numpy()
            fine_preds = fine_pre.cpu().numpy()
            fine2_preds = fine2_pre.cpu().numpy()

            #将batch加入总列表
            all_true_coarse_labels.extend(coarse_labels)
            all_pred_coarse_labels.extend(coarse_preds)
            all_true_fine_labels.extend(fine_labels)
            all_pred_fine_labels.extend(fine_preds)
            all_true_fine2_labels.extend(fine2_labels)
            all_pred_fine2_labels.extend(fine2_preds)

    true_coarse_labels = np.array(all_true_coarse_labels)
    pred_coarse_probs = np.array(all_pred_coarse_labels)
    true_fine_labels = np.array(all_true_fine_labels)
    pred_fine_probs = np.array(all_pred_fine_labels)
    true_fine2_labels = np.array(all_true_fine2_labels)
    pred_fine2_probs = np.array(all_pred_fine2_labels)

    pred_coarse_labels = np.where(pred_coarse_probs > 0.5, 1, 0)
    pred_fine_labels = np.where(pred_fine_probs > 0.5, 1, 0)
    pred_fine2_labels = np.where(pred_fine2_probs > 0.5, 1, 0)


    # for i in [0,1,2,5,6,9,12]:
    #     print("----------------------------------")
    #     print(f"第{i}个smiles:{smiles_lsit[i]}的预测值")
    #
    #     indices_pred = np.where(pred_coarse_labels[i] == 1)[0]
    #     print(indices_pred)
    #     print(f"第{i}个smiles:{smiles_lsit[i]}的真实值")
    #     indices_true = np.where(true_coarse_labels[i] == 1)[0]
    #     print(indices_true)
    # # print(true_coarse_labels[0])
    # sys.exit()
    #

    # best_threshold_coarse = tune_thresholds_per_label(true_coarse_labels,pred_coarse_labels)
    # print(best_threshold_coarse)
    # best_threshold_fine = tune_thresholds_per_label(true_fine_labels,pred_fine_labels)

    # pred_coarse = (pred_coarse_labels>best_threshold_coarse)
    # pred_fine = (pred_fine_labels>best_threshold_fine)
    # indices_coarse_pred = np.argwhere(pred_coarse_labels == 1)
    # print(indices_coarse_pred.T)
    # indices_coarse_true = np.argwhere(true_coarse_labels == 1)
    # print(indices_coarse_true.T)
    #
    # indices_fine_pred = np.argwhere(pred_fine_labels == 1)
    # print(indices_fine_pred.T)
    # indices_fine_true = np.argwhere(true_fine_labels == 1)
    # print(indices_fine_true.T)
    #
    # indices_fine2_pred = np.argwhere(pred_fine2_labels == 1)
    # print(indices_fine2_pred.T)
    # indices_fine2_true = np.argwhere(true_fine2_labels == 1)
    # print(indices_fine2_true.T)






    micro_precision_coarse = precision_score(true_coarse_labels, pred_coarse_labels, average='micro')
    micro_recall_coarse = recall_score(true_coarse_labels, pred_coarse_labels, average='micro')
    micro_f1_coarse = f1_score(true_coarse_labels, pred_coarse_labels, average='micro')
    micro_accuracy_coarse = accuracy_score(true_coarse_labels, pred_coarse_labels)
    auc_macro_coarse = roc_auc_score(true_coarse_labels, pred_coarse_probs, average="macro")
    auc_micro_coarse = roc_auc_score(true_coarse_labels, pred_coarse_probs, average="micro")
    auc_weighted_coarse = roc_auc_score(true_coarse_labels, pred_coarse_probs, average="weighted")
    rl_coarse = label_ranking_loss(true_coarse_labels, pred_coarse_probs)
    coarse_one_error = one_error(pred_coarse_probs,true_coarse_labels)
    coarse_MCC = MCC(pred_coarse_labels,true_coarse_labels)
    coarse_top3_acc,_ = top_k_success_rate(true_coarse_labels,pred_coarse_probs,3)
    coarse_top5_acc,_ = top_k_success_rate(true_coarse_labels,pred_coarse_probs,5)
    coarse_top10_acc,_ = top_k_success_rate(true_coarse_labels,pred_coarse_probs,10)
    coarse_coverage = coverage(true_coarse_labels,pred_coarse_probs)
    coarse_hamming_loss = hamming_loss(true_coarse_labels, pred_coarse_labels)

    micro_precision_fine = precision_score(true_fine_labels, pred_fine_labels, average='micro')
    micro_recall_fine = recall_score(true_fine_labels, pred_fine_labels, average='micro')
    micro_f1_fine = f1_score(true_fine_labels, pred_fine_labels, average='micro')
    micro_accuracy_fine = accuracy_score(true_fine_labels, pred_fine_labels)
    auc_macro_fine = roc_auc_score(true_fine_labels, pred_fine_probs, average="macro")
    auc_micro_fine = roc_auc_score(true_fine_labels, pred_fine_probs, average="micro")
    auc_weighted_fine = roc_auc_score(true_fine_labels, pred_fine_probs, average="weighted")
    rl_fine = label_ranking_loss(true_fine_labels, pred_fine_probs)
    fine_one_error = one_error(pred_fine_probs, true_fine_labels)
    fine_MCC = MCC(pred_fine_labels, true_fine_labels)
    fine_top3_acc,_ = top_k_success_rate(true_fine_labels, pred_fine_probs, 3)
    fine_top5_acc,_ = top_k_success_rate(true_fine_labels, pred_fine_probs, 5)
    fine_top10_acc,_ = top_k_success_rate(true_fine_labels, pred_fine_probs, 10)
    fine_top30_acc,_ = top_k_success_rate(true_fine_labels, pred_fine_probs, 30)
    fine_coverage = coverage(true_fine_labels, pred_fine_probs)
    fine_hamming_loss = hamming_loss(true_fine_labels, pred_fine_labels)

    micro_precision_fine2 = precision_score(true_fine2_labels, pred_fine2_labels, average='micro')
    micro_recall_fine2 = recall_score(true_fine2_labels, pred_fine2_labels, average='micro')
    micro_f1_fine2 = f1_score(true_fine2_labels, pred_fine2_labels, average='micro')
    micro_accuracy_fine2 = accuracy_score(true_fine2_labels, pred_fine2_labels)
    auc_macro_fine2 = roc_auc_score(true_fine2_labels, pred_fine2_probs, average="macro")
    auc_micro_fine2 = roc_auc_score(true_fine2_labels, pred_fine2_probs, average="micro")
    auc_weighted_fine2 = roc_auc_score(true_fine2_labels, pred_fine2_probs, average="weighted")
    rl_fine2 = label_ranking_loss(true_fine2_labels, pred_fine2_probs)
    fine2_one_error = one_error(pred_fine2_probs, true_fine2_labels)
    fine2_MCC = MCC(pred_fine2_labels, true_fine2_labels)
    fine2_top3_acc,_ = top_k_success_rate(true_fine2_labels, pred_fine2_probs, 3)
    fine2_top5_acc,_ = top_k_success_rate(true_fine2_labels, pred_fine2_probs, 5)
    fine2_top10_acc,_ = top_k_success_rate(true_fine2_labels, pred_fine2_probs, 10)
    fine2_top30_acc,_ = top_k_success_rate(true_fine2_labels, pred_fine2_probs, 30)
    fine2_coverage = coverage(true_fine2_labels, pred_fine2_probs)
    fine2_hamming_loss = hamming_loss(true_fine2_labels, pred_fine2_labels)

    print(f'micro_precision_coarse:{micro_precision_coarse:.4f}')
    print(f'micro_recall_coarse:{micro_recall_coarse:.4f}')
    print(f'micro_f1_coarse:{micro_f1_coarse:.4f}')
    print(f'micro_accuracy_coarse:{micro_accuracy_coarse:.4f}')
    print(f'auc_macro_coarse:{auc_macro_coarse:.4f}')
    print(f'auc_micro_coarse:{auc_micro_coarse:.4f}')
    print(f'auc_weighted_coarse:{auc_weighted_coarse:.4f}')
    print(f'rl_coarse:{rl_coarse:.4f}')
    print(f'coarse_one_error:{coarse_one_error:.4f}')
    print(f'coarse_MCC:{coarse_MCC:.4f}')
    print(f'coarse_top3_acc:{coarse_top3_acc:.4f}')
    print(f'coarse_top5_acc:{coarse_top5_acc:.4f}')
    print(f'coarse_top10_acc:{coarse_top10_acc:.4f}')
    print(f'coarse_coverage:{coarse_coverage:.4f}')
    print(f'coarse_hamming_loss:{coarse_hamming_loss:.4f}')



    print(f'micro_precision_fine:{micro_precision_fine:.4f}')
    print(f'micro_recall_fine:{micro_recall_fine:.4f}')
    print(f'micro_f1_fine:{micro_f1_fine:.4f}')
    print(f'micro_accuracy_fine:{micro_accuracy_fine:.4f}')
    print(f'auc_macro_fine:{auc_macro_fine:.4f}')
    print(f'auc_micro_fine:{auc_micro_fine:.4f}')
    print(f'auc_weighted_fine:{auc_weighted_fine:.4f}')
    print(f'rl_fine:{rl_fine:.4f}')
    print(f'fine_one_error:{fine_one_error:.4f}')
    print(f'fine_MCC:{fine_MCC:.4f}')
    print(f'fine_top3_acc:{fine_top3_acc:.4f}')
    print(f'fine_top5_acc:{fine_top5_acc:.4f}')
    print(f'fine_top10_acc:{fine_top10_acc:.4f}')
    print(f'fine_top30_acc:{fine_top30_acc:.4f}')
    print(f'fine_coverage:{fine_coverage:.4f}')
    print(f'fine_hamming_loss:{fine_hamming_loss:.4f}')


    print(f'micro_precision_fine2:{micro_precision_fine2:.4f}')
    print(f'micro_recall_fine2:{micro_recall_fine2:.4f}')
    print(f'micro_f1_fine2:{micro_f1_fine2:.4f}')
    print(f'micro_accuracy_fine2:{micro_accuracy_fine2:.4f}')
    print(f'auc_macro_fine2:{auc_macro_fine2:.4f}')
    print(f'auc_micro_fine2:{auc_micro_fine2:.4f}')
    print(f'auc_weighted_fine2:{auc_weighted_fine2:.4f}')
    print(f'rl_fine2:{rl_fine2:.4f}')
    print(f'fine2_one_error:{fine2_one_error:.4f}')
    print(f'fine2_MCC:{fine2_MCC:.4f}')
    print(f'fine2_top3_acc:{fine2_top3_acc:.4f}')
    print(f'fine2_top5_acc:{fine2_top5_acc:.4f}')
    print(f'fine2_top10_acc:{fine2_top10_acc:.4f}')
    print(f'fine2_top30_acc:{fine2_top30_acc}')
    print(f'fine2_coverage:{fine2_coverage:.4f}')
    print(f'fine2_hamming_loss:{fine2_hamming_loss:.4f}')





# ,micro_precision_fine2,micro_recall_fine2,micro_f1_fine2,micro_accuracy_fine2


test()







