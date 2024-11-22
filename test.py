import argparse
import pickle as pkl

import dgl
from torchsummary import summary
from dataprocess import Data_process_mid
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from TAHIN import TAHIN
import json
from utils import (
    evaluate_acc,
    evaluate_auc,
    evaluate_f1_score,
    evaluate_logloss,
)

def softmax(x):
    # 将输入张量作为输入，应用softmax函数
    softmax_x = F.softmax(x, dim=-1)
    return softmax_x


def main(args):
    # step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # step 2: Load data
    (
        g,
        train_loader,
        eval_loader,
        test_loader,
        meta_paths, #元路径 对于Amazon数据集    meta_paths = {
                                        #     "user": [["ui", "iu"]],
                                        #     "item": [["iu", "ui"], ["ic", "ci"], ["ib", "bi"], ["iv", "vi"]],
                                        # }
        user_key,
        item_key,
    ) = load_data(args.dataset, args.batch, args.num_workers, args.path) #Amazon数据集 batch
    g = g.to(device)         #把图放到cuda中
    print("Data loaded.")

    # step 3: Create model and training components
    model = TAHIN(          #模型就是TAHIN
        g, meta_paths, args.in_size, args.out_size, args.num_heads, args.dropout
    )
    # summary(model,input_size=[(128,128)])
    model = model.to(device)  #把模型放到cuda中进行训练
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print("Model created.")

    # step 4: Training
    # print("Start training.")
    # best_acc = 0.0
    # kill_cnt = 0
    # for epoch in range(args.epochs):
    #     # Training and validation using a full graph
    #     model.train()
    #     train_loss = []
    #     for step, batch in enumerate(train_loader):
    #         user, item, label = [_.to(device) for _ in batch]
    #         logits = model.forward(g, user_key, item_key, user, item)
    #
    #         # compute loss
    #         tr_loss = criterion(logits, label)
    #         train_loss.append(tr_loss)
    #
    #         # backward
    #         optimizer.zero_grad()
    #         tr_loss.backward()
    #         optimizer.step()
    #
    #     train_loss = torch.stack(train_loss).sum().cpu().item()
    #
    #     model.eval()
    #     with torch.no_grad():
    #         validate_loss = []
    #         validate_acc = []
    #         for step, batch in enumerate(eval_loader):
    #             user, item, label = [_.to(device) for _ in batch]
    #             logits = model.forward(g, user_key, item_key, user, item)
    #
    #             # compute loss
    #             val_loss = criterion(logits, label)
    #             val_acc = evaluate_acc(
    #                 logits.detach().cpu().numpy(), label.detach().cpu().numpy()
    #             )
    #             validate_loss.append(val_loss)
    #             validate_acc.append(val_acc)
    #
    #         validate_loss = torch.stack(validate_loss).sum().cpu().item()
    #         validate_acc = np.mean(validate_acc)
    #
    #         # validate
    #         if validate_acc > best_acc:
    #             best_acc = validate_acc
    #             best_epoch = epoch
    #             torch.save(model.state_dict(), "TAHIN" + "_" + args.dataset)
    #             kill_cnt = 0
    #             print("saving model...")
    #         else:
    #             kill_cnt += 1
    #             if kill_cnt > args.early_stop:
    #                 print("early stop.")
    #                 print("best epoch:{}".format(best_epoch))
    #                 break
    #
    #         print(
    #             "In epoch {}, Train Loss: {:.4f}, Valid Loss: {:.5}\n, Valid ACC: {:.5}".format(
    #                 epoch, train_loss, validate_loss, validate_acc
    #             )
    #         )

    # test use the best model
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load("TAHIN" + "_" + args.dataset))
        test_loss = []
        test_acc = []
        test_auc = []
        test_f1 = []
        test_logloss = []
        # re0, re1, re2, re3, re4 = Data_process_mid('./data/').read_mid(field0='uid_degree',field1='uid_industry',field2='uid_major',field3='uid_position',field4='industry_position')
        # uid_dict,industry_dict,major_dict,position_dict = Data_process_mid("./data/").demapper(re0, re1, re2, re3, re4)
        # 打开 JSON 文件并读取数据
        with open('uid_dict_data.json', 'r') as file:
            json_data = file.read()
        # 将 JSON 数据转换为字典
        uid_dict = json.loads(json_data)
        # 打开 JSON 文件并读取数据
        with open('position_dict_data.json', 'r') as file:
            json_data = file.read()
        # 将 JSON 数据转换为字典
        position_dict = json.loads(json_data)
        user_name = '林国瑞'     #输入用户名字（值）
        user_dict_key = uid_dict[user_name]     #找到对应的键
        user = torch.tensor([user_dict_key])      #将其转换成tensor 代入模型
        item_name = '市场专员' #市场专员:478    市场总监:37
        item_dict_key = position_dict[item_name]
        print(item_dict_key)
        # item = torch.tensor([item_dict_key])
        item = torch.tensor(range(1,23))       #输入职位的键
        # item = np.arange(1, 555)
        logits1 = model.forward(g,user_key,item_key,user,item)
        print('logits1:')
        print(logits1,user,item)
        user_id =user[0].item()
        ukeys = [key for key, val in uid_dict.items() if val == user_id]
        print(ukeys)

        item_id = item.tolist()
        j = 0

        for i in item_id:
            pkeys = [key for key, val in position_dict.items() if val == i]
            if logits1[j] >=0:
                print("用户{}".format(ukeys)+"对于职位{}".format(pkeys)+'的适配为{}'.format(logits1[j]))
            # print("用户{}".format(ukeys) + "对于职位{}".format(pkeys) + '的适配为{}'.format(logits1[j]))
            j = j+1
        print()

        softmax_y = softmax(logits1)
        print(softmax_y)

        # for step, batch in enumerate(test_loader):
        #     user, item, label = [_.to(device) for _ in batch]
        #     logits = model.forward(g, user_key, item_key, user, item)           #相当于TAHIN()
        #     print(logits,user,item)  #计算出user和item之间的匹配程度
        #
        #     # compute loss
        #     loss = criterion(logits, label)
        #     acc = evaluate_acc(
        #         logits.detach().cpu().numpy(), label.detach().cpu().numpy()
        #     )
        #     # auc = evaluate_auc(
        #     #     logits.detach().cpu().numpy(), label.detach().cpu().numpy()
        #     # )
        #     # f1 = evaluate_f1_score(
        #     #     logits.detach().cpu().numpy(), label.detach().cpu().numpy()
        #     # )
        #     # log_loss = evaluate_logloss(
        #     #     logits.detach().cpu().numpy(), label.detach().cpu().numpy()
        #     # )
        #
        #     test_loss.append(loss)
        #     test_acc.append(acc)
        #     # test_auc.append(auc)
        #     # test_f1.append(f1)
        #     # test_logloss.append(log_loss)
        #
        # test_loss = torch.stack(test_loss).sum().cpu().item()
        # test_acc = np.mean(test_acc)
        # # test_auc = np.mean(test_auc)
        # # test_f1 = np.mean(test_f1)
        # # test_logloss = np.mean(test_logloss)
        # print(
        #     "Test Loss: {:.5}\n, Test ACC: {:.5}\n".format(     # AUC: {:.5}\n, F1: {:.5}\n,, Logloss: {:.5}\n
        #         test_loss, test_acc  #test_auc, test_f1,, test_logloss
        #     )
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,    #使用 ArgumentDefaultsHelpFormatter 类作为 formatter_class 参数值，可以使 ArgumentParser 对象在生成帮助文档时包括每个参数的默认值。这有助于用户了解哪些参数是必需的，以及如何正确使用它们
    )

    parser.add_argument(
        "--dataset",
        default="jobmatch",
        help="Dataset to use, default: movielens",
    )
    parser.add_argument(
        "--path", default="./data", help="Path to save the data"
    )
    parser.add_argument("--model", default="TAHIN", help="Model Name")

    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--wd", type=float, default=0, help="L2 Regularization for Optimizer"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of processes to construct batches",
    )
    parser.add_argument(
        "--early_stop", default=15, type=int, help="Patience for early stop."
    )

    parser.add_argument(
        "--in_size",
        default=64,
        type=int,
        help="Initial dimension size for entities.",          #实体的初始维度大小
    )
    parser.add_argument(
        "--out_size",
        default=64,
        type=int,
        help="Output dimension size for entities.",             #实体的输出维度
    )

    parser.add_argument(
        "--num_heads", default=4, type=int, help="Number of attention heads"       #注意力的头数
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")

    args = parser.parse_args()

    print(args)

    main(args)

