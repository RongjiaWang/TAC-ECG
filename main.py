
#  MIMIC-IV-ECG数据库预训练的模型，通过adapter向PTB-XL数据库迁移
import numpy as np
import csv
import datetime
from torch.utils.data import Dataset
from torchinfo import summary
import torch.utils.tensorboard
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, \
    roc_auc_score, confusion_matrix
import os
import random

import dataset.PTB_XL_read
import utils.utils
import model.xresnet1d101_adapter

# 定义路径
# 定义根路径
project_path = "./"
# 定义日志目录
# 必须是启动web应用程序时指定的目录的子目录
# 建议使用日期时间作为子目录名称
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# 设置随机数种子
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
np.random.seed(seed)
random.seed(seed)

# 禁用非确定性操作
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 获取心电信号原始数据和标签原始数据
    X, Y, patients, k_folds = PTB_XL_read.get_data()
    data = PTB_XL_read.data_slice(X, Y, patients, k_folds)

    # 定义模型基本信息
    config = {
        'num_epochs': 100,  # epoch次数
        'batch_size': 128,  # batch_size大小
        'lr_start': 0.001,  # 初始学习率
        'lr_max': 0.01  # 最大学习率
    }

    accuracys = []
    precisions = []
    sensitivitys = []
    specificitys = []
    f1s = []
    AUCs = []
    # 划分训练集和测试集（5倍交叉验证）
    for k in range(1, 11):
        # k表示当前测试集的1折
        if k == 1:
            k_val = 10
        else:
            k_val = k-1
        # 构建训练集、验证集与测试集
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        for row in data:
            if int(row[12000]) == k:
                X_test.append(row[0:12000])
                y_test.append(row[12001])
            elif int(row[12000]) == k_val:
                X_val.append(row[0:12000])
                y_val.append(row[12001])
            else:
                X_train.append(row[0:12000])
                y_train.append(row[12001])

        # 划分batch_size=128
        train_dataloader, val_dataloader, test_dataloader = \
            utils.train_val_test(X_train, y_train, X_val, y_val, X_test, y_test, config['batch_size'])

        # 定义模型（残差收缩网络）
        model = xresnet1d101_adapter.xresnet1d101_adapter()
        model_path = project_path + f"xresnet1d101_12lead{k}.pt"
        model_path_pre = f"xresnet1d101_clip.pt"
        if os.path.exists(model_path):
            # 如果存在载入预训练模型
            print('Import the pre-trained model, skip the training process')
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
        else:
            # 加载预训练模型
            pretrained_dict = torch.load(model_path_pre)
            # 要删除的字符或子字符串
            remove_str = 'visual.'
            # 创建一个新的字典来存储修改后的键值对
            new_pretrained_dict = {}
            for key, value in pretrained_dict.items():
                # 删除键中的指定字符或子字符串
                new_key = key.replace(remove_str, '')
                new_pretrained_dict[new_key] = value
            # 加载自己的网络
            model_dict = model.state_dict()
            # 确保提取的参数与你的模型结构中对应层的形状一致(逐层更新模型的参数)
            for layer_name in model_dict:
                if (layer_name != 'fc.weight') and (layer_name != 'fc.bias') and (layer_name in new_pretrained_dict):
                    # 受分类类别数影响的若干全连接层不通过预训练模型进行参数初始化
                    model_dict[layer_name] = new_pretrained_dict[layer_name]
                    print(f" {layer_name} 层的参数量已完成初始化")
                else:
                    print(f" {layer_name} 层的参数量已发生改变 .")
            # 更新你的模型参数
            model.load_state_dict(model_dict)

            # 冻结网络参数
            for name, param in model.named_parameters():
                if (name != 'fc.weight') and (name != 'fc.bias') and (name in new_pretrained_dict):
                    param.requires_grad = False
            # 验证网络参数是否冻结
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

            # 写入模型信息
            filename1 = '12导联模型信息1.txt'
            with open(filename1, 'w') as f:
                f.write(str(summary(model, input_size=(1, 12, 1000))))
                f.write('\n')
                f.close()

            model = model.to(device)
            criterion = torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_start'])

            writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
            history = utils.train_epochs(train_dataloader, val_dataloader, model, criterion, optimizer, config, writer,
                                   model_path)
            writer.close()

            model_train_test.plot_history_torch(history, k)
        # 查看模型的信息
        print(summary(model, input_size=(1, 12, 1000)))

        # 在测试集预测
        y_pred = []
        preds = []
        model.load_state_dict(torch.load(model_path))
        print("读取模型，开始测试")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for step_index, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                X = X.reshape(-1, 12, 1000)
                pred = model(X)
                pred_result = (pred >= 0.5).detach().cpu().numpy()
                y_pred.extend(pred_result)
                preds.extend(pred.cpu().numpy())
                # 混淆矩阵热力图
        y_true = np.array(y_test)
        y_pred = np.array(y_pred)
        # 绘制混淆矩阵
        labels = ["NORM", "MI", "STTC", "CD", "HYP"]
        # 为每个标签绘制混淆矩阵
        for i, label in enumerate(labels):
            utils.plot_confusion_matrix(y_true[:, i], y_pred[:, i], label, k)
        # plt.show()
        # 计算accuracy
        print('accuracy_score', accuracy_score(y_true, y_pred))
        accuracys.append(accuracy_score(y_true, y_pred))
        # 计算多分类的precision、recall=sensitivity、specificity=1-假正例率、f1-score、AUC值
        print('precision', precision_score(y_true, y_pred, average='macro'))
        precisions.append(precision_score(y_true, y_pred, average='macro'))
        print('sensitivity', recall_score(y_true, y_pred, average='macro'))
        sensitivitys.append(recall_score(y_true, y_pred, average='macro'))
        # 计算特异性
        def calculate_specificity(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp)
            return specificity
        specificities = []
        for i in range(y_true.shape[1]):
            spec = calculate_specificity(y_true[:, i], y_pred[:, i])
            specificities.append(spec)
        average_spe = np.mean(specificities)
        print('specificity', average_spe)
        specificitys.append(average_spe)
        print('f1-score', f1_score(y_true, y_pred, average='macro'))
        f1s.append(f1_score(y_true, y_pred, average='macro'))
        # 计算Macro AUC
        preds = np.array(preds)
        # 减去最大值操作可以防止指数爆炸
        preds = np.exp(preds - np.max(preds, axis=-1, keepdims=True)) / np.sum(
            np.exp(preds - np.max(preds, axis=-1, keepdims=True)), axis=-1, keepdims=True)  # softmax函数
        macro_auc = roc_auc_score(y_true, preds, average='macro')
        print('AUC', macro_auc)
        AUCs.append(macro_auc)
        # 下面这个可以显示出每个类别的precision、recall、f1-score。
        class_report = classification_report(y_true, y_pred)
        print('classification_report\n', class_report)

        # 写入模型测试结果
        filename2 = f'12导联模型测试结果{k}.txt'
        with open(filename2, 'w') as f:
            # 设置格式tplt，15代表间隔距离，可根据自己需要调整
            tplt = "{:<15}\t{:<30}"
            # 按tplt格式写入抬头行
            f.write(tplt.format('性能指标', '结果', chr(255)))
            # 换行
            f.write('\n')
            # 写入第一行数据
            f.write(tplt.format('accuracy', accuracy_score(y_true, y_pred), chr(255)))
            f.write('\n')
            f.write(tplt.format('precision', precision_score(y_true, y_pred, average='macro'), chr(255)))
            f.write('\n')
            f.write(tplt.format('sensitivity', recall_score(y_true, y_pred, average='macro'), chr(255)))
            f.write('\n')
            f.write(tplt.format('specificity', average_spe, chr(255)))
            f.write('\n')
            f.write(tplt.format('f1-score', f1_score(y_true, y_pred, average='macro'), chr(255)))
            f.write('\n')
            f.write(tplt.format('AUC', macro_auc))
            f.write('\n')
            f.write('classification_report\n')
            f.write(class_report)
            f.close()
        k += 1

    # 写入模型测试结果
    filename3 = f'测试结果统计.csv'
    with open(filename3, mode='w', encoding='utf-8-sig') as f:
        # Create a CSV writer
        writer = csv.writer(f)
        writer.writerow(['PTB-XL数据库'])
        writer.writerow([' ', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1-score', 'AUC'])
        writer.writerow([' ', "{:.2f}".format((np.mean(accuracys)*100)), "{:.2f}".format((np.mean(precisions)*100)),
                         "{:.2f}".format((np.mean(sensitivitys)*100)), "{:.2f}".format((np.mean(specificitys)*100)),
                         "{:.2f}".format((np.mean(f1s)*100)), "{:.2f}".format((np.mean(AUCs)*100))])
