import numpy as np
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，无需图形界面
import matplotlib.pyplot as plt
import torch.utils.tensorboard
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns

"""定义整合数据集和类"""
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.x)

"""训练集、验证集和测试集的数据整合"""
def train_val_test(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    train_dataset, val_dataset, test_dataset = \
        ECGDataset(X_train, y_train), ECGDataset(X_val, y_val), ECGDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

# 定义训练功能和验证功能
def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()  # model.train()，作用是启用 batch normalization 和 dropout 。
    for step_index, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, 12, 1000)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()  # item()转化为标量
        train_loss.append(loss)
        pred_result = (pred >= 0.5).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)  # 进度条右边显示信息
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}

def test_steps(loop, model, criterion):
    test_loss = []
    test_acc = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, 12, 1000)
            pred = model(X)
            loss = criterion(pred, y).item()
            test_loss.append(loss)
            pred_result = (pred >= 0.5).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            test_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss),
            "acc": np.mean(test_acc)}

# 定义训练函数
def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer, model_path):
    num_epochs = config['num_epochs']
    lr_start = config['lr_start']
    lr = lr_start
    lr_max = config['lr_max']
    train_loss_ls = []
    train_loss_acc = []
    test_loss_ls = []
    test_loss_acc = []
    train_lr = []
    best_loss_ls = float('inf')
    epoch_patience = 0
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))  # 进度条
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')  # 进度条左边显示信息
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        """
        train_steps return
                 {"loss": np.mean(train_loss),
                "acc": np.mean(train_acc)}
        """
        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])
        train_lr.append(lr)

        # 打印每一轮的loss和acc
        print(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; '
              f'train acc: {train_metrix["acc"]}; ')
        print(f'Epoch {epoch + 1}: '
              f'val loss: {test_metrix["loss"]}; '
              f'val acc: {test_metrix["acc"]}')
        print(f'Epoch {epoch + 1}: '
              f'lr: {lr}; ')

        # SummaryWriter
        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar('validation/accuracy', test_metrix['acc'], epoch)

        # 采用warming up策略
        if epoch + 1 <= 10:
            lr = lr + (lr_max - lr_start) / 10
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            # 根据验证集损失函数调整学习率和早停策略
            if test_metrix["loss"] < best_loss_ls:
                print(f'原本的最小损失函数为{best_loss_ls}，本次epoch的损失函数为{test_metrix["loss"]}，保存模型')
                best_loss_ls = test_metrix["loss"]
                best_acc = test_metrix["acc"]
                epoch_patience = 0
                # 保存模型
                torch.save(model.state_dict(), model_path)
                # model.state_dict() 能够获取 模型中的所有参数，其返回值是一个有序字典 OrderedDict
            else:
                if (test_metrix["loss"] > best_loss_ls) and (test_metrix["loss"] < (1.1 * best_loss_ls)) and \
                        (test_metrix["acc"] > best_acc):
                    print("符合动态调整模型的条件，保存模型")
                    # 保存模型
                    torch.save(model.state_dict(), model_path)
                    # model.state_dict() 能够获取 模型中的所有参数，其返回值是一个有序字典 OrderedDict
                    best_acc = test_metrix["acc"]
                epoch_patience += 1

            if epoch_patience > 2 and epoch_patience < 8:  # 如果连续两个epoch不改善
                if lr > 0.00001:
                    lr = lr * 0.1
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    epoch_patience = 0
            elif epoch_patience == 8:  # 如果连续八个epoch不改善
                break
        """if epoch != 0 and (epoch + 1) % 5 == 0:
            lr = lr * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)"""

    return {'train_loss': train_loss_ls,
            'train_acc': train_loss_acc,
            'val_loss': test_loss_ls,
            'val_acc': test_loss_acc,
            'lr': train_lr}

# 绘制训练过程曲线
def plot_history_torch(history, k):
    history_len = len(history['train_acc'])
    x = np.linspace(1, history_len, history_len)
    x = list(x)

    plt.figure(figsize=(8, 8))
    plt.plot(x, history['train_acc'])
    plt.plot(x, history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'accuracy{k}.png')
    # plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(x, history['train_loss'])
    plt.plot(x, history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'loss{k}.png')
    # plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(x, history['lr'])
    plt.title('Model Lr')
    plt.ylabel('lr')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.savefig(f'lr{k}.png')
    # plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, label_name, k):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {label_name}')
    # plt.show()
    plt.savefig(f'confusion_earlystopping_{label_name}_{k}.png')

