
import datetime
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，无需图形界面
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.tensorboard
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

"""定义整合数据集和类"""
class ECGDataset(Dataset):
    def __init__(self, image, text):
        self.image = image
        self.text = text

    def __getitem__(self, index):
        image = torch.tensor(self.image[index], dtype=torch.float32)
        text = torch.tensor(self.text[index], dtype=torch.long)
        return image, text

    def __len__(self):
        return len(self.image)

"""训练集、验证集和测试集的数据整合"""
def train_test(image_train, text_train, image_test, text_test, batch_size=128):
    train_dataset, test_dataset = ECGDataset(image_train, text_train), ECGDataset(image_test, text_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

# 定义训练功能和验证功能
def train_steps(loop, model, optimizer):
    train_loss = []
    model.train()
    for step_index, (image, text) in loop:
        image = image.to(device)
        image = image.reshape(-1, 12, 1000)
        text = text.to(device)
        text = text.reshape(-1, 128)
        logits_per_image, logits_per_text = model(image, text)
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, device=device).long()
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        # loss = F.cross_entropy(logits_per_image, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()  # item()转化为标量
        train_loss.append(loss)
        loop.set_postfix(loss=loss)  # 进度条右边显示信息
    return {"loss": np.mean(train_loss)}

def test_steps(loop, model):
    test_loss = []
    model.eval()
    with torch.no_grad():
        for step_index, (image, text) in loop:
            image = image.to(device)
            image = image.reshape(-1, 12, 1000)
            text = text.to(device)
            text = text.reshape(-1, 128)
            logits_per_image, logits_per_text = model(image, text)
            batch_size = image.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            # loss = F.cross_entropy(logits_per_image, labels)
            loss = loss.item()
            test_loss.append(loss)
            loop.set_postfix(loss=loss)
    return {"loss": np.mean(test_loss)}

# 定义训练函数
def train_epochs(train_dataloader, test_dataloader, model, optimizer, config, writer, model_path):
    num_epochs = config['num_epochs']
    lr = config['lr']
    train_loss_ls = []
    test_loss_ls = []
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
        train_metrix = train_steps(train_loop, model, optimizer)
        test_metrix = test_steps(test_loop, model)

        train_loss_ls.append(train_metrix['loss'])
        test_loss_ls.append(test_metrix['loss'])
        train_lr.append(lr)

        # 打印每一轮的loss和acc
        print(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; ')
        print(f'Epoch {epoch + 1}: '
              f'val loss: {test_metrix["loss"]}; ')
        print(f'Epoch {epoch + 1}: '
              f'lr: {lr}; ')

        # SummaryWriter
        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)

        # 根据验证集损失函数调整学习率和早停策略
        if test_metrix["loss"] < best_loss_ls:
            print(f'原本的最小损失函数为{best_loss_ls}，本次epoch的损失函数为{test_metrix["loss"]}，保存模型')
            best_loss_ls = test_metrix["loss"]
            epoch_patience = 0
            # 保存模型
            torch.save(model.state_dict(), model_path)
            # model.state_dict() 能够获取 模型中的所有参数，其返回值是一个有序字典 OrderedDict
        else:
            epoch_patience += 1

        if epoch_patience > 2 and epoch_patience < 8:  # 如果连续两个epoch不改善
            if lr > 0.00001:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                epoch_patience = 0
        elif epoch_patience == 8:  # 如果连续八个epoch不改善
            break

    return {'train_loss': train_loss_ls,
            'val_loss': test_loss_ls,
            'lr': train_lr}

# 绘制训练过程曲线
def plot_history_torch(history):
    history_len = len(history['train_loss'])
    x = np.linspace(1, history_len, history_len)
    x = list(x)

    plt.figure(figsize=(8, 8))
    plt.plot(x, history['train_loss'])
    plt.plot(x, history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'loss.png')
    # plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(x, history['lr'])
    plt.title('Model Lr')
    plt.ylabel('lr')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.savefig(f'lr.png')
    # plt.show()
