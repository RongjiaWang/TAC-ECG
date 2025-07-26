
import datetime
import os
import torch.utils.tensorboard

import MIMIC_IV_read
import xresnet1d101_CLIP
import util
import bert

# 定义路径
# 定义根路径
project_path = "./"
# 定义日志目录
# 必须是启动web应用程序时指定的目录的子目录
# 建议使用日期时间作为子目录名称
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# CPU or GPU的选择
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print("Using {} device".format(device))

if __name__ == '__main__':
    # 获取心电信号原始数据和标签原始数据
    data, data_text_old = MIMIC_IV_read.get_data()
    data_text = bert.bert_text(data_text_old)
    data_counter = len(data)  # 心电信号的个数
    # 定义模型基本信息
    config = {
        'num_epochs': 100,  # epoch次数
        'batch_size': 128,  # batch_size大小
        'lr': 0.001,  # 初始学习率
    }

    # 构建训练集、验证集与测试集
    image_train = []
    text_train = []
    image_test = []
    text_test = []
    counter = 0
    for i in range(data_counter):
        if counter < int((data_counter*0.9)):
            image_train.append(data[i])
            text_train.append(data_text[i])
        else:
            image_test.append(data[i])
            text_test.append(data_text[i])
        counter += 1

    # 划分batch_size=128
    train_dataloader, test_dataloader = \
        util.train_test(image_train, text_train, image_test, text_test, config['batch_size'])

    # 定义模型（残差收缩网络）
    model = xresnet1d101_CLIP.CLIP()
    model_path = project_path + f"xresnet1d101_clip.pt"
    if os.path.exists(model_path):
        # 如果存在载入预训练模型
        print('Import the pre-trained model, skip the training process')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    else:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
        # train and evaluate model
        history = util.train_epochs(train_dataloader, test_dataloader, model, optimizer, config,
                                                writer, model_path)
        writer.close()
        model_train_test.plot_history_torch(history)

    # 模型测试
    model.load_state_dict(torch.load(model_path))
    print("读取模型，开始测试")
    model.to(device)
    model.eval()
    # 写入模型测试结果
    filename2 = f'对比训练模型测试结果.txt'
    with open(filename2, 'w') as f:
        with torch.no_grad():
            for step_index, (image, text) in enumerate(test_dataloader):
                image = image.to(device)
                image = image.reshape(-1, 12, 1000)
                text = text.to(device)
                text = text.reshape(-1, 128)
                logits_per_image, logits_per_text = model(image, text)
                # 找出每一行中最大值的索引
                max_indices = torch.argmax(logits_per_image, dim=1)
                logits_per_image = logits_per_image.to('cpu')
                logits_per_text = logits_per_text.to('cpu')
                max_indices = max_indices.to('cpu')
                f.write('第{}组'.format(step_index))
                f.write('\n')
                f.write(str(logits_per_image))
                f.write('\n')
                f.write(str(max_indices))
        f.close()
