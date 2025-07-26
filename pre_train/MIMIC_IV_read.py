
import datetime
import wfdb
import numpy as np
from scipy.signal import resample
import csv

"""12导联心电信号的降采样"""
def Signal_downsampling(signals):
    # 原ECG信号为（5000，12），数据类型为ndarray
    signals_down = []  # 存储降采样后的心电信号
    signals = signals.T  # 转置
    signals = list(signals)  # 第一维索引表示导联数
    for i in range(12):
        signal = signals[i]
        # Z-Score归一化
        if np.std(signal) != 0:
            signal = (signal - np.mean(signal)) / np.std(signal)

        # 将重采样后的信号统一规划到1000
        original_sampling_rate = 500  # 原始采样频率为500Hz
        target_sampling_rate = 100  # 目标采样频率为100Hz
        # 计算目标信号的长度
        target_length = int(len(signal)*target_sampling_rate/original_sampling_rate)
        # 使用scipy的resample函数进行重采样
        resampled_signal = resample(signal, target_length)
        resampled_signal = list(resampled_signal)

        if len(resampled_signal) >= 1000:
            for j in resampled_signal[0:1000]:
                signals_down.append(j)
        elif len(resampled_signal) < 1000:
            # 用0补全缺失的部分
            filled_list = resampled_signal + [0]*(1000-len(resampled_signal))
            for j in filled_list:
                signals_down.append(j)

    return signals_down

"""读取所有的12导联ECG数据"""
def get_data():
    # 读取信号记录统计文件
    records = []
    with open("/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            records.append(row)
        file.close()
    del records[0]  # 删除第一行（标题行）

    # 读取文本描述记录
    records_text = []
    with open('/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            records_text.append(row)
        file.close()
    del records_text[0]  # 删除第一行（标题行）

    # 逐个读取数据并整理
    orig_path = '/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
    ECG_Signals = []  # 所有的ECG-12导联心电信号
    text_Signals = []  # 所有的ECG-12导联心电信号的文本描述
    subject_ids = []  # 受试者数量
    study_ids = []  # ECG记录数量
    # 开始读取信号
    for i in range(len(records)):
        record = records[i]
        record_text = records_text[i]
        # 提取受试者id
        subject = record[0]
        # 每个受试者仅读取一条记录
        if subject not in subject_ids:
            # 提取ECG记录id
            study = record[1]
            # 读取心电信号
            specific_path = record[4]
            data = wfdb.rdrecord(orig_path+specific_path)
            data_ecg = data.p_signal
            data_ecg_down = Signal_downsampling(data_ecg)
            if not (np.isnan(np.array(data_ecg_down))).any():
                ECG_Signals.append(list(data_ecg_down))
                subject_ids.append(subject)
                study_ids.append(study)
                text_Signals.append(record_text[4:20])
            if len(ECG_Signals) >= 200000:
                break
            else:
                print(len(ECG_Signals))

    # 写入结果数据
    filename = 'MIMIC-IV-ECG数据库统计数据.txt'
    with open(filename, 'w') as f:
        # 设置格式tplt， 15表示间距
        tplt = "{:<20}\t{:20}"
        f.write(tplt.format('受试者人数', '记录数'))
        f.write('\n')
        f.write(tplt.format(len(subject_ids), len(study_ids)))
        f.close()

    return ECG_Signals, text_Signals

if __name__ == '__main__':
    data = get_data()
