import numpy as np
import pandas as pd
import datetime
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# 读取原始信号数据和标签数据
def get_data():
    # 设置读取文件路径
    path = '/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate = 100
    # 读取文件并转换标签
    Y = pd.read_csv(path + 'ptbxl_database.csv')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # 获取原始信号数据
    X = load_raw_data(Y, sampling_rate, path)

    # 获取scp_statements.csv中的诊断信息
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # 添加诊断信息
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    patients = Y.patient_id
    patients = np.array(patients)
    """ecg_ids = Y.ecg_id
    ecg_ids = np.array(ecg_ids)"""
    k_folds = Y.strat_fold
    k_folds = np.array(k_folds)
    Y = Y.diagnostic_superclass
    Y = np.array(Y)
    return X, Y, patients, k_folds

"""12导联心电信号的切片"""
def Data_preprocesse(signals):
    Single_beats = []  # 存储单心拍数据集
    signals = signals.T
    signals = list(signals)  # 第一维索引表示导联数
    for i in range(12):
        signal = signals[i]
        # Z-Score归一化（缓解基线漂移）
        if np.std(signal) != 0:
            signal = (signal - np.mean(signal)) / np.std(signal)

        # 将重采样后的信号统一规划到1000，并除去部分存在单个或若干个导联确实的记录
        if len(signal) >= 1000:
            for j in signal[0:1000]:
                Single_beats.append(j)
        elif len(signal) < 1000:
            filled_list = signal + [0] * (1000 - len(signal))
            for j in filled_list:
                Single_beats.append(j)

    return Single_beats

"""所有患者的12导联诊断用数据的切片及切片后统计"""
def data_slice(X, Y, patients, k_folds):
    Single_beat_sum = []  # 记录所有不同类型的心拍总数据
    # 记录不同类型疾病各自的心拍总数
    Single_beat_count = [0, 0, 0, 0, 0]
    # 统计患者人数和序号信息
    patient_count = [0, 0, 0, 0, 0]
    # 记录真实的记录数和患者数
    total_real = 0
    # 统计各类患者序号
    dic = {}
    dic[0] = []
    dic[1] = []
    dic[2] = []
    dic[3] = []
    dic[4] = []
    for i in range(len(X)):
        patient = patients[i]
        patient = int(patient)
        k_fold = k_folds[i]
        k_fold = int(k_fold)
        print(i)
        signals = X[i, :, :]
        # 摘出标签数据（可能有多个标签）
        labels = Y[i]
        Single_beats = Data_preprocesse(signals)
        if len(Single_beats) == 12000:
            label_count = 0
            Single_beats.append(k_fold)
            label_0_1 = [0, 0, 0, 0, 0]
            total_real += 1
            if len(labels) != 0:
                for label_i in range(len(labels)):
                    label = labels[label_i]
                    label_count += 1
                    if label == 'NORM':  # 健康组
                        Single_beat_count[0] = Single_beat_count[0] + 1
                        patient_count[0] = patient_count[0] + 1
                        dic[0].append(patient)
                        label_0_1[0] = 1
                    elif label == 'MI':  # 心肌梗死
                        Single_beat_count[1] = Single_beat_count[1] + 1
                        patient_count[1] = patient_count[1] + 1
                        dic[1].append(patient)
                        label_0_1[1] = 1
                    elif label == 'STTC':  # ST-T改变
                        Single_beat_count[2] = Single_beat_count[2] + 1
                        patient_count[2] = patient_count[2] + 1
                        dic[2].append(patient)
                        label_0_1[2] = 1
                    elif label == 'CD':  # 传导紊乱
                        Single_beat_count[3] = Single_beat_count[3] + 1
                        patient_count[3] = patient_count[3] + 1
                        dic[3].append(patient)
                        label_0_1[3] = 1
                    elif label == 'HYP':  # 肥大
                        Single_beat_count[4] = Single_beat_count[4] + 1
                        patient_count[4] = patient_count[4] + 1
                        dic[4].append(patient)
                        label_0_1[4] = 1
            Single_beats.append(label_0_1)
            Single_beat_sum.append(Single_beats)

    # 心拍总数
    total = 0
    for i in range(0, len(Single_beat_count)):
        total = total + Single_beat_count[i]

    # 患者记录总数
    total_patient = 0
    for i in range(0, len(patient_count)):
        total_patient = total_patient + patient_count[i]

    # 按键（即患者类别排序）
    sorted_dic = dict(sorted(dic.items(), key=lambda item: item[0]))

    # 写入患者数据
    filename1 = '12导联ECG数据统计.txt'
    with open(filename1, 'w') as f:
        # 设置格式tplt，15代表间隔距离，可根据自己需要调整
        tplt = "{:<15}\t{:<15}\t{:<15}\t{:<15}"
        # 按tplt格式写入抬头行
        f.write(tplt.format('疾病类别', '疾病记录数量', '心拍切片数量', '心拍占比', chr(255)))
        # 换行
        f.write('\n')
        f.write(tplt.format('健康（HC）', patient_count[0], Single_beat_count[0],
                            f"{Single_beat_count[0] * 100 / total_real}%", chr(255)))
        f.write('\n')
        f.write(tplt.format('心肌梗死（MI）', patient_count[1], Single_beat_count[1],
                            f"{Single_beat_count[1] * 100 / total_real}%", chr(255)))
        f.write('\n')
        # 写入第二行数据
        f.write(tplt.format('ST-T改变（STTC）', patient_count[2], Single_beat_count[2],
                            f"{Single_beat_count[2] * 100 / total_real}%", chr(255)))
        f.write('\n')
        f.write(tplt.format('传导紊乱（CD）', patient_count[3], Single_beat_count[3],
                            f"{Single_beat_count[3] * 100 / total_real}%", chr(255)))
        f.write('\n')
        f.write(tplt.format('肥大（HYP）', patient_count[4], Single_beat_count[4],
                            f"{Single_beat_count[4] * 100 / total_real}%", chr(255)))
        f.write('\n')
        f.write(tplt.format('总计（total）', total_real, total_real, f"100%", chr(255)))
        f.write('\n')
        f.close()

    with open(filename1, 'a') as f1:
        for item in sorted_dic.items():
            f1.write(str(item))
            f1.write('\n')
        f1.close()
    return Single_beat_sum

if __name__ == '__main__':
    X, Y, patients, k_folds = get_data()
    data = data_slice(X, Y, patients, k_folds)
