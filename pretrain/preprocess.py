import numpy as np
import pandas as pd
import wfdb
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
sys.path.append("../finetune/")
sys.path.append("../utils")

# set your meta path of mimic-ecg (use relative path)
meta_path = '../datasets/pretrain/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
report_csv = pd.read_csv(f'{meta_path}/machine_measurements.csv', low_memory=False)
record_csv = pd.read_csv(f'{meta_path}/record_list.csv', low_memory=False)

def process_report(row):
    # Select the relevant columns and filter out NaNs
    report = row[['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 
                  'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 
                  'report_10', 'report_11', 'report_12', 'report_13', 'report_14', 
                  'report_15', 'report_16', 'report_17']].dropna()
    # Concatenate the report
    report = '. '.join(report)
    # Replace and preprocess text
    report = report.replace('EKG', 'ECG').replace('ekg', 'ecg')
    report = report.strip(' ***').strip('*** ').strip('***').strip('=-').strip('=')
    # Convert to lowercase
    report = report.lower()

    # concatenate the report if the report length is not 0
    total_report = ''
    if len(report.split()) != 0:
        total_report = report
        total_report = total_report.replace('\n', ' ')
        total_report = total_report.replace('\r', ' ')
        total_report = total_report.replace('\t', ' ')
        total_report += '.'
    if len(report.split()) == 0:
        total_report = 'empty'
    # Calculate the length of the report in words
    return len(report.split()), total_report

tqdm.pandas()
report_csv['report_length'], report_csv['total_report'] = zip(*report_csv.progress_apply(process_report, axis=1))
# Filter out reports with less than 4 words
report_csv = report_csv[report_csv['report_length'] >= 4]

# you should get 771693 here
print(report_csv.shape)

# 对齐 record_csv 与 report_csv：按 study_id 精确对齐并保证行顺序一致
report_csv.reset_index(drop=True, inplace=True)
report_csv = report_csv.set_index('study_id')
record_csv = record_csv[record_csv['study_id'].isin(report_csv.index)]
record_csv.reset_index(drop=True, inplace=True)
report_csv = report_csv.loc[record_csv['study_id']].reset_index()

# build an empty numpy array to store the data, we use int16 to save the space
temp_npy = np.zeros((len(record_csv), 12, 5000), dtype=np.int16)

for sample_idx, p in enumerate(tqdm(record_csv['path'])):
    # read the data
    ecg_path = os.path.join(meta_path, p)
    record = wfdb.rdsamp(ecg_path)[0]
    # record shape: (n_samples, n_leads)
    record = record.T  # (n_leads, n_samples)

    if np.isnan(record).sum() == 0 and np.isinf(record).sum() == 0:
        # 无 NaN / Inf，直接归一化
        r_min = record.min()
        r_max = record.max()
        if r_max > r_min:
            record = (record - r_min) / (r_max - r_min)
        else:
            # 整条记录为常数，直接置零
            record = np.zeros_like(record)
        record *= 1000
        record = record.astype(np.int16)
        temp_npy[sample_idx] = record[:, :5000]
    else:
        # 对每个导联，在时间轴上用邻域均值填充 NaN 和 Inf
        n_leads, n_samples = record.shape
        for lead in range(n_leads):
            row = record[lead, :]
            bad_idx = np.where(~np.isfinite(row))[0]
            if bad_idx.size == 0:
                continue

            finite_row = row[np.isfinite(row)]
            for bad_pos in bad_idx:
                left = max(0, bad_pos - 6)
                right = min(n_samples, bad_pos + 7)  # 以当前位置为中心，左右各最多 6 个点
                window = row[left:right]
                window = window[np.isfinite(window)]

                if window.size > 0:
                    row[bad_pos] = window.mean()
                elif finite_row.size > 0:
                    # 邻域都不可用时退化为该导联的整体有限均值
                    row[bad_pos] = finite_row.mean()
                else:
                    # 整条导联都无有效值时退化为 0
                    row[bad_pos] = 0.0

            record[lead, :] = row

        # 替换完成后再进行归一化和缩放
        r_min = record.min()
        r_max = record.max()
        if r_max > r_min:
            record = (record - r_min) / (r_max - r_min)
        else:
            record = np.zeros_like(record)
        record *= 1000
        record = record.astype(np.int16)
        temp_npy[sample_idx] = record[:, :5000]


# split to train and val
train_npy, val_npy, train_csv, val_csv = train_test_split(temp_npy, report_csv, test_size=0.02, random_state=42)

train_csv.reset_index(drop=True, inplace=True)
val_csv.reset_index(drop=True, inplace=True)

#打印最终数量
print(f'total size {len(train_npy) + len(val_npy)}')

# save to your path
np.save("your_path_train.npy", train_npy)
np.save("your_path_val.npy", val_npy)
train_csv.to_csv("your_path_train.csv", index=False)
val_csv.to_csv("your_path_val.csv", index=False)
