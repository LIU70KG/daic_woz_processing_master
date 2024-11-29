# 作者：刘成广
# 时间：2024/8/9 下午5:10
import os
import os
import pickle
import librosa
import soundfile as sf
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import os
import pickle
import numpy as np
import utils.utilities as util
import h5py
import pandas as pd
import random
import math
import sys
from zipfile import ZipFile
import re
import torch
import copy

# -------------视觉特征----------------
# 列出视觉文件夹中以特定后缀结尾的文件名
DATASET = '/home/liu70kg/PycharmProjects/Depression/DAIC-woz/DAIC-woz-dataset'
suffixes = ['AUs.txt', 'gaze.txt', 'pose.txt']
CV_lack_pose = {'367_CLNF_pose.txt', '396_CLNF_pose.txt', '432_CLNF_pose.txt'}
CV_frame_lack ={402, 420}  # –在对话结束之前，视频记录被剪切约2分钟。 所以缺少后续视频帧

def get_meta_data(dataset_path):
    """
    从数据集中抓取元数据路径，包括文件夹列表、音频路径列表和数据集中所有文件的转录文件列表

    Input
        dataset_path: str - Location of the dataset

    Outputs
        folder_list: list - The complete list of folders in the dataset
        audio_paths: list - The complete list of audio locations for the data
        transcript_paths: list - The complete list of locations of the
                          transcriptions
    """
    folder_list = []
    audio_files = []
    audio_paths = []
    transcript_paths = []
    list_dir_dataset_path = os.listdir(dataset_path)
    counter = 0
    for file in list_dir_dataset_path:
        if file.endswith('.zip'):
            if counter == 0:
                print('Converting zip files...')
                counter += 1
            current_file = os.path.join(dataset_path, file)
            new_file = file.split('.')[0]
            try:
                with ZipFile(current_file, 'r') as zipObj:
                    # Extract all the contents of zip file in different directory
                    zipObj.extractall(os.path.join(dataset_path, new_file))
                    os.remove(current_file)
            except:
                print(f"The file {current_file} may not have downloaded "
                      f"correctly, please try re-downloading and running the "
                      f"pre-processing tool again")
                sys.exit()

    list_dir_dataset_path = os.listdir(dataset_path)
    list_dir_dataset_path.sort()
    for i in list_dir_dataset_path:
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'wav' in j:
                    audio_files.append(j)
                    audio_paths.append(os.path.join(dataset_path, i, j))
                if 'TRANSCRIPT' in j:
                    if 'lock' in j or '._' in j:
                        pass
                    else:
                        transcript_paths.append(os.path.join(dataset_path, i, j))

    return folder_list, audio_paths, transcript_paths


def autocorrelation(x, lag=1):
    """计算序列 x 的自相关性，滞后为 lag"""
    n = len(x)
    x_mean = np.mean(x)
    autocorr = np.correlate(x - x_mean, x - x_mean, mode='full')[n-1:]
    # 检查 autocorr[0] 是否为零
    if autocorr[0] == 0:
        return 0.0  # 返回 NaN 或其他适当的值，例如 0
    else:
        return autocorr[lag] / autocorr[0]  # 归一化



folder_list, _, transcript_paths = get_meta_data(DATASET)

# -------------语音特征----------------
audio_paragraph_features_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/audio_paragraph_features.pickle'
audio_paragraph_features = np.load(audio_paragraph_features_path, allow_pickle=True)
print("获取到测试者的说话段落的语音特征了，并且整合到变量：audio_paragraph_features！")

# -------------开始和截至时间----------------
folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/on_off_times.pickle'
on_off_times = np.load(folder_path, allow_pickle=True)
print("获取到测试者的说话段落的开始和截至时间了，并且整合到变量：on_off_times！")

# 一个一个样本的处理
visual_paragraph_features = []
for i, per_text in enumerate(folder_list):
    # if i!=0:  # 哪里出了问题？大杀器
    #     continue
    print(f"--------------------处理文件 ：{folder_list[i]}！-------------------------")
    # -------------------------------------获取该样本对应的视觉特征！（CV_fea, 所有帧)-------------------------------------------------
    CV_PATH = os.path.join(DATASET, folder_list[i])  # 文件路径
    assert os.path.exists(CV_PATH)
    cv_file_names = [file for file in os.listdir(CV_PATH) if any(file.endswith(suffix) for suffix in suffixes)]
    sorted_cv_file_names = sorted(cv_file_names)  # 视觉文件夹中以特定后缀结尾的文件
    # ---------1读取单个样本的视觉特征-------
    CV_fea_all = []
    for cv_file_name in sorted_cv_file_names:
        CV_fea_path = os.path.join(CV_PATH, cv_file_name)
        # ---------1.1读取单个视觉文件的特征-------
        CV_fea_one = []
        with open(CV_fea_path, "r") as file:
            CV_fea = file.readlines()
        for CV_fea_lines in CV_fea:
            temp = CV_fea_lines.split()
            if cv_file_name in CV_lack_pose:
                temp = ['0' if x == '-1.#IND,' or x == '-1.#IND' else x for x in temp]  # 将list里特定元素‘-1.#IND’替换成0
            CV_fea_one.append(temp)
        # ---------1.1------------
        CV_fea_all.append(CV_fea_one)
    # ---------1-------------
    if CV_fea_all.__len__() == 3:
        assert len(CV_fea_all[0]) == len(CV_fea_all[1]) == len(CV_fea_all[2])
        CV_fea_all_1 = [row[4:] for row in CV_fea_all[1]]  # 去除相同的'frame,', 'timestamp,', 'confidence,', 'success,'
        CV_fea_all_2 = [row[4:] for row in CV_fea_all[2]]  # 去除相同的'frame,', 'timestamp,', 'confidence,', 'success,'
        CV_fea_str = [a + b + c for a, b, c in zip(CV_fea_all[0], CV_fea_all_1, CV_fea_all_2)]  # 特征拼接到一起
        CV_fea_str = CV_fea_str[1:]
        CV_fea = [[float(cv_str.strip(',')) for cv_str in cv_list] for cv_list in CV_fea_str]
        timestamp = [row[1] for row in CV_fea]  # 各帧时间戳
        print(f"获取指定的视觉特征了，并且整合到变量：CV_fea了(所有帧) ！, 各帧时间戳在变量timestamp里！")
    elif CV_fea_all.__len__() == 2:
        assert len(CV_fea_all[0]) == len(CV_fea_all[1])
        CV_fea_all_1 = [row[4:] for row in CV_fea_all[1]]
        CV_fea_str = [a + b for a, b in zip(CV_fea_all[0], CV_fea_all_1)]
        CV_fea_str = CV_fea_str[1:]
        CV_fea = [[float(cv_str.strip(',')) for cv_str in cv_list] for cv_list in CV_fea_str]
        timestamp = [row[1] for row in CV_fea]
        print(f"获取指定的视觉特征了，并且整合到变量：CV_fea了(所有帧) ！, 各帧时间戳在变量timestamp里！")
    else:
        print("修改了suffixes里的个数，如果不是2个或则3个，记得这里也改一下哦！")
        sys.exit()

    sample_on_off_times = on_off_times[i]
    sample_audio_paragraph_features = audio_paragraph_features[i]

    # ------------------获取该样本 对应的段落特征！-----------------------------------------------------------------------
    cv_feature = []
    for j, audio_features_s in enumerate(sample_audio_paragraph_features):
        paragraph_time = sample_on_off_times[j]  # ----段落对应的戳时间范围
        features_num = audio_features_s.shape[0]      # 判断段落里特征的个数
        # -------------------------获取对齐的视觉特征（CV_fea_frame:(单词数, 38)）句子为单位---------------------------------
        # 初始化结果列表
        closest_timestamps1 = [ts for ts in timestamp if ts >= float(paragraph_time[0])]
        closest_timestamps2 = [ts for ts in timestamp if ts <= float(paragraph_time[1])]
        intersection = sorted(list(set(closest_timestamps1) & set(closest_timestamps2)))  # 可选取的帧
        if int(folder_list[i].split('_')[0]) in CV_frame_lack and intersection == []:   # 402 420 –在对话结束之前，视频记录被剪切。
            continue  # 缺少视觉模态，这些对话数据去除。

        time_frame = intersection
        # 获取所有视觉特征
        indices = [timestamp.index(value) for value in time_frame if value in timestamp]
        values = [CV_fea[index] for index in indices]
        CV_fea_frame = [row[4:] for row in values]
        CV_fea_frame = np.stack(CV_fea_frame)
        # # 检查是否为空
        # if len(CV_fea_frame) > 0:
        #     CV_fea_frame = np.stack(CV_fea_frame)
        # else:
        #     continue

        # --------视觉特征分配------
        rows_per_feature = CV_fea_frame.shape[0] // features_num  # 每份的基础行数
        extra_rows = CV_fea_frame.shape[0] % features_num  # 余数行数
        split_cv_frames_feature = []
        start_row = 0
        for pp in range(features_num):
            # 当前份额的行数：基础行数 + 如果有余数，前几份每份多分配一行
            current_rows = rows_per_feature + (1 if pp < extra_rows else 0)
            split_cv_frames_feature.append(CV_fea_frame[start_row:start_row + current_rows])
            start_row += current_rows

        # --------视觉特征形成-------
        # 初始化列表来存储最终的 1*76 特征向量
        cv_split_feature = []
        # 计算均值和方差，并拼接
        for frame in split_cv_frames_feature:
            frame_0 = frame[0]
            mean_vector = np.mean(frame, axis=0)  # 计算行上的均值，得到形状为 (38,)
            var_vector = np.var(frame, axis=0)  # 计算行上的方差，得到形状为 (38,)
            # 计算序列的自相关性
            autocorr_vectors = []
            for t in range(frame.shape[1]):
                feature_series = frame[:, t]  # 提取第 i 列（即第 i 个特征）
                # 计算该特征序列的自相关性，滞后为 1
                autocorr_value = autocorrelation(feature_series, lag=1)
                autocorr_vectors.append(autocorr_value)  # 保存自相关值
            autocorr_vector = np.array(autocorr_vectors)  # 将所有自相关性值组成向量

            # 将均值和方差拼接在一起，形成形状为 (76,) 的向量
            combined_vector = np.concatenate((frame_0, mean_vector, var_vector, autocorr_vector), axis=0)

            # 将拼接后的向量加入到 final_feature 列表
            cv_split_feature.append(combined_vector)

        # 将 final_feature 列表转换为一个 ndarray，形成形状为 (1, 76) 的最终特征向量
        cv_split_feature = np.array(cv_split_feature).reshape(len(cv_split_feature), -1)
        cv_feature.append(cv_split_feature)
    visual_paragraph_features.append(cv_feature)


folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph'
with open(os.path.join(folder_path, 'visual_paragraph_features.pickle'), 'wb') as f:
    pickle.dump(visual_paragraph_features, f)
print(f"视觉段落的特征处理完成，保存在{os.path.join(folder_path, 'visual_paragraph_features.pickle')}")