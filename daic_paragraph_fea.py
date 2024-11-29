# 作者：刘成广
# 时间：2024/4/23 下午2:43
# 作者：刘成广
# 时间：2024/4/22 下午10:39
import os
import pickle
import numpy as np
import utils.utilities as util
import sys
from zipfile import ZipFile


# ----------------------数据列表--------------------------------------------
DATASET = '/home/liu70kg/PycharmProjects/Depression/DAIC-woz/DAIC-woz-dataset'
TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH = os.path.join(DATASET, 'full_test_split.csv')
complete_Depression_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
# -------------------------------------------------------------------


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


folder_list, _, transcript_paths = get_meta_data(DATASET)


# 获得标签（id class score gen）
def data_to_array(file):
    data = util.csv_read(file)
    data = [lst for lst in data if lst]  # 原data是list，里面有空的list元素
    data_array = np.array([[d[0] for d in data], [d[1] for d in data],
                           [d[2] for d in data], [d[3] for d in
                                                  data]]).astype(int)
    return data_array, data


# 获取训练集、验证集、测试集的样本号，以及所有样本的信息（id class score gen）
train_labels,  _ = data_to_array(TRAIN_SPLIT_PATH)
train_list = train_labels[0]
valid_labels,  _ = data_to_array(DEV_SPLIT_PATH)
valid_list = valid_labels[0]
test_labels,  _ = data_to_array(TEST_SPLIT_PATH)
test_list = test_labels[0]
complete_labels, _ = data_to_array(complete_Depression_PATH)


#  读取处理好的特征实验一下
# features_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/fea_all_paragraph.pkl'
# with open(features_path, 'rb') as f:
#     # 使用pickle.load()方法从文件中加载数据
#     paragraph_features = pickle.load(f)
#     k=1

# -------------文本特征----------------
text_paragraph_features_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/text_paragraph_features.pickle'
with open(text_paragraph_features_path, 'rb') as f:
    # 使用pickle.load()方法从文件中加载数据
    text_paragraph_features = pickle.load(f)
    print("获取到测试者的说话段落的文本特征了，并且整合到变量：text_paragraph_features！")
# -------------语音特征----------------
audio_paragraph_features_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/audio_paragraph_features.pickle'
with open(audio_paragraph_features_path, 'rb') as f:
    # 使用pickle.load()方法从文件中加载数据
    audio_paragraph_features = pickle.load(f)
    print("获取到测试者的说话段落的语音特征了，并且整合到变量：audio_paragraph_features！")
# -------------视觉特征----------------
visual_paragraph_features_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/visual_paragraph_features.pickle'
with open(visual_paragraph_features_path, 'rb') as f:
    # 使用pickle.load()方法从文件中加载数据
    visual_paragraph_features = pickle.load(f)
    print("获取到测试者的说话段落的语音特征了，并且整合到变量：visual_paragraph_features！")


daic_fea_all = []
train_data = []
valid_data = []
train_valid_data = []
test_data = []
# 一个一个样本的处理
CV_frame_lack ={402, 420}  # –在对话结束之前，视频记录被剪切约2分钟。 所以缺少后续视频帧
for i, sample_visual_paragraph_features in enumerate(visual_paragraph_features):
    label = complete_labels[:, i]
    f = label[0]
    if f in CV_frame_lack:
        sample_audio_paragraph_features = audio_paragraph_features[i][:len(sample_visual_paragraph_features)]
        sample_text_paragraph_features = text_paragraph_features[i][:len(sample_visual_paragraph_features)]
    else:
        sample_audio_paragraph_features = audio_paragraph_features[i]
        sample_text_paragraph_features = text_paragraph_features[i]
    # -----------------------------取一个，可以考虑换--------------------------------
    visual_fea =[]
    for temp_features in sample_visual_paragraph_features:
        # visual_fea.append(temp_features[0])
        output = np.concatenate([temp_features[0], np.mean(temp_features, axis=0), np.max(temp_features, axis=0)])
        visual_fea.append(output)
    visual_fea = np.squeeze(np.vstack(visual_fea))

    audio_fea =[]
    for temp_features in sample_audio_paragraph_features:
        output = np.concatenate([temp_features[0], np.mean(temp_features, axis=0), np.max(temp_features, axis=0)])
        audio_fea.append(output)
    audio_fea = np.squeeze(np.vstack(audio_fea))
    txt_fea = np.squeeze(np.vstack(sample_text_paragraph_features))
    # -----------------------------文本特征复制，可以考虑换--------------------------------
    # txt_fea = []
    # for pr, temp_features in enumerate(sample_visual_paragraph_features):
    #     num = temp_features.shape[0]
    #     temp = np.tile(sample_text_paragraph_features[pr], (num, 1))
    #     txt_fea.append(temp)
    #
    # txt_fea = np.squeeze(np.vstack(txt_fea))
    # audio_fea = np.squeeze(np.vstack(sample_audio_paragraph_features))
    # visual_fea = np.squeeze(np.vstack(sample_visual_paragraph_features))
    # ----------------------------------------------------------------------------------

    # ------------------段落为单位的所有特征和信息组合在一起, paragraph_inforamtion------------------------------


    paragraph_inforamtion = [((visual_fea, audio_fea, txt_fea), label[1:], label[0])]
    daic_fea_all.extend(paragraph_inforamtion)
    if label[0] in train_list:
        train_data.extend(paragraph_inforamtion)
    elif label[0] in valid_list:
        valid_data.extend(paragraph_inforamtion)
    elif label[0] in test_list:
        test_data.extend(paragraph_inforamtion)
    else:
        raise ValueError(f"出错了啊，为什么样本{label[0]}不属于train_valid_test？")

    if label[0] in train_list or label[0] in valid_list:  # 训练和验证合并
        train_valid_data.extend(paragraph_inforamtion)
    print(f"--------------------文件 ：{folder_list[i]}处理成功，特征处理完成！-------------------------\n")

print(f"--------------------数据集DAIC-woz特征处理成功！在变量daic_fea_all里-------------------------")
#
#
# 保存为.pkl文件
description = "解释：(段落级视觉特征, 段落级语音特征, 段落级文本特征), (类别，分数，性别), (样本编号)"


daic_train_file = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/train_data_paragraph_concat.pkl'
daic_fea_train = {"train": train_data, "description": description}
with open(daic_train_file, 'wb') as file:
    pickle.dump(daic_fea_train, file)
print(f"数据集DAIC-woz训练部分 has been saved as {daic_train_file}.")

daic_valid_file = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/valid_data_paragraph_concat.pkl'
daic_fea_valid = {"valid": valid_data, "description": description}
with open(daic_valid_file, 'wb') as file:
    pickle.dump(daic_fea_valid, file)
print(f"数据集DAIC-woz验证部分 has been saved as {daic_valid_file}.")

daic_train_valid_file = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/train_valid_data_paragraph_concat.pkl'
daic_fea_train_valid = {"train_valid": train_valid_data, "description": description}
with open(daic_train_valid_file, 'wb') as file:
    pickle.dump(daic_fea_train_valid, file)
print(f"数据集DAIC-woz训练+验证部分 has been saved as {daic_train_valid_file}.")

daic_test_file = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/test_data_paragraph_concat.pkl'
daic_fea_test = {"test": test_data, "description": description}
with open(daic_test_file, 'wb') as file:
    pickle.dump(daic_fea_test, file)
print(f"数据集DAIC-woz测试部分 has been saved as {daic_test_file}.")


daic_fea_all_file = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/fea_all_paragraph_concat.pkl'
daic_fea = {"all": daic_fea_all, "description": description}
with open(daic_fea_all_file, 'wb') as file:
    pickle.dump(daic_fea, file)
print(f"数据集DAIC-woz全部部分 has been saved as {daic_fea_all_file}.")