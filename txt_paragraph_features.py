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


def next_power_of_two(num):
    # 初始化 power 为 1
    power = 1
    # 不断将 power 乘以 2，直到它大于或等于 num
    while power < num:
        power *= 2
    return power


# folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/audio_paragraph_features.pickle'
# audio_paragraph_features = np.load(folder_path, allow_pickle=True)
folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/paragraph_all.pickle'
paragraph_all = np.load(folder_path, allow_pickle=True)
folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/on_off_times.pickle'
on_off_times = np.load(folder_path, allow_pickle=True)

# 初始化一个变量来存储所有句子的总长度
total_length = 0
# 计算所有句子的数量
sentence_count = 0
# 遍历 paragraph_all 里的每个子列表
for paragraph in paragraph_all:
    for sentence in paragraph:
        # 将每个句子的长度（以字符数计）加到总长度中
        total_length += len(sentence)
        # 增加句子计数器
        sentence_count += 1
# 计算平均句子长度
average_sentence_length = total_length / sentence_count if sentence_count > 0 else 0
closest_power = next_power_of_two(average_sentence_length)
# ---------------------bert提取特征-----------------------
local_model_path = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

text_paragraph_features = []
for i, paragraph in enumerate(paragraph_all):
    print(f"处理样本：{i}")
    sample_paragraph = []
    for sentence in paragraph:
        # Tokenize the input text and get input IDs and attention masks
        inputs = tokenizer(sentence, return_tensors='pt', max_length=closest_power, truncation=True, padding='max_length')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get the output from the BERT model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # The last hidden state is the embedding
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

        # To get a single vector per sentence, we can take the [CLS] token's embedding (first token)
        sentence_embedding = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

        # Convert to numpy array (optional)
        sentence_embedding_np = sentence_embedding.detach().cpu().numpy()  # Should be (1, 768)
        sample_paragraph.append(sentence_embedding_np)
    text_paragraph_features.append(sample_paragraph)

folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph'
with open(os.path.join(folder_path, 'text_paragraph_features.pickle'), 'wb') as f:
    pickle.dump(text_paragraph_features, f)
print(f"文本段落的特征处理完成，保存在{os.path.join(folder_path, 'text_paragraph_features.pickle')}")