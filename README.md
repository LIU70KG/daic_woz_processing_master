1：项目daic_woz_processing-master里的obtain_paragraph.py ：

第一步：读取数据集的原始数据（转录文本和语音），获取转录文本中被采访者每次回答的开始和结束时间（以每次回答为准，即问题级段落）。每次回答的开始和结束时间存储在文件on_off_times.pickle里。同时，记录每次回答的内容，即转录文本，存储在文件paragraph_all.pickle里。
注1：只记录单次回答大于1秒的内容，小于1秒的单次回复信息较少且不适应于后续VGGish提取特征，所以抛弃。
注2：特殊数据特殊处理，例如，只有被采访者的语音被转录的样本special_case_2 = [451, 458, 480]，将其各回答组成若干个大于1秒的段落。存在中断的样本special_case = {373: [395, 428],444: [286, 387]}内的内容选择跳过。时间错位的样本special_case_3 = {318: 34.319917, 321: 3.8379167, 341: 6.1892, 362: 16.8582}进行时间矫正。

第二步：依据on_off_times内的时间，将原始数据里的wav语音进行对应的截断。各样本的单次回答语音存储在audio_paragraph下的sample_paragraph文件夹；各样本中被采访者完整语音存储在audio_paragraph下的audio_data文件夹。

2：项目torchvggish-master里的main.py：
依据在YouTube-8M预训练的VGGish模型，对各样本的单次回答语音提取n*128的特征，特征保存在文件audio_paragraph_features.pickle。
为什么是n*128?因为1秒的语音特征是1*128，如果单次回答是5.3秒，那特征就是5*128。
# torchvggish-master
torchvggish-master下载入口：。
[VGGish](https://github.com/LIU70KG/torchvggish/tree/main/torchvggish-master)<sup>[1]</sup>, 

3：项目daic_woz_processing-master里的txt_paragraph_features.py：
依据在英语bert-base-uncased预训练的bert模型，对各样本的单次回答文本内容提取1*768的特征，特征保存在文件text_paragraph_features.pickle。

4：项目daic_woz_processing-master里的visual_paragraph_features.py
依据openface特征，提取特定的特征，例如suffixes = ['AUs.txt', 'gaze.txt', 'pose.txt']共38个。与audio_paragraph_features.pickle一样，提取对应的单次回答相关的多帧openface的视觉帧特征，例如n*38。计算这n帧的第一帧特征、均值、方差、序列的自相关性，获得1*（4*38）即1*152特征。特征保存在文件visual_paragraph_features.pickle。
注：小于1秒的回答，同样抛弃，[1,2)秒对应特征是1*152，如果单次回答是[2,n]秒，那特征就是n*152。
注：缺少后续视频帧的样本{402, 420}对应的语句跳过。所以，后续对齐需要以视觉特征为准。

5：项目daic_woz_processing-master里的daic_paragraph_fea.py
将各模态特征对齐，对于视觉和语音的特征，若段落特征>1，暂时只取第一个特征。从而各样本的模态特征为：（视觉：段落*152；语音：段落*128；文本：段落*768)。数据分为训练、验证、训练+验证、测试、全部。分别为：train_data_paragraph_?_.pkl、valid_data_paragraph_?_.pkl、train_valid_data_paragraph_?_.pkl、test_data_paragraph_?_.pkl、fea_all_paragraph_?_.pkl。

处理说明：(可选)
xxx.pkl:若视觉和语音的段落特征>1，暂时只取第一个特征。
xxx_all.pkl:若视觉和语音的段落特征>1，文本特征对应数量复制。
xxx_concat.pkl:若视觉和语音的段落特征>1，特征拼接：np.concatenate([temp_features[0], np.mean(temp_features, axis=0), np.max(temp_features, axis=0)])。（代码最终选择处理方式）
