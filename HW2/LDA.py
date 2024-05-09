import jieba
import re
import random
from random import randint
import numpy as np
from sklearn.svm import SVC
import torch

Data_path = "jyxstxtqj_downcc.com"
Topics=["天龙八部","鹿鼎记","神雕侠侣","笑傲江湖","倚天屠龙记","射雕英雄传",
        "书剑恩仇录","碧血剑","飞狐外传","侠客行","连城诀","雪山飞狐","白马啸西风",
        "三十三剑客图","鸳鸯刀","越女剑"]
N = 1000   # 段落数，做训练集
D = 500  # 每段的单词数
T = 10   #主题数

def data_preprocess_sample(topic,is_word):
    data = open("%s/%s.txt" % (Data_path, topic), "r", encoding="gb18030")
    data = data.read()
    data_sample = []
    print(len(data))
    para_sample=[]

    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
          '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        data = data.replace(a, '')
    print(len(data))
    if is_word:
        for word in data:
            data_sample.append(word)
    else:
        for words in jieba.cut(data):
            data_sample.append(words)
    for i in range(N//T):
        rand_value = randint(0, len(data_sample) - D)
        new_context = data_sample[rand_value:rand_value + D]
        #data_sample = random.sample(data_sample, N // T)
        para_sample.append(new_context)
    print(-N // T//10)
    # data_sample = [jieba.lcut(d) for d in data_sample]#使用结巴分词，当以字为单位建模时注释掉
    print(len(para_sample[-N // T//10:]))
    return para_sample[:-N // T//10], para_sample[-N // T//10:]

training_txt = []
testing_txt = []
for i in range(T):
    training_txt0, testing_txt0 = data_preprocess_sample(Topics[i],0)
    training_txt = training_txt + training_txt0
    testing_txt = testing_txt + testing_txt0

Topic_All = []  # N 个文章，每个文章中的 D 个词是什么 topic
Topic_count = {}  # T 个 topic，每个 topic 当中有多少个词
# T 个 topic，每个 topic 中每个词的词频
Topic_fre0 = {}
Topic_fre1 = {}
Topic_fre2 = {}
Topic_fre3 = {}
Topic_fre4 = {}
Topic_fre5 = {}
Topic_fre6 = {}
Topic_fre7 = {}
Topic_fre8 = {}
Topic_fre9 = {}
Topic_fre10 = {}
Topic_fre11 = {}
Topic_fre12 = {}
Topic_fre13 = {}
Topic_fre14 = {}
Topic_fre15 = {}
Doc_fre = []  # N 个文章，每个文章中，T 个 topic 的词各有多少
Doc_count = []  # N个文章，每个文章中有多少个词

i = 0
for data in training_txt:

    topic = []  # 500 个词，每个词初始化一个 topic
    docfre = {}
    for k in range(T):
        docfre[k] = docfre.get(k, 0) + 0  # 统计每篇文章的词频
        #print(docfre)

    for word in data:
        a = random.randint(0, T - 1)
        topic.append(a)
        if '\u4e00' <= word <= '\u9fa5':
            Topic_count[a] = Topic_count.get(a, 0) + 1  # 统计每个 topic 总词数
            docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
            exec('Topic_fre{}[word]=Topic_fre{}.get(word, 0) + 1'.format(i//(9*N//T//10),i//(9*N//T//10)))  # 统计每个topic的词频
    Topic_All.append(topic)
    temp_docfre=docfre
    docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
    Doc_fre.append(docfre)
    Doc_count.append(sum(docfre))  # [500, 500, ...] 可惜很多乱码无法识别，实际不到 500
    print(i, i//(9*N//T//10))
    i = i + 1

Topic_count = list(dict(sorted(Topic_count.items(), key=lambda x: x[0], reverse=False)).values())
Doc_fre = np.array(Doc_fre)  # 转为array方便后续计算
Topic_count = np.array(Topic_count)  # 转为array方便后续计算
Doc_count = np.array(Doc_count)  # 转为array方便后续计算

Doc_pro = []  # 每个topic被选中的概率
Doc_pronew = []  # 记录每次迭代后每个topic被选中的新概率

for i in range(len(training_txt)):
    doc = np.divide(Doc_fre[i], Doc_count[i])
    Doc_pro.append(doc)

Doc_pro = np.array(Doc_pro)

stop = 0  # 迭代停止标志
loopcount = 1  # 迭代次数
while stop == 0:
    i = 0  # 文章数目
    for data in training_txt:
        top = Topic_All[i]
        for w in range(len(data)):
            word = data[w]
            pro = []
            topfre = []
            if '\u4e00' <= word <= '\u9fa5':
                for j in range(T):
                   exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数

                pro = Doc_pro[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                Doc_fre[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                Doc_fre[i][m] += 1
                Topic_count[top[w]] -= 1  # 更新每个topic的总词数
                Topic_count[m] += 1
                exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) - 1'.format(top[w], top[w]))  # 更新每个topic该词的频数
                exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) + 1'.format(m, m))
                top[w] = m
        Topic_All[i] = top
        i += 1
    # print(Doc_fre, 'new')
    # print(Topic_count, 'new')
    if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
        for i in range(len(training_txt)):
            doc = np.divide(Doc_fre[i], Doc_count[i])
            Doc_pronew.append(doc)
        Doc_pronew = np.array(Doc_pronew)
    else:
        for i in range(len(training_txt)):
            doc = np.divide(Doc_fre[i], Doc_count[i])
            Doc_pronew[i] = doc
    # print(Doc_pro)
    # print(Doc_pronew)
    if (Doc_pronew == Doc_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为模型已经训练完毕
        stop = 1
    else:
        Doc_pro = Doc_pronew.copy()
    loopcount += 1
    print(loopcount)
print(Doc_pronew)  # 输出最终训练的到的每篇文章选中各个topic的概率
print(loopcount)  # 输出迭代次数
print(Topic_count)
print('模型训练完毕！')
# 模型实际训练出了不同 Topic 中每一个 Word 的词频以及总词数，也就是 Topic 中，Word 的概率分布，在测试中直接调用就行。

Doc_count_test = []  # 每篇文章中有多少个词
Doc_fre_test = []  # 每篇文章有多少各个topic的词
Topic_All_test = []  # 每篇文章中的每个词来自哪个topic
i = 0
for data in testing_txt:
    topic = []
    docfre = {}
    for k in range(T):
        docfre[k] = docfre.get(k, 0) + 0  # 统计每篇文章的词频
    for word in data:
        a = random.randint(0, T - 1)  # 为每个单词赋予一个随机初始topic
        topic.append(a)
        if '\u4e00' <= word <= '\u9fa5':
            docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
    Topic_All_test.append(topic)
    docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
    Doc_fre_test.append(docfre)
    Doc_count_test.append(sum(docfre))  # 统计每篇文章的总词数
    i += 1
# print(Topic_All[0])
Doc_fre_test = np.array(Doc_fre_test)
Doc_count_test = np.array(Doc_count_test)
# print(Doc_fre_test)
# print(Doc_count_test)

Doc_pro_test = []  # 每个topic被选中的概率
Doc_pronew_test = []  # 记录每次迭代后每个topic被选中的新概率
for i in range(len(testing_txt)):
    doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
    Doc_pro_test.append(doc)
Doc_pro_test = np.array(Doc_pro_test)
# print(Doc_pro_test)

stop = 0  # 迭代停止标志
loopcount = 1  # 迭代次数
while stop == 0:
    i = 0
    for data in testing_txt:
        top = Topic_All_test[i]
        for w in range(len(data)):
            word = data[w]
            pro = []
            topfre = []
            if '\u4e00' <= word <= '\u9fa5':
                for j in range(T):
                    exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数

                pro = Doc_pro_test[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                Doc_fre_test[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                Doc_fre_test[i][m] += 1
                top[w] = m
        Topic_All_test[i] = top
        i += 1
    # print(Doc_fre_test, 'new')
    if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
        for i in range(len(testing_txt)):
            doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
            Doc_pronew_test.append(doc)
        Doc_pronew_test = np.array(Doc_pronew_test)
    else:
        for i in range(len(testing_txt)):
            doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
            Doc_pronew_test[i] = doc
    # print(Doc_pro_test)
    # print(Doc_pronew_test)
    if (Doc_pronew_test == Doc_pro_test).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为训练集已分类完毕
        stop = 1
    else:
        Doc_pro_test = Doc_pronew_test.copy()
    loopcount += 1
print(Doc_pronew)
print(Doc_pronew_test)
print(loopcount)
print('测试集测试完毕！')


#为分类准备数据
# train_index=[i for i in range(int(9*N//10))]
# test_index=[i for i in range(int(N//10))]
train_index=[i for i in range(int(N//T*9//10*T))]
test_index=[i for i in range(int(N//T//10*T))]
train_index=[i for i in range(len(training_txt))]
test_index=[i for i in range(len(testing_txt))]

label=[]
for i in range(T):
    label=label+[i for j in range(int(len(training_txt)/T))]
labels = np.zeros(Doc_pronew.shape)
for i in range(len(label)):
    labels[i, label[i]] = 1

label_test=[]
for i in range(T):
    #label_test=label_test+[i for j in range(int(N//T//10))]
    label_test=label_test+[i for j in range(int(len(testing_txt)/T))]
labels_test=np.zeros(Doc_pronew_test.shape)
for i in range(len(label_test)):
    labels_test[i, label_test[i]] = 1

train_data = Doc_pronew[train_index, :]
train_label = labels[train_index]
test_data = Doc_pronew_test[test_index, :]
test_label = labels_test[test_index]

train_data = torch.from_numpy(train_data.astype(np.float32))
train_label = torch.from_numpy(train_label.astype(np.float32))
test_data = torch.from_numpy(test_data.astype(np.float32))
test_label = torch.from_numpy(test_label.astype(np.float32))

print(train_label.shape)
print(test_label.shape)


print("训练SVM分类器")
# train_label = np.array(train_label)
# test_label = np.array(test_label)
classifier = SVC(kernel='linear', probability=True)
classifier.fit(train_data, np.argmax(train_label.numpy(), 1))
print("训练集的精确度为： {:.4f}.".format(sum(classifier.predict(train_data) == np.argmax(train_label.numpy(), 1)) / len(np.argmax(train_label.numpy(), 1))))
#print("测试集的精确度为 {:.4f}.".format(sum(classifier.predict(test_data) == np.argmax(test_label.numpy(), 1)) / len(np.argmax(test_label.numpy(), 1))))
a=0
for i in range(10):
    a=a+sum(classifier.predict(test_data) == np.argmax(test_label.numpy(), 1)) / len(np.argmax(test_label.numpy(), 1))
    #print(a)
print("测试集的精确度为 {:.4f}.".format(a/10))
