# -*- coding: utf-8 -*-
import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter
import os

folder_path = 'jyxstxtqj_downcc.com'
floder_result_path='Zipf_Result'
def get_files():
    # 获取语料库路径
    with open(folder_path + '/inf.txt', encoding='gb18030', mode='r') as f:
        names = str(f.read())
        return names

def words_count(name):
    # 使用结巴分词并计数
    txt = open(folder_path + os.sep + name + '.txt', "r", encoding="gb18030").read()
    txt = txt.replace('\n','')
    txt = txt.replace('\u3000', '')
    txt = txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
    words = jieba.lcut(txt)#结巴分词
    counts = {}#设置初始字典
    for word in words:#开始遍历计数
        counts[word] = counts.get(word, 0) + 1
    return counts

def del_process(counts):
    #去除标点符号和停用词
    with open('cn_punctuation.txt', 'r', encoding='utf-8') as f:#读取标点符号
        punctuation = [line.strip() for line in f]
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:#读取停词
        stopwords = [line.strip() for line in f]
    for word in punctuation:#去除标点符号
        if (word in counts) == True:
            del counts[word]
    for word in stopwords:#去除停词
        if (word in counts) == True:
            del counts[word]
    return counts

def result_write(counts,name):
    #排序并计算结果
    items = list(counts.items())#返回遍历得分所有键与值
    items.sort(key=lambda x: x[1], reverse=True)#根据词出现次序进行排序
    sort_list = sorted(counts.values(), reverse=True)#sort_list用于绘图时的数据列表
    file = open(floder_result_path + os.sep + name + 'data.txt', mode='w', encoding='utf-8')#将数据写入txt文本
    for i in range(len(items)):#输出词语与词频
        word, count = items[i]
        new_context = word + "   " + str(count) + '\n'#写入txt文件
        file.write(new_context)
    file.close()
    return sort_list

def plot(sort_list,name):
    #用matplotlib验证Zipf-Law
    plt.title('Zipf-Law', fontsize=18)
    plt.xlabel('rank', fontsize=18)#排名
    plt.ylabel('freq', fontsize=18)#频率
    plt.yticks([pow(10, i) for i in range(0, 4)])#设置y刻度
    plt.xticks([pow(10, i) for i in range(0, 4)])#设置x刻度
    x = [i for i in range(len(sort_list))]
    plt.yscale('log')#设置坐标的缩放
    plt.xscale('log')
    plt.plot(x, sort_list, 'r')
    plt.savefig(floder_result_path+os.sep+name+'Zipf_Law.jpg')
    plt.clf()

if __name__ == '__main__':
    names=get_files()
    for name in names.split(','):
        counts=words_count(name)#读取文件并进行分词和计数
        counts=del_process(counts)#删除标点与停词
        sort_list=result_write(counts,name)#计算结果并写入文档
        plot(sort_list,name)#绘制图形Pyh